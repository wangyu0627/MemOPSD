import gc
from logging import getLogger
from typing import Union
import torch
import os
from accelerate import Accelerator
from torch.utils.data import DataLoader

from genrec.dataset import AbstractDataset
from genrec.model import AbstractModel
from genrec.tokenizer import AbstractTokenizer
from genrec.utils import get_config, init_seed, init_logger, init_device, \
    get_dataset, get_tokenizer, get_model, get_trainer, log, config_for_log, \
    get_file_name, build_dataloader_kwargs


class Pipeline:
    def __init__(
        self,
        model_name: Union[str, AbstractModel],
        dataset_name: Union[str, AbstractDataset],
        checkpoint_path: str = None,
        tokenizer: AbstractTokenizer = None,
        trainer = None,
        config_dict: dict = None,
        config_file: str = None,
    ):
        self.config = get_config(
            model_name=model_name,
            dataset_name=dataset_name,
            config_file=config_file,
            config_dict=config_dict
        )
        self.checkpoint_path = checkpoint_path

        # Accelerator. Wandb is opt-in so shell scripts can run without login prompts.
        use_wandb = self._config_bool(self.config.get('use_wandb', False))
        self.config['use_wandb'] = use_wandb
        self.accelerator = Accelerator(
            log_with='wandb' if use_wandb else None,
            mixed_precision='no',
        )
        self.config['accelerator'] = self.accelerator
        self.config['device'] = self.accelerator.device  # use accelerate's device instead of init_device()
        self.config['use_ddp'] = (self.accelerator.num_processes > 1)

        # Seed and Logger
        init_seed(self.config['rand_seed'], self.config['reproducibility'])
        init_logger(self.config)
        self.logger = getLogger()
        self.log(f'Device: {self.config["device"]}')
        
        # Initialize wandb tracker only when explicitly requested.
        if use_wandb:
            wandb_project = self.config.get('wandb_project')
            if not wandb_project:
                raise ValueError('use_wandb=True requires wandb_project to be set.')
            wandb_group = f"{self.config['dataset']}-{self.config['model']}"
            wandb_run_name = self.config.get(
                'wandb_run_name', get_file_name(self.config, suffix='')
            )
            
            self.accelerator.init_trackers(
                project_name=wandb_project,
                config=config_for_log(self.config),
                init_kwargs={
                    "wandb": {
                        "name": wandb_run_name,
                        "group": wandb_group,
                        "tags": [self.config['dataset'], self.config['model']],
                    }
                },
            )

        # Dataset
        self.raw_dataset = get_dataset(dataset_name)(self.config)
        self.log(self.raw_dataset)
        self.split_datasets = self.raw_dataset.split()

        # Tokenizer
        if tokenizer is not None:
            self.tokenizer = tokenizer(self.config, self.raw_dataset)
        else:
            assert isinstance(model_name, str), 'Tokenizer must be provided if model_name is not a string.'
            self.tokenizer = get_tokenizer(model_name)(self.config, self.raw_dataset)
        self.tokenized_datasets = self.tokenizer.tokenize(self.split_datasets)

        # Model
        with self.accelerator.main_process_first():
            self.model = get_model(model_name)(self.config, self.raw_dataset, self.tokenizer)
            if checkpoint_path is not None:
                self._load_model_checkpoint(checkpoint_path)
                self.log(f'Loaded model checkpoint from {checkpoint_path}')
        self.log(self.model)
        self.log(self.model.n_parameters)

        # Trainer
        if trainer is not None:
            self.trainer = trainer
        else:
            self.trainer = get_trainer(model_name)(self.config, self.model, self.tokenizer, self.split_datasets)

    @staticmethod
    def _config_bool(value) -> bool:
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            return value.strip().lower() in {'1', 'true', 'yes', 'y', 'on'}
        return bool(value)

    def _resolve_tune_split(self):
        tune_split = str(self.config.get('tune_split', 'val')).strip().lower()
        aliases = {'valid': 'val', 'validation': 'val'}
        tune_split = aliases.get(tune_split, tune_split)
        if tune_split not in {'val', 'test'}:
            raise ValueError("--tune_split must be one of {'val', 'test'}.")
        return tune_split

    def _build_eval_dataloader(self, split):
        batch_size = (
            self.config.get('test_batch_size', self.config['eval_batch_size'])
            if split == 'test'
            else self.config['eval_batch_size']
        )
        return DataLoader(
            self.tokenized_datasets[split],
            batch_size=batch_size,
            shuffle=False,
            collate_fn=self.tokenizer.collate_fn[split],
            **build_dataloader_kwargs(
                self.config,
                split,
                dataset=self.tokenized_datasets[split],
                batch_size=batch_size,
            )
        )

    def run(self):
        # DataLoader
        test_dataloader = self._build_eval_dataloader('test')

        if self.config.get('eval_only', False):
            if self.checkpoint_path is None:
                raise ValueError('--eval_only=True requires --checkpoint_path=/path/to/checkpoint.pth')

            self.log(f'Running eval-only with checkpoint: {self.checkpoint_path}')
            self.model, test_dataloader = self.accelerator.prepare(
                self.model, test_dataloader
            )
            self.trainer.model = self.model
            test_results = self.trainer.evaluate(
                test_dataloader, split='test', step=0, epoch=0
            )

            if self.accelerator.is_main_process:
                for key in test_results:
                    self.accelerator.log({f'Test_Metric/{key}': test_results[key]})
            self.log(f'Test Results: {self.trainer._results_for_log(test_results)}')
            self.trainer.end()
            return

        train_dataloader = DataLoader(
            self.tokenized_datasets['train'],
            batch_size=self.config['train_batch_size'],
            shuffle=True,
            collate_fn=self.tokenizer.collate_fn['train'],
            **build_dataloader_kwargs(
                self.config,
                'train',
                dataset=self.tokenized_datasets['train'],
                batch_size=self.config['train_batch_size'],
            )
        )
        tune_split = self._resolve_tune_split()
        tune_dataloader = self._build_eval_dataloader(tune_split)
        self.config['active_tune_split'] = tune_split
        if tune_split == 'test':
            self.log(
                'Using test split for training-time checkpoint selection. '
                'This intentionally tunes on the test set.'
            )

        self.trainer.fit(train_dataloader, tune_dataloader)

        self.accelerator.wait_for_everyone()
        if self.config['use_ddp']:
            # make sure all processes have the same checkpoint path
            import torch.distributed as dist
            ckpt_path_container = [self.trainer.saved_model_ckpt]
            dist.broadcast_object_list(ckpt_path_container, src=0)
            self.trainer.saved_model_ckpt = ckpt_path_container[0]

        self._release_training_state_for_eval()
        train_dataloader = None
        tune_dataloader = None

        if self.config['load_best_ckpt'] and self.checkpoint_path is None:
            self._load_model_checkpoint(self.trainer.saved_model_ckpt)
            eval_epoch = self.trainer.best_epoch
            if self.accelerator.is_main_process:
                self.log(f'Loaded best model checkpoint from {self.trainer.saved_model_ckpt}')
        else:
            eval_epoch = self.trainer.last_epoch

        self.model, test_dataloader = self.accelerator.prepare(
            self.model, test_dataloader
        )
        self.trainer.model = self.model

        test_results = self.trainer.evaluate(
            test_dataloader, split='test',
            step=self.trainer.current_step, epoch=eval_epoch)

        if self.accelerator.is_main_process:
            for key in test_results:
                self.accelerator.log({f'Test_Metric/{key}': test_results[key]})
        self.log(f'Test Results: {self.trainer._results_for_log(test_results)}')

        self.trainer.end()

    def _release_training_state_for_eval(self):
        self.accelerator.wait_for_everyone()
        self.model = self.accelerator.unwrap_model(self.trainer.model)
        self.trainer.model = self.model
        self.accelerator.free_memory()
        self._clear_cuda_cache()

    def _load_model_checkpoint(self, checkpoint_path):
        state_dict = torch.load(checkpoint_path, map_location='cpu')
        self.model.load_state_dict(state_dict)
        del state_dict
        self._clear_cuda_cache()

    def _clear_cuda_cache(self):
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            try:
                torch.cuda.ipc_collect()
            except RuntimeError:
                pass

    def log(self, message, level='info'):
        return log(message, self.config['accelerator'], self.logger, level=level)

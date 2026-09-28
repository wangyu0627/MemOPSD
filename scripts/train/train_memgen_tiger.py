import argparse
import json
import os
import sys
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

SUPPORTED_MODELS = ("TIGER", "CARE", "LETTER", "LatentR3")
MODEL_NAMES = {model_name.upper(): model_name for model_name in SUPPORTED_MODELS}


@dataclass(frozen=True)
class TargetSpec:
    name: str
    data_dir: Path
    run_suffix: str


def normalize_model_name(model_name: str) -> str:
    normalized = MODEL_NAMES.get(str(model_name).upper())
    if normalized is None:
        raise ValueError(
            "model must be one of TIGER, CARE, LETTER, or LatentR3, "
            f"got {model_name!r}"
        )
    return normalized


def target_specs(
    data_root,
    target: str,
    model_name: str = "TIGER",
) -> list[TargetSpec]:
    data_root = Path(data_root)
    target = str(target).lower()
    model_suffix = normalize_model_name(model_name).lower()
    specs = {
        "memory": TargetSpec(
            name="memory",
            data_dir=data_root / "memory_tiger",
            run_suffix=f"memory_{model_suffix}",
        ),
        "generalize": TargetSpec(
            name="generalize",
            data_dir=data_root / "generalize_tiger",
            run_suffix=f"generalize_{model_suffix}",
        ),
    }
    if target == "both":
        return [specs["memory"], specs["generalize"]]
    if target not in specs:
        raise ValueError("target must be one of: memory, generalize, both")
    return [specs[target]]


def read_jsonl_rows(path) -> dict[str, list]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"MemGen TIGER data file not found: {path}")

    users = []
    item_seqs = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if "user" not in row or "item_seq" not in row:
                raise ValueError(f"{path}:{line_no} must contain user and item_seq.")
            item_seq = list(row["item_seq"])
            if len(item_seq) < 2:
                continue
            users.append(row["user"])
            item_seqs.append(item_seq)

    if not users:
        raise ValueError(f"No usable rows with len(item_seq) >= 2 in {path}.")
    return {"user": users, "item_seq": item_seqs}


class PreExpandedTIGERTokenizer:
    def _tokenize_preexpanded(self, examples: dict, split: str, indices=None) -> dict:
        all_input_ids, all_attention_mask, all_labels, all_indices = [], [], [], []
        for i in range(len(examples["user"])):
            cur_example = {
                "user": examples["user"][i],
                "item_seq": examples["item_seq"][i],
            }
            input_ids, attention_mask, labels = self._tokenize_once(cur_example)
            all_input_ids.append(input_ids)
            all_attention_mask.append(attention_mask)
            all_labels.append(labels)
            if indices is not None:
                all_indices.append(int(indices[i]))

        output = {
            "input_ids": all_input_ids,
            "attention_mask": all_attention_mask,
            "labels": all_labels,
        }
        if indices is not None:
            output["idx"] = all_indices
        return output

    def tokenize(self, datasets: dict) -> dict:
        batch_size = 128
        tokenized_datasets = {}
        for split in datasets:
            if split == "train":
                tokenized_datasets[split] = datasets[split].map(
                    lambda t, idx: self._tokenize_preexpanded(t, split, idx),
                    batched=True,
                    batch_size=batch_size,
                    with_indices=True,
                    remove_columns=datasets[split].column_names,
                    num_proc=self.config["num_proc"],
                    desc=f"Tokenizing pre-expanded {split} set: ",
                )
            else:
                tokenized_datasets[split] = datasets[split].map(
                    lambda t, idx: self.tokenize_function(t, split, idx),
                    batched=True,
                    batch_size=batch_size,
                    with_indices=True,
                    remove_columns=datasets[split].column_names,
                    num_proc=self.config["num_proc"],
                    desc=f"Tokenizing {split} set: ",
                )
        for split in datasets:
            tokenized_datasets[split].set_format(type="torch")
        return tokenized_datasets


class _DummyAccelerator:
    is_main_process = True
    device = "cpu"
    num_processes = 1

    @contextmanager
    def main_process_first(self):
        yield


def parse_command_line_overrides(unparsed: list[str]) -> dict:
    args = {}
    for text_arg in unparsed:
        if "=" not in text_arg:
            raise ValueError(f"Invalid command line argument: {text_arg}")
        key, value = text_arg.split("=", 1)
        key = key[len("--") :] if key.startswith("--") else key
        try:
            value = eval(value)
        except Exception:
            pass
        args[key] = value
    return args


def make_preexpanded_tokenizer_class(tokenizer_cls):
    class RuntimePreExpandedTokenizer(PreExpandedTIGERTokenizer, tokenizer_cls):
        pass

    return RuntimePreExpandedTokenizer


def dataset_from_rows(rows: dict):
    from datasets import Dataset

    return Dataset.from_dict(rows)


def load_target_split_datasets(target_dir: Path, splits: list[str]) -> dict:
    split_datasets = {}
    for split in splits:
        split_datasets[split] = dataset_from_rows(
            read_jsonl_rows(target_dir / f"{split}.jsonl")
        )
    if "train" not in split_datasets or "val" not in split_datasets:
        raise ValueError("Training requires train.jsonl and val.jsonl.")
    return split_datasets


def build_dataloader(config, tokenizer, tokenized_datasets, split: str, batch_size: int):
    from torch.utils.data import DataLoader
    from genrec.utils import build_dataloader_kwargs

    return DataLoader(
        tokenized_datasets[split],
        batch_size=batch_size,
        shuffle=(split == "train"),
        collate_fn=tokenizer.collate_fn[split],
        **build_dataloader_kwargs(
            config,
            split,
            dataset=tokenized_datasets[split],
            batch_size=batch_size,
        ),
    )


def train_one_target(args, config_overrides: dict, spec: TargetSpec):
    from accelerate import Accelerator
    import torch

    from genrec.utils import (
        get_config,
        get_dataset,
        get_model,
        get_tokenizer,
        get_trainer,
        init_logger,
        init_seed,
    )

    config = get_config(
        model_name=args.model,
        dataset_name=args.dataset,
        config_file=args.config,
        config_dict=config_overrides,
    )
    config["run_id"] = f"{config['run_id']}_{spec.run_suffix}"
    if args.ckpt_dir is not None:
        config["ckpt_dir"] = args.ckpt_dir
    if args.log_dir is not None:
        config["log_dir"] = args.log_dir

    use_wandb = _config_bool(config.get("use_wandb", False))
    accelerator = Accelerator(
        log_with="wandb" if use_wandb else None,
        mixed_precision="no",
    )
    config["accelerator"] = accelerator
    config["device"] = accelerator.device
    config["use_ddp"] = accelerator.num_processes > 1

    init_seed(config["rand_seed"], config["reproducibility"])
    init_logger(config)

    raw_dataset = get_dataset(args.dataset)(config)
    split_datasets = load_target_split_datasets(
        spec.data_dir,
        splits=[split.strip() for split in args.splits.split(",") if split.strip()],
    )

    tokenizer_cls = make_preexpanded_tokenizer_class(
        get_tokenizer(args.model)
    )
    tokenizer = tokenizer_cls(config, raw_dataset)
    tokenized_datasets = tokenizer.tokenize(split_datasets)

    with accelerator.main_process_first():
        model = get_model(args.model)(config, raw_dataset, tokenizer)
    trainer = get_trainer(args.model)(config, model, tokenizer, split_datasets)

    train_dataloader = build_dataloader(
        config,
        tokenizer,
        tokenized_datasets,
        "train",
        batch_size=config["train_batch_size"],
    )
    val_dataloader = build_dataloader(
        config,
        tokenizer,
        tokenized_datasets,
        "val",
        batch_size=config["eval_batch_size"],
    )
    trainer.fit(train_dataloader, val_dataloader)

    results = {
        "target": spec.name,
        "checkpoint": trainer.saved_model_ckpt,
        "train_rows": len(split_datasets["train"]),
        "val_rows": len(split_datasets["val"]),
    }
    if "test" in split_datasets and args.eval_test:
        if config["load_best_ckpt"] and os.path.exists(trainer.saved_model_ckpt):
            state_dict = torch.load(trainer.saved_model_ckpt, map_location="cpu")
            accelerator.unwrap_model(trainer.model).load_state_dict(state_dict)
        test_dataloader = build_dataloader(
            config,
            tokenizer,
            tokenized_datasets,
            "test",
            batch_size=config.get("test_batch_size", config["eval_batch_size"]),
        )
        results["test"] = trainer.evaluate(test_dataloader, split="test")
    return results


def _config_bool(value) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y", "on"}
    return bool(value)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description=(
            "Train Memory/Generalize TIGER, CARE, LETTER, or LatentR3 "
            "from MemGen JSONL data."
        )
    )
    parser.add_argument(
        "--model",
        type=normalize_model_name,
        choices=SUPPORTED_MODELS,
        default="TIGER",
    )
    parser.add_argument("--dataset", type=str, default="AmazonReviews2023")
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--memgen_data_dir", type=str, required=True)
    parser.add_argument(
        "--target",
        type=str,
        default="both",
        choices=["memory", "generalize", "both"],
    )
    parser.add_argument("--splits", type=str, default="train,val,test")
    parser.add_argument("--ckpt_dir", type=str, default=None)
    parser.add_argument("--log_dir", type=str, default=None)
    parser.add_argument("--eval_test", action="store_true")
    args, unparsed = parser.parse_known_args(argv)
    return args, parse_command_line_overrides(unparsed)


def main():
    args, config_overrides = parse_args()
    results = []
    for spec in target_specs(args.memgen_data_dir, args.target, args.model):
        results.append(train_one_target(args, config_overrides, spec))
    print(json.dumps(results, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

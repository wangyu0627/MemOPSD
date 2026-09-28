from logging import getLogger

import numpy as np
import torch
from torch.nn.utils import clip_grad_norm_
from torch.optim import AdamW
from tqdm import tqdm
from transformers.optimization import get_scheduler

from genrec.trainer import Trainer
from genrec.utils import get_total_steps, log


class S3RecTrainer(Trainer):
    def fit(self, train_dataloader, val_dataloader):
        optimizer = AdamW(
            self.model.parameters(),
            lr=self.config["lr"] * (self.accelerator.num_processes ** 0.5),
            weight_decay=self.config["weight_decay"],
        )

        finetune_steps = get_total_steps(self.config, train_dataloader)
        pretrain_epochs = self.config.get("pretrain_epochs", 0)
        pretrain_steps = pretrain_epochs * len(train_dataloader)
        total_n_steps = finetune_steps + pretrain_steps
        if total_n_steps == 0:
            self.log("No training steps needed.")
            return

        scheduler = get_scheduler(
            name="cosine",
            optimizer=optimizer,
            num_warmup_steps=self.config["warmup_steps"]
            // self.accelerator.num_processes,
            num_training_steps=total_n_steps // self.accelerator.num_processes,
        )

        self.model, optimizer, train_dataloader, val_dataloader, scheduler = (
            self.accelerator.prepare(
                self.model,
                optimizer,
                train_dataloader,
                val_dataloader,
                scheduler,
            )
        )

        self.current_step = 0
        if pretrain_epochs > 0:
            self._pretrain(train_dataloader, optimizer, scheduler, pretrain_epochs)

        n_epochs = np.ceil(
            finetune_steps / (len(train_dataloader) * self.accelerator.num_processes)
        ).astype(int)
        self.best_epoch = 0
        best_val_score = -1
        budget_epochs = self.config.get("budget_epochs", None)
        n_epochs = budget_epochs or n_epochs

        for epoch in range(n_epochs):
            self.model.train()
            total_loss = torch.zeros((), device=self.accelerator.device)
            train_progress_bar = tqdm(
                train_dataloader,
                total=len(train_dataloader),
                desc=f"Training - [Epoch {epoch + 1}]",
            )
            for batch in train_progress_bar:
                optimizer.zero_grad()
                outputs = self.model(batch)
                loss = outputs.loss
                self.accelerator.backward(loss)
                if self.config["max_grad_norm"] is not None:
                    clip_grad_norm_(
                        self.model.parameters(),
                        self.config["max_grad_norm"],
                )
                optimizer.step()
                scheduler.step()
                total_loss = total_loss + loss.detach()
                self.current_step += 1

            avg_train_loss = (total_loss / len(train_dataloader)).item()
            self.accelerator.log(
                {"Loss/train_loss": avg_train_loss},
                step=epoch + 1,
            )
            self.log(f"[Epoch {epoch + 1}] Train Loss: {avg_train_loss}")

            if (epoch + 1) % self.config["eval_interval"] == 0:
                all_results = self.evaluate(
                    val_dataloader,
                    split="val",
                    step=self.current_step,
                    epoch=epoch + 1,
                )
                if self.accelerator.is_main_process:
                    for key in all_results:
                        self.accelerator.log(
                            {f"Val_Metric/{key}": all_results[key]},
                            step=epoch + 1,
                        )
                    self.log(
                        f"[Epoch {epoch + 1}] Val Results: "
                        f"{self._results_for_log(all_results)}"
                    )
                val_score = all_results[self.config["val_metric"]]
                if val_score > best_val_score:
                    best_val_score = val_score
                    self.best_epoch = epoch + 1
                    if self.accelerator.is_main_process:
                        if self.config["use_ddp"]:
                            unwrapped_model = self.accelerator.unwrap_model(self.model)
                            torch.save(unwrapped_model.state_dict(), self.saved_model_ckpt)
                        else:
                            torch.save(self.model.state_dict(), self.saved_model_ckpt)
                        self.log(
                            f"[Epoch {epoch + 1}] Saved model checkpoint to "
                            f"{self.saved_model_ckpt}"
                        )

                if (
                    self.config["patience"] is not None
                    and epoch + 1 - self.best_epoch >= self.config["patience"]
                ):
                    self.log(f"Early stopping at epoch {epoch + 1}")
                    break

        self.last_epoch = epoch + 1
        self.log(f"Best epoch: {self.best_epoch}, Best val score: {best_val_score}")

        if self.do_fine_grained_eval and self.accelerator.is_main_process:
            self._log_eval_results_artifact()

    def _pretrain(self, train_dataloader, optimizer, scheduler, pretrain_epochs):
        loss_names = ["loss", "aap_loss", "mip_loss", "map_loss", "sp_loss"]
        for epoch in range(pretrain_epochs):
            self.model.train()
            losses = {
                name: torch.zeros((), device=self.accelerator.device)
                for name in loss_names
            }
            progress_bar = tqdm(
                train_dataloader,
                total=len(train_dataloader),
                desc=f"S3Rec Pretrain - [Epoch {epoch + 1}]",
            )
            for batch in progress_bar:
                optimizer.zero_grad()
                model = self.model.module if self.config["use_ddp"] else self.model
                outputs = model.pretrain(batch)
                self.accelerator.backward(outputs.loss)
                if self.config["max_grad_norm"] is not None:
                    clip_grad_norm_(
                        self.model.parameters(),
                        self.config["max_grad_norm"],
                    )
                optimizer.step()
                scheduler.step()
                self.current_step += 1

                for name in loss_names:
                    losses[name] = losses[name] + getattr(outputs, name).detach()

            averages = {
                name: (value / len(train_dataloader)).item()
                for name, value in losses.items()
            }
            self.accelerator.log(
                {f"S3RecPretrain/{k}": v for k, v in averages.items()},
                step=epoch + 1,
            )
            self.log(f"[S3Rec Pretrain Epoch {epoch + 1}] {averages}")

    def log(self, message, level="info"):
        return log(message, self.config["accelerator"], getLogger(), level=level)

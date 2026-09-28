import os
from logging import getLogger

import numpy as np
import torch
from torch.nn.utils import clip_grad_norm_
from torch.optim import AdamW
from tqdm import tqdm
from transformers.optimization import get_scheduler

from genrec.post_training import OPRDTrainer
from genrec.utils import get_total_steps, log


class LatentR3Trainer(OPRDTrainer):
    """LatentR3 GRPO by default, universal SFT/OPRD when explicitly selected."""

    legacy_config_prefix = "latentr3"
    stage_log_name = "LatentR3"

    def _uses_explicit_universal_stage(self) -> bool:
        explicit_keys = set(self.config.get("_explicit_config_keys", []))
        return "post_train_stage" in explicit_keys

    def fit(self, train_dataloader, val_dataloader):
        if self._uses_explicit_universal_stage():
            return super().fit(train_dataloader, val_dataloader)
        return self._fit_sft_then_grpo(train_dataloader, val_dataloader)

    def _fit_sft_then_grpo(self, train_dataloader, val_dataloader):
        if self._should_resume_from_sft_checkpoint():
            train_dataloader, val_dataloader = self._prepare_from_sft_checkpoint(
                train_dataloader,
                val_dataloader,
            )
        else:
            train_dataloader, val_dataloader = self._fit_sft(
                train_dataloader,
                val_dataloader,
            )
        if self.config.get("latentr3_use_grpo", True):
            self._load_sft_checkpoint_if_available()
            self._fit_grpo(train_dataloader, val_dataloader)
        elif self.accelerator.is_main_process:
            self.log("[LatentR3] GRPO disabled by config.")

    def _fit_sft(self, train_dataloader, val_dataloader):
        optimizer = AdamW(
            self.model.parameters(),
            lr=self.config["lr"] * (self.accelerator.num_processes ** 0.5),
            weight_decay=self.config["weight_decay"],
        )
        total_n_steps = get_total_steps(self.config, train_dataloader)
        if total_n_steps == 0:
            self.log("No SFT training steps needed.")
            self.best_epoch = 0
            self.last_epoch = 0
            self.current_step = 0
            self.model, train_dataloader, val_dataloader = self.accelerator.prepare(
                self.model,
                train_dataloader,
                val_dataloader,
            )
            return train_dataloader, val_dataloader

        scheduler = get_scheduler(
            name="cosine",
            optimizer=optimizer,
            num_warmup_steps=self.config["warmup_steps"]
            // self.accelerator.num_processes,
            num_training_steps=total_n_steps // self.accelerator.num_processes,
        )
        (
            self.model,
            optimizer,
            train_dataloader,
            val_dataloader,
            scheduler,
        ) = self.accelerator.prepare(
            self.model,
            optimizer,
            train_dataloader,
            val_dataloader,
            scheduler,
        )

        n_epochs = np.ceil(
            total_n_steps / (len(train_dataloader) * self.accelerator.num_processes)
        ).astype(int)
        n_epochs = self.config.get("budget_epochs", None) or n_epochs
        self.best_epoch = 0
        self.last_epoch = 0
        self.current_step = 0
        best_val_score = -1

        for epoch in range(n_epochs):
            self.model.train()
            total_loss = torch.zeros((), device=self.accelerator.device)
            progress = tqdm(
                train_dataloader,
                total=len(train_dataloader),
                desc=f"LatentR3 SFT - [Epoch {epoch + 1}]",
            )
            for batch in progress:
                optimizer.zero_grad()
                outputs = self.model(batch)
                loss = outputs.loss
                self.accelerator.backward(loss)
                if self.config["max_grad_norm"] is not None:
                    clip_grad_norm_(self.model.parameters(), self.config["max_grad_norm"])
                optimizer.step()
                scheduler.step()
                total_loss = total_loss + loss.detach()
                self.current_step += 1

            avg_sft_loss = (total_loss / len(train_dataloader)).item()
            self.accelerator.log(
                {"Loss/sft_train_loss": avg_sft_loss},
                step=epoch + 1,
            )
            self.log(
                f"[LatentR3 SFT Epoch {epoch + 1}] "
                f"Train Loss: {avg_sft_loss}"
            )

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
                        f"[LatentR3 SFT Epoch {epoch + 1}] "
                        f"Val Results: {self._results_for_log(all_results)}"
                    )
                val_score = all_results[self.config["val_metric"]]
                if val_score > best_val_score:
                    best_val_score = val_score
                    self.best_epoch = epoch + 1
                    self._save_current_model(
                        epoch + 1,
                        path=self.sft_saved_model_ckpt,
                        stage="SFT",
                    )
                    self._save_current_model(
                        epoch + 1,
                        path=self.saved_model_ckpt,
                        stage="SFT",
                    )

                if (
                    self.config["patience"] is not None
                    and epoch + 1 - self.best_epoch >= self.config["patience"]
                ):
                    self.log(f"Early stopping SFT at epoch {epoch + 1}")
                    break

        self.last_epoch = epoch + 1
        self.sft_best_score = best_val_score
        self.sft_best_epoch = self.best_epoch
        self.log(
            f"[LatentR3 SFT] Best epoch: {self.best_epoch}, "
            f"Best val score: {best_val_score}"
        )
        return train_dataloader, val_dataloader

    def _fit_grpo(self, train_dataloader, val_dataloader):
        grpo_epochs = self.config.get("latentr3_grpo_epochs", 1)
        if grpo_epochs <= 0:
            self.log("[LatentR3 GRPO] No GRPO epochs requested.")
            return

        unwrapped = self.accelerator.unwrap_model(self.model)
        self._set_grpo_trainable(unwrapped)
        trainable_params = [p for p in self.model.parameters() if p.requires_grad]
        if not trainable_params:
            self.log("[LatentR3 GRPO] No trainable parameters; skipping.")
            return

        optimizer = AdamW(
            trainable_params,
            lr=self.config.get("latentr3_grpo_lr", self.config["lr"])
            * (self.accelerator.num_processes ** 0.5),
            weight_decay=self.config.get("latentr3_grpo_weight_decay", 0.0),
        )
        total_steps = grpo_epochs * len(train_dataloader)
        scheduler = get_scheduler(
            name="cosine",
            optimizer=optimizer,
            num_warmup_steps=self.config.get("latentr3_grpo_warmup_steps", 0)
            // self.accelerator.num_processes,
            num_training_steps=max(1, total_steps // self.accelerator.num_processes),
        )
        optimizer, scheduler = self.accelerator.prepare(optimizer, scheduler)

        self.log(
            "[LatentR3 GRPO] Trainable parameters: "
            f"{sum(p.numel() for p in trainable_params)}"
        )
        self.log(
            "[LatentR3 GRPO] Reward type: "
            f"{self.config.get('latentr3_grpo_reward_type', 'token_match')}"
        )
        best_grpo_score = self._initial_grpo_best_score(val_dataloader)
        grpo_best_epoch = 0
        last_grpo_epoch = 0
        grpo_patience = self.config.get("latentr3_grpo_patience", 1)
        grpo_min_delta = self.config.get("latentr3_grpo_min_delta", 0.0)

        for epoch in range(grpo_epochs):
            last_grpo_epoch = epoch + 1
            self.model.train()
            total_loss = torch.zeros((), device=self.accelerator.device)
            total_reward = torch.zeros((), device=self.accelerator.device)
            progress = tqdm(
                train_dataloader,
                total=len(train_dataloader),
                desc=f"LatentR3 GRPO - [Epoch {epoch + 1}]",
            )
            for batch in progress:
                optimizer.zero_grad()
                loss, mean_reward = self._grpo_batch_loss(batch)
                self.accelerator.backward(loss)
                if self.config["max_grad_norm"] is not None:
                    clip_grad_norm_(trainable_params, self.config["max_grad_norm"])
                optimizer.step()
                scheduler.step()
                total_loss = total_loss + loss.detach()
                total_reward = total_reward + mean_reward.detach()
                self.current_step += 1

            log_epoch = self.last_epoch + epoch + 1
            avg_grpo_loss = (total_loss / len(train_dataloader)).item()
            avg_grpo_reward = (total_reward / len(train_dataloader)).item()
            self.accelerator.log(
                {
                    "Loss/grpo_loss": avg_grpo_loss,
                    "Reward/grpo_reward": avg_grpo_reward,
                },
                step=log_epoch,
            )
            self.log(
                f"[LatentR3 GRPO Epoch {epoch + 1}] "
                f"Loss: {avg_grpo_loss}, "
                f"Reward: {avg_grpo_reward}"
            )

            all_results = self.evaluate(
                val_dataloader,
                split="val",
                step=self.current_step,
                epoch=log_epoch,
            )
            if self.accelerator.is_main_process:
                for key in all_results:
                    self.accelerator.log(
                        {f"Val_Metric/{key}": all_results[key]},
                        step=log_epoch,
                    )
                self.log(
                    f"[LatentR3 GRPO Epoch {epoch + 1}] "
                    f"Val Results: {self._results_for_log(all_results)}"
                )

            val_score = all_results[self.config["val_metric"]]
            if val_score > best_grpo_score + grpo_min_delta:
                best_grpo_score = val_score
                grpo_best_epoch = epoch + 1
                self.best_epoch = log_epoch
                self._save_current_model(log_epoch, stage="GRPO")
            elif (
                grpo_patience is not None
                and epoch + 1 - grpo_best_epoch >= grpo_patience
            ):
                self.log(
                    f"[LatentR3 GRPO] Early stopping at epoch {epoch + 1} "
                    f"after {grpo_patience} epoch(s) without improvement."
                )
                break

        self.last_epoch = self.last_epoch + last_grpo_epoch
        self.log(
            f"[LatentR3 GRPO] Best epoch: {grpo_best_epoch}, "
            f"Best val score: {best_grpo_score}"
        )
        if grpo_best_epoch == 0:
            self.log(
                "[LatentR3 GRPO] No GRPO checkpoint improved over the SFT "
                "baseline; restoring SFT checkpoint before continuing."
            )
            self._load_sft_checkpoint_if_available()

    def _initial_grpo_best_score(self, val_dataloader):
        if hasattr(self, "sft_best_score"):
            self.log(
                "[LatentR3 GRPO] Using SFT best score as GRPO baseline: "
                f"{self.sft_best_score}"
            )
            return self.sft_best_score

        if self.config.get("latentr3_grpo_eval_sft_baseline", True):
            results = self.evaluate(
                val_dataloader,
                split="val",
                step=self.current_step,
                epoch=self.last_epoch,
            )
            score = results[self.config["val_metric"]]
            self.log(
                "[LatentR3 GRPO] Evaluated loaded SFT checkpoint baseline: "
                f"{self._results_for_log(results)}"
            )
            return score

        return -1

    def _set_grpo_trainable(self, model):
        train_backbone = self.config.get("latentr3_grpo_train_backbone", False)
        if train_backbone:
            for param in model.parameters():
                param.requires_grad = True
            return

        for param in model.t5.parameters():
            param.requires_grad = False
        for param in model.attention.parameters():
            param.requires_grad = True

    def _grpo_batch_loss(self, batch):
        model_for_sampling = self.accelerator.unwrap_model(self.model)
        n_generations = self.config.get("latentr3_grpo_num_generations", 8)
        temperature = self.config.get("latentr3_grpo_temperature", 1.0)
        epsilon = self.config.get("latentr3_grpo_epsilon", 0.2)

        was_training = self.model.training
        self.model.eval()
        try:
            completions, _ = model_for_sampling.sample(
                batch=batch,
                n_return_sequences=n_generations,
                temperature=temperature,
            )
            with torch.no_grad():
                old_logps = self.model(
                    batch,
                    grpo_sequences=completions,
                    grpo_temperature=temperature,
                ).logps.sum(dim=-1)

            rewards = self._rewards(batch, completions, old_logps)
            advantages = self._group_advantages(rewards)
            current_logps = self.model(
                batch,
                grpo_sequences=completions,
                grpo_temperature=temperature,
            ).logps.sum(dim=-1)
        finally:
            if was_training:
                self.model.train()

        ratio = torch.exp(current_logps - old_logps)
        clipped_ratio = torch.clamp(ratio, 1.0 - epsilon, 1.0 + epsilon)
        objective = torch.min(ratio * advantages, clipped_ratio * advantages)
        loss = -objective.mean()
        return loss, rewards.mean()

    def _rewards(self, batch, completions, old_logps):
        reward_type = self.config.get("latentr3_grpo_reward_type", "token_match")
        if reward_type == "likelihood":
            return -torch.exp(-old_logps / max(1, completions.size(-1)))
        if reward_type == "token_match":
            targets = self._target_tokens(batch, completions)
            return completions.eq(targets).float().mean(dim=-1)
        return self._exact_match_rewards(batch, completions)

    def _exact_match_rewards(self, batch, completions):
        targets = self._target_tokens(batch, completions)
        return completions.eq(targets).all(dim=-1).float()

    def _target_tokens(self, batch, completions):
        targets = batch["labels"][:, : self.accelerator.unwrap_model(self.model).n_digit]
        return targets.unsqueeze(1).expand_as(completions)

    def _group_advantages(self, rewards):
        advantages = rewards - rewards.mean(dim=1, keepdim=True)
        if self.config.get("latentr3_grpo_normalize_advantages", True):
            advantages = advantages / rewards.std(
                dim=1,
                keepdim=True,
                unbiased=False,
            ).clamp(min=1e-6)
        return advantages

    def _load_sft_checkpoint_if_available(self):
        self.accelerator.wait_for_everyone()
        if not os.path.exists(self.sft_saved_model_ckpt):
            stage = "OPRD" if self._uses_explicit_universal_stage() else "GRPO"
            self.log(
                f"[LatentR3 {stage}] No SFT checkpoint found; continuing from "
                "current SFT parameters.",
                level="warning",
            )
            return

        target_model = self.accelerator.unwrap_model(self.model)
        state_dict = torch.load(
            self.sft_saved_model_ckpt,
            map_location="cpu",
        )
        target_model.load_state_dict(state_dict)
        del state_dict
        self.accelerator.wait_for_everyone()
        stage = "OPRD" if self._uses_explicit_universal_stage() else "GRPO"
        self.log(
            f"[LatentR3 {stage}] Loaded SFT checkpoint before {stage}: "
            f"{self.sft_saved_model_ckpt}"
        )

    def _should_resume_from_sft_checkpoint(self):
        if not self._stage_config("skip_sft_if_checkpoint", True):
            return False
        if not os.path.exists(self.sft_saved_model_ckpt):
            return False
        return self._checkpoint_matches_current_model(self.sft_saved_model_ckpt)

    def _checkpoint_matches_current_model(self, path):
        try:
            state_dict = torch.load(path, map_location="cpu")
        except Exception as error:
            self.log(
                "[LatentR3] Could not inspect SFT checkpoint; will retrain SFT: "
                f"{path} ({error})",
                level="warning",
            )
            return False

        target_model = self.accelerator.unwrap_model(self.model)
        expected = target_model.state_dict()
        for key in ("t5.shared.weight", "t5.lm_head.weight"):
            if key in state_dict and key in expected:
                if tuple(state_dict[key].shape) != tuple(expected[key].shape):
                    self.log(
                        "[LatentR3] Existing SFT checkpoint was created by an "
                        "incompatible LatentR3 architecture; retraining SFT "
                        f"instead of resuming: {path}",
                        level="warning",
                    )
                    return False
        return True

    def _prepare_from_sft_checkpoint(self, train_dataloader, val_dataloader):
        self._load_state_dict_from_sft_checkpoint()
        self.model, train_dataloader, val_dataloader = self.accelerator.prepare(
            self.model,
            train_dataloader,
            val_dataloader,
        )
        self.best_epoch = 0
        self.last_epoch = 0
        self.current_step = 0
        self.log(
            "[LatentR3] Found existing SFT checkpoint; skipped SFT and will "
            f"continue from {self.sft_saved_model_ckpt}"
        )
        return train_dataloader, val_dataloader

    def _load_state_dict_from_sft_checkpoint(self):
        if not os.path.exists(self.sft_saved_model_ckpt):
            raise FileNotFoundError(
                "LatentR3 OPRD requires a trained SFT checkpoint. "
                "Set --sft_checkpoint_path=/path/to/checkpoint.pth or train "
                f"SFT first. Missing: {self.sft_saved_model_ckpt}"
            )
        target_model = self.accelerator.unwrap_model(self.model)
        state_dict = torch.load(
            self.sft_saved_model_ckpt,
            map_location="cpu",
        )
        target_model.load_state_dict(state_dict)
        del state_dict

    def _save_current_model(self, epoch, path=None, stage=None):
        if not self.accelerator.is_main_process:
            return
        path = path or self.saved_model_ckpt
        if self.config["use_ddp"]:
            model = self.accelerator.unwrap_model(self.model)
            torch.save(model.state_dict(), path)
        else:
            torch.save(self.model.state_dict(), path)
        prefix = f"[{stage} Epoch {epoch}]" if stage else f"[Epoch {epoch}]"
        self.log(f"{prefix} Saved model checkpoint to {path}")

    def log(self, message, level="info"):
        return log(message, self.config["accelerator"], getLogger(), level=level)

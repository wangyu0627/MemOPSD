import ast
import csv
import json
import math
import os
import time
from logging import getLogger

import numpy as np
import torch
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_
from torch.optim import AdamW
from tqdm import tqdm
from transformers.optimization import get_scheduler

from genrec.trainer import Trainer
from genrec.utils import get_total_steps, log
from genrec.post_training.sid_kd import confidence_weight, soft_sid_targets_from_topk


class OPRDTrainer(Trainer):
    legacy_config_prefix = None
    stage_log_name = "OPRD"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        ckpt_root, ckpt_ext = os.path.splitext(self.saved_model_ckpt)
        configured_sft_ckpt = self._stage_config("sft_checkpoint_path", None)
        self.sft_saved_model_ckpt = configured_sft_ckpt or f"{ckpt_root}.sft{ckpt_ext}"
        self.oprd_teacher = None
        self.memory_oprd_teacher = None
        self.memory_oprd_label_table = None
        self.sasrec_sid_kd_table = None

    def _stage_config(self, key, default=None):
        explicit_keys = set(self.config.get("_explicit_config_keys", []))
        generic_value = self.config.get(key, None)
        if key in explicit_keys and generic_value is not None:
            return generic_value

        legacy_key = self._legacy_key_for(key)
        if legacy_key is not None and self.config.get(legacy_key, None) is not None:
            return self.config.get(legacy_key, default)
        if generic_value is not None:
            return generic_value
        return default

    def _legacy_key_for(self, key):
        if not self.legacy_config_prefix:
            return None
        if key == "post_train_stage":
            return f"{self.legacy_config_prefix}_train_stage"
        if key == "sft_checkpoint_path":
            return f"{self.legacy_config_prefix}_sft_checkpoint_path"
        if key == "skip_sft_if_checkpoint":
            return f"{self.legacy_config_prefix}_skip_sft_if_checkpoint"
        if key.startswith("oprd_"):
            return f"{self.legacy_config_prefix}_opd_{key[len('oprd_'):]}"
        return f"{self.legacy_config_prefix}_{key}"

    def _normalize_stage_name(self, stage):
        stage = str(stage).lower()
        if stage == "opd":
            return "oprd"
        if stage == "sft_opd":
            return "sft_oprd"
        return stage

    def _prepare_sft_auxiliary_state(self, total_n_steps):
        """Extension hook for read-only SFT diagnostics.

        The default implementation is intentionally a no-op so ordinary SFT
        runs retain their existing data flow and behavior.
        """

    def _before_sft_optimizer_step(self, step, epoch, total_n_steps, batch=None):
        """Run an optional diagnostic before the standard SFT update."""

    def _prepare_oprd_auxiliary_state(self, total_n_steps):
        """Extension hook for read-only OPRD diagnostics."""

    def _before_oprd_optimizer_step(
        self,
        step,
        epoch,
        total_n_steps,
        batch=None,
    ):
        """Run an optional diagnostic before the OPRD update."""

    def _after_oprd_training(self, step, epoch, total_n_steps):
        """Finalize OPRD diagnostics before any checkpoint fallback restore."""

    def fit(self, train_dataloader, val_dataloader):
        raw_stage = str(self._stage_config("post_train_stage", "sft")).lower()
        valid_stages = {"sft", "opd", "oprd", "sft_opd", "sft_oprd"}
        if raw_stage not in valid_stages:
            raise ValueError(
                "post_train_stage must be one of "
                f"{sorted(valid_stages)}, got {raw_stage}."
            )
        stage = self._normalize_stage_name(raw_stage)

        if stage in {"sft", "sft_oprd"}:
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
        else:
            train_dataloader, val_dataloader = self._prepare_oprd_only(
                train_dataloader,
                val_dataloader,
            )

        if stage in {"oprd", "sft_oprd"}:
            self._load_sft_checkpoint_if_available()
            self._fit_oprd(train_dataloader, val_dataloader)

    def _fit_sft(self, train_dataloader, val_dataloader):
        optimizer = AdamW(
            self.model.parameters(),
            lr=self.config["lr"] * (self.accelerator.num_processes ** 0.5),
            weight_decay=self.config["weight_decay"],
        )
        total_n_steps = get_total_steps(self.config, train_dataloader)
        if total_n_steps == 0:
            self.best_epoch = 0
            self.last_epoch = 0
            self.current_step = 0
            self.model, train_dataloader, val_dataloader = self.accelerator.prepare(
                self.model,
                train_dataloader,
                val_dataloader,
            )
            self.log(f"[{self.stage_log_name} SFT] No SFT training steps needed.")
            return train_dataloader, val_dataloader

        scheduler = get_scheduler(
            name="cosine",
            optimizer=optimizer,
            num_warmup_steps=self.config["warmup_steps"] // self.accelerator.num_processes,
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
        planned_sft_steps = int(n_epochs) * len(train_dataloader)
        self._prepare_sft_auxiliary_state(total_n_steps=planned_sft_steps)
        self.best_epoch = 0
        self.last_epoch = 0
        self.current_step = 0
        best_val_score = -1
        tune_split = self._active_tune_split()
        tune_label = self._split_log_label(tune_split)

        for epoch in range(n_epochs):
            self.model.train()
            total_loss = torch.zeros((), device=self.accelerator.device)
            progress = tqdm(
                train_dataloader,
                total=len(train_dataloader),
                desc=f"{self.stage_log_name} SFT - [Epoch {epoch + 1}]",
            )
            for batch in progress:
                optimizer.zero_grad()
                self._before_sft_optimizer_step(
                    step=self.current_step,
                    epoch=epoch + 1,
                    total_n_steps=planned_sft_steps,
                    batch=batch,
                )
                loss, _, _ = self._sft_batch_loss(batch)
                self.accelerator.backward(loss)
                if self.config["max_grad_norm"] is not None:
                    clip_grad_norm_(self.model.parameters(), self.config["max_grad_norm"])
                optimizer.step()
                scheduler.step()
                total_loss = total_loss + loss.detach()
                self.current_step += 1

            avg_sft_loss = (total_loss / len(train_dataloader)).item()
            self.accelerator.log(
                {f"Loss/{self.stage_log_name.lower()}_sft_train_loss": avg_sft_loss},
                step=epoch + 1,
            )
            self.log(
                f"[{self.stage_log_name} SFT Epoch {epoch + 1}] "
                f"Train Loss: {avg_sft_loss}"
            )

            if (epoch + 1) % self.config["eval_interval"] == 0:
                all_results = self.evaluate(
                    val_dataloader,
                    split=tune_split,
                    step=self.current_step,
                    epoch=epoch + 1,
                )
                if self.accelerator.is_main_process:
                    for key in all_results:
                        self.accelerator.log(
                            {f"{tune_label}_Metric/{key}": all_results[key]},
                            step=epoch + 1,
                        )
                    self.log(
                        f"[{self.stage_log_name} SFT Epoch {epoch + 1}] "
                        f"{tune_label} Results: {self._results_for_log(all_results)}"
                    )
                val_score = all_results[self.config["val_metric"]]
                if val_score > best_val_score:
                    best_val_score = val_score
                    self.best_epoch = epoch + 1
                    self._save_current_model(
                        epoch + 1,
                        path=self.sft_saved_model_ckpt,
                        stage=f"{self.stage_log_name} SFT",
                    )
                    self._save_current_model(
                        epoch + 1,
                        path=self.saved_model_ckpt,
                        stage=f"{self.stage_log_name} SFT",
                    )

                if (
                    self.config["patience"] is not None
                    and epoch + 1 - self.best_epoch >= self.config["patience"]
                ):
                    self.log(
                        f"[{self.stage_log_name} SFT] Early stopping at epoch {epoch + 1}"
                    )
                    break

        self.last_epoch = epoch + 1
        self.sft_best_score = best_val_score
        self.sft_best_epoch = self.best_epoch
        self.log(
            f"[{self.stage_log_name} SFT] Best epoch: {self.best_epoch}, "
            f"Best val score: {best_val_score} "
            f"(selection split: {tune_split})"
        )
        return train_dataloader, val_dataloader

    def _fit_oprd(self, train_dataloader, val_dataloader):
        self._validate_memory_controls()
        oprd_epochs = int(self._stage_config("oprd_epochs", 1))
        if oprd_epochs <= 0:
            self.log(f"[{self.stage_log_name} OPRD] No OPRD epochs requested.")
            return

        oprd_started_at = time.perf_counter()
        ce_weight = float(self._stage_config("oprd_ce_weight", 1.0))
        memory_logit_weight = float(self._stage_config("memory_oprd_logit_weight", 0.0))
        memory_ce_weight = float(self._stage_config("memory_oprd_ce_weight", 0.0))
        oprd_lr = float(self._stage_config("oprd_lr", self.config["lr"]))
        oprd_num_rollouts = int(self._stage_config("oprd_num_rollouts", 1))
        if self._use_memory_oprd_teacher() and memory_logit_weight > 0:
            self.memory_oprd_teacher = self._build_memory_oprd_teacher()
            self.oprd_teacher = None
        else:
            self.oprd_teacher = None
            self.memory_oprd_teacher = None

        unwrapped = self.accelerator.unwrap_model(self.model)
        self._set_oprd_trainable(unwrapped)
        trainable_params = [p for p in self.model.parameters() if p.requires_grad]
        if not trainable_params:
            raise ValueError("OPRD trainability left no trainable parameters.")

        optimizer = AdamW(
            trainable_params,
            lr=oprd_lr * (self.accelerator.num_processes ** 0.5),
            weight_decay=float(self._stage_config("oprd_weight_decay", 0.0)),
        )
        total_steps = oprd_epochs * len(train_dataloader)
        scheduler = get_scheduler(
            name="cosine",
            optimizer=optimizer,
            num_warmup_steps=int(self._stage_config("oprd_warmup_steps", 0))
            // self.accelerator.num_processes,
            num_training_steps=max(1, total_steps // self.accelerator.num_processes),
        )
        optimizer, scheduler = self.accelerator.prepare(optimizer, scheduler)
        self._prepare_oprd_auxiliary_state(total_n_steps=total_steps)

        self.log(
            f"[{self.stage_log_name} OPRD] Trainable parameters: "
            f"{sum(p.numel() for p in trainable_params)}"
        )
        self.log(
            f"[{self.stage_log_name} OPRD] Loss weights: "
            f"Logit={memory_logit_weight}, LR={oprd_lr}, "
            f"CE={ce_weight}, Rollouts={oprd_num_rollouts}"
        )
        self.log(
            f"[{self.stage_log_name} OPRD] Memory controls: "
            + json.dumps({
                "memory_oprd_ce_weight": memory_ce_weight,
                "memory_oprd_logit_weight": memory_logit_weight,
                "memory_oprd_normalization": self._memory_normalization(),
                "memory_oprd_non_memory": self._stage_config("memory_oprd_non_memory", "skip"),
                "memory_oprd_teacher_path": self._stage_config("memory_oprd_teacher_path", None),
                "memory_oprd_labels_path": self._stage_config("memory_oprd_labels_path", None),
                "sft_checkpoint_path": self.sft_saved_model_ckpt,
                "memory_oprd_temperature": self._stage_config("memory_oprd_temperature", 2.0),
                "oprd_rollout_temperature": self._stage_config("oprd_rollout_temperature", 1.0),
                "backbone_temperature": self.config.get("temperature", 1.0),
                "oprd_ce_weight": ce_weight,
                "oprd_num_rollouts": oprd_num_rollouts,
                "oprd_lr": oprd_lr,
                "oprd_epochs": oprd_epochs,
                "oprd_patience": self._stage_config("oprd_patience", 1),
                "rand_seed": self.config.get("rand_seed"),
                "train_batch_size": self.config.get("train_batch_size"),
                "planned_optimizer_steps": total_steps,
            }, sort_keys=True)
        )
        best_oprd_score = self._initial_oprd_best_score(val_dataloader)
        oprd_best_epoch = 0
        last_oprd_epoch = 0
        oprd_patience = self._stage_config("oprd_patience", 1)
        oprd_min_delta = float(self._stage_config("oprd_min_delta", 0.0))
        tune_split = self._active_tune_split()
        tune_label = self._split_log_label(tune_split)
        oprd_step = 0

        for epoch in range(oprd_epochs):
            last_oprd_epoch = epoch + 1
            self.model.train()
            total_loss = torch.zeros((), device=self.accelerator.device)
            total_ce_loss = torch.zeros((), device=self.accelerator.device)
            total_memory_kd_loss = torch.zeros((), device=self.accelerator.device)
            total_memory_ce_loss = torch.zeros((), device=self.accelerator.device)
            progress = tqdm(
                train_dataloader,
                total=len(train_dataloader),
                desc=f"{self.stage_log_name} OPRD - [Epoch {epoch + 1}]",
            )
            for batch in progress:
                optimizer.zero_grad()
                self._before_oprd_optimizer_step(
                    step=oprd_step,
                    epoch=epoch + 1,
                    total_n_steps=total_steps,
                    batch=batch,
                )
                (
                    loss,
                    ce_loss,
                    memory_kd_loss,
                ) = self._oprd_batch_loss(batch)
                self.accelerator.backward(loss)
                if self.config["max_grad_norm"] is not None:
                    clip_grad_norm_(trainable_params, self.config["max_grad_norm"])
                optimizer.step()
                scheduler.step()
                total_loss = total_loss + loss.detach()
                total_ce_loss = total_ce_loss + ce_loss.detach()
                total_memory_kd_loss = total_memory_kd_loss + memory_kd_loss.detach()
                total_memory_ce_loss = total_memory_ce_loss + self._last_memory_ce_loss
                self.current_step += 1
                oprd_step += 1

            log_epoch = self.last_epoch + epoch + 1
            avg_oprd_loss = (total_loss / len(train_dataloader)).item()
            avg_ce_loss = (total_ce_loss / len(train_dataloader)).item()
            avg_memory_kd_loss = (total_memory_kd_loss / len(train_dataloader)).item()
            avg_memory_ce_loss = (total_memory_ce_loss / len(train_dataloader)).item()
            log_prefix = self.stage_log_name.lower()
            self.accelerator.log(
                {
                    f"Loss/{log_prefix}_oprd_loss": avg_oprd_loss,
                    f"Loss/{log_prefix}_oprd_ce_loss": avg_ce_loss,
                    f"Loss/{log_prefix}_oprd_logit_loss": avg_memory_kd_loss,
                    f"Loss/{log_prefix}_oprd_memory_ce_loss": avg_memory_ce_loss,
                },
                step=log_epoch,
            )
            self.log(
                f"[{self.stage_log_name} OPRD Epoch {epoch + 1}] "
                f"Loss: {avg_oprd_loss}, "
                f"CE: {avg_ce_loss}, "
                f"Logit: {avg_memory_kd_loss}, "
                f"MemoryCE: {avg_memory_ce_loss}"
            )

            all_results = self.evaluate(
                val_dataloader,
                split=tune_split,
                step=self.current_step,
                epoch=log_epoch,
            )
            if self.accelerator.is_main_process:
                for key in all_results:
                    self.accelerator.log(
                        {f"{tune_label}_Metric/{key}": all_results[key]},
                        step=log_epoch,
                    )
                self.log(
                    f"[{self.stage_log_name} OPRD Epoch {epoch + 1}] "
                    f"{tune_label} Results: {self._results_for_log(all_results)}"
                )

            val_score = all_results[self.config["val_metric"]]
            if val_score > best_oprd_score + oprd_min_delta:
                best_oprd_score = val_score
                oprd_best_epoch = epoch + 1
                self.best_epoch = log_epoch
                self._save_current_model(log_epoch, stage=f"{self.stage_log_name} OPRD")
            elif (
                oprd_patience is not None
                and epoch + 1 - oprd_best_epoch >= oprd_patience
            ):
                self.log(
                    f"[{self.stage_log_name} OPRD] Early stopping at epoch {epoch + 1} "
                    f"after {oprd_patience} epoch(s) without improvement."
                )
                break

        self.last_epoch = self.last_epoch + last_oprd_epoch
        self.log(
            f"[{self.stage_log_name} OPRD] Best epoch: {oprd_best_epoch}, "
            f"Best val score: {best_oprd_score} "
            f"(selection split: {tune_split})"
        )
        self._after_oprd_training(
            step=oprd_step,
            epoch=last_oprd_epoch,
            total_n_steps=total_steps,
        )
        if oprd_best_epoch == 0:
            self.log(
                f"[{self.stage_log_name} OPRD] No OPRD checkpoint improved over "
                "the SFT baseline; restoring SFT checkpoint before continuing."
            )
            self._load_sft_checkpoint_if_available()
            self._save_current_model(
                self.last_epoch,
                stage=f"{self.stage_log_name} OPRD Fallback",
            )
            self.accelerator.wait_for_everyone()

        self.accelerator.wait_for_everyone()
        self.log(
            f"[{self.stage_log_name} OPRD] Budget: optimizer_steps={oprd_step}, "
            f"completed_epochs={last_oprd_epoch}, "
            f"elapsed_seconds={time.perf_counter() - oprd_started_at:.3f} "
            "(student stage including setup/validation/checkpointing; "
            "excludes teacher pretraining and final test)"
        )

    def _initial_oprd_best_score(self, val_dataloader):
        if hasattr(self, "sft_best_score"):
            self.log(
                f"[{self.stage_log_name} OPRD] Using SFT best score as OPRD baseline: "
                f"{self.sft_best_score}"
            )
            return self.sft_best_score

        if self._stage_config("oprd_eval_sft_baseline", False):
            tune_split = self._active_tune_split()
            results = self.evaluate(
                val_dataloader,
                split=tune_split,
                step=self.current_step,
                epoch=self.last_epoch,
            )
            score = results[self.config["val_metric"]]
            self.log(
                f"[{self.stage_log_name} OPRD] Evaluated loaded SFT checkpoint baseline: "
                f"{self._results_for_log(results)}"
            )
            return score
        return -1

    def _build_oprd_teacher(self):
        student_model = self.accelerator.unwrap_model(self.model)
        teacher = type(student_model)(
            student_model.config,
            student_model.dataset,
            student_model.tokenizer,
        )
        if os.path.exists(self.sft_saved_model_ckpt):
            state_dict = torch.load(
                self.sft_saved_model_ckpt,
                map_location="cpu",
            )
            teacher.load_state_dict(state_dict)
            del state_dict
        else:
            teacher.load_state_dict(student_model.state_dict())
        teacher.to(self.accelerator.device)
        teacher.eval()
        for param in teacher.parameters():
            param.requires_grad = False
        return teacher

    def _use_memory_oprd_teacher(self):
        memory_path = self._stage_config("memory_oprd_teacher_path", None)
        return bool(memory_path)

    def _build_memory_oprd_teacher(self):
        memory_path = self._stage_config("memory_oprd_teacher_path", None)
        return self._load_teacher_from_checkpoint(memory_path)

    def _load_teacher_from_checkpoint(self, checkpoint_path):
        if not checkpoint_path or not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Dual OPRD teacher checkpoint not found: {checkpoint_path}")

        student_model = self.accelerator.unwrap_model(self.model)
        teacher = type(student_model)(
            student_model.config,
            student_model.dataset,
            student_model.tokenizer,
        )
        state_dict = torch.load(checkpoint_path, map_location="cpu")
        teacher.load_state_dict(state_dict)
        del state_dict
        teacher.to(self.accelerator.device)
        teacher.eval()
        for param in teacher.parameters():
            param.requires_grad = False
        return teacher

    def _set_oprd_trainable(self, model):
        train_backbone = bool(self._stage_config("oprd_train_backbone", True))
        if train_backbone:
            for param in model.parameters():
                param.requires_grad = True
            return

        for param in model.parameters():
            param.requires_grad = False
        if not hasattr(model, "set_oprd_trainable"):
            raise ValueError(
                f"{type(model).__name__} does not implement set_oprd_trainable(). "
                "Use --oprd_train_backbone=True or add the adapter method."
            )
        model.set_oprd_trainable()

    def _extract_ce_loss(self, student_outputs):
        if hasattr(student_outputs, "rec_loss"):
            return student_outputs.rec_loss
        if hasattr(student_outputs, "loss"):
            return student_outputs.loss
        raise AttributeError("Model outputs must provide .loss or .rec_loss for OPRD.")

    def _sft_batch_loss(self, batch):
        student_outputs = self.model(batch)
        ce_loss = self._extract_ce_loss(student_outputs)
        sid_kd_loss = self._sasrec_sid_kd_loss(student_outputs, batch)
        ce_weight = float(self._stage_config("sasrec_sid_kd_ce_weight", 1.0))
        sid_kd_weight = float(self._stage_config("sasrec_sid_kd_weight", 0.0))
        loss = ce_weight * ce_loss + sid_kd_weight * sid_kd_loss
        return loss, ce_loss, sid_kd_loss

    def _oprd_batch_loss(self, batch):
        self._validate_memory_controls()
        student_outputs = self.model(batch)
        ce_loss = self._extract_ce_loss(student_outputs)
        ce_weight = float(self._stage_config("oprd_ce_weight", 1.0))
        memory_logit_weight = float(self._stage_config("memory_oprd_logit_weight", 0.0))
        memory_ce_weight = float(self._stage_config("memory_oprd_ce_weight", 0.0))
        self._last_memory_ce_loss = torch.zeros_like(ce_loss).detach()
        memory_kd_loss = torch.zeros_like(ce_loss)
        if memory_logit_weight > 0:
            sample_weights = None
            if self._use_memory_oprd_teacher():
                sample_weights = self._memory_teacher_selection(batch)
            memory_kd_loss = self._memory_teacher_rollout_logit_loss(
                batch=batch,
                sample_weights=sample_weights,
                reference_loss=ce_loss,
            )
        loss = ce_weight * ce_loss + memory_logit_weight * memory_kd_loss
        if memory_ce_weight > 0:
            memory_ce_loss = self._memory_sid_ce_loss(student_outputs, batch, ce_loss)
            loss = loss + memory_ce_weight * memory_ce_loss
            self._last_memory_ce_loss = memory_ce_loss.detach()
        return loss, ce_loss, memory_kd_loss

    def _memory_normalization(self):
        normalization = str(self._stage_config("memory_oprd_normalization", "selected")).lower()
        if normalization not in {"selected", "batch"}:
            raise ValueError("memory_oprd_normalization must be 'selected' or 'batch'.")
        return normalization

    def _validate_memory_controls(self):
        normalization = self._memory_normalization()
        ce_weight = float(self._stage_config("memory_oprd_ce_weight", 0.0))
        if not math.isfinite(ce_weight) or ce_weight < 0:
            raise ValueError("memory_oprd_ce_weight must be finite and non-negative.")
        if ce_weight > 0:
            if str(self.config.get("model", "")).upper() != "TIGER":
                raise ValueError("Extra Memory SID CE is currently supported only for TIGER.")
            if str(self._stage_config("memory_oprd_non_memory", "skip")).lower() != "skip":
                raise ValueError("Memory CE requires memory_oprd_non_memory=skip.")
            if float(self._stage_config("memory_oprd_logit_weight", 0.0)) != 0.0:
                raise ValueError("Memory CE control requires memory_oprd_logit_weight=0.")
        if ce_weight > 0 or normalization == "batch":
            if getattr(getattr(self, "accelerator", None), "num_processes", 1) != 1:
                raise ValueError("W2 MemCE/batch-normalized controls currently require single-process training.")

    def _memory_sid_ce_loss(self, student_outputs, batch, reference_loss):
        """Extra gold-prefix SID CE; the original backbone objective retains EOS."""
        self._validate_memory_controls()
        logits = getattr(student_outputs, "logits", None)
        labels = batch.get("labels")
        n_digit = int(getattr(self.tokenizer, "n_digit", 0))
        if (
            logits is None or labels is None or n_digit <= 0
            or logits.ndim != 3 or labels.ndim != 2
            or logits.shape[0] != labels.shape[0]
            or logits.shape[1] < n_digit or labels.shape[1] < n_digit
        ):
            raise ValueError("Memory CE requires aligned logits/labels with all SID positions.")
        sample_weights = self._memory_teacher_selection(batch)
        if sample_weights.numel() != labels.shape[0]:
            raise ValueError("Memory CE sample indices must match the SID batch dimension.")
        selected = sample_weights.to(device=logits.device).bool()
        if not bool(selected.any()):
            return torch.zeros_like(reference_loss)
        sid_labels = labels.to(device=logits.device)[selected, :n_digit]
        eos_token = getattr(self.tokenizer, "eos_token", None)
        padding_token = getattr(self.tokenizer, "padding_token", None)
        if (
            sid_labels.dtype != torch.long
            or bool((sid_labels < 0).any())
            or bool((sid_labels >= logits.shape[-1]).any())
            or (eos_token is not None and bool((sid_labels == eos_token).any()))
            or (padding_token is not None and bool((sid_labels == padding_token).any()))
        ):
            raise ValueError("Memory CE requires valid gold SID targets, without padding or EOS.")
        temperature = float(self.config.get("temperature", 1.0))
        if not math.isfinite(temperature) or temperature <= 0:
            raise ValueError("Memory CE backbone temperature must be finite and positive.")
        sid_logits = logits[selected, :n_digit].float() / temperature
        token_ce = F.cross_entropy(
            sid_logits.reshape(-1, sid_logits.shape[-1]),
            sid_labels.reshape(-1), reduction="none",
        ).reshape(-1, n_digit)
        return token_ce.mean(dim=1).mean()

    def _stage_bool(self, key, default=False):
        value = self._stage_config(key, default)
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            return value.strip().lower() in {"1", "true", "yes", "y", "on"}
        return bool(value)

    def _load_sasrec_sid_kd_table(self):
        if self.sasrec_sid_kd_table is not None:
            return self.sasrec_sid_kd_table

        path = self._stage_config("sasrec_sid_kd_path", None)
        if path in (None, "", "None"):
            self.sasrec_sid_kd_table = {}
            return self.sasrec_sid_kd_table
        if not os.path.exists(path):
            raise FileNotFoundError(f"sasrec_sid_kd_path does not exist: {path}")

        top_k = int(self._stage_config("sasrec_sid_kd_top_k", 50))
        confidence_key = str(self._stage_config("sasrec_sid_kd_confidence_key", "confidence_msp"))
        tau_low = float(self._stage_config("sasrec_sid_kd_confidence_tau_low", 0.0))
        tau_high = float(self._stage_config("sasrec_sid_kd_confidence_tau_high", tau_low))

        table = {}
        confidences = []
        rows = []
        with open(path, newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                sample_id = int(row.get("sample_idx", row.get("idx", row.get("sample_id"))))
                items = ast.literal_eval(row["top_items"])[:top_k]
                scores = ast.literal_eval(row["top_scores"])[:top_k]
                confidence = float(row.get(confidence_key, row.get("confidence_msp", 0.0)))
                rows.append((sample_id, items, scores, confidence))
                confidences.append(confidence)

        weights = confidence_weight(confidences, tau_low=tau_low, tau_high=tau_high)
        for (sample_id, items, scores, confidence), weight in zip(rows, weights):
            table[sample_id] = {
                "top_items": items,
                "top_scores": scores,
                "confidence": confidence,
                "weight": float(weight),
            }
        self.sasrec_sid_kd_table = table
        self.log(f"[{self.stage_log_name} SID-KD] Loaded {len(table)} SASRec rows from {path}")
        return self.sasrec_sid_kd_table

    def _internal_item2sid_tokens(self):
        model = self.accelerator.unwrap_model(self.model)
        id2item = model.dataset.id_mapping["id2item"]
        item2tokens = {}
        for item_id in range(1, len(id2item)):
            item = id2item[item_id]
            if item in self.tokenizer.item2tokens:
                item2tokens[item_id] = tuple(int(token) for token in self.tokenizer.item2tokens[item])
        return item2tokens

    def _sasrec_sid_kd_targets(self, batch, device, dtype):
        if "idx" not in batch:
            return None
        if not hasattr(self.tokenizer, "n_digit"):
            return None

        table = self._load_sasrec_sid_kd_table()
        if not table:
            return None

        sample_ids = [int(value) for value in batch["idx"].detach().cpu().tolist()]
        rows = [table.get(sample_id) for sample_id in sample_ids]
        if not any(row is not None and row["weight"] > 0.0 for row in rows):
            return None

        n_digit = int(self.tokenizer.n_digit)
        vocab_size = int(self.tokenizer.vocab_size)
        top_items = [row["top_items"] if row is not None else [] for row in rows]
        top_scores = [row["top_scores"] if row is not None else [] for row in rows]
        target_lists, mask_lists = soft_sid_targets_from_topk(
            top_items=top_items,
            top_scores=top_scores,
            item2tokens=self._internal_item2sid_tokens(),
            n_digit=n_digit,
            vocab_size=vocab_size,
            temperature=float(self._stage_config("sasrec_sid_kd_temperature", 1.0)),
            scores_are_logits=self._stage_bool("sasrec_sid_kd_scores_are_logits", True),
        )

        targets = torch.tensor(target_lists, device=device, dtype=dtype)
        digit_mask = torch.tensor(mask_lists, device=device, dtype=dtype)
        sample_weights = torch.tensor(
            [0.0 if row is None else float(row["weight"]) for row in rows],
            device=device,
            dtype=dtype,
        ).view(-1, 1)
        valid_weights = digit_mask * sample_weights
        if float(valid_weights.sum().detach().cpu()) <= 0.0:
            return None
        return targets, digit_mask, sample_weights, valid_weights

    def _sasrec_sid_kd_loss(self, student_outputs, batch):
        sid_kd_weight = float(self._stage_config("sasrec_sid_kd_weight", 0.0))
        logits = getattr(student_outputs, "logits", None)
        if sid_kd_weight <= 0.0 or logits is None:
            return torch.zeros_like(self._extract_ce_loss(student_outputs))
        target_pack = self._sasrec_sid_kd_targets(batch, logits.device, logits.dtype)
        if target_pack is None:
            return torch.zeros_like(self._extract_ce_loss(student_outputs))

        targets, _, _, valid_weights = target_pack
        n_digit = int(self.tokenizer.n_digit)
        token_logits = logits[:, :n_digit, :].float()
        log_probs = F.log_softmax(token_logits, dim=-1)
        kl_div = getattr(F, "kl_div")
        per_digit_kl = kl_div(log_probs, targets.float(), reduction="none").sum(dim=-1)
        return (per_digit_kl * valid_weights.float()).sum() / valid_weights.float().sum().clamp_min(1e-12)

    def _sasrec_sid_kd_rollout_loss(self, batch, sequences, reference_loss):
        if float(self._stage_config("sasrec_sid_kd_weight", 0.0)) <= 0.0:
            return torch.zeros_like(reference_loss)
        if sequences.dim() == 2:
            sequences = sequences.unsqueeze(1)
        if sequences.dim() != 3:
            return torch.zeros_like(reference_loss)
        if not hasattr(self.tokenizer, "n_digit"):
            return torch.zeros_like(reference_loss)

        model = self.accelerator.unwrap_model(self.model)
        if not hasattr(model, "_oprd_encoder_context") or not hasattr(model, "_oprd_stage_logits"):
            return torch.zeros_like(reference_loss)

        n_digit = int(self.tokenizer.n_digit)
        batch_size, n_rollouts, rollout_digits = sequences.shape
        n_digit = min(n_digit, int(rollout_digits))
        target_pack = self._sasrec_sid_kd_targets(
            batch,
            device=sequences.device,
            dtype=reference_loss.dtype,
        )
        if target_pack is None:
            return torch.zeros_like(reference_loss)
        targets, digit_mask, sample_weights, _ = target_pack

        encoder_hidden_states, encoder_attention_mask = model._oprd_encoder_context(batch)
        flat_encoder_hidden = encoder_hidden_states.repeat_interleave(n_rollouts, dim=0)
        flat_encoder_mask = encoder_attention_mask.repeat_interleave(n_rollouts, dim=0)
        flat_sequences = sequences.to(flat_encoder_hidden.device).long().reshape(
            batch_size * n_rollouts,
            rollout_digits,
        )

        losses = []
        weights = []
        kl_div = getattr(F, "kl_div")
        for digit_idx in range(n_digit):
            prev_tokens = flat_sequences[:, :digit_idx]
            token_logits = model._oprd_stage_logits(
                encoder_hidden_states=flat_encoder_hidden,
                encoder_attention_mask=flat_encoder_mask,
                prev_tokens=prev_tokens,
            )
            log_probs = F.log_softmax(token_logits.float(), dim=-1)
            digit_targets = targets[:, digit_idx, :].repeat_interleave(n_rollouts, dim=0)
            digit_weights = (
                digit_mask[:, digit_idx].view(-1, 1) * sample_weights
            ).view(-1).repeat_interleave(n_rollouts)
            digit_kl = kl_div(log_probs, digit_targets.float(), reduction="none").sum(dim=-1)
            losses.append(digit_kl)
            weights.append(digit_weights)

        all_losses = torch.cat(losses)
        all_weights = torch.cat(weights).float()
        denom = all_weights.sum()
        if float(denom.detach().cpu()) <= 0.0:
            return torch.zeros_like(reference_loss)
        return (all_losses * all_weights).sum() / denom.clamp_min(1e-12)

    def _load_memory_oprd_labels(self):
        if self.memory_oprd_label_table is not None:
            return self.memory_oprd_label_table

        path = self._stage_config("memory_oprd_labels_path", None)
        if not path:
            raise ValueError("memory_oprd_labels_path is required for memory-only OPRD.")
        if not os.path.exists(path):
            raise FileNotFoundError(f"memory_oprd_labels_path does not exist: {path}")

        train_idx_stride = int(self.config.get("train_idx_stride", 100000))
        table = {}
        with open(path, "r", encoding="utf-8") as handle:
            for line_no, line in enumerate(handle, start=1):
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                if "sample_id" in row:
                    sample_id = int(row["sample_id"])
                elif "idx" in row:
                    sample_id = int(row["idx"])
                else:
                    source_row_idx = int(row["source_row_idx"])
                    prefix_idx = int(row.get("prefix_idx", 0))
                    sample_id = int(source_row_idx * train_idx_stride + prefix_idx)

                group = str(row.get("group", ""))
                if not group:
                    labels = set(row.get("labels", []))
                    if "memorization" in labels:
                        group = "memory"
                    elif "generalization" in labels:
                        group = "generalization"
                    else:
                        group = "uncategorized"
                table[sample_id] = group

        self.memory_oprd_label_table = table
        self.log(
            f"[{self.stage_log_name} Memory-OPRD] Loaded {len(table)} labels from {path}"
        )
        return self.memory_oprd_label_table

    def _memory_teacher_selection(self, batch):
        if "idx" not in batch:
            raise ValueError(
                "Memory-only OPRD requires batch['idx']; keep train tokenization with indices."
            )

        label_table = self._load_memory_oprd_labels()
        non_memory_policy = str(
            self._stage_config("memory_oprd_non_memory", "skip")
        ).lower()
        if non_memory_policy not in {"skip", "align"}:
            raise ValueError("memory_oprd_non_memory must be one of {'skip', 'align'}.")

        sample_ids = [int(value) for value in batch["idx"].detach().cpu().tolist()]
        sample_weights = []
        for sample_id in sample_ids:
            group = str(label_table.get(sample_id, "uncategorized")).lower()
            if group == "memory":
                sample_weights.append(1.0)
            elif non_memory_policy == "align":
                sample_weights.append(1.0)
            else:  # skip
                sample_weights.append(0.0)

        return torch.tensor(
            sample_weights,
            device=batch["idx"].device,
            dtype=torch.float32,
        )

    def _memory_teacher_logit_loss(
        self,
        student_outputs,
        batch,
        sample_weights=None,
        reference_loss=None,
    ):
        normalization = self._memory_normalization()
        if reference_loss is None:
            reference_loss = self._extract_ce_loss(student_outputs)

        memory_logit_weight = float(self._stage_config("memory_oprd_logit_weight", 0.0))
        if memory_logit_weight <= 0 or not self._use_memory_oprd_teacher():
            return torch.zeros_like(reference_loss)

        student_logits = getattr(student_outputs, "logits", None)
        labels = batch.get("labels", None) if isinstance(batch, dict) else None
        if student_logits is None or labels is None:
            return torch.zeros_like(reference_loss)

        if self.memory_oprd_teacher is None:
            self.memory_oprd_teacher = self._build_memory_oprd_teacher()
        if sample_weights is None:
            sample_weights = self._memory_teacher_selection(batch)

        with torch.no_grad():
            self.memory_oprd_teacher.eval()
            teacher_outputs = self.memory_oprd_teacher(batch)
            teacher_logits = getattr(teacher_outputs, "logits", None)

        if teacher_logits is None:
            raise AttributeError("Memory OPRD teacher outputs must provide logits.")
        if student_logits.shape[-1] != teacher_logits.shape[-1]:
            raise ValueError(
                "Memory OPRD logit distillation requires student and teacher "
                "to share the same tokenizer vocabulary."
            )

        seq_len = min(student_logits.shape[1], teacher_logits.shape[1], labels.shape[1])
        if seq_len <= 0:
            return torch.zeros_like(reference_loss)

        student_token_logits = student_logits[:, :seq_len, :].float()
        teacher_token_logits = teacher_logits[:, :seq_len, :].to(
            student_token_logits.device
        ).float()
        labels = labels[:, :seq_len].to(student_token_logits.device)
        valid_mask = labels != -100

        sample_weights = sample_weights.to(
            device=student_token_logits.device,
            dtype=student_token_logits.dtype,
        )
        if sample_weights.numel() != student_token_logits.shape[0]:
            raise ValueError(
                "Memory OPRD sample weights must match the logit batch dimension."
            )
        token_weights = valid_mask.to(student_token_logits.dtype) * sample_weights.view(-1, 1)
        if float(token_weights.sum().detach().cpu()) <= 0.0:
            return torch.zeros_like(reference_loss)

        temperature = max(
            float(self._stage_config("memory_oprd_temperature", 2.0)),
            1e-6,
        )
        log_probs = F.log_softmax(student_token_logits / temperature, dim=-1)
        target_probs = F.softmax(teacher_token_logits / temperature, dim=-1)
        kl_div = getattr(F, "kl_div")
        token_kl = kl_div(log_probs, target_probs, reduction="none").sum(dim=-1)
        token_kl = token_kl * (temperature ** 2)
        denominator = (
            token_weights.sum() if normalization == "selected"
            else valid_mask.to(token_weights.dtype).sum()
        )
        return (token_kl * token_weights).sum() / denominator.clamp_min(1e-12)

    def _memory_teacher_rollout_logit_loss(
        self,
        batch,
        sample_weights=None,
        reference_loss=None,
    ):
        if reference_loss is None:
            raise ValueError("reference_loss is required for memory OPRD rollout KL.")

        n_rollouts = int(self._stage_config("oprd_num_rollouts", 1))
        if not self._use_memory_oprd_teacher():
            return torch.zeros_like(reference_loss)

        rollouts = self._sample_oprd_rollouts(batch, n_rollouts=n_rollouts)
        rollout_batch, rollout_weights = self._batch_with_oprd_rollout_labels(
            batch=batch,
            rollouts=rollouts,
            sample_weights=sample_weights,
        )
        rollout_outputs = self.model(rollout_batch)
        rollout_logit_loss = self._memory_teacher_logit_loss(
            student_outputs=rollout_outputs,
            batch=rollout_batch,
            sample_weights=rollout_weights,
            reference_loss=reference_loss,
        )
        return rollout_logit_loss

    def _batch_with_oprd_rollout_labels(self, batch, rollouts, sample_weights=None):
        if not isinstance(batch, dict):
            raise TypeError("OPRD rollout logit KD requires batch to be a dict.")
        if rollouts.dim() != 3:
            raise ValueError("OPRD rollouts must have shape [batch, rollouts, digits].")

        batch_size, n_rollouts, n_digit = rollouts.shape
        rollout_batch = {}
        for key, value in batch.items():
            if isinstance(value, torch.Tensor) and value.shape[:1] == (batch_size,):
                rollout_batch[key] = value.repeat_interleave(n_rollouts, dim=0)
            else:
                rollout_batch[key] = value

        rollout_batch["labels"] = rollouts.reshape(batch_size * n_rollouts, n_digit)
        rollout_weights = None
        if sample_weights is not None:
            rollout_weights = sample_weights.repeat_interleave(n_rollouts)
        return rollout_batch, rollout_weights

    def _normalize_oprd_rollouts(self, sampled):
        if isinstance(sampled, torch.Tensor):
            sequences = sampled
        elif isinstance(sampled, (tuple, list)) and sampled:
            sequences = sampled[0]
        elif isinstance(sampled, dict):
            sequences = sampled.get("sequences", sampled.get("preds"))
        else:
            raise TypeError(
                "OPRD sample() must return a tensor, a non-empty tuple/list, "
                "or a mapping containing 'sequences' or 'preds'."
            )

        if not isinstance(sequences, torch.Tensor):
            raise TypeError("Normalized OPRD rollouts must be a torch.Tensor.")
        if sequences.dim() != 3:
            raise ValueError(
                "Normalized OPRD rollouts must have three dimensions "
                "[batch, rollouts, digits]."
            )
        return sequences

    def _sample_oprd_rollouts(self, batch, n_rollouts=None):
        model_for_sampling = self.accelerator.unwrap_model(self.model)
        if not hasattr(model_for_sampling, "sample"):
            raise AttributeError(
                f"{type(model_for_sampling).__name__} does not implement sample()."
            )
        if n_rollouts is None:
            n_rollouts = int(self._stage_config("oprd_num_rollouts", 1))
        n_rollouts = int(n_rollouts)
        if n_rollouts <= 0:
            raise ValueError("oprd_num_rollouts must be positive.")
        temperature = float(self._stage_config("oprd_rollout_temperature", 1.0))
        was_training = self.model.training
        self.model.eval()
        try:
            sampled = model_for_sampling.sample(
                batch=batch,
                n_return_sequences=n_rollouts,
                temperature=temperature,
            )
        finally:
            if was_training:
                self.model.train()
        return self._normalize_oprd_rollouts(sampled)

    def _prepare_oprd_only(self, train_dataloader, val_dataloader):
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
            f"[{self.stage_log_name} OPRD] Starting OPRD-only training from "
            f"{self.sft_saved_model_ckpt}"
        )
        return train_dataloader, val_dataloader

    def _load_sft_checkpoint_if_available(self):
        self.accelerator.wait_for_everyone()
        if not os.path.exists(self.sft_saved_model_ckpt):
            self.log(
                f"[{self.stage_log_name} OPRD] No SFT checkpoint found; "
                "continuing from current SFT parameters.",
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
        self.log(
            f"[{self.stage_log_name} OPRD] Loaded SFT checkpoint before OPRD: "
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
                f"[{self.stage_log_name}] Could not inspect SFT checkpoint; "
                f"will retrain SFT: {path} ({error})",
                level="warning",
            )
            return False

        target_model = self.accelerator.unwrap_model(self.model)
        expected = target_model.state_dict()
        try:
            for key, value in state_dict.items():
                if key in expected and tuple(value.shape) != tuple(expected[key].shape):
                    self.log(
                        f"[{self.stage_log_name}] Existing SFT checkpoint is "
                        f"incompatible; retraining SFT instead of resuming: {path}",
                        level="warning",
                    )
                    return False
            return True
        finally:
            del state_dict

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
            f"[{self.stage_log_name}] Found existing SFT checkpoint; skipped SFT "
            f"and will continue from {self.sft_saved_model_ckpt}"
        )
        return train_dataloader, val_dataloader

    def _load_state_dict_from_sft_checkpoint(self):
        if not os.path.exists(self.sft_saved_model_ckpt):
            raise FileNotFoundError(
                f"{self.stage_log_name} OPRD requires a trained SFT checkpoint. "
                "Set --sft_checkpoint_path=/path/to/checkpoint.pth or train SFT "
                f"first. Missing: {self.sft_saved_model_ckpt}"
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

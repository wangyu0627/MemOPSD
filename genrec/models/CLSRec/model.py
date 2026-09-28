import math
import random
from types import SimpleNamespace

import torch
import torch.nn.functional as F
from transformers import GPT2Config, GPT2LMHeadModel

from genrec.dataset import AbstractDataset
from genrec.model import AbstractModel
from genrec.tokenizer import AbstractTokenizer


class CLSRec(AbstractModel):
    """
    CL4SRec adapted to the local SASRec interface.

    The recommendation backbone and tokenizer are matched to SASRec. Training
    adds CL4SRec's two augmented sequence views and an InfoNCE objective over
    the final valid sequence representation.
    """

    def __init__(
        self,
        config: dict,
        dataset: AbstractDataset,
        tokenizer: AbstractTokenizer,
    ):
        super(CLSRec, self).__init__(config, dataset, tokenizer)
        self.cl_aug_type = config["cl_aug_type"]
        self.cl_aug_rate = config["cl_aug_rate"]
        self.cl_loss_weight = config["cl_loss_weight"]
        self.cl_temperature = config["cl_temperature"]

        valid_item_mask = torch.zeros(tokenizer.vocab_size, dtype=torch.bool)
        valid_item_mask[1 : dataset.n_items] = True
        self.register_buffer("valid_item_mask", valid_item_mask)
        self.loss_fct = torch.nn.CrossEntropyLoss(
            ignore_index=tokenizer.ignored_label,
        )

        gpt2config = GPT2Config(
            vocab_size=tokenizer.vocab_size,
            n_positions=tokenizer.max_token_seq_len,
            n_embd=config["n_embd"],
            n_layer=config["n_layer"],
            n_head=config["n_head"],
            n_inner=config["n_inner"],
            activation_function=config["activation_function"],
            resid_pdrop=config["resid_pdrop"],
            embd_pdrop=config["embd_pdrop"],
            attn_pdrop=config["attn_pdrop"],
            layer_norm_epsilon=config["layer_norm_epsilon"],
            initializer_range=config["initializer_range"],
            eos_token_id=tokenizer.eos_token,
        )
        self.gpt2 = GPT2LMHeadModel(gpt2config)

    @property
    def n_parameters(self) -> str:
        total_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        emb_params = sum(
            p.numel()
            for p in self.gpt2.get_input_embeddings().parameters()
            if p.requires_grad
        )
        return (
            f"#Embedding parameters: {emb_params}\n"
            f"#Non-embedding parameters: {total_params - emb_params}\n"
            f"#Total trainable parameters: {total_params}\n"
        )

    def _gather_index(self, output: torch.Tensor, index: torch.Tensor) -> torch.Tensor:
        index = index.view(-1, 1, 1).expand(-1, -1, output.shape[-1])
        return output.gather(dim=1, index=index).squeeze(1)

    def _sample_aug_type(self):
        aug_type = self.cl_aug_type
        if isinstance(aug_type, str):
            aug_type = aug_type.lower()
            if aug_type == "random":
                return random.choice([0, 1, 2])
            if aug_type == "crop":
                return 0
            if aug_type == "reorder":
                return 1
            if aug_type == "mask":
                return 2
        return int(aug_type)

    def _augment_one_sequence(
        self,
        tokens: list[int],
        aug_type: int,
    ) -> list[int]:
        length = len(tokens)
        if length <= 1:
            return tokens

        if aug_type == 0:
            crop_len = max(1, math.floor(length * self.cl_aug_rate))
            crop_len = min(crop_len, length)
            start = random.randint(0, length - crop_len)
            return tokens[start : start + crop_len]

        if aug_type == 1:
            reorder_len = max(1, math.floor(length * self.cl_aug_rate))
            reorder_len = min(reorder_len, length)
            start = random.randint(0, length - reorder_len)
            reordered = tokens[:]
            segment = reordered[start : start + reorder_len]
            random.shuffle(segment)
            reordered[start : start + reorder_len] = segment
            return reordered

        mask_num = max(1, math.floor(length * self.cl_aug_rate))
        mask_num = min(mask_num, length)
        masked = tokens[:]
        mask_positions = random.sample(range(length), mask_num)
        for pos in mask_positions:
            masked[pos] = self.tokenizer.eos_token
        return masked

    def _augment_batch(
        self,
        input_ids: torch.Tensor,
        seq_lens: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch_size, max_len = input_ids.shape
        augmented = torch.zeros_like(input_ids)
        attention_mask = torch.zeros_like(input_ids)
        augmented_lens = torch.zeros_like(seq_lens)

        for i in range(batch_size):
            seq_len = int(seq_lens[i].item())
            tokens = input_ids[i, :seq_len].tolist()
            aug_tokens = self._augment_one_sequence(tokens, self._sample_aug_type())
            aug_len = min(len(aug_tokens), max_len)
            if aug_len > 0:
                augmented[i, :aug_len] = torch.tensor(
                    aug_tokens[:aug_len],
                    dtype=input_ids.dtype,
                    device=input_ids.device,
                )
                attention_mask[i, :aug_len] = 1
            augmented_lens[i] = aug_len

        return augmented, attention_mask, augmented_lens

    def _info_nce_loss(
        self,
        z_i: torch.Tensor,
        z_j: torch.Tensor,
    ) -> torch.Tensor:
        batch_size = z_i.size(0)
        if batch_size <= 1:
            return torch.zeros((), dtype=z_i.dtype, device=z_i.device)

        z = F.normalize(torch.cat([z_i, z_j], dim=0), dim=1)
        logits = torch.matmul(z, z.T) / max(self.cl_temperature, 1e-8)
        logits.fill_diagonal_(-10000.0)

        labels = torch.arange(2 * batch_size, device=z.device)
        labels = (labels + batch_size) % (2 * batch_size)
        return F.cross_entropy(logits, labels)

    def _gpt2_sequence_representations(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        seq_lens: torch.Tensor,
    ) -> torch.Tensor:
        outputs = self.gpt2(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            return_dict=True,
        )
        return self._gather_index(outputs.hidden_states[-1], seq_lens - 1)

    def _contrastive_loss(self, batch: dict) -> torch.Tensor:
        aug_ids1, aug_mask1, aug_lens1 = self._augment_batch(
            batch["input_ids"],
            batch["seq_lens"],
        )
        aug_ids2, aug_mask2, aug_lens2 = self._augment_batch(
            batch["input_ids"],
            batch["seq_lens"],
        )

        z_i = self._gpt2_sequence_representations(aug_ids1, aug_mask1, aug_lens1)
        z_j = self._gpt2_sequence_representations(aug_ids2, aug_mask2, aug_lens2)
        return self._info_nce_loss(z_i, z_j)

    def forward(self, batch: dict):
        outputs = self.gpt2(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            output_hidden_states=True,
            return_dict=True,
        )
        rec_loss = self.loss_fct(
            outputs.logits.reshape(-1, outputs.logits.shape[-1]),
            batch["labels"].reshape(-1),
        )
        if self.cl_loss_weight > 0:
            cl_loss = self._contrastive_loss(batch)
            loss = rec_loss + self.cl_loss_weight * cl_loss
        else:
            cl_loss = torch.zeros((), device=batch["input_ids"].device)
            loss = rec_loss

        return SimpleNamespace(
            loss=loss,
            logits=outputs.logits,
            hidden_states=outputs.hidden_states[-1],
            rec_loss=rec_loss,
            cl_loss=cl_loss,
        )

    def generate(self, batch: dict, n_return_sequences: int = 1, num_beams=None):
        outputs = self.gpt2(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
        )
        logits = self._gather_index(outputs.logits, batch["seq_lens"] - 1)
        preds = logits.topk(n_return_sequences, dim=-1).indices
        return preds.unsqueeze(-1)

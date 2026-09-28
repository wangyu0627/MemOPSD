from types import SimpleNamespace

import torch
import torch.nn as nn
import torch.nn.functional as F

from genrec.dataset import AbstractDataset
from genrec.model import AbstractModel
from genrec.models.S3Rec.layers import Encoder, LayerNorm
from genrec.tokenizer import AbstractTokenizer


class S3RecConfig:
    def __init__(self, config: dict):
        self.hidden_size = config["hidden_size"]
        self.num_hidden_layers = config["num_hidden_layers"]
        self.num_attention_heads = config["num_attention_heads"]
        self.hidden_act = config["hidden_act"]
        self.hidden_dropout_prob = config["hidden_dropout_prob"]
        self.attention_probs_dropout_prob = config["attention_probs_dropout_prob"]
        self.initializer_range = config["initializer_range"]
        self.layer_norm_epsilon = config["layer_norm_epsilon"]


class S3Rec(AbstractModel):
    def __init__(
        self,
        config: dict,
        dataset: AbstractDataset,
        tokenizer: AbstractTokenizer,
    ):
        super(S3Rec, self).__init__(config, dataset, tokenizer)
        self.rec_loss_fct = nn.CrossEntropyLoss(ignore_index=tokenizer.ignored_label)
        valid_item_mask = torch.zeros(tokenizer.vocab_size, dtype=torch.bool)
        valid_item_mask[1 : dataset.n_items] = True
        self.register_buffer("valid_item_mask", valid_item_mask)

        self.s3rec_config = S3RecConfig(config)

        self.item_embeddings = nn.Embedding(
            tokenizer.vocab_size,
            config["hidden_size"],
            padding_idx=tokenizer.padding_token,
        )
        self.attribute_embeddings = nn.Embedding(
            tokenizer.attribute_size,
            config["hidden_size"],
            padding_idx=0,
        )
        self.position_embeddings = nn.Embedding(
            tokenizer.max_token_seq_len,
            config["hidden_size"],
        )
        self.item_encoder = Encoder(self.s3rec_config)
        self.LayerNorm = LayerNorm(
            config["hidden_size"],
            eps=config["layer_norm_epsilon"],
        )
        self.dropout = nn.Dropout(config["hidden_dropout_prob"])

        self.aap_norm = nn.Linear(config["hidden_size"], config["hidden_size"])
        self.mip_norm = nn.Linear(config["hidden_size"], config["hidden_size"])
        self.map_norm = nn.Linear(config["hidden_size"], config["hidden_size"])
        self.sp_norm = nn.Linear(config["hidden_size"], config["hidden_size"])

        self.ssl_loss_fct = nn.BCELoss(reduction="none")

        item_attribute_matrix = torch.zeros(
            dataset.n_items + 1,
            tokenizer.attribute_size,
            dtype=torch.float32,
        )
        for item_id, attrs in enumerate(tokenizer.item_id2attributes):
            if item_id >= item_attribute_matrix.size(0):
                continue
            for attr in attrs:
                if 0 < attr < tokenizer.attribute_size:
                    item_attribute_matrix[item_id, attr] = 1.0
        self.register_buffer("item_attribute_matrix", item_attribute_matrix)

        self.apply(self.init_weights)

    @property
    def n_parameters(self) -> str:
        total_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        emb_params = sum(
            p.numel() for p in self.item_embeddings.parameters() if p.requires_grad
        )
        emb_params += sum(
            p.numel() for p in self.position_embeddings.parameters() if p.requires_grad
        )
        emb_params += sum(
            p.numel()
            for p in self.attribute_embeddings.parameters()
            if p.requires_grad
        )
        return (
            f"#Embedding parameters: {emb_params}\n"
            f"#Non-embedding parameters: {total_params - emb_params}\n"
            f"#Total trainable parameters: {total_params}\n"
        )

    def init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(
                mean=0.0,
                std=self.config["initializer_range"],
            )
        elif isinstance(module, LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    def add_position_embedding(self, input_ids: torch.Tensor) -> torch.Tensor:
        seq_length = input_ids.size(1)
        position_ids = torch.arange(
            seq_length,
            dtype=torch.long,
            device=input_ids.device,
        )
        position_ids = position_ids.unsqueeze(0).expand_as(input_ids)
        sequence_emb = (
            self.item_embeddings(input_ids)
            + self.position_embeddings(position_ids)
        )
        sequence_emb = self.LayerNorm(sequence_emb)
        return self.dropout(sequence_emb)

    def _build_attention_mask(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor = None,
        causal: bool = True,
    ) -> torch.Tensor:
        if attention_mask is None:
            attention_mask = (input_ids > 0).long()
        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)

        if causal:
            max_len = attention_mask.size(-1)
            causal_mask = torch.ones(
                (1, max_len, max_len),
                dtype=torch.long,
                device=input_ids.device,
            ).tril()
            extended_attention_mask = (
                extended_attention_mask * causal_mask.unsqueeze(1)
            )

        extended_attention_mask = extended_attention_mask.to(
            dtype=next(self.parameters()).dtype,
        )
        return (1.0 - extended_attention_mask) * -10000.0

    def encode(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor = None,
        causal: bool = True,
    ) -> torch.Tensor:
        sequence_emb = self.add_position_embedding(input_ids)
        extended_attention_mask = self._build_attention_mask(
            input_ids=input_ids,
            attention_mask=attention_mask,
            causal=causal,
        )
        encoded_layers = self.item_encoder(
            sequence_emb,
            extended_attention_mask,
            output_all_encoded_layers=True,
        )
        return encoded_layers[-1]

    def finetune(self, input_ids: torch.Tensor, attention_mask: torch.Tensor = None):
        return self.encode(input_ids, attention_mask=attention_mask, causal=True)

    def _compute_logits(self, sequence_output: torch.Tensor) -> torch.Tensor:
        logits = torch.matmul(sequence_output, self.item_embeddings.weight.T)
        mask_shape = [1] * logits.dim()
        mask_shape[-1] = -1
        return logits.masked_fill(
            ~self.valid_item_mask.view(*mask_shape),
            -1e4,
        )

    def associated_attribute_prediction(self, sequence_output, attribute_embedding):
        sequence_output = self.aap_norm(sequence_output)
        sequence_output = sequence_output.view(-1, self.config["hidden_size"], 1)
        score = torch.matmul(attribute_embedding, sequence_output)
        return torch.sigmoid(score.squeeze(-1))

    def masked_item_prediction(self, sequence_output, target_item):
        sequence_output = self.mip_norm(
            sequence_output.view(-1, self.config["hidden_size"])
        )
        target_item = target_item.view(-1, self.config["hidden_size"])
        score = torch.mul(sequence_output, target_item)
        return torch.sigmoid(torch.sum(score, -1))

    def masked_attribute_prediction(self, sequence_output, attribute_embedding):
        sequence_output = self.map_norm(sequence_output)
        sequence_output = sequence_output.view(-1, self.config["hidden_size"], 1)
        score = torch.matmul(attribute_embedding, sequence_output)
        return torch.sigmoid(score.squeeze(-1))

    def segment_prediction(self, context, segment):
        context = self.sp_norm(context)
        score = torch.mul(context, segment)
        return torch.sigmoid(torch.sum(score, dim=-1))

    def _last_valid_hidden(self, hidden_states, input_ids):
        lengths = (input_ids != 0).sum(dim=-1).clamp(min=1)
        gather_idx = lengths.sub(1).view(-1, 1, 1).expand(
            -1,
            1,
            hidden_states.size(-1),
        )
        return hidden_states.gather(1, gather_idx).squeeze(1)

    def _safe_mean(self, loss, mask, feature_dim=1):
        denom = mask.sum() * feature_dim
        if denom.item() == 0:
            return loss.sum() * 0.0
        return loss.sum() / denom.clamp(min=1.0)

    def pretrain(self, batch: dict):
        masked_item_sequence = batch["masked_item_sequence"]
        pos_items = batch["pos_items"]
        neg_items = batch["neg_items"]
        masked_segment_sequence = batch["masked_segment_sequence"]
        pos_segment = batch["pos_segment"]
        neg_segment = batch["neg_segment"]

        sequence_output = self.encode(masked_item_sequence, causal=False)
        attribute_embeddings = self.attribute_embeddings.weight
        attributes = self.item_attribute_matrix[
            pos_items.clamp(max=self.item_attribute_matrix.size(0) - 1)
        ]

        if self.tokenizer.attribute_size > 1:
            aap_score = self.associated_attribute_prediction(
                sequence_output,
                attribute_embeddings,
            )
            aap_loss = self.ssl_loss_fct(
                aap_score,
                attributes.view(-1, self.tokenizer.attribute_size),
            )
            aap_mask = (
                (masked_item_sequence != self.tokenizer.mask_token)
                & (masked_item_sequence != 0)
            ).float()
            aap_loss = self._safe_mean(
                aap_loss * aap_mask.flatten().unsqueeze(-1),
                aap_mask,
                self.tokenizer.attribute_size,
            )

            map_score = self.masked_attribute_prediction(
                sequence_output,
                attribute_embeddings,
            )
            map_loss = self.ssl_loss_fct(
                map_score,
                attributes.view(-1, self.tokenizer.attribute_size),
            )
            map_mask = (masked_item_sequence == self.tokenizer.mask_token).float()
            map_loss = self._safe_mean(
                map_loss * map_mask.flatten().unsqueeze(-1),
                map_mask,
                self.tokenizer.attribute_size,
            )
        else:
            aap_loss = sequence_output.sum() * 0.0
            map_loss = sequence_output.sum() * 0.0

        pos_item_embs = self.item_embeddings(pos_items)
        neg_item_embs = self.item_embeddings(neg_items)
        pos_score = self.masked_item_prediction(sequence_output, pos_item_embs)
        neg_score = self.masked_item_prediction(sequence_output, neg_item_embs)
        mip_distance = torch.sigmoid(pos_score - neg_score)
        mip_loss = self.ssl_loss_fct(
            mip_distance,
            torch.ones_like(mip_distance),
        )
        mip_mask = (masked_item_sequence == self.tokenizer.mask_token).float()
        mip_loss = self._safe_mean(mip_loss * mip_mask.flatten(), mip_mask)

        segment_context = self.encode(masked_segment_sequence, causal=False)
        segment_context = self._last_valid_hidden(
            segment_context,
            masked_segment_sequence,
        )
        pos_segment_emb = self.encode(pos_segment, causal=False)
        pos_segment_emb = self._last_valid_hidden(pos_segment_emb, pos_segment)
        neg_segment_emb = self.encode(neg_segment, causal=False)
        neg_segment_emb = self._last_valid_hidden(neg_segment_emb, neg_segment)

        pos_segment_score = self.segment_prediction(segment_context, pos_segment_emb)
        neg_segment_score = self.segment_prediction(segment_context, neg_segment_emb)
        sp_distance = torch.sigmoid(pos_segment_score - neg_segment_score)
        sp_loss = self.ssl_loss_fct(sp_distance, torch.ones_like(sp_distance)).mean()

        joint_loss = (
            self.config["aap_weight"] * aap_loss
            + self.config["mip_weight"] * mip_loss
            + self.config["map_weight"] * map_loss
            + self.config["sp_weight"] * sp_loss
        )
        return SimpleNamespace(
            loss=joint_loss,
            aap_loss=aap_loss,
            mip_loss=mip_loss,
            map_loss=map_loss,
            sp_loss=sp_loss,
        )

    def forward(self, batch: dict):
        sequence_output = self.finetune(
            batch["input_ids"],
            attention_mask=batch["attention_mask"],
        )
        logits = self._compute_logits(sequence_output)
        loss = self.rec_loss_fct(
            logits.view(-1, logits.shape[-1]),
            batch["labels"].view(-1),
        )
        return SimpleNamespace(loss=loss, logits=logits, hidden_states=sequence_output)

    def gather_index(self, output, index):
        index = index.view(-1, 1, 1).expand(-1, -1, output.shape[-1])
        return output.gather(dim=1, index=index).squeeze(1)

    def generate(self, batch: dict, n_return_sequences: int = 1, num_beams=None):
        sequence_output = self.finetune(
            batch["input_ids"],
            attention_mask=batch["attention_mask"],
        )
        final_state = self.gather_index(sequence_output, batch["seq_lens"] - 1)
        logits = self._compute_logits(final_state)
        preds = logits.topk(n_return_sequences, dim=-1).indices
        return preds.unsqueeze(-1)

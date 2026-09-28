from types import SimpleNamespace

import torch
import torch.nn as nn

from genrec.dataset import AbstractDataset
from genrec.model import AbstractModel
from genrec.models.S3Rec.layers import Encoder, LayerNorm
from genrec.tokenizer import AbstractTokenizer


class FDSAConfig:
    def __init__(
        self,
        config: dict,
        num_hidden_layers: int,
        num_attention_heads: int,
    ):
        self.hidden_size = config["hidden_size"]
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.hidden_act = config["hidden_act"]
        self.hidden_dropout_prob = config["hidden_dropout_prob"]
        self.attention_probs_dropout_prob = config["attention_probs_dropout_prob"]
        self.initializer_range = config["initializer_range"]
        self.layer_norm_epsilon = config["layer_norm_epsilon"]


class FDSA(AbstractModel):
    """
    Feature-level Deeper Self-Attention Network for sequential recommendation.

    This implementation keeps the paper's two-branch structure: an item-level
    self-attention encoder, a feature-level self-attention encoder, vanilla
    attention over each item's heterogeneous features, and a linear fusion
    layer before tied item-embedding prediction.
    """

    def __init__(
        self,
        config: dict,
        dataset: AbstractDataset,
        tokenizer: AbstractTokenizer,
    ):
        super(FDSA, self).__init__(config, dataset, tokenizer)
        self.loss_fct = nn.CrossEntropyLoss(ignore_index=tokenizer.ignored_label)
        valid_item_mask = torch.zeros(tokenizer.vocab_size, dtype=torch.bool)
        valid_item_mask[1 : dataset.n_items] = True
        self.register_buffer("valid_item_mask", valid_item_mask)

        self.hidden_size = config["hidden_size"]
        self.item_embeddings = nn.Embedding(
            tokenizer.vocab_size,
            self.hidden_size,
            padding_idx=tokenizer.padding_token,
        )
        self.attribute_embeddings = nn.Embedding(
            tokenizer.feature_size,
            self.hidden_size,
            padding_idx=0,
        )
        self.position_embeddings = nn.Embedding(
            tokenizer.max_token_seq_len,
            self.hidden_size,
        )

        self.item_layer_norm = LayerNorm(
            self.hidden_size,
            eps=config["layer_norm_epsilon"],
        )
        self.feature_layer_norm = LayerNorm(
            self.hidden_size,
            eps=config["layer_norm_epsilon"],
        )
        self.dropout = nn.Dropout(config["hidden_dropout_prob"])

        item_config = FDSAConfig(
            config,
            num_hidden_layers=config["num_item_layers"],
            num_attention_heads=config["num_item_heads"],
        )
        feature_config = FDSAConfig(
            config,
            num_hidden_layers=config["num_feature_layers"],
            num_attention_heads=config["num_feature_heads"],
        )
        self.item_encoder = Encoder(item_config)
        self.feature_encoder = Encoder(feature_config)

        self.feature_attention = nn.Linear(self.hidden_size, self.hidden_size)
        self.feature_attention_score = nn.Linear(self.hidden_size, 1, bias=False)
        self.fusion = nn.Linear(self.hidden_size * 2, self.hidden_size)

        self.apply(self.init_weights)

    @property
    def n_parameters(self) -> str:
        total_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        item_emb_params = sum(
            p.numel() for p in self.item_embeddings.parameters() if p.requires_grad
        )
        attr_emb_params = sum(
            p.numel()
            for p in self.attribute_embeddings.parameters()
            if p.requires_grad
        )
        pos_emb_params = sum(
            p.numel()
            for p in self.position_embeddings.parameters()
            if p.requires_grad
        )
        emb_params = item_emb_params + attr_emb_params + pos_emb_params
        return (
            f"#Item embedding parameters: {item_emb_params}\n"
            f"#Feature embedding parameters: {attr_emb_params}\n"
            f"#Position embedding parameters: {pos_emb_params}\n"
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
        if isinstance(module, nn.Embedding) and module.padding_idx is not None:
            with torch.no_grad():
                module.weight[module.padding_idx].zero_()

    def _position_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        seq_length = input_ids.size(1)
        position_ids = torch.arange(
            seq_length,
            dtype=torch.long,
            device=input_ids.device,
        )
        return position_ids.unsqueeze(0).expand_as(input_ids)

    def _build_attention_mask(
        self,
        sequence_mask: torch.Tensor,
        causal: bool = True,
    ) -> torch.Tensor:
        extended_attention_mask = sequence_mask.unsqueeze(1).unsqueeze(2)
        if causal:
            max_len = sequence_mask.size(-1)
            causal_mask = torch.ones(
                (1, max_len, max_len),
                dtype=torch.long,
                device=sequence_mask.device,
            ).tril()
            extended_attention_mask = (
                extended_attention_mask * causal_mask.unsqueeze(1)
            )

        extended_attention_mask = extended_attention_mask.to(
            dtype=next(self.parameters()).dtype,
        )
        return (1.0 - extended_attention_mask) * -10000.0

    def _item_sequence_embedding(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        embeddings = (
            self.item_embeddings(input_ids)
            + self.position_embeddings(self._position_ids(input_ids))
        )
        embeddings = self.item_layer_norm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings * attention_mask.unsqueeze(-1).to(embeddings.dtype)

    def _feature_sequence_embedding(
        self,
        feature_ids: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        feature_mask = feature_ids.ne(0)
        feature_emb = self.attribute_embeddings(feature_ids)
        attention_hidden = torch.tanh(self.feature_attention(feature_emb))
        feature_scores = self.feature_attention_score(attention_hidden).squeeze(-1)
        feature_scores = feature_scores.masked_fill(~feature_mask, -10000.0)

        feature_weights = torch.softmax(feature_scores, dim=-1)
        feature_weights = feature_weights * feature_mask.to(feature_weights.dtype)
        feature_weights = feature_weights / feature_weights.sum(
            dim=-1,
            keepdim=True,
        ).clamp(min=1e-12)

        feature_repr = torch.sum(feature_weights.unsqueeze(-1) * feature_emb, dim=2)
        has_feature = feature_mask.any(dim=-1)
        feature_sequence_mask = has_feature & attention_mask.bool()

        embeddings = (
            feature_repr
            + self.position_embeddings(self._position_ids(input_ids))
        )
        embeddings = self.feature_layer_norm(embeddings)
        embeddings = self.dropout(embeddings)
        embeddings = embeddings * feature_sequence_mask.unsqueeze(-1).to(
            embeddings.dtype
        )
        return embeddings, feature_sequence_mask.long()

    def encode(
        self,
        input_ids: torch.Tensor,
        feature_ids: torch.Tensor,
        attention_mask: torch.Tensor = None,
    ) -> torch.Tensor:
        if attention_mask is None:
            attention_mask = input_ids.ne(self.tokenizer.padding_token).long()

        item_emb = self._item_sequence_embedding(input_ids, attention_mask)
        item_attention_mask = self._build_attention_mask(attention_mask)
        item_layers = self.item_encoder(
            item_emb,
            item_attention_mask,
            output_all_encoded_layers=True,
        )
        item_output = item_layers[-1] * attention_mask.unsqueeze(-1).to(
            item_layers[-1].dtype
        )

        feature_emb, feature_sequence_mask = self._feature_sequence_embedding(
            feature_ids,
            input_ids,
            attention_mask,
        )
        feature_attention_mask = self._build_attention_mask(feature_sequence_mask)
        feature_layers = self.feature_encoder(
            feature_emb,
            feature_attention_mask,
            output_all_encoded_layers=True,
        )
        feature_output = feature_layers[-1] * feature_sequence_mask.unsqueeze(-1).to(
            feature_layers[-1].dtype
        )

        fused = self.fusion(torch.cat([item_output, feature_output], dim=-1))
        return fused * attention_mask.unsqueeze(-1).to(fused.dtype)

    def _compute_logits(self, sequence_output: torch.Tensor) -> torch.Tensor:
        logits = torch.matmul(sequence_output, self.item_embeddings.weight.T)
        mask_shape = [1] * logits.dim()
        mask_shape[-1] = -1
        return logits.masked_fill(
            ~self.valid_item_mask.view(*mask_shape),
            -10000.0,
        )

    def _filter_seen_items(self, logits: torch.Tensor, batch: dict) -> torch.Tensor:
        if not self.config.get("filter_seen_items", False):
            return logits

        seen_items = batch["input_ids"]
        seen_mask = (seen_items > 0) & (seen_items < self.dataset.n_items)
        if "labels" in batch:
            labels = batch["labels"]
            if labels.dim() == 1:
                labels = labels.unsqueeze(1)
            labels = labels.masked_fill(labels == self.tokenizer.ignored_label, 0)
            seen_mask = seen_mask & (
                seen_items.unsqueeze(-1) != labels.unsqueeze(1)
            ).all(dim=-1)

        seen_items = seen_items.masked_fill(~seen_mask, 0)
        logits = logits.clone()
        logits.scatter_(1, seen_items, -10000.0)
        logits[:, self.tokenizer.padding_token] = -10000.0
        return logits

    def forward(self, batch: dict):
        sequence_output = self.encode(
            input_ids=batch["input_ids"],
            feature_ids=batch["feature_ids"],
            attention_mask=batch["attention_mask"],
        )
        logits = self._compute_logits(sequence_output)
        loss = self.loss_fct(
            logits.view(-1, logits.shape[-1]),
            batch["labels"].view(-1),
        )
        return SimpleNamespace(loss=loss, logits=logits, hidden_states=sequence_output)

    def gather_index(self, output: torch.Tensor, index: torch.Tensor) -> torch.Tensor:
        index = index.view(-1, 1, 1).expand(-1, -1, output.shape[-1])
        return output.gather(dim=1, index=index).squeeze(1)

    def generate(self, batch: dict, n_return_sequences: int = 1, num_beams=None):
        sequence_output = self.encode(
            input_ids=batch["input_ids"],
            feature_ids=batch["feature_ids"],
            attention_mask=batch["attention_mask"],
        )
        final_state = self.gather_index(sequence_output, batch["seq_lens"] - 1)
        logits = self._compute_logits(final_state)
        logits = self._filter_seen_items(logits, batch)
        preds = logits.topk(n_return_sequences, dim=-1).indices
        return preds.unsqueeze(-1)

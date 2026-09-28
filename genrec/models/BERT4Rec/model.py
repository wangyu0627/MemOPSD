from types import SimpleNamespace

import torch
import torch.nn as nn
import torch.nn.functional as F

from genrec.dataset import AbstractDataset
from genrec.model import AbstractModel
from genrec.models.S3Rec.layers import ACT2FN, Encoder, LayerNorm
from genrec.tokenizer import AbstractTokenizer


class BERT4RecConfig:
    def __init__(self, config: dict):
        self.hidden_size = config["hidden_size"]
        self.num_hidden_layers = config["num_hidden_layers"]
        self.num_attention_heads = config["num_attention_heads"]
        self.hidden_act = config["hidden_act"]
        self.hidden_dropout_prob = config["hidden_dropout_prob"]
        self.attention_probs_dropout_prob = config["attention_probs_dropout_prob"]
        self.initializer_range = config["initializer_range"]
        self.layer_norm_epsilon = config["layer_norm_epsilon"]


class BERT4Rec(AbstractModel):
    """
    BERT4Rec adapted to the local genrec interface.

    This is a bidirectional Transformer encoder trained with masked item
    prediction. Evaluation masks the final item in each sequence and ranks real
    item ids from the hidden state at the masked position.
    """

    def __init__(
        self,
        config: dict,
        dataset: AbstractDataset,
        tokenizer: AbstractTokenizer,
    ):
        super(BERT4Rec, self).__init__(config, dataset, tokenizer)
        self.loss_fct = nn.CrossEntropyLoss(ignore_index=tokenizer.ignored_label)
        valid_item_mask = torch.zeros(tokenizer.vocab_size, dtype=torch.bool)
        valid_item_mask[1 : dataset.n_items] = True
        self.register_buffer("valid_item_mask", valid_item_mask)

        self.bert_config = BERT4RecConfig(config)

        self.item_embeddings = nn.Embedding(
            tokenizer.vocab_size,
            config["hidden_size"],
            padding_idx=tokenizer.padding_token,
        )
        self.position_embeddings = nn.Embedding(
            tokenizer.max_token_seq_len,
            config["hidden_size"],
        )
        self.token_type_embeddings = nn.Embedding(
            config["type_vocab_size"],
            config["hidden_size"],
        )
        self.LayerNorm = LayerNorm(
            config["hidden_size"],
            eps=config["layer_norm_epsilon"],
        )
        self.dropout = nn.Dropout(config["hidden_dropout_prob"])
        self.encoder = Encoder(self.bert_config)

        self.prediction_dense = nn.Linear(config["hidden_size"], config["hidden_size"])
        self.prediction_layer_norm = LayerNorm(
            config["hidden_size"],
            eps=config["layer_norm_epsilon"],
        )
        self.output_bias = nn.Parameter(torch.zeros(tokenizer.vocab_size))

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
            for p in self.token_type_embeddings.parameters()
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

    def add_embedding(self, input_ids: torch.Tensor) -> torch.Tensor:
        seq_length = input_ids.size(1)
        position_ids = torch.arange(
            seq_length,
            dtype=torch.long,
            device=input_ids.device,
        )
        position_ids = position_ids.unsqueeze(0).expand_as(input_ids)
        token_type_ids = torch.zeros_like(input_ids)

        embeddings = (
            self.item_embeddings(input_ids)
            + self.position_embeddings(position_ids)
            + self.token_type_embeddings(token_type_ids)
        )
        embeddings = self.LayerNorm(embeddings)
        return self.dropout(embeddings)

    def _build_attention_mask(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor = None,
    ) -> torch.Tensor:
        if attention_mask is None:
            attention_mask = (input_ids != self.tokenizer.padding_token).long()
        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
        extended_attention_mask = extended_attention_mask.to(
            dtype=next(self.parameters()).dtype,
        )
        return (1.0 - extended_attention_mask) * -10000.0

    def encode(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor = None,
    ) -> torch.Tensor:
        sequence_emb = self.add_embedding(input_ids)
        extended_attention_mask = self._build_attention_mask(
            input_ids,
            attention_mask=attention_mask,
        )
        encoded_layers = self.encoder(
            sequence_emb,
            extended_attention_mask,
            output_all_encoded_layers=True,
        )
        return encoded_layers[-1]

    def _prediction_scores(self, sequence_output: torch.Tensor) -> torch.Tensor:
        hidden_states = self.prediction_dense(sequence_output)
        hidden_states = ACT2FN[self.config["hidden_act"]](hidden_states)
        hidden_states = self.prediction_layer_norm(hidden_states)
        logits = F.linear(hidden_states, self.item_embeddings.weight, self.output_bias)

        mask_shape = [1] * logits.dim()
        mask_shape[-1] = -1
        return logits.masked_fill(
            ~self.valid_item_mask.view(*mask_shape),
            -1e4,
        )

    def _filter_seen_items(self, logits: torch.Tensor, batch: dict) -> torch.Tensor:
        if not self.config.get("filter_seen_items", True):
            return logits
        if "input_ids" not in batch:
            return logits

        seen_items = batch.get("seen_item_ids", batch["input_ids"])
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
        logits.scatter_(1, seen_items, -1e4)
        logits[:, self.tokenizer.padding_token] = -1e4
        return logits

    def forward(self, batch: dict):
        sequence_output = self.encode(
            batch["input_ids"],
            attention_mask=batch["attention_mask"],
        )
        logits = self._prediction_scores(sequence_output)
        loss = self.loss_fct(
            logits.reshape(-1, logits.shape[-1]),
            batch["labels"].reshape(-1),
        )
        return SimpleNamespace(loss=loss, logits=logits, hidden_states=sequence_output)

    def gather_index(self, output: torch.Tensor, index: torch.Tensor) -> torch.Tensor:
        index = index.view(-1, 1, 1).expand(-1, -1, output.shape[-1])
        return output.gather(dim=1, index=index).squeeze(1)

    def generate(self, batch: dict, n_return_sequences: int = 1, num_beams=None):
        sequence_output = self.encode(
            batch["input_ids"],
            attention_mask=batch["attention_mask"],
        )
        if "mask_positions" in batch:
            gather_positions = batch["mask_positions"]
        else:
            gather_positions = batch["seq_lens"] - 1
        final_state = self.gather_index(sequence_output, gather_positions)
        logits = self._prediction_scores(final_state)
        logits = self._filter_seen_items(logits, batch)
        preds = logits.topk(n_return_sequences, dim=-1).indices
        return preds.unsqueeze(-1)

import math
from types import SimpleNamespace

import torch
import torch.nn as nn
import torch.nn.functional as F

from genrec.dataset import AbstractDataset
from genrec.model import AbstractModel
from genrec.tokenizer import AbstractTokenizer


class RelativePositionalBias(nn.Module):
    """Shared relative positional bias used by the HSTU attention kernel."""

    def __init__(self, max_seq_len: int, initializer_range: float) -> None:
        super().__init__()
        self.max_seq_len = max_seq_len
        self.weight = nn.Parameter(torch.empty(2 * max_seq_len - 1))
        nn.init.normal_(self.weight, mean=0.0, std=initializer_range)

    def forward(self, seq_len: int) -> torch.Tensor:
        positions = torch.arange(seq_len, device=self.weight.device)
        rel_pos = positions[:, None] - positions[None, :]
        rel_pos = rel_pos.clamp(
            min=-(self.max_seq_len - 1),
            max=self.max_seq_len - 1,
        )
        return self.weight[rel_pos + self.max_seq_len - 1]


class HSTUBlock(nn.Module):
    """
    Dense PyTorch adaptation of the HSTU sequential transduction unit.

    The block follows the public Meta generative-recommenders implementation:
    layer norm -> UVQK projection -> SiLU attention -> gated output projection
    -> residual connection.
    """

    def __init__(
        self,
        embedding_dim: int,
        num_heads: int,
        linear_dim: int,
        attention_dim: int,
        max_seq_len: int,
        linear_dropout_rate: float,
        attn_dropout_rate: float,
        linear_activation: str,
        enable_relative_attention_bias: bool,
        layer_norm_epsilon: float,
        initializer_range: float,
    ) -> None:
        super().__init__()
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.linear_dim = linear_dim
        self.attention_dim = attention_dim
        self.linear_activation = linear_activation
        self.layer_norm_epsilon = layer_norm_epsilon
        self.attn_dropout_rate = attn_dropout_rate

        projection_dim = num_heads * (2 * linear_dim + 2 * attention_dim)
        self.uvqk = nn.Parameter(torch.empty(embedding_dim, projection_dim))
        nn.init.normal_(self.uvqk, mean=0.0, std=initializer_range)

        self.output = nn.Linear(num_heads * linear_dim, embedding_dim)
        nn.init.xavier_uniform_(self.output.weight)
        nn.init.zeros_(self.output.bias)

        self.dropout = nn.Dropout(linear_dropout_rate)
        self.relative_attention_bias = (
            RelativePositionalBias(max_seq_len, initializer_range)
            if enable_relative_attention_bias
            else None
        )

    def _linear_projection(self, x: torch.Tensor):
        uvqk = torch.matmul(
            F.layer_norm(
                x,
                normalized_shape=(self.embedding_dim,),
                eps=self.layer_norm_epsilon,
            ),
            self.uvqk,
        )
        if self.linear_activation == "silu":
            uvqk = F.silu(uvqk)
        elif self.linear_activation != "none":
            raise ValueError(f"Unknown HSTU linear activation: {self.linear_activation}")

        split_sizes = [
            self.num_heads * self.linear_dim,
            self.num_heads * self.linear_dim,
            self.num_heads * self.attention_dim,
            self.num_heads * self.attention_dim,
        ]
        return torch.split(uvqk, split_sizes, dim=-1)

    def forward(self, x: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape
        u, v, q, k = self._linear_projection(x)

        u = u.view(batch_size, seq_len, self.num_heads, self.linear_dim)
        v = v.view(batch_size, seq_len, self.num_heads, self.linear_dim)
        q = q.view(batch_size, seq_len, self.num_heads, self.attention_dim)
        k = k.view(batch_size, seq_len, self.num_heads, self.attention_dim)

        qk_attn = torch.einsum("bnhd,bmhd->bhnm", q, k)
        if self.relative_attention_bias is not None:
            qk_attn = qk_attn + self.relative_attention_bias(seq_len).view(
                1, 1, seq_len, seq_len
            )

        qk_attn = F.silu(qk_attn) / seq_len
        causal_mask = torch.ones(
            (seq_len, seq_len),
            device=x.device,
            dtype=torch.bool,
        ).tril()
        valid_mask = (
            causal_mask.view(1, 1, seq_len, seq_len)
            & attention_mask.view(batch_size, 1, 1, seq_len).bool()
            & attention_mask.view(batch_size, 1, seq_len, 1).bool()
        )
        qk_attn = qk_attn * valid_mask.to(qk_attn.dtype)
        if self.attn_dropout_rate > 0.0:
            qk_attn = F.dropout(
                qk_attn,
                p=self.attn_dropout_rate,
                training=self.training,
            )

        attn_output = torch.einsum("bhnm,bmhd->bnhd", qk_attn, v)
        attn_output = attn_output.reshape(
            batch_size,
            seq_len,
            self.num_heads * self.linear_dim,
        )
        attn_output = F.layer_norm(
            attn_output,
            normalized_shape=(self.num_heads * self.linear_dim,),
            eps=self.layer_norm_epsilon,
        )

        gated_output = u.reshape(batch_size, seq_len, -1) * attn_output
        output = self.output(self.dropout(gated_output))
        output = output + x
        return output * attention_mask.unsqueeze(-1).to(output.dtype)


class HSTU(AbstractModel):
    """
    HSTU for the local genrec framework.

    This ports the sequential HSTU encoder from Meta's generative-recommenders
    codebase into the same autoregressive next-item interface used by SASRec.
    """

    def __init__(
        self,
        config: dict,
        dataset: AbstractDataset,
        tokenizer: AbstractTokenizer,
    ) -> None:
        super(HSTU, self).__init__(config, dataset, tokenizer)
        self.loss_fct = nn.CrossEntropyLoss(ignore_index=tokenizer.ignored_label)
        valid_item_mask = torch.zeros(tokenizer.vocab_size, dtype=torch.bool)
        valid_item_mask[1 : dataset.n_items] = True
        self.register_buffer("valid_item_mask", valid_item_mask)

        self.item_embedding = nn.Embedding(
            tokenizer.vocab_size,
            config["n_embd"],
            padding_idx=tokenizer.padding_token,
        )
        nn.init.normal_(
            self.item_embedding.weight,
            mean=0.0,
            std=config["initializer_range"],
        )
        with torch.no_grad():
            self.item_embedding.weight[tokenizer.padding_token].zero_()

        self.position_embedding = nn.Embedding(
            tokenizer.max_token_seq_len,
            config["n_embd"],
        )
        nn.init.normal_(
            self.position_embedding.weight,
            mean=0.0,
            std=math.sqrt(1.0 / config["n_embd"]),
        )
        self.embedding_dropout = nn.Dropout(config["dropout_rate"])

        self.blocks = nn.ModuleList(
            [
                HSTUBlock(
                    embedding_dim=config["n_embd"],
                    num_heads=config["num_heads"],
                    linear_dim=config["dv"],
                    attention_dim=config["dqk"],
                    max_seq_len=tokenizer.max_token_seq_len,
                    linear_dropout_rate=config["linear_dropout_rate"],
                    attn_dropout_rate=config["attn_dropout_rate"],
                    linear_activation=config["linear_activation"],
                    enable_relative_attention_bias=config[
                        "enable_relative_attention_bias"
                    ],
                    layer_norm_epsilon=config["layer_norm_epsilon"],
                    initializer_range=config["initializer_range"],
                )
                for _ in range(config["num_blocks"])
            ]
        )

        self.temperature = config["temperature"]
        self.item_l2_norm = config["item_l2_norm"]
        self.user_l2_norm = config["user_l2_norm"]

    @property
    def n_parameters(self) -> str:
        total_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        emb_params = sum(
            p.numel() for p in self.item_embedding.parameters() if p.requires_grad
        )
        emb_params += sum(
            p.numel() for p in self.position_embedding.parameters() if p.requires_grad
        )
        return (
            f"#Embedding parameters: {emb_params}\n"
            f"#Non-embedding parameters: {total_params - emb_params}\n"
            f"#Total trainable parameters: {total_params}\n"
        )

    def _encode(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        batch_size, seq_len = input_ids.shape
        positions = torch.arange(seq_len, device=input_ids.device).unsqueeze(0)
        hidden_states = (
            self.item_embedding(input_ids) * math.sqrt(self.config["n_embd"])
            + self.position_embedding(positions.expand(batch_size, -1))
        )
        hidden_states = self.embedding_dropout(hidden_states)
        hidden_states = hidden_states * attention_mask.unsqueeze(-1).to(
            hidden_states.dtype
        )

        for block in self.blocks:
            hidden_states = block(hidden_states, attention_mask)

        if self.user_l2_norm:
            hidden_states = F.normalize(hidden_states, dim=-1)
        return hidden_states

    def _compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
        item_embeddings = self.item_embedding.weight
        if self.item_l2_norm:
            item_embeddings = F.normalize(item_embeddings, dim=-1)
        logits = torch.matmul(hidden_states, item_embeddings.T) / self.temperature
        mask_shape = [1] * logits.dim()
        mask_shape[-1] = -1
        return logits.masked_fill(
            ~self.valid_item_mask.view(*mask_shape),
            -1e4,
        )

    def forward(self, batch: dict):
        hidden_states = self._encode(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
        )
        logits = self._compute_logits(hidden_states)
        loss = self.loss_fct(
            logits.view(-1, logits.shape[-1]),
            batch["labels"].view(-1),
        )
        return SimpleNamespace(loss=loss, logits=logits, hidden_states=hidden_states)

    def gather_index(self, output: torch.Tensor, index: torch.Tensor) -> torch.Tensor:
        index = index.view(-1, 1, 1).expand(-1, -1, output.shape[-1])
        return output.gather(dim=1, index=index).squeeze(1)

    def generate(
        self,
        batch: dict,
        n_return_sequences: int = 1,
        num_beams=None,
    ) -> torch.Tensor:
        hidden_states = self._encode(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
        )
        final_state = self.gather_index(hidden_states, batch["seq_lens"] - 1)
        logits = self._compute_logits(final_state)
        preds = logits.topk(n_return_sequences, dim=-1).indices
        return preds.unsqueeze(-1)

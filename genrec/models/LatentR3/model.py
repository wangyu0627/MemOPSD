import math
from types import SimpleNamespace

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import T5Config, T5ForConditionalGeneration

from genrec.dataset import AbstractDataset
from genrec.model import AbstractModel
from genrec.tokenizer import AbstractTokenizer


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    half = x.shape[-1] // 2
    x1 = x[..., :half]
    x2 = x[..., half:]
    return torch.cat((-x2, x1), dim=-1)


def apply_vanilla_rope(
    query: torch.Tensor,
    key: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    return (query * cos) + (rotate_half(query) * sin), (key * cos) + (
        rotate_half(key) * sin
    )


class VanillaRoPE(nn.Module):
    def __init__(self, hidden_size: int, base: float = 1000.0) -> None:
        super().__init__()
        if hidden_size % 2 != 0:
            raise ValueError("LatentR3 requires an even hidden size for RoPE.")
        inv_freq = 1.0 / (
            base ** (torch.arange(0, hidden_size, 2).float() / hidden_size)
        )
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    @torch.no_grad()
    def forward(
        self,
        x: torch.Tensor,
        position_ids: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        inv_freq = self.inv_freq.to(device=x.device)
        inv_freq_expanded = inv_freq[None, :, None].float().expand(
            position_ids.shape[0],
            -1,
            1,
        )
        position_ids_expanded = position_ids[:, None, :].float()
        device_type = x.device.type
        device_type = device_type if device_type != "mps" else "cpu"
        with torch.autocast(device_type=device_type, enabled=False):
            freqs = (
                inv_freq_expanded.float() @ position_ids_expanded.float()
            ).transpose(1, 2)
            emb = torch.cat((freqs, freqs), dim=-1)
            cos = emb.cos()
            sin = emb.sin()
        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)


class LatentSelfAttention(nn.Module):
    """
    R3 latent-thought attention over hidden states before the thought position.

    The attention layer is ported from R3-main, but the surrounding backbone is
    the local lightweight T5 so model-size comparisons stay aligned with TIGER.
    """

    def __init__(
        self,
        hidden_size: int,
        end_k: int = -1,
        rope_base: float = 1000.0,
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.end_k = end_k
        self.query = nn.Linear(hidden_size, hidden_size)
        self.key = nn.Linear(hidden_size, hidden_size)
        self.value = nn.Linear(hidden_size, hidden_size)
        self.scale_score = 1.0 / math.sqrt(hidden_size)
        self.rope = VanillaRoPE(hidden_size, base=rope_base)

    @staticmethod
    def mask_to_weights(
        attention_mask: torch.Tensor,
        thought_id_idx: torch.Tensor,
        end_k: int = -1,
    ) -> torch.Tensor:
        if attention_mask.size(0) != thought_id_idx.size(0):
            raise ValueError("attention_mask must match thought_id_idx batch size.")

        visible = torch.zeros_like(attention_mask, dtype=torch.bool)
        base_mask = attention_mask.bool()
        for i in range(attention_mask.size(0)):
            idx = int(thought_id_idx[i].item())
            start_idx = max(0, idx - end_k) if end_k != -1 else 0
            if start_idx < idx:
                visible[i, start_idx:idx] = base_mask[i, start_idx:idx]
            if not visible[i].any() and idx > 0:
                visible[i, idx - 1] = base_mask[i, idx - 1]

        attention_weight = torch.zeros(
            attention_mask.shape,
            dtype=torch.float32,
            device=attention_mask.device,
        )
        return attention_weight.masked_fill(~visible, -10000.0)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
        thought_id_idx: torch.Tensor,
    ) -> torch.Tensor:
        attention_weights = self.mask_to_weights(
            attention_mask=attention_mask,
            thought_id_idx=thought_id_idx,
            end_k=self.end_k,
        )
        position_ids = torch.arange(
            hidden_states.shape[1],
            device=hidden_states.device,
        ).unsqueeze(0)
        cos, sin = self.rope(hidden_states, position_ids=position_ids)

        query = self.query(hidden_states)
        key = self.key(hidden_states)
        value = self.value(hidden_states)
        query, key = apply_vanilla_rope(query, key, cos, sin)

        batch_size, seq_len, _ = hidden_states.shape
        indices = thought_id_idx - 1
        if (indices < 0).any() or (indices >= seq_len).any():
            raise ValueError("thought_id_idx - 1 is outside the sequence.")

        query_selected = query[
            torch.arange(batch_size, device=hidden_states.device),
            indices,
        ].unsqueeze(1)
        attn_scores = (
            torch.matmul(query_selected, key.transpose(-2, -1)) * self.scale_score
        )
        attn_scores = attn_scores.float() + attention_weights.unsqueeze(1)
        attn_weights = F.softmax(attn_scores, dim=-1).to(value.dtype)
        return torch.matmul(attn_weights, value).squeeze(1)


class LatentR3(AbstractModel):
    """
    LatentR3 with the same T5 backbone size and semantic tokenizer as TIGER.

    The local T5 adaptation places the latent thought on the decoder side:
    decoder_start, latent_thought, target semantic IDs. This keeps the vocabulary
    exactly equal to TIGER while forcing the latent thought state to predict the
    first target code, matching the role it plays in decoder-only R3.
    """

    def __init__(
        self,
        config: dict,
        dataset: AbstractDataset,
        tokenizer: AbstractTokenizer,
    ) -> None:
        super().__init__(config, dataset, tokenizer)

        self.n_digit = tokenizer.n_digit
        self.model_vocab_size = tokenizer.vocab_size
        self.constrained_generation = config.get(
            "latentr3_constrained_generation",
            True,
        )

        t5config = T5Config(
            num_layers=config["num_layers"],
            num_decoder_layers=config["num_decoder_layers"],
            d_model=config["d_model"],
            d_ff=config["d_ff"],
            num_heads=config["num_heads"],
            d_kv=config["d_kv"],
            dropout_rate=config["dropout_rate"],
            activation_function=config["activation_function"],
            vocab_size=self.model_vocab_size,
            pad_token_id=tokenizer.padding_token,
            eos_token_id=tokenizer.eos_token,
            decoder_start_token_id=0,
            feed_forward_proj=config["feed_forward_proj"],
            n_positions=tokenizer.max_token_seq_len,
        )
        self.t5 = T5ForConditionalGeneration(config=t5config)
        self.attention = LatentSelfAttention(
            hidden_size=config["d_model"],
            end_k=config.get("latentr3_end_k", config.get("end_k", -1)),
            rope_base=config.get("latentr3_rope_base", 1000.0),
        )
        self.prefix_trie = self._build_prefix_trie()

        item_token_ids = sorted(
            {
                int(token)
                for tokens in tokenizer.item2tokens.values()
                for token in tokens
            }
        )
        self.register_buffer(
            "item_token_ids",
            torch.tensor(item_token_ids, dtype=torch.long),
            persistent=False,
        )
        self.register_buffer(
            "all_token_ids",
            torch.arange(tokenizer.vocab_size, dtype=torch.long),
            persistent=False,
        )

    def _build_prefix_trie(self) -> dict:
        root = {}
        for tokens in self.tokenizer.item2tokens.values():
            node = root
            for token in tokens:
                node = node.setdefault(int(token), {})
        return root

    @property
    def n_parameters(self) -> str:
        total_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        emb_params = sum(
            p.numel()
            for p in self.t5.get_input_embeddings().parameters()
            if p.requires_grad
        )
        latent_attention_params = sum(
            p.numel() for p in self.attention.parameters() if p.requires_grad
        )
        backbone_non_emb = total_params - emb_params - latent_attention_params
        return (
            f"#Backbone: T5 matched to TIGER\n"
            f"#Embedding parameters: {emb_params}\n"
            f"#Latent attention parameters: {latent_attention_params}\n"
            f"#Backbone non-embedding parameters: {backbone_non_emb}\n"
            f"#Non-embedding parameters: {total_params - emb_params}\n"
            f"#Total trainable parameters: {total_params}\n"
        )

    def _latent_decoder_context(
        self,
        batch: dict,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        input_ids = batch["input_ids"].long()
        attention_mask = batch["attention_mask"].long()
        encoder_outputs = self.t5.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True,
        )
        encoder_hidden_states = encoder_outputs.last_hidden_state

        thought_positions = attention_mask.sum(dim=1).long().clamp(min=1)
        latent_embeds = self.attention(
            hidden_states=encoder_hidden_states,
            attention_mask=attention_mask,
            thought_id_idx=thought_positions,
        )
        return encoder_hidden_states, attention_mask, latent_embeds

    def _decoder_inputs_for_labels(
        self,
        labels: torch.Tensor,
        latent_embeds: torch.Tensor,
    ) -> torch.Tensor:
        batch_size = labels.shape[0]
        device = labels.device
        start_ids = torch.full(
            (batch_size, 1),
            self.t5.config.decoder_start_token_id,
            dtype=torch.long,
            device=device,
        )
        label_prefix = labels[:, :-1].masked_fill(
            labels[:, :-1] == -100,
            self.tokenizer.padding_token,
        )
        parts = [
            self.t5.shared(start_ids),
            latent_embeds.unsqueeze(1),
            self.t5.shared(label_prefix),
        ]
        return torch.cat(parts, dim=1)

    def forward(
        self,
        batch: dict,
        grpo_sequences: torch.Tensor = None,
        grpo_temperature: float = 1.0,
    ):
        if grpo_sequences is not None:
            return SimpleNamespace(
                logps=self.sequence_log_probs(
                    batch=batch,
                    sequences=grpo_sequences,
                    temperature=grpo_temperature,
                )
            )

        encoder_hidden_states, encoder_attention_mask, latent_embeds = (
            self._latent_decoder_context(batch)
        )
        labels = batch["labels"].to(encoder_hidden_states.device).long()
        decoder_inputs_embeds = self._decoder_inputs_for_labels(labels, latent_embeds)
        decoder_attention_mask = torch.ones(
            decoder_inputs_embeds.shape[:2],
            dtype=encoder_attention_mask.dtype,
            device=decoder_inputs_embeds.device,
        )

        decoder_outputs = self.t5.decoder(
            inputs_embeds=decoder_inputs_embeds,
            attention_mask=decoder_attention_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            use_cache=False,
            return_dict=True,
        )
        sequence_output = decoder_outputs.last_hidden_state[:, 1:, :]
        if self.t5.config.tie_word_embeddings:
            sequence_output = sequence_output * (self.t5.model_dim**-0.5)
        logits = self.t5.lm_head(sequence_output)
        loss = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            labels.reshape(-1),
            ignore_index=-100,
        )
        return SimpleNamespace(loss=loss, logits=logits)

    def _allowed_tokens_for_prefix(self, prefix: list[int]) -> list[int]:
        node = self.prefix_trie
        for token in prefix:
            node = node.get(int(token))
            if node is None:
                return []
        return list(node.keys())

    def _candidate_token_ids(self, prefix: list[int], device: torch.device):
        if not self.constrained_generation:
            return self.all_token_ids.to(device)

        allowed = self._allowed_tokens_for_prefix(prefix)
        if not allowed:
            return None
        return torch.tensor(allowed, dtype=torch.long, device=device)

    def _mask_logits_for_prefixes(
        self,
        logits: torch.Tensor,
        prefixes: torch.Tensor,
    ) -> torch.Tensor:
        if not self.constrained_generation:
            return logits

        masked_logits = logits.new_full(logits.shape, -10000.0)
        for row, prefix in enumerate(prefixes.detach().cpu().tolist()):
            allowed = self._candidate_token_ids(prefix, logits.device)
            if allowed is None or allowed.numel() == 0:
                allowed = self.item_token_ids.to(logits.device)
            masked_logits[row, allowed] = logits[row, allowed]
        return masked_logits

    def _beam_step(
        self,
        logits: torch.Tensor,
        prefixes: torch.Tensor,
        beam_scores: torch.Tensor,
        num_beams: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size, cur_beams, stage = prefixes.shape
        vocab_size = logits.shape[-1]
        token_scores = torch.log_softmax(logits, dim=-1).view(
            batch_size,
            cur_beams,
            vocab_size,
        )

        next_prefixes = []
        next_scores = []
        fallback = self.item_token_ids.to(logits.device)

        for batch_idx in range(batch_size):
            batch_scores = []
            batch_beams = []
            batch_tokens = []

            for beam_idx in range(cur_beams):
                prefix = prefixes[batch_idx, beam_idx].detach().cpu().tolist()
                allowed = self._candidate_token_ids(prefix, logits.device)
                if allowed is None or allowed.numel() == 0:
                    continue

                scores = token_scores[batch_idx, beam_idx, allowed]
                scores = scores + beam_scores[batch_idx, beam_idx]
                batch_scores.append(scores)
                batch_beams.append(torch.full_like(allowed, beam_idx))
                batch_tokens.append(allowed)

            if not batch_scores:
                scores = token_scores[batch_idx, 0, fallback] + beam_scores[
                    batch_idx,
                    0,
                ]
                batch_scores.append(scores)
                batch_beams.append(torch.zeros_like(fallback))
                batch_tokens.append(fallback)

            candidate_scores = torch.cat(batch_scores, dim=0)
            candidate_beams = torch.cat(batch_beams, dim=0)
            candidate_tokens = torch.cat(batch_tokens, dim=0)

            keep = min(num_beams, candidate_scores.numel())
            top_scores, top_indices = candidate_scores.topk(keep, dim=0)
            selected_beams = candidate_beams[top_indices]
            selected_tokens = candidate_tokens[top_indices]
            selected_prefixes = prefixes[batch_idx, selected_beams]
            selected_prefixes = torch.cat(
                [selected_prefixes, selected_tokens.unsqueeze(-1)],
                dim=-1,
            )

            if keep < num_beams:
                pad_count = num_beams - keep
                selected_prefixes = torch.cat(
                    [
                        selected_prefixes,
                        selected_prefixes[:1].expand(pad_count, stage + 1),
                    ],
                    dim=0,
                )
                top_scores = torch.cat(
                    [top_scores, top_scores[:1].expand(pad_count)],
                    dim=0,
                )

            next_prefixes.append(selected_prefixes)
            next_scores.append(top_scores)

        return torch.stack(next_prefixes, dim=0), torch.stack(next_scores, dim=0)

    def _repeat_context(
        self,
        encoder_hidden_states: torch.Tensor,
        encoder_attention_mask: torch.Tensor,
        latent_embeds: torch.Tensor,
        repeats: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return (
            encoder_hidden_states.repeat_interleave(repeats, dim=0),
            encoder_attention_mask.repeat_interleave(repeats, dim=0),
            latent_embeds.repeat_interleave(repeats, dim=0),
        )

    def _normalize_distill_layers(self, distill_layers) -> list[int]:
        if distill_layers is None:
            return [-1]
        if isinstance(distill_layers, int):
            return [distill_layers]
        if isinstance(distill_layers, str):
            cleaned = distill_layers.strip().strip("[]")
            if not cleaned:
                return [-1]
            return [int(layer.strip()) for layer in cleaned.split(",")]
        return [int(layer) for layer in distill_layers]

    def _oprd_decoder_inputs(
        self,
        flat_sequences: torch.Tensor,
        latent_embeds: torch.Tensor,
    ) -> torch.Tensor:
        batch_size = flat_sequences.shape[0]
        start_ids = torch.full(
            (batch_size, 1),
            self.t5.config.decoder_start_token_id,
            dtype=torch.long,
            device=flat_sequences.device,
        )
        parts = [
            self.t5.shared(start_ids),
            latent_embeds.unsqueeze(1),
        ]
        if flat_sequences.shape[1] > 1:
            parts.append(self.t5.shared(flat_sequences[:, :-1]))
        return torch.cat(parts, dim=1)

    def set_oprd_trainable(self) -> None:
        for param in self.attention.parameters():
            param.requires_grad = True

    def oprd_hidden_states(
        self,
        batch: dict,
        sequences: torch.Tensor,
        distill_layers=None,
    ) -> tuple[torch.Tensor, ...]:
        if sequences.dim() == 2:
            sequences = sequences.unsqueeze(1)
        if sequences.dim() != 3:
            raise ValueError(
                "sequences must have shape [batch, samples, n_digit]."
            )

        batch_size, n_samples, n_digit = sequences.shape
        if n_digit != self.n_digit:
            raise ValueError(f"Expected {self.n_digit} SID digits, got {n_digit}.")

        encoder_hidden, encoder_mask, latent = self._latent_decoder_context(batch)
        flat_encoder_hidden, flat_encoder_mask, flat_latent = self._repeat_context(
            encoder_hidden,
            encoder_mask,
            latent,
            n_samples,
        )
        flat_sequences = sequences.to(flat_encoder_hidden.device).long().reshape(
            batch_size * n_samples,
            n_digit,
        )
        decoder_inputs = self._oprd_decoder_inputs(flat_sequences, flat_latent)
        decoder_attention_mask = torch.ones(
            decoder_inputs.shape[:2],
            dtype=flat_encoder_mask.dtype,
            device=decoder_inputs.device,
        )
        decoder_outputs = self.t5.decoder(
            inputs_embeds=decoder_inputs,
            attention_mask=decoder_attention_mask,
            encoder_hidden_states=flat_encoder_hidden,
            encoder_attention_mask=flat_encoder_mask,
            use_cache=False,
            return_dict=True,
            output_hidden_states=True,
        )

        selected = []
        for layer_idx in self._normalize_distill_layers(distill_layers):
            states = decoder_outputs.hidden_states[layer_idx]
            states = states[:, 1:, :]
            selected.append(
                states.view(batch_size, n_samples, n_digit, -1)
            )
        return tuple(selected)

    def sample(
        self,
        batch: dict,
        n_return_sequences: int,
        temperature: float = 1.0,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size = batch["input_ids"].shape[0]
        temperature = max(float(temperature), 1e-6)
        with torch.no_grad():
            encoder_hidden_states, encoder_attention_mask, latent_embeds = (
                self._latent_decoder_context(batch)
            )
            prefixes = torch.empty(
                batch_size,
                n_return_sequences,
                0,
                dtype=torch.long,
                device=batch["input_ids"].device,
            )
            logps = torch.empty(
                batch_size,
                n_return_sequences,
                0,
                dtype=torch.float,
                device=batch["input_ids"].device,
            )
            flat_encoder_hidden, flat_encoder_mask, flat_latent = self._repeat_context(
                encoder_hidden_states,
                encoder_attention_mask,
                latent_embeds,
                n_return_sequences,
            )

            for stage in range(self.n_digit):
                flat_prefixes = prefixes.reshape(
                    batch_size * n_return_sequences,
                    stage,
                )
                logits = self._stage_logits(
                    encoder_hidden_states=flat_encoder_hidden,
                    encoder_attention_mask=flat_encoder_mask,
                    latent_embeds=flat_latent,
                    prev_tokens=flat_prefixes,
                )
                logits = self._mask_logits_for_prefixes(logits, flat_prefixes)
                token_logps = torch.log_softmax(logits / temperature, dim=-1)
                next_tokens = torch.multinomial(token_logps.exp(), num_samples=1)
                next_logps = token_logps.gather(1, next_tokens)
                next_tokens = next_tokens.view(batch_size, n_return_sequences)
                next_logps = next_logps.view(batch_size, n_return_sequences)
                prefixes = torch.cat([prefixes, next_tokens.unsqueeze(-1)], dim=-1)
                logps = torch.cat([logps, next_logps.unsqueeze(-1)], dim=-1)

        return prefixes, logps

    def sequence_log_probs(
        self,
        batch: dict,
        sequences: torch.Tensor,
        temperature: float = 1.0,
    ) -> torch.Tensor:
        batch_size, n_sequences, seq_len = sequences.shape
        if seq_len != self.n_digit:
            raise ValueError(
                f"Expected sequences with {self.n_digit} digits, got {seq_len}."
            )

        temperature = max(float(temperature), 1e-6)
        encoder_hidden_states, encoder_attention_mask, latent_embeds = (
            self._latent_decoder_context(batch)
        )
        flat_encoder_hidden, flat_encoder_mask, flat_latent = self._repeat_context(
            encoder_hidden_states,
            encoder_attention_mask,
            latent_embeds,
            n_sequences,
        )
        step_logps = []
        for stage in range(self.n_digit):
            flat_prefixes = sequences[:, :, :stage].reshape(
                batch_size * n_sequences,
                stage,
            )
            logits = self._stage_logits(
                encoder_hidden_states=flat_encoder_hidden,
                encoder_attention_mask=flat_encoder_mask,
                latent_embeds=flat_latent,
                prev_tokens=flat_prefixes,
            )
            logits = self._mask_logits_for_prefixes(logits, flat_prefixes)
            token_logps = torch.log_softmax(logits / temperature, dim=-1)
            target_tokens = sequences[:, :, stage].reshape(-1, 1)
            step_logps.append(
                token_logps.gather(1, target_tokens).view(batch_size, n_sequences)
            )
        return torch.stack(step_logps, dim=-1)

    def _stage_logits(
        self,
        encoder_hidden_states: torch.Tensor,
        encoder_attention_mask: torch.Tensor,
        latent_embeds: torch.Tensor,
        prev_tokens: torch.Tensor,
    ) -> torch.Tensor:
        batch_size = prev_tokens.shape[0]
        device = prev_tokens.device
        start_ids = torch.full(
            (batch_size, 1),
            self.t5.config.decoder_start_token_id,
            dtype=torch.long,
            device=device,
        )
        parts = [self.t5.shared(start_ids), latent_embeds.unsqueeze(1)]
        if prev_tokens.shape[1] > 0:
            parts.append(self.t5.shared(prev_tokens))
        decoder_inputs_embeds = torch.cat(parts, dim=1)
        decoder_attention_mask = torch.ones(
            decoder_inputs_embeds.shape[:2],
            dtype=encoder_attention_mask.dtype,
            device=device,
        )

        decoder_outputs = self.t5.decoder(
            inputs_embeds=decoder_inputs_embeds,
            attention_mask=decoder_attention_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            use_cache=False,
            return_dict=True,
        )
        sequence_output = decoder_outputs.last_hidden_state[:, -1, :]
        if self.t5.config.tie_word_embeddings:
            sequence_output = sequence_output * (self.t5.model_dim**-0.5)
        return self.t5.lm_head(sequence_output)

    def generate(
        self,
        batch: dict,
        n_return_sequences: int = 1,
        return_scores: bool = False,
        num_beams: int = None,
    ):
        num_beams = self.config["num_beams"] if num_beams is None else num_beams
        num_beams = max(num_beams, n_return_sequences)
        batch_size = batch["input_ids"].shape[0]

        with torch.no_grad():
            encoder_hidden_states, encoder_attention_mask, latent_embeds = (
                self._latent_decoder_context(batch)
            )
            prefixes = torch.empty(
                batch_size,
                1,
                0,
                dtype=torch.long,
                device=batch["input_ids"].device,
            )
            beam_scores = torch.zeros(
                batch_size,
                1,
                dtype=torch.float,
                device=batch["input_ids"].device,
            )

            for stage in range(self.n_digit):
                cur_beams = prefixes.shape[1]
                flat_prefixes = prefixes.reshape(batch_size * cur_beams, stage)
                flat_encoder_hidden, flat_encoder_mask, flat_latent = (
                    self._repeat_context(
                        encoder_hidden_states,
                        encoder_attention_mask,
                        latent_embeds,
                        cur_beams,
                    )
                )
                logits = self._stage_logits(
                    encoder_hidden_states=flat_encoder_hidden,
                    encoder_attention_mask=flat_encoder_mask,
                    latent_embeds=flat_latent,
                    prev_tokens=flat_prefixes,
                )
                prefixes, beam_scores = self._beam_step(
                    logits=logits,
                    prefixes=prefixes,
                    beam_scores=beam_scores,
                    num_beams=num_beams,
                )

        preds = prefixes[:, :n_return_sequences, :]
        scores = beam_scores[:, :n_return_sequences]
        if return_scores:
            return {"preds": preds, "scores": scores}
        return preds

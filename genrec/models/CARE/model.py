import torch
import torch.nn as nn
from transformers import T5Config, T5ForConditionalGeneration
from transformers.modeling_outputs import Seq2SeqLMOutput

from genrec.dataset import AbstractDataset
from genrec.model import AbstractModel
from genrec.tokenizer import AbstractTokenizer


class CARE(AbstractModel):
    """
    CARE adapted to the local TIGER-style generative recommender.

    This keeps the local TIGER tokenizer and lightweight T5 backbone while
    adding CARE's query-anchored reasoning.
    """

    def __init__(
        self,
        config: dict,
        dataset: AbstractDataset,
        tokenizer: AbstractTokenizer,
    ):
        super(CARE, self).__init__(config, dataset, tokenizer)
        t5config = T5Config(
            num_layers=config["num_layers"],
            num_decoder_layers=config["num_decoder_layers"],
            d_model=config["d_model"],
            d_ff=config["d_ff"],
            num_heads=config["num_heads"],
            d_kv=config["d_kv"],
            dropout_rate=config["dropout_rate"],
            activation_function=config["activation_function"],
            vocab_size=tokenizer.vocab_size,
            pad_token_id=tokenizer.padding_token,
            eos_token_id=tokenizer.eos_token,
            decoder_start_token_id=0,
            feed_forward_proj=config["feed_forward_proj"],
            n_positions=tokenizer.max_token_seq_len,
        )
        self.t5 = T5ForConditionalGeneration(config=t5config)
        hidden_size = config["d_model"]

        self.n_digit = tokenizer.n_digit
        self.query_list = self._fit_stage_list(
            config["care_query_list"],
            fill_value=1,
        )
        self.progressive_list = self._fit_stage_list(
            config["care_progressive_list"],
            fill_value=True,
        )
        self.progressive_attn = config["care_progressive_attn"]
        self.query_div_scale = config["care_query_div_scale"]

        self.n_query = sum(self.query_list)
        if self.n_query > 0:
            self.query_vector = nn.Embedding(self.n_query, hidden_size)
        else:
            self.query_vector = None

        self.loss_fct = nn.CrossEntropyLoss()
        self._validated_token_id_names = set()

    def _fit_stage_list(self, values: list, fill_value):
        values = list(values)
        if len(values) < self.n_digit:
            values.extend([fill_value] * (self.n_digit - len(values)))
        return values[: self.n_digit]

    def _validate_token_ids(self, name: str, token_ids: torch.Tensor):
        if name in self._validated_token_id_names or token_ids.numel() == 0:
            return

        min_id = int(token_ids.detach().min().cpu().item())
        max_id = int(token_ids.detach().max().cpu().item())
        max_valid_id = self.tokenizer.vocab_size - 1
        if min_id < 0 or max_id > max_valid_id:
            raise ValueError(
                f"CARE received {name} outside tokenizer vocabulary "
                f"[0, {max_valid_id}]: min={min_id}, max={max_id}. "
                "Check the cached semantic IDs and tokenizer config."
            )
        self._validated_token_id_names.add(name)

    @property
    def n_parameters(self) -> str:
        total_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        emb_params = sum(
            p.numel()
            for p in self.t5.get_input_embeddings().parameters()
            if p.requires_grad
        )
        query_params = (
            sum(p.numel() for p in self.query_vector.parameters() if p.requires_grad)
            if self.query_vector is not None
            else 0
        )
        backbone_non_emb = total_params - emb_params - query_params
        return (
            f"#Embedding parameters: {emb_params}\n"
            f"#CARE query parameters: {query_params}\n"
            f"#Backbone non-embedding parameters: {backbone_non_emb}\n"
            f"#Non-embedding parameters: {total_params - emb_params}\n"
            f"#Total trainable parameters: {total_params}\n"
        )

    def _progressive_encoder_mask(
        self,
        attention_mask: torch.Tensor,
        stage: int,
    ) -> torch.Tensor:
        if not self.progressive_attn or not self.progressive_list[stage]:
            return attention_mask

        base_mask = attention_mask.bool()
        batch_size, seq_len = attention_mask.shape
        positions = torch.arange(seq_len, device=attention_mask.device).unsqueeze(0)
        eos_pos = attention_mask.long().sum(dim=1, keepdim=True) - 1

        is_user = positions == 0
        is_eos = positions == eos_pos
        is_item_token = (positions > 0) & (positions < eos_pos)
        code_pos = (positions - 1).remainder(self.n_digit)
        visible_item = is_item_token & (code_pos <= stage)

        visible = (is_user | is_eos | visible_item) & base_mask
        return visible.to(dtype=attention_mask.dtype)

    def _decoder_inputs_embeds(
        self,
        prev_tokens: torch.Tensor,
        stage: int,
    ) -> torch.Tensor:
        batch_size = prev_tokens.shape[0]
        device = prev_tokens.device
        start_ids = torch.full(
            (batch_size, 1),
            self.t5.config.decoder_start_token_id,
            dtype=torch.long,
            device=device,
        )
        parts = [self.t5.shared(start_ids)]
        if prev_tokens.shape[1] > 0:
            parts.append(self.t5.shared(prev_tokens))

        n_query = self.query_list[stage]
        if n_query > 0:
            query_start = sum(self.query_list[:stage])
            query_ids = torch.arange(
                query_start,
                query_start + n_query,
                device=device,
            ).unsqueeze(0)
            query_ids = query_ids.expand(batch_size, -1)
            parts.append(self.query_vector(query_ids))

        return torch.cat(parts, dim=1)

    def _stage_logits(
        self,
        encoder_hidden_states: torch.Tensor,
        encoder_attention_mask: torch.Tensor,
        prev_tokens: torch.Tensor,
        stage: int,
    ) -> torch.Tensor:
        decoder_inputs_embeds, query_ranges = self._inference_decoder_inputs(
            prev_tokens,
            stage,
        )
        decoder_attention_mask = torch.ones(
            decoder_inputs_embeds.shape[:2],
            dtype=encoder_attention_mask.dtype,
            device=decoder_inputs_embeds.device,
        )
        stage_encoder_mask = self._training_encoder_attention_mask(
            attention_mask=encoder_attention_mask,
            decoder_len=decoder_inputs_embeds.shape[1],
            query_ranges=query_ranges,
        )

        decoder_outputs = self.t5.decoder(
            inputs_embeds=decoder_inputs_embeds,
            attention_mask=decoder_attention_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=stage_encoder_mask,
            use_cache=False,
            return_dict=True,
        )
        sequence_output = decoder_outputs.last_hidden_state[:, -1, :]
        if self.t5.config.tie_word_embeddings:
            sequence_output = sequence_output * (self.t5.model_dim**-0.5)
        return self.t5.lm_head(sequence_output)

    def _inference_decoder_inputs(
        self,
        prev_tokens: torch.Tensor,
        stage: int,
    ) -> tuple[torch.Tensor, list[tuple[int, int, int]]]:
        batch_size = prev_tokens.shape[0]
        device = prev_tokens.device
        start_ids = torch.full(
            (batch_size, 1),
            self.t5.config.decoder_start_token_id,
            dtype=torch.long,
            device=device,
        )
        parts = [self.t5.shared(start_ids)]
        query_ranges = []
        cur_pos = 1

        for cur_stage in range(stage + 1):
            n_query = self.query_list[cur_stage]
            query_start_pos = cur_pos
            if n_query > 0:
                query_start = sum(self.query_list[:cur_stage])
                query_ids = torch.arange(
                    query_start,
                    query_start + n_query,
                    device=device,
                ).unsqueeze(0)
                query_ids = query_ids.expand(batch_size, -1)
                parts.append(self.query_vector(query_ids))
                cur_pos += n_query
                query_ranges.append((cur_stage, query_start_pos, cur_pos))

            if cur_stage < stage:
                parts.append(self.t5.shared(prev_tokens[:, cur_stage : cur_stage + 1]))
                cur_pos += 1

        return torch.cat(parts, dim=1), query_ranges

    def _training_decoder_inputs(
        self,
        labels: torch.Tensor,
    ) -> tuple[torch.Tensor, list[int], list[tuple[int, int, int]]]:
        labels = labels.long()
        self._validate_token_ids("labels", labels)
        batch_size = labels.shape[0]
        device = labels.device
        start_ids = torch.full(
            (batch_size, 1),
            self.t5.config.decoder_start_token_id,
            dtype=torch.long,
            device=device,
        )
        parts = [self.t5.shared(start_ids)]
        pred_positions = []
        query_ranges = []
        cur_pos = 1

        for stage in range(self.n_digit):
            n_query = self.query_list[stage]
            query_start_pos = cur_pos
            if n_query > 0:
                query_start = sum(self.query_list[:stage])
                query_ids = torch.arange(
                    query_start,
                    query_start + n_query,
                    device=device,
                ).unsqueeze(0)
                query_ids = query_ids.expand(batch_size, -1)
                parts.append(self.query_vector(query_ids))
                cur_pos += n_query
                pred_positions.append(cur_pos - 1)
                query_ranges.append((stage, query_start_pos, cur_pos))
            else:
                pred_positions.append(cur_pos - 1)

            gold_code = labels[:, stage : stage + 1]
            parts.append(self.t5.shared(gold_code))
            cur_pos += 1

        return torch.cat(parts, dim=1), pred_positions, query_ranges

    def _training_encoder_attention_mask(
        self,
        attention_mask: torch.Tensor,
        decoder_len: int,
        query_ranges: list[tuple[int, int, int]],
    ) -> torch.Tensor:
        cross_mask = attention_mask[:, None, :].expand(
            -1,
            decoder_len,
            -1,
        ).clone()
        for stage, start, end in query_ranges:
            stage_mask = self._progressive_encoder_mask(attention_mask, stage)
            cross_mask[:, start:end, :] = stage_mask[:, None, :]
        return cross_mask

    def _query_diversity_loss(self) -> torch.Tensor:
        if self.query_vector is None or self.n_query <= 1:
            return torch.zeros((), device=next(self.parameters()).device)

        qv = nn.functional.normalize(self.query_vector.weight, dim=1)
        sim = torch.matmul(qv, qv.T)
        off_diag = ~torch.eye(sim.size(0), dtype=torch.bool, device=sim.device)
        return sim[off_diag].mean()

    def _query_embeddings(
        self,
        batch_size: int,
        stage: int,
        device: torch.device,
    ) -> torch.Tensor:
        n_query = self.query_list[stage]
        query_start = sum(self.query_list[:stage])
        query_ids = torch.arange(
            query_start,
            query_start + n_query,
            device=device,
        ).unsqueeze(0)
        query_ids = query_ids.expand(batch_size, -1)
        return self.query_vector(query_ids)

    def set_oprd_trainable(self) -> None:
        for module in (
            self.t5.decoder.block,
            self.t5.decoder.final_layer_norm,
        ):
            for param in module.parameters():
                param.requires_grad = True

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

    def sample(
        self,
        batch: dict,
        n_return_sequences: int = 1,
        temperature: float = 1.0,
    ) -> torch.Tensor:
        batch_size = batch["input_ids"].shape[0]
        device = batch["input_ids"].device
        temperature = max(float(temperature), 1e-6)

        with torch.no_grad():
            encoder_outputs = self.t5.encoder(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                return_dict=True,
            )
            encoder_hidden_states = encoder_outputs.last_hidden_state
            encoder_attention_mask = batch["attention_mask"]
            flat_encoder_hidden = encoder_hidden_states.repeat_interleave(
                n_return_sequences,
                dim=0,
            )
            flat_encoder_mask = encoder_attention_mask.repeat_interleave(
                n_return_sequences,
                dim=0,
            )
            prefixes = torch.empty(
                batch_size * n_return_sequences,
                0,
                dtype=torch.long,
                device=device,
            )
            for stage in range(self.n_digit):
                logits = self._stage_logits(
                    encoder_hidden_states=flat_encoder_hidden,
                    encoder_attention_mask=flat_encoder_mask,
                    prev_tokens=prefixes,
                    stage=stage,
                )
                probs = torch.softmax(logits / temperature, dim=-1)
                next_tokens = torch.multinomial(probs, num_samples=1)
                prefixes = torch.cat([prefixes, next_tokens], dim=-1)
        return prefixes.view(batch_size, n_return_sequences, self.n_digit)

    def oprd_hidden_states(
        self,
        batch: dict,
        sequences: torch.Tensor,
        distill_layers=None,
    ) -> tuple[torch.Tensor, ...]:
        if sequences.dim() == 2:
            sequences = sequences.unsqueeze(1)
        if sequences.dim() != 3:
            raise ValueError("sequences must have shape [batch, samples, n_digit].")

        batch_size, n_samples, n_digit = sequences.shape
        if n_digit != self.n_digit:
            raise ValueError(f"Expected {self.n_digit} SID digits, got {n_digit}.")

        encoder_outputs = self.t5.encoder(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            return_dict=True,
        )
        encoder_hidden_states = encoder_outputs.last_hidden_state.repeat_interleave(
            n_samples,
            dim=0,
        )
        flat_attention_mask = batch["attention_mask"].repeat_interleave(
            n_samples,
            dim=0,
        )
        flat_sequences = sequences.to(encoder_hidden_states.device).long().reshape(
            batch_size * n_samples,
            n_digit,
        )
        decoder_inputs_embeds, pred_positions, query_ranges = self._training_decoder_inputs(flat_sequences)
        decoder_attention_mask = torch.ones(
            decoder_inputs_embeds.shape[:2],
            dtype=flat_attention_mask.dtype,
            device=decoder_inputs_embeds.device,
        )
        encoder_attention_mask = self._training_encoder_attention_mask(
            attention_mask=flat_attention_mask,
            decoder_len=decoder_inputs_embeds.shape[1],
            query_ranges=query_ranges,
        )
        decoder_outputs = self.t5.decoder(
            inputs_embeds=decoder_inputs_embeds,
            attention_mask=decoder_attention_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            use_cache=False,
            return_dict=True,
            output_hidden_states=True,
        )

        selected_states = []
        for layer_idx in self._normalize_distill_layers(distill_layers):
            layer_states = decoder_outputs.hidden_states[layer_idx]
            sid_states = layer_states[:, pred_positions, :]
            selected_states.append(sid_states.view(batch_size, n_samples, n_digit, -1))
        return tuple(selected_states)

    def forward(self, batch: dict) -> Seq2SeqLMOutput:
        input_ids = batch["input_ids"].long()
        attention_mask = batch["attention_mask"].long()
        labels = batch["labels"][:, : self.n_digit].long().to(input_ids.device)
        self._validate_token_ids("input_ids", input_ids)
        self._validate_token_ids("labels", labels)

        encoder_outputs = self.t5.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True,
        )
        encoder_hidden_states = encoder_outputs.last_hidden_state

        decoder_inputs_embeds, pred_positions, query_ranges = (
            self._training_decoder_inputs(labels)
        )
        decoder_attention_mask = torch.ones(
            decoder_inputs_embeds.shape[:2],
            dtype=attention_mask.dtype,
            device=decoder_inputs_embeds.device,
        )
        encoder_attention_mask = self._training_encoder_attention_mask(
            attention_mask=attention_mask,
            decoder_len=decoder_inputs_embeds.shape[1],
            query_ranges=query_ranges,
        )
        decoder_outputs = self.t5.decoder(
            inputs_embeds=decoder_inputs_embeds,
            attention_mask=decoder_attention_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            use_cache=False,
            return_dict=True,
        )
        sequence_output = decoder_outputs.last_hidden_state[:, pred_positions, :]
        if self.t5.config.tie_word_embeddings:
            sequence_output = sequence_output * (self.t5.model_dim**-0.5)
        logits = self.t5.lm_head(sequence_output)
        loss = self.loss_fct(
            logits.reshape(-1, logits.size(-1)),
            labels.reshape(-1),
        )
        loss = loss + self.query_div_scale * self._query_diversity_loss()
        return Seq2SeqLMOutput(loss=loss, logits=logits)

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
            encoder_outputs = self.t5.encoder(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                return_dict=True,
            )
            encoder_hidden_states = encoder_outputs.last_hidden_state
            encoder_attention_mask = batch["attention_mask"]

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
                flat_encoder_hidden = encoder_hidden_states.repeat_interleave(
                    cur_beams,
                    dim=0,
                )
                flat_encoder_mask = encoder_attention_mask.repeat_interleave(
                    cur_beams,
                    dim=0,
                )
                logits = self._stage_logits(
                    encoder_hidden_states=flat_encoder_hidden,
                    encoder_attention_mask=flat_encoder_mask,
                    prev_tokens=flat_prefixes,
                    stage=stage,
                )
                token_scores = torch.log_softmax(logits, dim=-1)
                vocab_size = token_scores.shape[-1]
                candidate_scores = (
                    beam_scores.reshape(batch_size, cur_beams, 1)
                    + token_scores.reshape(batch_size, cur_beams, vocab_size)
                )
                top_scores, top_indices = candidate_scores.reshape(
                    batch_size,
                    cur_beams * vocab_size,
                ).topk(num_beams, dim=1)
                next_beam_indices = torch.div(
                    top_indices,
                    vocab_size,
                    rounding_mode="floor",
                )
                next_tokens = top_indices % vocab_size

                if stage > 0:
                    gather_index = next_beam_indices.unsqueeze(-1).expand(
                        -1,
                        -1,
                        stage,
                    )
                    prefixes = prefixes.gather(1, gather_index)
                else:
                    prefixes = prefixes.expand(batch_size, num_beams, 0)
                prefixes = torch.cat([prefixes, next_tokens.unsqueeze(-1)], dim=-1)
                beam_scores = top_scores

        preds = prefixes[:, :n_return_sequences, :]
        scores = beam_scores[:, :n_return_sequences]
        if return_scores:
            return {"preds": preds, "scores": scores}
        return preds

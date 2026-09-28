import torch
from torch.nn import CrossEntropyLoss
from transformers import T5Config, T5ForConditionalGeneration

from genrec.model import AbstractModel
from genrec.dataset import AbstractDataset
from genrec.tokenizer import AbstractTokenizer


class TIGER(AbstractModel):
    """
    TIGER model from Rajput et al. "Recommender Systems with Generative Retrieval." NeurIPS 2023.

    Args:
        config (dict): Configuration parameters for the model.
        dataset (AbstractDataset): The dataset object.
        tokenizer (AbstractTokenizer): The tokenizer object.

    Attributes:
        t5 (T5ForConditionalGeneration): The T5 model for conditional generation.
    """
    def __init__(
        self,
        config: dict,
        dataset: AbstractDataset,
        tokenizer: AbstractTokenizer,
    ):
        super(TIGER, self).__init__(config, dataset, tokenizer)
        t5config = T5Config(
            num_layers=config['num_layers'],
            num_decoder_layers=config['num_decoder_layers'],
            d_model=config['d_model'],
            d_ff=config['d_ff'],
            num_heads=config['num_heads'],
            d_kv=config['d_kv'],
            dropout_rate=config['dropout_rate'],
            activation_function=config['activation_function'],
            vocab_size=tokenizer.vocab_size,
            pad_token_id=tokenizer.padding_token,
            eos_token_id=tokenizer.eos_token,
            decoder_start_token_id=0,
            feed_forward_proj=config['feed_forward_proj'],
            n_positions=tokenizer.max_token_seq_len,
        )

        self.t5 = T5ForConditionalGeneration(config=t5config)
        self.temperature = config.get("temperature", 1.0)
        self.loss_fct = CrossEntropyLoss(ignore_index=-100)
        self.prefix_trie = self._build_prefix_trie()

    @property
    def n_parameters(self) -> str:
        """
        Calculates the number of trainable parameters in the model.

        Returns:
            str: A string containing the number of embedding parameters, non-embedding parameters, and total trainable parameters.
        """
        total_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        emb_params = sum(p.numel() for p in self.t5.get_input_embeddings().parameters() if p.requires_grad)
        return f'#Embedding parameters: {emb_params}\n' \
                f'#Non-embedding parameters: {total_params - emb_params}\n' \
                f'#Total trainable parameters: {total_params}\n'

    def _build_prefix_trie(self):
        root = {}
        for tokens in self.tokenizer.item2tokens.values():
            node = root
            for token in list(tokens) + [self.tokenizer.eos_token]:
                node = node.setdefault(int(token), {})
        return root

    def _prefix_allowed_tokens_fn(self, batch_id, input_ids):
        node = self.prefix_trie
        for token in input_ids.tolist()[1:]:
            if token not in node:
                return [self.tokenizer.eos_token]
            node = node[token]
        return list(node.keys()) or [self.tokenizer.eos_token]

    def _validate_token_ids(self, name: str, token_ids: torch.Tensor):
        if token_ids.numel() == 0:
            return

        valid_token_ids = token_ids[token_ids != -100]
        if valid_token_ids.numel() == 0:
            return

        min_id = int(valid_token_ids.detach().min().cpu().item())
        max_id = int(valid_token_ids.detach().max().cpu().item())
        max_valid_id = self.tokenizer.vocab_size - 1
        if min_id < 0 or max_id > max_valid_id:
            raise ValueError(
                f"TIGER received {name} outside tokenizer vocabulary "
                f"[0, {max_valid_id}]: min={min_id}, max={max_id}. "
                "Check the cached semantic IDs and tokenizer config."
            )

    def _decoder_input_ids_from_labels(self, labels: torch.Tensor) -> torch.Tensor:
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
        if labels.shape[1] <= 1:
            decoder_input_ids = start_ids
        else:
            label_prefix = labels[:, :-1]
            if label_prefix.numel() > 0:
                min_id = int(label_prefix.detach().min().cpu().item())
                if min_id < 0:
                    label_prefix = label_prefix.masked_fill(
                        label_prefix == -100,
                        self.tokenizer.padding_token,
                    )
            decoder_input_ids = torch.cat([start_ids, label_prefix], dim=1)
        self._validate_token_ids("decoder_input_ids", decoder_input_ids)
        return decoder_input_ids

    def forward(self, batch: dict) -> torch.Tensor:
        """
        Forward pass of the model. Returns the output logits and the loss value.

        Args:
            batch (dict): A dictionary containing the input data for the model.

        Returns:
            outputs (ModelOutput): 
                The output of the model, which includes:
                - loss (torch.Tensor)
                - logits (torch.Tensor)
        """
        input_ids = batch["input_ids"].long()
        attention_mask = batch["attention_mask"].long()
        labels = batch["labels"].long().to(input_ids.device)
        self._validate_token_ids("input_ids", input_ids)
        self._validate_token_ids("labels", labels)
        decoder_input_ids = self._decoder_input_ids_from_labels(labels)

        outputs = self.t5(
            input_ids=input_ids,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            labels=labels,
        )
        outputs.loss = self.loss_fct(
            (outputs.logits / self.temperature).reshape(-1, outputs.logits.size(-1)),
            labels.reshape(-1),
        )
        return outputs

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

    def _oprd_encoder_context(self, batch: dict) -> tuple[torch.Tensor, torch.Tensor]:
        input_ids = batch["input_ids"].long()
        attention_mask = batch["attention_mask"].long()
        encoder_outputs = self.t5.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True,
        )
        return encoder_outputs.last_hidden_state, attention_mask

    def _oprd_decoder_input_ids_for_prefixes(self, prev_tokens: torch.Tensor):
        batch_size = prev_tokens.shape[0]
        device = prev_tokens.device
        start_ids = torch.full(
            (batch_size, 1),
            self.t5.config.decoder_start_token_id,
            dtype=torch.long,
            device=device,
        )
        if prev_tokens.shape[1] == 0:
            decoder_input_ids = start_ids
        else:
            decoder_input_ids = torch.cat([start_ids, prev_tokens.long()], dim=1)
        return decoder_input_ids

    def _oprd_stage_logits(
        self,
        encoder_hidden_states: torch.Tensor,
        encoder_attention_mask: torch.Tensor,
        prev_tokens: torch.Tensor,
    ) -> torch.Tensor:
        decoder_input_ids = self._oprd_decoder_input_ids_for_prefixes(prev_tokens)
        decoder_outputs = self.t5.decoder(
            input_ids=decoder_input_ids,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            use_cache=False,
            return_dict=True,
        )
        sequence_output = decoder_outputs.last_hidden_state[:, -1, :]
        if self.t5.config.tie_word_embeddings:
            sequence_output = sequence_output * (self.t5.model_dim**-0.5)
        return self.t5.lm_head(sequence_output)

    def _oprd_mask_logits_for_prefixes(
        self,
        logits: torch.Tensor,
        prefixes: torch.Tensor,
    ) -> torch.Tensor:
        if not self.config.get("constrained_generation", False):
            return logits
        masked_logits = logits.new_full(logits.shape, -10000.0)
        for row, prefix in enumerate(prefixes.detach().cpu().tolist()):
            prefix_ids = torch.tensor(
                [self.t5.config.decoder_start_token_id] + prefix,
                dtype=torch.long,
            )
            allowed = self._prefix_allowed_tokens_fn(row, prefix_ids)
            allowed = torch.tensor(allowed, dtype=torch.long, device=logits.device)
            masked_logits[row, allowed] = logits[row, allowed]
        return masked_logits

    def sample(
        self,
        batch: dict,
        n_return_sequences: int = 1,
        temperature: float = 1.0,
    ) -> torch.Tensor:
        batch_size = batch["input_ids"].shape[0]
        device = batch["input_ids"].device
        n_digit = self.tokenizer.n_digit
        temperature = max(float(temperature), 1e-6)

        with torch.no_grad():
            encoder_hidden_states, encoder_attention_mask = self._oprd_encoder_context(batch)
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
            for _ in range(n_digit):
                logits = self._oprd_stage_logits(
                    encoder_hidden_states=flat_encoder_hidden,
                    encoder_attention_mask=flat_encoder_mask,
                    prev_tokens=prefixes,
                )
                logits = self._oprd_mask_logits_for_prefixes(logits, prefixes)
                probs = torch.softmax(logits / temperature, dim=-1)
                next_tokens = torch.multinomial(probs, num_samples=1)
                prefixes = torch.cat([prefixes, next_tokens], dim=-1)
        return prefixes.view(batch_size, n_return_sequences, n_digit)

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
        if n_digit != self.tokenizer.n_digit:
            raise ValueError(f"Expected {self.tokenizer.n_digit} SID digits, got {n_digit}.")

        encoder_hidden_states, encoder_attention_mask = self._oprd_encoder_context(batch)
        flat_encoder_hidden = encoder_hidden_states.repeat_interleave(n_samples, dim=0)
        flat_encoder_mask = encoder_attention_mask.repeat_interleave(n_samples, dim=0)
        flat_sequences = sequences.to(flat_encoder_hidden.device).long().reshape(
            batch_size * n_samples,
            n_digit,
        )
        self._validate_token_ids("oprd_sequences", flat_sequences)
        decoder_input_ids = self._oprd_decoder_input_ids_for_prefixes(flat_sequences[:, :-1])
        decoder_outputs = self.t5.decoder(
            input_ids=decoder_input_ids,
            encoder_hidden_states=flat_encoder_hidden,
            encoder_attention_mask=flat_encoder_mask,
            use_cache=False,
            return_dict=True,
            output_hidden_states=True,
        )

        selected_states = []
        for layer_idx in self._normalize_distill_layers(distill_layers):
            layer_states = decoder_outputs.hidden_states[layer_idx]
            selected_states.append(layer_states.view(batch_size, n_samples, n_digit, -1))
        return tuple(selected_states)

    def generate(
        self,
        batch: dict,
        n_return_sequences: int = 1,
        return_scores: bool = False,
        num_beams: int = None,
    ):
        """
        Generates sequences using beam search algorithm.

        Args:
            batch (dict): A dictionary containing input_ids and attention_mask.
            n_return_sequences (int): The number of sequences to generate.
            return_scores (bool): Whether to return the beam scores (log probs).

        Returns:
            torch.Tensor or dict: The generated sequences, or a dict if return_scores=True.
        """
        n_digit = self.tokenizer.n_digit
        num_beams = self.config['num_beams'] if num_beams is None else num_beams
        num_beams = max(num_beams, n_return_sequences)
        batch_size = batch['input_ids'].shape[0]

        input_ids = batch["input_ids"].long()
        attention_mask = batch["attention_mask"].long()
        self._validate_token_ids("input_ids", input_ids)

        with torch.no_grad():
            outputs = self.t5.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=n_digit + 1,
                num_beams=num_beams,
                num_return_sequences=n_return_sequences,
                return_dict_in_generate=True,
                output_scores=True,
                use_cache=True,
            )

        # 1. Process Sequences
        sequences = outputs.sequences
        seq_len = sequences.shape[1]
        
        # Reshape to (batch_size, n_return_sequences, seq_len)
        sequences = sequences.reshape(batch_size, n_return_sequences, seq_len)
        
        # Extract semantic tokens (skip decoder_start_token)
        pred_tokens = sequences[:, :, 1:1+n_digit]

        # 2. Return Logic (Minimal Change)
        if return_scores:
            # sequences_scores contains the sum of log probs for the generated sequence
            # Shape: (batch_size * n_return_sequences,) -> Reshape to (batch_size, n_return_sequences)
            scores = outputs.sequences_scores.reshape(batch_size, n_return_sequences)
            return {
                'preds': pred_tokens, 
                'scores': scores
            }
        
        return pred_tokens

import random
import re
from collections import Counter
from typing import Any

from genrec.dataset import AbstractDataset
from genrec.tokenizer import AbstractTokenizer


class S3RecTokenizer(AbstractTokenizer):
    """
    Tokenizer for S3Rec.

    Item ids keep the framework convention: 0 is padding and 1..n_items-1 are
    real items. S3Rec adds one extra mask token at n_items.
    """

    def __init__(self, config: dict, dataset: AbstractDataset):
        super(S3RecTokenizer, self).__init__(config, dataset)

        self.dataset = dataset
        self.item2tokens = dataset.item2id
        self.mask_token = dataset.n_items
        self.eos_token = self.mask_token
        self.ignored_label = -100
        self.item_id2attributes, self.attribute_size = self._build_item_attributes()
        self.long_sequence = []

    def _init_tokenizer(self):
        pass

    def _extract_attribute_tokens(self, metadata: Any) -> list[str]:
        tokens = []
        if metadata is None:
            return tokens
        if isinstance(metadata, dict):
            for key, value in metadata.items():
                if key.lower() in {"brand", "category", "categories"}:
                    tokens.extend(self._extract_attribute_tokens(value))
        elif isinstance(metadata, list):
            for value in metadata:
                tokens.extend(self._extract_attribute_tokens(value))
        else:
            text = str(metadata).lower()
            tokens.extend(re.findall(r"[a-z0-9][a-z0-9_+-]{1,}", text))
        return tokens

    def _build_item_attributes(self):
        item2meta = getattr(self.dataset, "item2meta", None)
        item_id2raw_attrs = [[] for _ in range(self.dataset.n_items + 1)]
        if not item2meta:
            return item_id2raw_attrs, 1

        counter = Counter()
        for raw_item, item_id in self.dataset.item2id.items():
            if item_id == 0:
                continue
            attrs = self._extract_attribute_tokens(item2meta.get(raw_item))
            if not attrs:
                continue
            attrs = attrs[: self.config["max_attributes_per_item"]]
            item_id2raw_attrs[item_id] = attrs
            counter.update(set(attrs))

        min_freq = self.config["attribute_min_freq"]
        max_vocab = self.config["attribute_vocab_size"]
        kept = [
            attr
            for attr, freq in counter.most_common(max_vocab)
            if freq >= min_freq
        ]
        attr2id = {attr: idx + 1 for idx, attr in enumerate(kept)}
        item_id2attributes = [[] for _ in range(self.dataset.n_items + 1)]
        for item_id, attrs in enumerate(item_id2raw_attrs):
            item_id2attributes[item_id] = [
                attr2id[attr]
                for attr in attrs
                if attr in attr2id
            ][: self.config["max_attributes_per_item"]]
        return item_id2attributes, len(attr2id) + 1

    def _neg_sample(self, item_set: set[int]) -> int:
        if self.dataset.n_items <= 2:
            return 0
        for _ in range(100):
            item = random.randint(1, self.dataset.n_items - 1)
            if item not in item_set:
                return item
        return random.randint(1, self.dataset.n_items - 1)

    def _pad(self, seq: list[int], value: int = 0) -> list[int]:
        seq = seq[: self.max_token_seq_len]
        return seq + [value] * (self.max_token_seq_len - len(seq))

    def _make_pretrain_fields(self, input_ids: list[int]) -> dict:
        sequence = [item for item in input_ids if item != 0]
        if not sequence:
            sequence = [0]

        item_set = set(sequence)
        masked_item_sequence = []
        neg_items = []
        for item in sequence[:-1]:
            if random.random() < self.config["mask_p"]:
                masked_item_sequence.append(self.mask_token)
                neg_items.append(self._neg_sample(item_set))
            else:
                masked_item_sequence.append(item)
                neg_items.append(item)
        masked_item_sequence.append(self.mask_token)
        neg_items.append(self._neg_sample(item_set))

        if len(sequence) < 2:
            masked_segment_sequence = sequence
            pos_segment = sequence
            neg_segment = sequence
        else:
            sample_length = random.randint(1, max(1, len(sequence) // 2))
            start_id = random.randint(0, len(sequence) - sample_length)
            if len(self.long_sequence) >= sample_length:
                neg_start_id = random.randint(
                    0,
                    len(self.long_sequence) - sample_length,
                )
                neg_core = self.long_sequence[neg_start_id : neg_start_id + sample_length]
            else:
                neg_core = sequence[start_id : start_id + sample_length]
            pos_core = sequence[start_id : start_id + sample_length]
            masked_segment_sequence = (
                sequence[:start_id]
                + [self.mask_token] * sample_length
                + sequence[start_id + sample_length :]
            )
            pos_segment = (
                [self.mask_token] * start_id
                + pos_core
                + [self.mask_token] * (len(sequence) - start_id - sample_length)
            )
            neg_segment = (
                [self.mask_token] * start_id
                + neg_core
                + [self.mask_token] * (len(sequence) - start_id - sample_length)
            )

        return {
            "masked_item_sequence": self._pad(masked_item_sequence),
            "pos_items": self._pad(sequence),
            "neg_items": self._pad(neg_items),
            "masked_segment_sequence": self._pad(masked_segment_sequence),
            "pos_segment": self._pad(pos_segment),
            "neg_segment": self._pad(neg_segment),
        }

    def _tokenize_first_n_items(self, item_seq: list) -> tuple:
        input_ids = [self.item2tokens[item] for item in item_seq[:-1]]
        seq_lens = len(input_ids)
        attention_mask = [1] * seq_lens
        labels = [self.item2tokens[item] for item in item_seq[1:]]

        pad_lens = self.max_token_seq_len - seq_lens
        input_ids.extend([0] * pad_lens)
        attention_mask.extend([0] * pad_lens)
        labels.extend([self.ignored_label] * pad_lens)
        return input_ids, attention_mask, labels, seq_lens

    def _tokenize_later_items(self, item_seq: list, pad_labels: bool = True) -> tuple:
        input_ids = [self.item2tokens[item] for item in item_seq[:-1]]
        seq_lens = len(input_ids)
        attention_mask = [1] * seq_lens
        labels = [self.ignored_label] * seq_lens
        labels[-1] = self.item2tokens[item_seq[-1]]

        pad_lens = self.max_token_seq_len - seq_lens
        input_ids.extend([0] * pad_lens)
        attention_mask.extend([0] * pad_lens)
        if pad_labels:
            labels.extend([self.ignored_label] * pad_lens)
        return input_ids, attention_mask, labels, seq_lens

    def tokenize_function(self, examples: dict, split: str) -> dict:
        all_input_ids, all_attention_mask, all_labels, all_seq_lens = [], [], [], []
        pretrain_fields = {
            "masked_item_sequence": [],
            "pos_items": [],
            "neg_items": [],
            "masked_segment_sequence": [],
            "pos_segment": [],
            "neg_segment": [],
        }

        max_item_seq_len = self.config["max_item_seq_len"]
        for item_seq in examples["item_seq"]:
            if split == "train":
                n_return_examples = max(len(item_seq) - max_item_seq_len, 1)

                outputs = self._tokenize_first_n_items(
                    item_seq=item_seq[: min(len(item_seq), max_item_seq_len + 1)]
                )
                input_ids, attention_mask, labels, seq_lens = outputs
                all_input_ids.append(input_ids)
                all_attention_mask.append(attention_mask)
                all_labels.append(labels)
                all_seq_lens.append(seq_lens)
                for key, value in self._make_pretrain_fields(input_ids).items():
                    pretrain_fields[key].append(value)

                for j in range(1, n_return_examples):
                    cur_item_seq = item_seq[j : j + max_item_seq_len + 1]
                    outputs = self._tokenize_later_items(cur_item_seq)
                    input_ids, attention_mask, labels, seq_lens = outputs
                    all_input_ids.append(input_ids)
                    all_attention_mask.append(attention_mask)
                    all_labels.append(labels)
                    all_seq_lens.append(seq_lens)
                    for key, value in self._make_pretrain_fields(input_ids).items():
                        pretrain_fields[key].append(value)
            else:
                outputs = self._tokenize_later_items(
                    item_seq=item_seq[-(max_item_seq_len + 1) :],
                    pad_labels=False,
                )
                input_ids, attention_mask, labels, seq_lens = outputs
                all_input_ids.append(input_ids)
                all_attention_mask.append(attention_mask)
                all_labels.append(labels[-1:])
                all_seq_lens.append(seq_lens)

        tokenized = {
            "input_ids": all_input_ids,
            "attention_mask": all_attention_mask,
            "labels": all_labels,
            "seq_lens": all_seq_lens,
        }
        if split == "train":
            tokenized.update(pretrain_fields)
        return tokenized

    def tokenize(self, datasets: dict) -> dict:
        self.long_sequence = [
            self.item2tokens[item]
            for seq in datasets["train"]["item_seq"]
            for item in seq
            if item in self.item2tokens
        ]

        batch_size = 128
        tokenized_datasets = {}
        for split in datasets:
            if split in ["val", "test"]:
                tokenized_datasets[split] = datasets[split].map(
                    lambda t, idx: {**self.tokenize_function(t, split), "idx": idx},
                    batched=True,
                    batch_size=batch_size,
                    with_indices=True,
                    remove_columns=datasets[split].column_names,
                    num_proc=self.config["num_proc"],
                    desc=f"Tokenizing {split} set: ",
                )
            else:
                tokenized_datasets[split] = datasets[split].map(
                    lambda t: self.tokenize_function(t, split),
                    batched=True,
                    batch_size=batch_size,
                    remove_columns=datasets[split].column_names,
                    num_proc=self.config["num_proc"],
                    desc=f"Tokenizing {split} set: ",
                )

        for split in datasets:
            tokenized_datasets[split].set_format(type="torch")
        return tokenized_datasets

    @property
    def vocab_size(self) -> int:
        return self.mask_token + 1

    @property
    def max_token_seq_len(self) -> int:
        return self.config["max_item_seq_len"]


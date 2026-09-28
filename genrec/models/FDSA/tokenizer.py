import re
from collections import Counter
from typing import Any

from genrec.dataset import AbstractDataset
from genrec.tokenizer import AbstractTokenizer


class FDSATokenizer(AbstractTokenizer):
    """
    Tokenizer for FDSA.

    Item ids follow the framework convention: 0 is padding and 1..n_items-1
    are real item tokens. FDSA additionally builds a bounded attribute
    vocabulary from dataset.item2meta and returns per-position feature ids.
    """

    def __init__(self, config: dict, dataset: AbstractDataset):
        super(FDSATokenizer, self).__init__(config, dataset)

        self.dataset = dataset
        self.item2tokens = dataset.item2id
        self.eos_token = dataset.n_items
        self.ignored_label = -100
        self.max_features_per_item = max(1, config["max_features_per_item"])
        self.feature2id = {}
        self.item_id2features, self.feature_size = self._build_item_features()

    def _init_tokenizer(self):
        pass

    def _extract_feature_tokens(self, metadata: Any, field: str = None) -> list[str]:
        if metadata is None:
            return []

        if isinstance(metadata, dict):
            tokens = []
            feature_fields = self.config.get("feature_fields", None)
            for key, value in metadata.items():
                if feature_fields is not None and key not in feature_fields:
                    continue
                next_field = str(key).lower()
                tokens.extend(self._extract_feature_tokens(value, next_field))
            return tokens

        if isinstance(metadata, (list, tuple, set)):
            tokens = []
            for value in metadata:
                tokens.extend(self._extract_feature_tokens(value, field))
            return tokens

        raw_tokens = re.findall(
            r"[a-z0-9][a-z0-9_+-]{1,}",
            str(metadata).lower(),
        )
        if field and self.config.get("prefix_feature_fields", True):
            return [f"{field}:{token}" for token in raw_tokens]
        return raw_tokens

    @staticmethod
    def _dedupe_preserve_order(values: list[str]) -> list[str]:
        seen = set()
        output = []
        for value in values:
            if value in seen:
                continue
            seen.add(value)
            output.append(value)
        return output

    def _build_item_features(self):
        item2meta = getattr(self.dataset, "item2meta", None)
        item_id2raw_features = [[] for _ in range(self.dataset.n_items)]
        if not item2meta:
            return item_id2raw_features, 1

        counter = Counter()
        for raw_item, item_id in self.dataset.item2id.items():
            if item_id == 0 or item_id >= len(item_id2raw_features):
                continue
            features = self._extract_feature_tokens(item2meta.get(raw_item))
            features = self._dedupe_preserve_order(features)
            features = features[: self.max_features_per_item]
            item_id2raw_features[item_id] = features
            counter.update(set(features))

        min_freq = self.config["feature_min_freq"]
        max_vocab = self.config["feature_vocab_size"]
        kept_features = [
            feature
            for feature, freq in counter.most_common(max_vocab)
            if freq >= min_freq
        ]
        self.feature2id = {feature: idx + 1 for idx, feature in enumerate(kept_features)}

        item_id2features = [[] for _ in range(self.dataset.n_items)]
        for item_id, features in enumerate(item_id2raw_features):
            item_id2features[item_id] = [
                self.feature2id[feature]
                for feature in features
                if feature in self.feature2id
            ][: self.max_features_per_item]

        return item_id2features, len(self.feature2id) + 1

    def _pad(self, seq: list[int], value: int = 0) -> list[int]:
        seq = seq[: self.max_token_seq_len]
        return seq + [value] * (self.max_token_seq_len - len(seq))

    def _to_item_ids(self, item_seq: list) -> list[int]:
        return [
            self.item2tokens[item]
            for item in item_seq
            if item in self.item2tokens and self.item2tokens[item] > 0
        ]

    def _features_for_input(self, input_ids: list[int]) -> list[list[int]]:
        feature_ids = []
        for item_id in input_ids[: self.max_token_seq_len]:
            if 0 < item_id < len(self.item_id2features):
                features = self.item_id2features[item_id]
            else:
                features = []
            features = features[: self.max_features_per_item]
            padded = features + [0] * (self.max_features_per_item - len(features))
            feature_ids.append(padded)

        pad_len = self.max_token_seq_len - len(feature_ids)
        feature_ids.extend([[0] * self.max_features_per_item for _ in range(pad_len)])
        return feature_ids

    def _tokenize_first_n_items(self, item_ids: list[int]) -> tuple:
        input_ids = item_ids[:-1]
        labels = item_ids[1:]
        seq_lens = len(input_ids)
        attention_mask = [1] * seq_lens

        padded_input_ids = self._pad(input_ids)
        padded_attention_mask = self._pad(attention_mask)
        padded_labels = self._pad(labels, self.ignored_label)
        feature_ids = self._features_for_input(input_ids)
        return (
            padded_input_ids,
            padded_attention_mask,
            padded_labels,
            seq_lens,
            feature_ids,
        )

    def _tokenize_later_items(
        self,
        item_ids: list[int],
        pad_labels: bool = True,
    ) -> tuple:
        input_ids = item_ids[:-1]
        seq_lens = len(input_ids)
        attention_mask = [1] * seq_lens
        labels = [self.ignored_label] * seq_lens
        if labels:
            labels[-1] = item_ids[-1]

        padded_input_ids = self._pad(input_ids)
        padded_attention_mask = self._pad(attention_mask)
        if pad_labels:
            output_labels = self._pad(labels, self.ignored_label)
        else:
            output_labels = labels
        feature_ids = self._features_for_input(input_ids)
        return (
            padded_input_ids,
            padded_attention_mask,
            output_labels,
            seq_lens,
            feature_ids,
        )

    def tokenize_function(self, examples: dict, split: str) -> dict:
        all_input_ids = []
        all_attention_mask = []
        all_labels = []
        all_seq_lens = []
        all_feature_ids = []

        max_item_seq_len = self.config["max_item_seq_len"]
        for item_seq in examples["item_seq"]:
            item_ids = self._to_item_ids(item_seq)
            if len(item_ids) < 2:
                continue

            if split == "train":
                n_return_examples = max(len(item_ids) - max_item_seq_len, 1)

                outputs = self._tokenize_first_n_items(
                    item_ids=item_ids[: min(len(item_ids), max_item_seq_len + 1)]
                )
                input_ids, attention_mask, labels, seq_lens, feature_ids = outputs
                all_input_ids.append(input_ids)
                all_attention_mask.append(attention_mask)
                all_labels.append(labels)
                all_seq_lens.append(seq_lens)
                all_feature_ids.append(feature_ids)

                for j in range(1, n_return_examples):
                    outputs = self._tokenize_later_items(
                        item_ids[j : j + max_item_seq_len + 1]
                    )
                    input_ids, attention_mask, labels, seq_lens, feature_ids = outputs
                    all_input_ids.append(input_ids)
                    all_attention_mask.append(attention_mask)
                    all_labels.append(labels)
                    all_seq_lens.append(seq_lens)
                    all_feature_ids.append(feature_ids)
            else:
                outputs = self._tokenize_later_items(
                    item_ids=item_ids[-(max_item_seq_len + 1) :],
                    pad_labels=False,
                )
                input_ids, attention_mask, labels, seq_lens, feature_ids = outputs
                all_input_ids.append(input_ids)
                all_attention_mask.append(attention_mask)
                all_labels.append(labels[-1:])
                all_seq_lens.append(seq_lens)
                all_feature_ids.append(feature_ids)

        return {
            "input_ids": all_input_ids,
            "attention_mask": all_attention_mask,
            "labels": all_labels,
            "seq_lens": all_seq_lens,
            "feature_ids": all_feature_ids,
        }

    def tokenize(self, datasets: dict) -> dict:
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
        return self.dataset.n_items

    @property
    def max_token_seq_len(self) -> int:
        return self.config["max_item_seq_len"]

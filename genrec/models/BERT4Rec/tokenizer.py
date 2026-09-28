import random

from genrec.dataset import AbstractDataset
from genrec.tokenizer import AbstractTokenizer


class BERT4RecTokenizer(AbstractTokenizer):
    """
    Tokenizer for BERT4Rec.

    Item ids follow the framework convention: 0 is padding and 1..n_items-1
    are real item tokens. BERT4Rec adds a single [MASK] token at n_items.
    """

    def __init__(self, config: dict, dataset: AbstractDataset):
        super(BERT4RecTokenizer, self).__init__(config, dataset)

        self.dataset = dataset
        self.item2tokens = dataset.item2id
        self.mask_token = dataset.n_items
        self.eos_token = self.mask_token
        self.ignored_label = -100

    def _init_tokenizer(self):
        pass

    def _pad(self, seq: list[int], value: int = 0) -> list[int]:
        seq = seq[: self.max_token_seq_len]
        return seq + [value] * (self.max_token_seq_len - len(seq))

    def _pad_seen_items(self, seq: list[int]) -> list[int]:
        max_len = self.config.get("filter_seen_max_len", self.max_token_seq_len)
        seq = seq[-max_len:]
        return seq + [0] * (max_len - len(seq))

    def _to_item_ids(self, item_seq: list) -> list[int]:
        return [
            self.item2tokens[item]
            for item in item_seq
            if item in self.item2tokens and self.item2tokens[item] > 0
        ]

    def _random_item(self, original_item: int) -> int:
        if self.dataset.n_items <= 2:
            return original_item
        for _ in range(100):
            item = random.randint(1, self.dataset.n_items - 1)
            if item != original_item:
                return item
        return random.randint(1, self.dataset.n_items - 1)

    def _replace_masked_item(self, item: int) -> int:
        mask_prob = self.config["mask_prob"]
        if random.random() < mask_prob:
            return self.mask_token
        if random.random() < 0.5:
            return item
        return self._random_item(item)

    def _make_masked_example(self, item_ids: list[int], force_last: bool = False):
        item_ids = item_ids[-self.max_token_seq_len :]
        input_ids = list(item_ids)
        labels = [self.ignored_label] * len(input_ids)

        if not input_ids:
            input_ids = [self.mask_token]
            labels = [self.ignored_label]
        elif force_last:
            last_idx = len(input_ids) - 1
            labels[last_idx] = input_ids[last_idx]
            input_ids[last_idx] = self.mask_token
        else:
            candidate_indices = list(range(len(input_ids)))
            random.shuffle(candidate_indices)
            n_to_predict = min(
                self.config["max_predictions_per_seq"],
                max(
                    1,
                    int(round(len(input_ids) * self.config["masked_lm_prob"])),
                ),
            )
            for idx in candidate_indices[:n_to_predict]:
                labels[idx] = input_ids[idx]
                input_ids[idx] = self._replace_masked_item(input_ids[idx])

        seq_lens = len(input_ids)
        attention_mask = [1] * seq_lens
        mask_positions = [
            idx
            for idx, label in enumerate(labels)
            if label != self.ignored_label
        ]
        mask_position = mask_positions[-1] if mask_positions else seq_lens - 1

        return (
            self._pad(input_ids),
            self._pad(attention_mask),
            self._pad(labels, self.ignored_label),
            seq_lens,
            mask_position,
        )

    def _windows_for_training(self, item_ids: list[int]) -> list[list[int]]:
        if len(item_ids) <= self.max_token_seq_len:
            return [item_ids]

        prop = self.config.get("prop_sliding_window", 0.1)
        if prop == -1.0:
            sliding_step = self.max_token_seq_len
        else:
            sliding_step = max(1, int(prop * self.max_token_seq_len))

        starts = list(
            range(
                len(item_ids) - self.max_token_seq_len,
                0,
                -sliding_step,
            )
        )
        starts.append(0)
        return [
            item_ids[start : start + self.max_token_seq_len]
            for start in starts[::-1]
        ]

    def _prefix_windows_for_training(self, item_ids: list[int]) -> list[list[int]]:
        return [
            item_ids[
                max(0, target_idx - self.config["max_item_seq_len"]) : target_idx + 1
            ]
            for target_idx in range(1, len(item_ids))
        ]

    def tokenize_function(self, examples: dict, split: str) -> dict:
        all_input_ids = []
        all_attention_mask = []
        all_labels = []
        all_seq_lens = []
        all_mask_positions = []
        all_seen_item_ids = []

        for item_seq in examples["item_seq"]:
            item_ids = self._to_item_ids(item_seq)

            if split == "train":
                for window in self._windows_for_training(item_ids):
                    for _ in range(self.config["dupe_factor"]):
                        outputs = self._make_masked_example(
                            window,
                            force_last=False,
                        )
                        input_ids, attention_mask, labels, seq_lens, mask_pos = outputs
                        all_input_ids.append(input_ids)
                        all_attention_mask.append(attention_mask)
                        all_labels.append(labels)
                        all_seq_lens.append(seq_lens)
                        all_mask_positions.append(mask_pos)

                    if (
                        self.config["train_mask_last"]
                        and not self.config.get("train_all_prefixes", False)
                    ):
                        outputs = self._make_masked_example(window, force_last=True)
                        input_ids, attention_mask, labels, seq_lens, mask_pos = outputs
                        all_input_ids.append(input_ids)
                        all_attention_mask.append(attention_mask)
                        all_labels.append(labels)
                        all_seq_lens.append(seq_lens)
                        all_mask_positions.append(mask_pos)

                if self.config.get("train_all_prefixes", False):
                    for prefix_window in self._prefix_windows_for_training(item_ids):
                        outputs = self._make_masked_example(
                            prefix_window,
                            force_last=True,
                        )
                        input_ids, attention_mask, labels, seq_lens, mask_pos = outputs
                        all_input_ids.append(input_ids)
                        all_attention_mask.append(attention_mask)
                        all_labels.append(labels)
                        all_seq_lens.append(seq_lens)
                        all_mask_positions.append(mask_pos)
            else:
                outputs = self._make_masked_example(
                    item_ids[-self.max_token_seq_len :],
                    force_last=True,
                )
                input_ids, attention_mask, labels, seq_lens, mask_pos = outputs
                all_input_ids.append(input_ids)
                all_attention_mask.append(attention_mask)
                all_labels.append([labels[mask_pos]])
                all_seq_lens.append(seq_lens)
                all_mask_positions.append(mask_pos)
                all_seen_item_ids.append(self._pad_seen_items(item_ids[:-1]))

        tokenized = {
            "input_ids": all_input_ids,
            "attention_mask": all_attention_mask,
            "labels": all_labels,
            "seq_lens": all_seq_lens,
            "mask_positions": all_mask_positions,
        }
        if split in ["val", "test"]:
            tokenized["seen_item_ids"] = all_seen_item_ids
        return tokenized

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
        return self.mask_token + 1

    @property
    def max_token_seq_len(self) -> int:
        return self.config["max_item_seq_len"] + 1

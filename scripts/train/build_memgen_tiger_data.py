import argparse
import json
import sys
from collections import Counter, defaultdict
from contextlib import contextmanager
from pathlib import Path
from typing import Iterable


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _empty_counts(max_hop: int):
    return {hop: defaultdict(Counter) for hop in range(1, max_hop + 1)}


def build_transition_counts(rows: Iterable[dict], max_hop: int):
    rules = _empty_counts(max_hop)
    reverse_rules = _empty_counts(max_hop)
    for row in rows:
        item_seq = list(row["item_seq"])
        for i in range(1, len(item_seq)):
            v = item_seq[i]
            for ctx_length in range(1, max_hop + 1):
                if i - ctx_length < 0:
                    continue
                u = item_seq[i - ctx_length]
                for hop in range(ctx_length, max_hop + 1):
                    rules[hop][u][v] += 1
                    reverse_rules[hop][v][u] += 1
    return rules, reverse_rules


class CountBasedMemGenLabeler:
    def __init__(self, support_rows: Iterable[dict], max_hop: int = 4):
        self.max_hop = int(max_hop)
        self.rules, self.reverse_rules = build_transition_counts(
            support_rows,
            self.max_hop,
        )

    def counts_for_row(self, row: dict):
        return build_transition_counts([row], self.max_hop)

    def _count(self, table, hop, u, v, excluded=None, reverse=False) -> int:
        count = table[hop].get(u, {}).get(v, 0)
        if excluded is not None:
            excluded_table = excluded[1] if reverse else excluded[0]
            count -= excluded_table[hop].get(u, {}).get(v, 0)
        return count

    def _has_rule(self, hop, u, v, excluded=None) -> bool:
        return self._count(self.rules, hop, u, v, excluded=excluded) > 0

    def _targets(self, table, hop, u, excluded=None, reverse=False):
        for v in table[hop].get(u, {}):
            if self._count(table, hop, u, v, excluded=excluded, reverse=reverse) > 0:
                yield v

    def _has_intersection(self, left, right, hop, left_u, right_u, excluded=None) -> bool:
        left_values = set(
            self._targets(
                left,
                hop,
                left_u,
                excluded=excluded,
                reverse=(left is self.reverse_rules),
            )
        )
        if not left_values:
            return False
        for value in self._targets(
            right,
            hop,
            right_u,
            excluded=excluded,
            reverse=(right is self.reverse_rules),
        ):
            if value in left_values:
                return True
        return False

    def get_case_labels(self, item_seq: list, excluded=None) -> set[str]:
        labels = set()
        if len(item_seq) < 2:
            labels.add("uncategorized")
            return labels

        v = item_seq[-1]
        prev = item_seq[-2]
        if self._has_rule(1, prev, v, excluded=excluded):
            return {"memorization"}

        for hop in range(1, self.max_hop + 1):
            for dist in range(1, hop + 1):
                if len(item_seq) - dist - 1 < 0:
                    continue
                u = item_seq[-dist - 1]

                if hop >= 2 and self._has_rule(hop, u, v, excluded=excluded):
                    labels.add(f"substitutability_{hop}")

                if self._has_rule(hop, v, u, excluded=excluded):
                    labels.add(f"symmetry_{hop}")

                if self._has_intersection(
                    self.rules,
                    self.reverse_rules,
                    hop,
                    u,
                    v,
                    excluded=excluded,
                ):
                    labels.add(f"transitivity_{hop}")

                has_2nd_symmetry = (
                    self._has_intersection(
                        self.rules,
                        self.reverse_rules,
                        hop,
                        v,
                        u,
                        excluded=excluded,
                    )
                    or self._has_intersection(
                        self.rules,
                        self.rules,
                        hop,
                        v,
                        u,
                        excluded=excluded,
                    )
                    or self._has_intersection(
                        self.reverse_rules,
                        self.reverse_rules,
                        hop,
                        v,
                        u,
                        excluded=excluded,
                    )
                )
                if has_2nd_symmetry:
                    labels.add(f"2nd-symmetry_{hop}")

        if labels:
            labels.add("generalization")
        else:
            labels.add("uncategorized")
        return labels


def label_group(labels: Iterable[str]) -> str:
    labels = set(labels)
    if "memorization" in labels:
        return "memory"
    if "generalization" in labels:
        return "generalization"
    return "uncategorized"


def build_labeled_cases(
    rows: list[dict],
    support_rows: list[dict],
    split: str,
    max_hop: int = 4,
    leave_one_sequence_out: bool = False,
) -> list[dict]:
    labeler = CountBasedMemGenLabeler(support_rows, max_hop=max_hop)
    source_counts = (
        [labeler.counts_for_row(row) for row in rows]
        if leave_one_sequence_out
        else [None for _ in rows]
    )
    cases = []
    for row_idx, row in enumerate(rows):
        user = row["user"]
        item_seq = list(row["item_seq"])
        if split == "train":
            case_ranges = range(max(0, len(item_seq) - 1))
        else:
            case_ranges = [len(item_seq) - 2] if len(item_seq) >= 2 else []

        for prefix_idx in case_ranges:
            case_seq = item_seq[: prefix_idx + 2]
            excluded = source_counts[row_idx]
            labels = sorted(labeler.get_case_labels(case_seq, excluded=excluded))
            cases.append(
                {
                    "user": user,
                    "item_seq": case_seq,
                    "target_item": case_seq[-1],
                    "context_item": case_seq[-2] if len(case_seq) >= 2 else None,
                    "labels": labels,
                    "group": label_group(labels),
                    "source_split": split,
                    "source_row_idx": row_idx,
                    "prefix_idx": prefix_idx,
                }
            )
    return cases


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_grouped_outputs(
    split_cases: dict[str, list[dict]],
    output_dir,
    include_uncategorized_in_generalize: bool = False,
) -> dict:
    output_dir = Path(output_dir)
    summary = {
        "format_version": 1,
        "memory_tiger_dir": str(output_dir / "memory_tiger"),
        "generalize_tiger_dir": str(output_dir / "generalize_tiger"),
        "include_uncategorized_in_generalize": include_uncategorized_in_generalize,
        "splits": {},
    }

    for split, cases in split_cases.items():
        memory_cases = [case for case in cases if case["group"] == "memory"]
        generalize_cases = [
            case
            for case in cases
            if case["group"] == "generalization"
            or (include_uncategorized_in_generalize and case["group"] == "uncategorized")
        ]

        all_path = output_dir / "all_labeled" / f"{split}.jsonl"
        memory_path = output_dir / "memory_tiger" / f"{split}.jsonl"
        generalize_path = output_dir / "generalize_tiger" / f"{split}.jsonl"
        _write_jsonl(all_path, cases)
        _write_jsonl(memory_path, memory_cases)
        _write_jsonl(generalize_path, generalize_cases)

        counts = Counter(case["group"] for case in cases)
        summary["splits"][split] = {
            "total": len(cases),
            "memory": counts["memory"],
            "generalization": counts["generalization"],
            "uncategorized": counts["uncategorized"],
            "all_labeled_file": str(all_path),
            "memory_tiger_file": str(memory_path),
            "generalize_tiger_file": str(generalize_path),
        }

    summary_path = output_dir / "summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return summary


class _DummyAccelerator:
    is_main_process = True
    device = "cpu"
    num_processes = 1

    @contextmanager
    def main_process_first(self):
        yield


def _dataset_id(config: dict) -> str:
    if config.get("category"):
        return f"{config['dataset']}-{config['category']}"
    if config.get("version"):
        return f"{config['dataset']}-{config['version']}"
    return str(config["dataset"])


def _rows_from_hf_dataset(dataset) -> list[dict]:
    return [
        {"user": user, "item_seq": list(item_seq)}
        for user, item_seq in zip(dataset["user"], dataset["item_seq"])
    ]


def load_split_rows_from_genrec(args, config_overrides: dict):
    from genrec.utils import get_config, get_dataset

    config = get_config(
        model_name="TIGER",
        dataset_name=args.dataset,
        config_file=args.config,
        config_dict=config_overrides,
    )
    config["accelerator"] = _DummyAccelerator()
    config.setdefault("device", "cpu")
    config["use_ddp"] = False

    raw_dataset = get_dataset(args.dataset)(config)
    split_datasets = raw_dataset.split()
    return {
        split: _rows_from_hf_dataset(split_datasets[split])
        for split in split_datasets
    }, config


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Build Memory-TIGER and Generalize-TIGER JSONL data with MemGen "
            "memorization/generalization labels."
        )
    )
    parser.add_argument("--dataset", type=str, default="AmazonReviews2023")
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--max_hop", type=int, default=4)
    parser.add_argument(
        "--splits",
        type=str,
        default="train,val,test",
        help="Comma-separated split names to export.",
    )
    parser.add_argument(
        "--include_uncategorized_in_generalize",
        action="store_true",
        help="Also place uncategorized cases in generalize_tiger/*.jsonl.",
    )
    parser.add_argument(
        "--no_leave_one_sequence_out",
        action="store_true",
        help=(
            "Disable leave-one-sequence-out labels for train cases. This makes "
            "almost every train prefix memory and is mainly for ablation/debugging."
        ),
    )
    args, unparsed = parser.parse_known_args()

    from genrec.utils import parse_command_line_args

    return args, parse_command_line_args(unparsed)


def main():
    args, config_overrides = parse_args()
    split_rows, config = load_split_rows_from_genrec(args, config_overrides)
    requested_splits = [split.strip() for split in args.splits.split(",") if split.strip()]
    missing = [split for split in requested_splits if split not in split_rows]
    if missing:
        raise ValueError(f"Requested split(s) not found: {missing}. Available: {sorted(split_rows)}")

    output_dir = Path(args.output_dir) if args.output_dir else Path(
        "data",
        "memgen_tiger",
        _dataset_id(config),
    )
    support_rows = split_rows["train"]
    split_cases = {}
    for split in requested_splits:
        split_cases[split] = build_labeled_cases(
            rows=split_rows[split],
            support_rows=support_rows,
            split=split,
            max_hop=args.max_hop,
            leave_one_sequence_out=(split == "train" and not args.no_leave_one_sequence_out),
        )

    summary = write_grouped_outputs(
        split_cases,
        output_dir=output_dir,
        include_uncategorized_in_generalize=args.include_uncategorized_in_generalize,
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

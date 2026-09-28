"""
Analyze semantic ID codebook and prefix coverage.

Example:
    python analysis/semantic_id_coverage.py \
        --sem-ids cache/AmazonReviews2023/Industrial_and_Scientific/processed/sidau_vq_uniform_sentence-t5-base_256,256,256,256.sem_ids \
        --codebook-size 256
"""

import argparse
import json
import math
from collections import Counter
from pathlib import Path


def _normalize_semantic_ids(item2sem_ids):
    sem_ids = []
    for item, raw_sid in item2sem_ids.items():
        if isinstance(raw_sid, int):
            sid = (int(raw_sid),)
        else:
            sid = tuple(int(token) for token in raw_sid)
        if not sid:
            raise ValueError(f"Empty semantic ID for item {item!r}")
        sem_ids.append(sid)

    if not sem_ids:
        raise ValueError("No semantic IDs found")

    n_layers = len(sem_ids[0])
    bad_lengths = sorted({len(sid) for sid in sem_ids if len(sid) != n_layers})
    if bad_lengths:
        raise ValueError(
            "Semantic IDs must have the same length. "
            f"Expected {n_layers}, found {bad_lengths}."
        )
    return sem_ids


def _entropy(counts):
    total = sum(counts.values())
    if total == 0:
        return 0.0
    value = 0.0
    for count in counts.values():
        prob = count / total
        value -= prob * math.log(prob)
    return value


def _coverage(n_used, capacity):
    if capacity is None or capacity <= 0:
        return None
    return n_used / capacity


def _prefix_capacity(codebook_sizes, prefix_len):
    if codebook_sizes is None:
        return None
    capacity = 1
    for size in codebook_sizes[:prefix_len]:
        capacity *= size
    return capacity


def analyze_semantic_ids(item2sem_ids, codebook_sizes=None, topk=10):
    sem_ids = _normalize_semantic_ids(item2sem_ids)
    n_layers = len(sem_ids[0])

    if codebook_sizes is not None:
        if len(codebook_sizes) == 1:
            codebook_sizes = codebook_sizes * n_layers
        if len(codebook_sizes) != n_layers:
            raise ValueError(
                f"Expected {n_layers} codebook sizes, got {len(codebook_sizes)}"
            )
        codebook_sizes = [int(size) for size in codebook_sizes]

    layer_reports = []
    for layer_idx in range(n_layers):
        values = [sid[layer_idx] for sid in sem_ids]
        counts = Counter(values)
        capacity = codebook_sizes[layer_idx] if codebook_sizes else None
        ent = _entropy(counts)
        entropy_base = capacity if capacity else len(counts)
        layer_reports.append(
            {
                "layer": layer_idx + 1,
                "capacity": capacity,
                "n_used": len(counts),
                "coverage": _coverage(len(counts), capacity),
                "entropy": ent,
                "entropy_ratio": ent / math.log(entropy_base)
                if entropy_base and entropy_base > 1
                else 0.0,
                "min_value": min(counts),
                "max_value": max(counts),
                "min_count": min(counts.values()),
                "max_count": max(counts.values()),
                "top_values": counts.most_common(topk),
            }
        )

    prefix_reports = []
    for prefix_len in range(1, n_layers + 1):
        prefixes = Counter(tuple(sid[:prefix_len]) for sid in sem_ids)
        capacity = _prefix_capacity(codebook_sizes, prefix_len)
        prefix_reports.append(
            {
                "prefix_len": prefix_len,
                "capacity": capacity,
                "n_used": len(prefixes),
                "coverage": _coverage(len(prefixes), capacity),
                "min_count": min(prefixes.values()),
                "max_count": max(prefixes.values()),
                "top_prefixes": prefixes.most_common(topk),
            }
        )

    full_counts = Counter(sem_ids)
    return {
        "n_items": len(sem_ids),
        "n_layers": n_layers,
        "n_unique_full_ids": len(full_counts),
        "max_items_per_full_id": max(full_counts.values()),
        "layers": layer_reports,
        "prefixes": prefix_reports,
    }


def _format_ratio(value):
    if value is None:
        return "-"
    return f"{value * 100:.2f}%"


def format_report(report):
    lines = []
    lines.append("Semantic ID coverage")
    lines.append(f"Items: {report['n_items']}")
    lines.append(f"Layers: {report['n_layers']}")
    lines.append(
        "Unique full IDs: "
        f"{report['n_unique_full_ids']} "
        f"(max items per full ID: {report['max_items_per_full_id']})"
    )

    lines.append("")
    lines.append("Layer coverage")
    lines.append(
        f"{'Layer':>5} {'Capacity':>10} {'Used':>8} {'Coverage':>10} "
        f"{'Entropy':>10} {'Ent/Max':>8} {'MinCnt':>8} {'MaxCnt':>8} "
        "Top values"
    )
    for row in report["layers"]:
        top_values = ", ".join(f"{value}:{count}" for value, count in row["top_values"])
        lines.append(
            f"{row['layer']:>5} "
            f"{str(row['capacity']) if row['capacity'] else '-':>10} "
            f"{row['n_used']:>8} "
            f"{_format_ratio(row['coverage']):>10} "
            f"{row['entropy']:>10.4f} "
            f"{row['entropy_ratio']:>8.4f} "
            f"{row['min_count']:>8} "
            f"{row['max_count']:>8} "
            f"{top_values}"
        )

    lines.append("")
    lines.append("Prefix coverage")
    lines.append(
        f"{'Prefix':>6} {'Capacity':>16} {'Used':>8} {'Coverage':>10} "
        f"{'MinCnt':>8} {'MaxCnt':>8} Top prefixes"
    )
    for row in report["prefixes"]:
        top_prefixes = ", ".join(
            f"{prefix}:{count}" for prefix, count in row["top_prefixes"]
        )
        lines.append(
            f"{row['prefix_len']:>6} "
            f"{str(row['capacity']) if row['capacity'] else '-':>16} "
            f"{row['n_used']:>8} "
            f"{_format_ratio(row['coverage']):>10} "
            f"{row['min_count']:>8} "
            f"{row['max_count']:>8} "
            f"{top_prefixes}"
        )
    return "\n".join(lines)


def print_report(report):
    print(format_report(report))


def _parse_codebook_sizes(args):
    if args.codebook_sizes:
        return [int(value) for value in args.codebook_sizes.split(",")]
    if args.codebook_size:
        return [int(args.codebook_size)]
    return None


def main():
    parser = argparse.ArgumentParser(
        description="Analyze per-layer and prefix coverage of semantic IDs."
    )
    parser.add_argument("--sem-ids", required=True, help="Path to a .sem_ids JSON file.")
    parser.add_argument(
        "--codebook-size",
        type=int,
        default=None,
        help="Single codebook size repeated for every semantic ID layer, e.g. 256.",
    )
    parser.add_argument(
        "--codebook-sizes",
        default=None,
        help="Comma-separated per-layer codebook sizes, e.g. 256,256,256,256.",
    )
    parser.add_argument(
        "--topk",
        type=int,
        default=10,
        help="Number of most frequent codewords or prefixes to print.",
    )
    parser.add_argument(
        "--json-output",
        default=None,
        help="Optional path to save the full report as JSON.",
    )
    args = parser.parse_args()

    sem_ids_path = Path(args.sem_ids)
    with sem_ids_path.open("r", encoding="utf-8") as f:
        item2sem_ids = json.load(f)

    report = analyze_semantic_ids(
        item2sem_ids,
        codebook_sizes=_parse_codebook_sizes(args),
        topk=args.topk,
    )
    print_report(report)

    if args.json_output:
        output_path = Path(args.json_output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)


if __name__ == "__main__":
    main()

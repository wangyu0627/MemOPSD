import math
from typing import Dict, Iterable, List, Sequence, Tuple


def _as_rows(values: Sequence) -> List[List]:
    return [list(row) for row in values]


def _softmax(values: Sequence[float], temperature: float) -> List[float]:
    temperature = max(float(temperature), 1e-12)
    logits = [float(value) / temperature for value in values]
    max_logit = max(logits)
    exp_values = [math.exp(value - max_logit) for value in logits]
    total = sum(exp_values)
    if total <= 0.0:
        return [0.0 for _ in exp_values]
    return [value / total for value in exp_values]


def confidence_weight(
    confidence: Iterable[float],
    tau_low: float,
    tau_high: float,
) -> List[float]:
    tau_low = float(tau_low)
    tau_high = float(tau_high)
    weights = []
    for value in confidence:
        value = float(value)
        if tau_high <= tau_low:
            weights.append(1.0 if value >= tau_low else 0.0)
        else:
            weights.append(min(1.0, max(0.0, (value - tau_low) / (tau_high - tau_low))))
    return weights


def soft_sid_targets_from_topk(
    top_items: Sequence[Sequence[int]],
    top_scores: Sequence[Sequence[float]],
    item2tokens: Dict[int, Sequence[int]],
    n_digit: int,
    vocab_size: int,
    temperature: float = 1.0,
    scores_are_logits: bool = True,
) -> Tuple[List[List[List[float]]], List[List[bool]]]:
    item_rows = _as_rows(top_items)
    score_rows = _as_rows(top_scores)
    n_digit = int(n_digit)
    vocab_size = int(vocab_size)

    all_targets: List[List[List[float]]] = []
    all_masks: List[List[bool]] = []
    for items, scores in zip(item_rows, score_rows):
        valid_items = []
        valid_scores = []
        for item, score in zip(items, scores):
            item = int(item)
            tokens = item2tokens.get(item)
            if tokens is None:
                continue
            if len(tokens) < n_digit:
                continue
            valid_items.append(item)
            valid_scores.append(float(score))

        targets = [[0.0 for _ in range(vocab_size)] for _ in range(n_digit)]
        mask = [False for _ in range(n_digit)]
        if valid_items:
            probs = (
                _softmax(valid_scores, temperature)
                if scores_are_logits
                else [float(score) for score in valid_scores]
            )
            if not scores_are_logits:
                total = sum(max(0.0, prob) for prob in probs)
                probs = [max(0.0, prob) / total for prob in probs] if total > 0.0 else probs

            for item, prob in zip(valid_items, probs):
                tokens = item2tokens[int(item)]
                for digit_idx in range(n_digit):
                    token = int(tokens[digit_idx])
                    if 0 <= token < vocab_size:
                        targets[digit_idx][token] += prob
                        mask[digit_idx] = True

            for digit_idx in range(n_digit):
                total = sum(targets[digit_idx])
                if total > 0.0:
                    targets[digit_idx] = [value / total for value in targets[digit_idx]]
                else:
                    mask[digit_idx] = False

        all_targets.append(targets)
        all_masks.append(mask)

    return all_targets, all_masks

"""Pure metric functions (no I/O) shared by bench_clothing, bench_count and agreement."""
from collections import Counter


def accuracy(gold: list, pred: list) -> float:
    """pred item None = no valid answer -> counted as wrong."""
    return sum(1 for g, p in zip(gold, pred) if g == p) / len(gold) if gold else float("nan")


def f1_per_class(gold: list, pred: list) -> dict:
    """Labels = every label seen in gold or in (valid) predictions. None predictions are only misses."""
    labels = sorted(set(gold) | {p for p in pred if p is not None})
    out = {}
    for c in labels:
        tp = sum(1 for g, p in zip(gold, pred) if g == c and p == c)
        fp = sum(1 for g, p in zip(gold, pred) if p == c and g != c)
        fn = sum(1 for g, p in zip(gold, pred) if g == c and p != c)
        out[c] = 2 * tp / (2 * tp + fp + fn) if (tp + fp + fn) else 0.0
    return out


def macro_f1(gold: list, pred: list) -> float:
    f = f1_per_class(gold, pred)
    return sum(f.values()) / len(f) if f else float("nan")


def top2_accuracy(gold: list, pred1: list, pred2: list) -> float:
    """Gold is hit by the first or the second guess. pred2 None = no second guess."""
    return sum(1 for g, a, b in zip(gold, pred1, pred2) if g == a or (b is not None and g == b)) / len(gold) \
        if gold else float("nan")


def confusion(gold: list, pred: list) -> tuple:
    """-> (labels, matrix) ; matrix[gold_label][pred_label] = count. No-answer predictions go to '__none__'."""
    pred = [p if p is not None else "__none__" for p in pred]
    labels = sorted(set(gold) | set(pred))
    m = {g: {p: 0 for p in labels} for g in labels}
    for g, p in zip(gold, pred):
        m[g][p] += 1
    return labels, m


def share(values: list, label) -> float:
    return sum(1 for v in values if v == label) / len(values) if values else float("nan")


def cohen_kappa(a: list, b: list) -> float:
    """Cohen's kappa of two raters. If chance agreement is 1 (both always give one same label):
    1.0 when they agree everywhere, otherwise 0.0 (the formula is 0/0 there)."""
    n = len(a)
    if n == 0:
        return float("nan")
    po = sum(1 for x, y in zip(a, b) if x == y) / n
    ca, cb = Counter(a), Counter(b)
    pe = sum(ca[c] / n * cb[c] / n for c in set(ca) | set(cb))
    if pe >= 1.0:
        return 1.0 if po == 1.0 else 0.0
    return (po - pe) / (1 - pe)


def count_metrics(true: list, pred: list) -> dict:
    """Counting metrics over pairs with a valid prediction (pred None = failed, reported separately)."""
    pairs = [(t, p) for t, p in zip(true, pred) if p is not None]
    n = len(pairs)
    out = {"n": len(true), "n_failed": len(true) - n}
    if not n:
        return {**out, "mae": float("nan"), "exact": float("nan"), "under": float("nan"),
                "over": float("nan"), "bias": float("nan")}
    out.update(mae=sum(abs(p - t) for t, p in pairs) / n,
               exact=sum(1 for t, p in pairs if p == t) / n,
               under=sum(1 for t, p in pairs if p < t) / n,
               over=sum(1 for t, p in pairs if p > t) / n,
               bias=sum(p - t for t, p in pairs) / n)
    return out


def bucket_of(box_h) -> str:
    if box_h is None:
        return "unknown_h"
    return "<80" if box_h < 80 else ("80-150" if box_h <= 150 else ">150")

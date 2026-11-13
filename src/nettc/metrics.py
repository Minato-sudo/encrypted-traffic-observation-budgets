"""Evaluation (Phase 2 Steps 8 and 9).

Two commitments from the methodology shape this module.

Per-class precision, recall, F1 and support are *always* reported together,
never replaced by an aggregate. The review found this failure repeatedly:
P2's own confusion matrix shows AIM at 0% and SFTP at 50% while the headline
macro-F1 looks healthy, and P10's File and VoIP classes sit far below its
98.25% average. The macro-to-weighted F1 gap is reported explicitly as a
direct measure of what aggregates conceal.

Everything is also reportable by stratum -- flow length, class support, and
encryption status -- because a single number averaged over a corpus that is
94.7% unencrypted (P5's measurement of USTC-TFC2016) does not describe
encrypted traffic classification.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional, Sequence

import numpy as np

EPS = 1e-12


@dataclass
class ClassMetrics:
    name: str
    support: int
    precision: float
    recall: float
    f1: float
    predicted: int


@dataclass
class Result:
    """Everything we report for one (model, budget, seed, stratum) cell."""
    accuracy: float
    macro_f1: float
    weighted_f1: float
    balanced_accuracy: float
    per_class: list
    n: int
    confusion: Optional[list] = None
    meta: dict = field(default_factory=dict)

    @property
    def concealment_gap(self) -> float:
        """Weighted F1 minus macro F1: how much the aggregate flatters the model."""
        return self.weighted_f1 - self.macro_f1

    def summary_line(self) -> str:
        return (f"acc {self.accuracy:.4f}  macroF1 {self.macro_f1:.4f}  "
                f"wF1 {self.weighted_f1:.4f}  gap {self.concealment_gap:+.4f}  n={self.n}")

    def to_dict(self) -> dict:
        d = asdict(self)
        d["concealment_gap"] = self.concealment_gap
        return d


def evaluate(y_true: np.ndarray, y_pred: np.ndarray, class_names: Sequence[str],
             with_confusion: bool = True, meta: Optional[dict] = None) -> Result:
    """Core metric computation. No sklearn dependency, so the arithmetic is visible."""
    y_true = np.asarray(y_true).astype(np.int64)
    y_pred = np.asarray(y_pred).astype(np.int64)
    k = len(class_names)
    cm = np.zeros((k, k), dtype=np.int64)
    np.add.at(cm, (y_true, y_pred), 1)

    support = cm.sum(axis=1)
    predicted = cm.sum(axis=0)
    tp = np.diag(cm).astype(np.float64)
    precision = tp / np.maximum(predicted, EPS)
    recall = tp / np.maximum(support, EPS)
    f1 = 2 * precision * recall / np.maximum(precision + recall, EPS)

    # Classes absent from the test set contribute nothing to macro averages;
    # including them as zeros would understate every model equally but make
    # the number depend on the split rather than the model.
    present = support > 0
    per_class = [
        ClassMetrics(name=class_names[i], support=int(support[i]),
                     precision=float(precision[i]), recall=float(recall[i]),
                     f1=float(f1[i]), predicted=int(predicted[i]))
        for i in range(k)
    ]
    n = int(support.sum())
    return Result(
        accuracy=float(tp.sum() / max(n, 1)),
        macro_f1=float(f1[present].mean()) if present.any() else 0.0,
        weighted_f1=float((f1 * support).sum() / max(n, 1)),
        balanced_accuracy=float(recall[present].mean()) if present.any() else 0.0,
        per_class=[asdict(c) for c in per_class],
        n=n,
        confusion=cm.tolist() if with_confusion else None,
        meta=meta or {},
    )


def per_class_table(result: Result, min_support: int = 0) -> str:
    rows = [f"{'class':<24}{'support':>8}{'prec':>8}{'recall':>8}{'F1':>8}"]
    for c in sorted(result.per_class, key=lambda r: r["support"]):
        if c["support"] < min_support:
            continue
        rows.append(f"{c['name']:<24}{c['support']:>8}{c['precision']:>8.3f}"
                    f"{c['recall']:>8.3f}{c['f1']:>8.3f}")
    rows.append(f"{'-- aggregate':<24}{result.n:>8}{'':>8}{'':>8}{result.macro_f1:>8.3f}")
    return "\n".join(rows)


# ---------------------------------------------------------------- stratification

def length_bins(values: np.ndarray, edges: Sequence[int] = (0, 128, 256, 512, 784, 1024, 1500, 1 << 30)
                ) -> tuple[np.ndarray, list[str]]:
    """Bin sessions by true length, with cut points at the budgets we sweep."""
    edges = list(edges)
    idx = np.clip(np.searchsorted(edges, values, side="right") - 1, 0, len(edges) - 2)
    names = [f"[{edges[i]},{edges[i+1]})" if edges[i + 1] < (1 << 30) else f"[{edges[i]},inf)"
             for i in range(len(edges) - 1)]
    return idx, names


def support_bins(store, edges: Optional[Sequence[int]] = None, n_bins: int = 4,
                 idx: Optional[np.ndarray] = None) -> tuple[np.ndarray, list[str]]:
    """Bin each session by how many sessions its class has.

    Gap 2 of the review is that minority classes fail while aggregate metrics
    conceal it, and the review's evidence was P2's AIM class at 0% accuracy on
    26 sessions. Fixed cut points cannot express that across corpora: on
    USTC-TFC2016 the *smallest* class holds several thousand sessions, so any
    edge list ending at 1000 puts every class in one bin and the
    stratification says nothing. The edges are therefore derived from the
    corpus unless given explicitly.

    Pass ``idx`` to count support within a split rather than the whole store.
    """
    counts = np.bincount(store.label[idx] if idx is not None else store.label,
                         minlength=store.n_classes)
    per_session = counts[store.label]

    if edges is None:
        present = counts[counts > 0]
        if present.size == 0:
            return np.zeros(store.n, dtype=int), ["no classes"]
        qs = np.unique(np.quantile(present, np.linspace(0, 1, n_bins + 1)).astype(np.int64))
        if qs.size < 2:
            return (np.zeros(store.n, dtype=int),
                    [f"all classes ~{int(present[0])} sessions"])
        edges = list(qs)
        edges[0] = 0
        edges[-1] = int(present.max()) + 1
    else:
        edges = list(edges)

    idx_bin = np.clip(np.searchsorted(edges, per_session, side="right") - 1,
                      0, len(edges) - 2)
    names = [f"{edges[i]}-{edges[i + 1] - 1}" if edges[i + 1] < (1 << 30)
             else f"{edges[i]}+" for i in range(len(edges) - 1)]
    return idx_bin, names


def stratified(y_true: np.ndarray, y_pred: np.ndarray, strata: np.ndarray,
               stratum_names: Sequence[str], class_names: Sequence[str],
               min_n: int = 10) -> dict[str, Result]:
    """Evaluate within each stratum. Strata below ``min_n`` are reported, flagged."""
    out: dict[str, Result] = {}
    for s, name in enumerate(stratum_names):
        m = strata == s
        if not m.any():
            continue
        res = evaluate(y_true[m], y_pred[m], class_names, with_confusion=False,
                       meta={"stratum": name, "underpowered": bool(m.sum() < min_n)})
        out[name] = res
    return out


# ---------------------------------------------------------------- comparisons

def mcnemar(correct_a: np.ndarray, correct_b: np.ndarray) -> dict:
    """Exact-ish McNemar test on paired per-sample correctness (after P12).

    Uses the binomial tail on the discordant pairs, which is exact and does
    not need the continuity-corrected chi-square approximation to be valid at
    small discordant counts.
    """
    a = np.asarray(correct_a).astype(bool)
    b = np.asarray(correct_b).astype(bool)
    n01 = int((~a & b).sum())   # b right, a wrong
    n10 = int((a & ~b).sum())   # a right, b wrong
    n = n01 + n10
    if n == 0:
        return {"n01": 0, "n10": 0, "p_value": 1.0, "note": "no discordant pairs"}
    # Accumulate the binomial tail in log space. Forming comb(n, i) and 2**n
    # separately overflows a float as soon as the test set is large -- n here
    # is the discordant count, which reached 5,000-odd on a 52,000-session
    # split and raised OverflowError. Working with log-gamma keeps every
    # intermediate finite and the result is the same exact tail.
    from math import exp, lgamma, log
    k = min(n01, n10)
    log_half = -n * log(2.0)
    tail = 0.0
    for i in range(k + 1):
        log_term = (lgamma(n + 1) - lgamma(i + 1) - lgamma(n - i + 1)) + log_half
        tail += exp(log_term)
    return {"n01": n01, "n10": n10, "p_value": float(min(1.0, 2.0 * tail)),
            "n_discordant": n}


def seed_summary(values: Sequence[float]) -> dict:
    """Mean and standard deviation over seeds, as Step 9 requires."""
    arr = np.asarray(values, dtype=np.float64)
    return {"mean": float(arr.mean()), "std": float(arr.std(ddof=1) if arr.size > 1 else 0.0),
            "n_seeds": int(arr.size), "values": [float(v) for v in arr]}


def compare_budgets(a: dict, b: dict, metric: str = "macro_f1") -> dict:
    """Difference between two budgets, declared inconclusive inside seed noise.

    Step 9: "a difference smaller than the seed-to-seed standard deviation is
    reported as inconclusive". This returns that verdict rather than a p-value
    alone, because with three seeds a p-value is not the honest instrument.
    """
    diff = a[metric]["mean"] - b[metric]["mean"]
    noise = max(a[metric]["std"], b[metric]["std"])
    return {"metric": metric, "diff": float(diff), "seed_noise": float(noise),
            "verdict": "inconclusive" if abs(diff) <= noise else
                       ("a_better" if diff > 0 else "b_better")}


# ---------------------------------------------------------------- frontier

def pareto_frontier(points: Sequence[tuple[float, float]]) -> list[int]:
    """Indices of points not dominated on (cost down, accuracy up)."""
    order = sorted(range(len(points)), key=lambda i: (points[i][0], -points[i][1]))
    keep, best = [], -np.inf
    for i in order:
        if points[i][1] > best:
            keep.append(i)
            best = points[i][1]
    return keep


def save_json(obj, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    def default(o):
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, (np.floating,)):
            return float(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        if hasattr(o, "to_dict"):
            return o.to_dict()
        if hasattr(o, "__dict__"):
            return o.__dict__
        raise TypeError(f"not JSON-serialisable: {type(o)}")

    path.write_text(json.dumps(obj, indent=2, default=default))

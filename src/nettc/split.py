"""Group-wise train/val/test splitting (Phase 2 Step 3f).

Splitting is group-wise by source capture file so that no session from one
capture can appear on both sides. That matters more than it sounds: in
USTC-TFC2016 one pcap *is* one class, so a file-level split is impossible
there and the methodology's temporal fallback is the normal case rather than
the exception. Both paths are implemented and which one was used for each
class is recorded in the manifest, because the two give different guarantees
and a reader needs to know which applies.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

TRAIN, VAL, TEST = 0, 1, 2
SPLIT_NAMES = ("train", "val", "test")


@dataclass
class SplitManifest:
    assignment: np.ndarray                  # per-session TRAIN/VAL/TEST
    strategy: dict = field(default_factory=dict)   # class name -> "group" | "temporal"
    fractions: tuple = (0.70, 0.15, 0.15)
    seed: int = 0

    def mask(self, split: int) -> np.ndarray:
        return self.assignment == split

    def indices(self, split: int) -> np.ndarray:
        return np.flatnonzero(self.assignment == split)

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps({
            "assignment": self.assignment.astype(int).tolist(),
            "strategy": self.strategy,
            "fractions": list(self.fractions),
            "seed": self.seed,
        }))

    @staticmethod
    def load(path: str | Path) -> "SplitManifest":
        d = json.loads(Path(path).read_text())
        return SplitManifest(
            assignment=np.asarray(d["assignment"], dtype=np.int64),
            strategy=d["strategy"], fractions=tuple(d["fractions"]), seed=d["seed"],
        )


def make_split(store, fractions=(0.70, 0.15, 0.15), seed: int = 0,
               min_groups_for_group_split: int = 3) -> SplitManifest:
    """Assign every session to train, val or test.

    For each class independently: if its sessions come from at least
    ``min_groups_for_group_split`` capture files, whole files are assigned to
    splits greedily so the resulting session counts approach ``fractions``.
    Otherwise the class falls back to a temporal split on first-packet
    timestamp, which keeps a time boundary between train and test even though
    it cannot keep a capture boundary.
    """
    assert abs(sum(fractions) - 1.0) < 1e-9, "fractions must sum to 1"
    rng = np.random.default_rng(seed)
    assignment = np.full(store.n, -1, dtype=np.int64)
    strategy: dict[str, str] = {}

    for cls in range(store.n_classes):
        cls_idx = np.flatnonzero(store.label == cls)
        if cls_idx.size == 0:
            continue
        name = store.label_names[cls]
        groups = np.unique(store.src_file[cls_idx])

        if groups.size >= min_groups_for_group_split:
            strategy[name] = "group"
            # Largest-remainder greedy fill: shuffle files, then give each to
            # whichever split is furthest below its target share.
            order = rng.permutation(groups)
            targets = np.asarray(fractions) * cls_idx.size
            filled = np.zeros(3)
            for g in order:
                g_idx = cls_idx[store.src_file[cls_idx] == g]
                deficit = targets - filled
                dest = int(np.argmax(deficit))
                assignment[g_idx] = dest
                filled[dest] += g_idx.size
            # A split left empty by an unlucky file-size distribution would
            # make the class unevaluable; fall back rather than report on it.
            if any((assignment[cls_idx] == s).sum() == 0 for s in (TRAIN, VAL, TEST)):
                strategy[name] = "temporal (group split left a split empty)"
            else:
                continue

        if name not in strategy or strategy[name].startswith("temporal"):
            strategy.setdefault(name, "temporal (single capture file)")
        order = cls_idx[np.argsort(store.first_ts[cls_idx], kind="stable")]
        n = order.size
        n_train = int(round(fractions[0] * n))
        n_val = int(round(fractions[1] * n))
        # Guarantee at least one session per split once a class has three.
        if n >= 3:
            n_train = min(max(n_train, 1), n - 2)
            n_val = min(max(n_val, 1), n - n_train - 1)
        assignment[order[:n_train]] = TRAIN
        assignment[order[n_train:n_train + n_val]] = VAL
        assignment[order[n_train + n_val:]] = TEST

    leftover = int((assignment < 0).sum())
    if leftover:
        assignment[assignment < 0] = TRAIN
    return SplitManifest(assignment=assignment, strategy=strategy,
                         fractions=tuple(fractions), seed=seed)


def cap_per_class(store, manifest: SplitManifest, max_per_class: int,
                  seed: int = 0) -> SplitManifest:
    """Cap each class to ``max_per_class`` sessions *within each split*.

    USTC-TFC2016 is severely unbalanced as distributed -- a single class can
    contribute an order of magnitude more sessions than another, because the
    capture files differ in size by that much. Training on it as-is makes the
    budget sweep partly a study of the two largest classes, and makes each run
    long enough that three seeds across two budget axes is impractical.

    Capping is applied after splitting, never before, so it cannot move a
    session across the train/test boundary. It is recorded in the manifest and
    must be reported: it changes the class priors, so absolute accuracies here
    are not comparable to papers that use the full corpus.
    """
    rng = np.random.default_rng(seed)
    assignment = manifest.assignment.copy()
    dropped = 0
    for split in (TRAIN, VAL, TEST):
        for cls in range(store.n_classes):
            idx = np.flatnonzero((store.label == cls) & (assignment == split))
            if idx.size <= max_per_class:
                continue
            drop = rng.choice(idx, size=idx.size - max_per_class, replace=False)
            assignment[drop] = -1
            dropped += drop.size
    out = SplitManifest(assignment=assignment, strategy=dict(manifest.strategy),
                        fractions=manifest.fractions, seed=manifest.seed)
    out.strategy["_cap"] = (f"capped at {max_per_class} per class per split "
                            f"(seed {seed}); {dropped} sessions set aside")
    return out


def summarise(store, manifest: SplitManifest) -> str:
    lines = ["split      sessions   classes"]
    for s, sname in enumerate(SPLIT_NAMES):
        idx = manifest.indices(s)
        lines.append(f"{sname:<10} {idx.size:>8}   {len(np.unique(store.label[idx])):>7}")
    lines.append("")
    lines.append(f"{'class':<22}{'train':>8}{'val':>7}{'test':>7}   strategy")
    for cls in range(store.n_classes):
        name = store.label_names[cls]
        cls_mask = store.label == cls
        counts = [int((cls_mask & manifest.mask(s)).sum()) for s in range(3)]
        lines.append(f"{name:<22}{counts[0]:>8}{counts[1]:>7}{counts[2]:>7}   "
                     f"{manifest.strategy.get(name, '-')}")
    return "\n".join(lines)

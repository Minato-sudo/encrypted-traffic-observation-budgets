"""Smoke test for the split manifest and the budget slicer."""
import sys, numpy as np
from pathlib import Path
from nettc.store import Store
from nettc.split import make_split, summarise, TRAIN, VAL, TEST
from nettc.slicer import (BudgetSpec, take, resolve_packet_width,
                          length_balanced_indices, length_balance_plan, PAD_SYMBOL)

s = Store(Path(sys.argv[1]) / "store.h5")
man = make_split(s, seed=0)
print(summarise(s, man)); print()

tr, te = man.indices(TRAIN), man.indices(TEST)
assert len(np.intersect1d(tr, te)) == 0, "train and test overlap"
print("no train/test index overlap: OK")

print("\nbyte budgets")
for b in (64, 128, 256, 512, 784, 1024, 1500):
    sl = take(s, te, BudgetSpec(n_bytes=b))
    print(f"  {b:>5}B  x{tuple(sl.x.shape)}  mean pad frac {sl.padding_fraction.mean():.3f}"
          f"  truncated {sl.truncated.mean():.3f}")

print("\npacket budgets (width from TRAIN split only)")
for p in (1, 2, 3, 5, 8, 15, 16, 20):
    spec = resolve_packet_width(s, BudgetSpec(n_packets=p), tr)
    sl = take(s, te, spec)
    print(f"  {p:>3}P  width {spec.width:>5}  x{tuple(sl.x.shape)}  mean pad frac {sl.padding_fraction.mean():.3f}")

print("\nmonotonicity: bytes available must not decrease with packet budget")
prev = None
for p in (1, 2, 3, 5, 8, 15, 20):
    spec = resolve_packet_width(s, BudgetSpec(n_packets=p), tr)
    sl = take(s, te, spec)
    mean_valid = float(np.minimum(sl.valid_len, s.byte_ceiling).mean())
    assert prev is None or mean_valid >= prev - 1e-9, f"non-monotonic at {p}P"
    prev = mean_valid
print("  OK")

print("\npad modes")
sl0 = take(s, te[:50], BudgetSpec(n_bytes=1024), pad_mode="zero")
slm = take(s, te[:50], BudgetSpec(n_bytes=1024), pad_mode="mask")
assert slm.x.dtype == np.int16 and (slm.x == PAD_SYMBOL).any(), "mask mode produced no PAD symbols"
real = np.arange(1024)[None, :] < np.minimum(sl0.valid_len, 1024)[:, None]
assert (sl0.x[real] == slm.x[real]).all(), "mask mode altered real bytes"
assert (slm.x[~real] == PAD_SYMBOL).all(), "mask mode left a padded position unmarked"
print("  zero/mask agree on real bytes, differ only on padding: OK")

print("\nident source (leakage control input)")
sli = take(s, te[:20], BudgetSpec(n_bytes=64), source="ident")
print(f"  x{tuple(sli.x.shape)}  nonzero rows {int((sli.x != 0).any(axis=1).sum())}/20")

print("\nlength-balanced resample")
rep = {}
bal = length_balanced_indices(s, tr, n_bins=6, seed=0, report=rep)
import collections
before = np.bincount(s.label[tr], minlength=s.n_classes)
after = np.bincount(s.label[bal], minlength=s.n_classes)
print("  class counts before:", before.tolist())
print("  class counts after :", after.tolist())
print("  equalised:", bool(len(set(after[after > 0].tolist())) == 1))
# Only the classes the control actually retained; it may exclude some, and
# a median over an empty class is nan, not a spread of zero.
kept = np.unique(s.label[bal]) if bal.size else np.array([], dtype=int)
med_before = [float(np.median(s.true_length[tr][s.label[tr]==c])) for c in kept]
med_after  = [float(np.median(s.true_length[bal][s.label[bal]==c])) for c in kept]
print(f"  retained {len(kept)}/{s.n_classes} classes, {bal.size} sessions"
      f"  ({rep.get('outcome','')})")
if kept.size:
    print("  per-class median length spread before: %.0f  after: %.0f"
          % (max(med_before)-min(med_before), max(med_after)-min(med_after)))
    assert max(med_after)-min(med_after) <= max(med_before)-min(med_before) + 1e-9, \
        "balancing must not widen the per-class length spread"
else:
    print("  control could not be formed on this corpus (reported, not silent)")

# Regression guard. Derived independently per split, the bins and the retained
# class set disagree between train and test, and a model trained on one label
# set and scored against another collapses to chance -- which reads exactly
# like the finding this control exists to detect. length_balance_plan derives
# both from train so every split shares them. See the docstring in slicer.py.
print("\nlength balance: every split must share one class set")
va = man.indices(VAL)
edges, keep = length_balance_plan(s, tr, n_bins=6)
shared = [length_balanced_indices(s, idx, seed=0, edges=edges, classes=keep)
          for idx in (tr, va, te)]
sets = [set(np.unique(s.label[b]).tolist()) for b in shared if b.size]
assert sets, "the plan produced no usable split; cannot verify agreement"
assert all(t == sets[0] for t in sets), (
    f"splits disagree on classes: {sets} -- a model trained on one and scored "
    f"against another collapses to chance and looks like a finding")
print(f"  train {shared[0].size} / val {shared[1].size} / test {shared[2].size} "
      f"sessions, {len(sets[0])} classes, identical across splits: OK")

# And the bug itself: independently derived plans are allowed to disagree, so
# the test asserts the *fix* is what prevents it, not that disagreement is
# impossible.
independent = [set(np.unique(s.label[length_balanced_indices(s, idx, n_bins=6, seed=0)]).tolist())
               for idx in (tr, te)]
if independent[0] != independent[1]:
    print(f"  (per-split derivation would have disagreed: {independent[0]} vs "
          f"{independent[1]} -- the shared plan is what prevents it)")
else:
    print("  (per-split derivation happens to agree on this corpus; the shared "
          "plan is still required, see slicer.py)")
s.close()
print("\nall checks passed")

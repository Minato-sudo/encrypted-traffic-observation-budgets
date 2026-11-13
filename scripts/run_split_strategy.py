"""Does a within-capture split inflate the result, and by how much?

Every number this study reports on ISCX rests on a split that is group-wise
for some classes and temporal for others, and USTC's is temporal for 19 of its
20 classes. A temporal split draws train and test from the same capture file;
a group-wise split does not. The first is the easier problem, the corpus that
scores higher here is also the one split the easier way, and the main study
cannot separate the two. That is its largest limitation.

It is separable on the subset of ISCX classes that have enough capture files
to be split either way. This script holds the classes, the budget, the model,
the epochs, the seeds *and the per-class split sizes* fixed, and varies only
where the train/test boundary falls:

  group     whole capture files to one side or the other. No session in the
            test set shares a capture with any training session.
  temporal  the same sessions, split at a timestamp. Train and test come from
            the same captures, but no test session precedes a training one.
  random    the same sessions, shuffled. The same captures on both sides and
            no time boundary either -- what a paper that says nothing about
            its split policy has most likely done.

Matching the per-class counts is what makes this a measurement rather than
another confound: the group arm's sizes fall where the capture files happen to
land, so the other two arms are cut to those same sizes instead of to a tidy
70/15/15. The only thing that differs between arms is the boundary.

The three arms score different test sets, so there is no paired test to run
here and the verdict is the seed-noise rule from Step 9. Reported as
``results/<corpus>/split_strategy.json``.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from nettc.metrics import save_json, seed_summary
from nettc.slicer import BudgetSpec
from nettc.split import TEST, TRAIN, VAL, SplitManifest, make_split
from nettc.store import Store
from nettc.train import TrainConfig, evaluate_model, train_model

CORPORA = {
    "iscx": dict(store="stores/iscx.h5", out="results/iscx",
                 budget=256, epochs=15, cap=5000, pad_mode="embed"),
    "ustc": dict(store="stores/ustc.h5", out="results/ustc",
                 budget=512, epochs=40, cap=0, pad_mode="embed"),
}
ARMS = ("group", "temporal", "random")


def group_splittable(manifest: SplitManifest, store) -> list[int]:
    """Classes the main split handled group-wise, as class indices."""
    return [c for c in range(store.n_classes)
            if manifest.strategy.get(store.label_names[c]) == "group"]


def build_arms(store, manifest: SplitManifest, classes: list[int],
               cap: int, seed: int = 0) -> tuple[dict, dict]:
    """Three (train, val, test) index triples over the same sessions.

    The group arm is lifted straight from the study's own manifest, so it is
    the split the rest of the results were produced under. The other two
    reproduce its per-class sizes exactly, cutting the same pool of sessions
    at a different boundary.
    """
    rng = np.random.default_rng(seed)
    arms: dict[str, list[list[np.ndarray]]] = {a: [[], [], []] for a in ARMS}
    sizes: dict[str, list[int]] = {}

    for c in classes:
        cls_idx = np.flatnonzero(store.label == c)
        got = [cls_idx[manifest.assignment[cls_idx] == s] for s in (TRAIN, VAL, TEST)]
        n = [g.size for g in got]

        # Cap inside each split, as the main study does, before the other arms
        # are cut to these sizes -- so all three stay matched.
        if cap:
            got = [g if g.size <= cap else rng.choice(g, cap, replace=False)
                   for g in got]
            n = [g.size for g in got]
        sizes[store.label_names[c]] = n

        by_time = cls_idx[np.argsort(store.first_ts[cls_idx], kind="stable")]
        shuffled = rng.permutation(cls_idx)
        for arm, pool in (("temporal", by_time), ("random", shuffled)):
            a, b = n[0], n[0] + n[1]
            for s, part in enumerate((pool[:a], pool[a:b], pool[b:b + n[2]])):
                arms[arm][s].append(part)
        for s in range(3):
            arms["group"][s].append(got[s])

    out = {}
    for arm in ARMS:
        out[arm] = tuple(np.concatenate(p) if p else np.array([], dtype=np.int64)
                         for p in arms[arm])
    return out, sizes


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--corpus", choices=sorted(CORPORA), default="iscx")
    ap.add_argument("--budget", type=int, default=None)
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--model", default="wang1d")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--min-classes", type=int, default=4,
                    help="refuse to report below this many group-splittable classes")
    args = ap.parse_args()

    conf = CORPORA[args.corpus]
    budget = args.budget or conf["budget"]
    epochs = args.epochs or conf["epochs"]
    store = Store(conf["store"])
    manifest = make_split(store)

    classes = group_splittable(manifest, store)
    names = [store.label_names[c] for c in classes]
    print(f"{args.corpus}: {len(classes)} of {store.n_classes} classes can be "
          f"split group-wise\n  {', '.join(names)}")
    if len(classes) < args.min_classes:
        plural = "" if len(classes) == 1 else "es"
        print(f"\nrefusing to run: {len(classes)} group-splittable class{plural} is too "
              f"few to compare split strategies on. This corpus cannot answer "
              f"the question; that is itself the finding.", file=sys.stderr)
        return 2

    arms, sizes = build_arms(store, manifest, classes, conf["cap"])
    chance = 1.0 / len(classes)
    for arm in ARMS:
        tr, va, te = arms[arm]
        print(f"  {arm:<9} train {tr.size:>6}  val {va.size:>5}  test {te.size:>5}")
    assert len({tuple(a.size for a in arms[x]) for x in ARMS}) == 1, \
        "arms are not size-matched"

    cfg = TrainConfig(epochs=epochs, batch_size=args.batch_size,
                      pad_mode=conf["pad_mode"], patience=max(4, epochs // 5))
    spec = BudgetSpec(n_bytes=budget)
    print(f"\nbudget {spec.label}  model {args.model}  epochs {epochs}  "
          f"seeds {tuple(args.seeds)}  chance {chance:.4f}")

    t0 = time.time()
    out: dict = {
        "corpus": args.corpus, "budget": spec.label, "model": args.model,
        "epochs": epochs, "seeds": list(args.seeds), "pad_mode": conf["pad_mode"],
        "classes": names, "n_classes_total": int(store.n_classes),
        "chance_rate": chance, "per_class_split_sizes": sizes,
        "n_train": int(arms["group"][0].size), "n_test": int(arms["group"][2].size),
        "arms": {},
    }

    for arm in ARMS:
        tr, va, te = arms[arm]
        accs, macros = [], []
        for seed in args.seeds:
            model, _h, _t, _v = train_model(store, tr, va, spec, args.model,
                                            cfg, seed=seed)
            res, _pred, _view = evaluate_model(model, store, te, spec, cfg)
            accs.append(res.accuracy)
            macros.append(res.macro_f1)
            print(f"  {arm:<9} seed {seed}  {res.summary_line()}", flush=True)
        out["arms"][arm] = {"accuracy": seed_summary(accs),
                            "macro_f1": seed_summary(macros)}

    # Everything is measured against the group arm, because that is the only
    # one of the three whose test sessions are guaranteed not to share a
    # capture with anything the model trained on.
    base = out["arms"]["group"]["macro_f1"]
    for arm in ARMS:
        if arm == "group":
            continue
        cell = out["arms"][arm]
        diff = cell["macro_f1"]["mean"] - base["mean"]
        noise = max(cell["macro_f1"]["std"], base["std"])
        cell["inflation_macro_f1"] = float(diff)
        cell["inflation_accuracy"] = float(
            cell["accuracy"]["mean"] - out["arms"]["group"]["accuracy"]["mean"])
        cell["seed_noise"] = float(noise)
        cell["verdict"] = ("inconclusive" if abs(diff) <= noise else
                           ("inflated" if diff > 0 else "deflated"))

    out["seconds"] = time.time() - t0
    print()
    print(f"{'arm':<10}{'macro-F1':>10}{'sd':>8}{'vs group':>13}  verdict")
    for arm in ARMS:
        c = out["arms"][arm]
        d = (f"{c['inflation_macro_f1']:+.4f}" if arm != "group" else "(reference)")
        v = c.get("verdict", "")
        print(f"{arm:<10}{c['macro_f1']['mean']:>10.4f}"
              f"{c['macro_f1']['std']:>8.4f}{d:>13}  {v}")

    path = Path(conf["out"]) / "split_strategy.json"
    save_json(out, path)
    print(f"\nwrote {path} in {out['seconds']/60:.1f} min")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

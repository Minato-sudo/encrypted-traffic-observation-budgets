#!/usr/bin/env python3
"""Paired significance tests for the comparisons the study actually rests on.

Step 9 reports a difference smaller than the seed-to-seed spread as
inconclusive, which is the right default with three seeds: a p-value computed
across three runs is not an honest instrument. But two of this study's central
claims are *paired*, the same test sessions scored by two models, and for
those McNemar's exact test is both applicable and much stronger than a
seed-noise band, because it works per sample rather than per run.

Two pairs are tested, one per claim:

1. **Saturation (Gap 1).** The smallest budget the sweep saturates at against
   the best larger budget. The claim is that reading more bytes buys nothing;
   a seed-noise verdict can only fail to detect a difference, whereas McNemar
   can say how many sessions actually changed answer and in which direction.

2. **Port dependence (Gap 4).** The same budget with the four port bytes
   retained and with them masked in every packet. The claim is that the port
   shortcut is redundantly encoded rather than load-bearing.

Both are scored on identical test indices with identical seeds, so the pairing
is exact. The granularity ablation is deliberately *not* tested here: sessions
and flows are different units, so their test sets differ in size and McNemar
does not apply.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import numpy as np

from nettc.experiments import ExperimentPaths
from nettc.metrics import mcnemar, save_json
from nettc.slicer import BudgetSpec
from nettc.split import SplitManifest, TEST, TRAIN, VAL, cap_per_class
from nettc.store import Store
from nettc.train import TrainConfig, evaluate_model, train_model

CORPORA = {
    "ustc": dict(store="stores/ustc.h5", out="results/ustc", epochs=40, cap=0),
    "iscx": dict(store="stores/iscx.h5", out="results/iscx", epochs=15, cap=5000),
}


def budget_of(label: str) -> int:
    """'784B' -> 784."""
    return int("".join(c for c in label if c.isdigit()))


def fit_and_score(store, tr, va, te, spec, cfg, seed, source):
    model, _h, _t, _v = train_model(store, tr, va, spec, "wang1d", cfg,
                                    seed=seed, source=source)
    res, pred, view = evaluate_model(model, store, te, spec, cfg, source=source)
    return res, np.asarray(pred) == np.asarray(view.y)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--corpus", required=True, choices=sorted(CORPORA))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    spec_cfg = CORPORA[args.corpus]
    paths = ExperimentPaths(Path(spec_cfg["out"]))
    target = paths.json("significance")
    if target.exists() and not args.force:
        print(f"already at {target} (--force to re-run)")
        return 0

    sweep = json.loads(paths.json("e2_budget_sweep_wang1d").read_text())
    sat = sweep["bytes_saturation"]
    small, large = budget_of(sat["at"]), budget_of(sat["best_larger_budget"])

    store = Store(spec_cfg["store"], lazy_bytes=True)
    manifest = SplitManifest.load(spec_cfg["store"] + ".split.json")
    if spec_cfg["cap"]:
        manifest = cap_per_class(store, manifest, spec_cfg["cap"], seed=0)
    tr, va, te = manifest.indices(TRAIN), manifest.indices(VAL), manifest.indices(TEST)
    cfg = TrainConfig(epochs=spec_cfg["epochs"], patience=4, batch_size=512,
                      pad_mode="embed")

    out = {"corpus": args.corpus, "seed": args.seed, "n_test": int(te.size),
           "note": ("McNemar's exact test on paired per-sample correctness. The "
                    "seed-noise verdicts reported elsewhere answer a different "
                    "question, whether a difference exceeds run-to-run variation, "
                    "and both are given because either alone is misleading."),
           "pairs": {}}
    t0 = time.time()

    print(f"=== saturation: {small}B against {large}B, {te.size} paired test sessions")
    res_a, ok_a = fit_and_score(store, tr, va, te, BudgetSpec(n_bytes=small), cfg,
                                args.seed, "bytes")
    res_b, ok_b = fit_and_score(store, tr, va, te, BudgetSpec(n_bytes=large), cfg,
                                args.seed, "bytes")
    test = mcnemar(ok_a, ok_b)
    out["pairs"]["saturation"] = {
        "question": f"does {large}B beat the saturation point {small}B?",
        "a": {"label": f"{small}B", "accuracy": res_a.accuracy, "macro_f1": res_a.macro_f1},
        "b": {"label": f"{large}B", "accuracy": res_b.accuracy, "macro_f1": res_b.macro_f1},
        "mcnemar": test,
        "bytes_ratio": large / small,
    }
    print(f"    {small}B acc {res_a.accuracy:.4f} | {large}B acc {res_b.accuracy:.4f} | "
          f"n01={test['n01']} n10={test['n10']} p={test['p_value']:.4g}")

    print(f"=== ports: retained against masked at {small}B")
    res_p, ok_p = fit_and_score(store, tr, va, te, BudgetSpec(n_bytes=small), cfg,
                                args.seed, "bytes")
    res_m, ok_m = fit_and_score(store, tr, va, te, BudgetSpec(n_bytes=small), cfg,
                                args.seed, "no_ports")
    test2 = mcnemar(ok_p, ok_m)
    out["pairs"]["ports"] = {
        "question": f"does masking the port bytes change the result at {small}B?",
        "a": {"label": "ports retained", "accuracy": res_p.accuracy,
              "macro_f1": res_p.macro_f1},
        "b": {"label": "ports masked", "accuracy": res_m.accuracy,
              "macro_f1": res_m.macro_f1},
        "mcnemar": test2,
    }
    print(f"    retained acc {res_p.accuracy:.4f} | masked acc {res_m.accuracy:.4f} | "
          f"n01={test2['n01']} n10={test2['n10']} p={test2['p_value']:.4g}")

    out["seconds"] = time.time() - t0
    save_json(out, target)
    print(f"wrote {target} in {out['seconds'] / 60:.1f} min")
    store.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""The two pieces the main run could not produce, for either corpus.

1. A stratified evaluation *without* the per-class cap. The cap exists to
   control runtime and class imbalance, but it caps the test set too, which
   flattens exactly the minority-class behaviour Gap 2 is about. The uncapped
   run restores the native imbalance so the gap can actually be examined.

2. Step 10(iii) granularity: bidirectional session against unidirectional
   flow. The unit of analysis is fixed at preprocessing time, so this needs a
   second store built with --directional, and is skipped if it is absent.

    scripts/run_granularity_and_uncapped.py --corpus iscx
    scripts/run_granularity_and_uncapped.py --corpus ustc --skip-granularity
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from nettc.ablations import ablate_granularity, summarise
from nettc.experiments import ExperimentPaths, run_e3
from nettc.metrics import save_json
from nettc.split import SplitManifest, make_split, summarise as split_summary
from nettc.store import Store
from nettc.train import TrainConfig

CORPORA = {
    "ustc": dict(store="stores/ustc.h5", flows="stores/ustc_flows.h5",
                 out="results/ustc", root="data/USTC-TFC2016"),
    "iscx": dict(store="stores/iscx.h5", flows="stores/iscx_flows.h5",
                 out="results/iscx", root="data/ISCX-VPN-nonVPN"),
}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--corpus", required=True, choices=sorted(CORPORA))
    ap.add_argument("--budget", type=int, default=784)
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--batch-size", type=int, default=512)
    ap.add_argument("--skip-uncapped", action="store_true")
    ap.add_argument("--skip-granularity", action="store_true")
    ap.add_argument("--force", action="store_true",
                    help="recompute a result that is already on disk")
    args = ap.parse_args()

    spec = CORPORA[args.corpus]
    out = Path(spec["out"])
    paths = ExperimentPaths(out)
    cfg = TrainConfig(epochs=args.epochs, patience=4,
                      batch_size=args.batch_size, pad_mode="embed")

    store = Store(spec["store"], lazy_bytes=False)
    manifest = SplitManifest.load(spec["store"] + ".split.json")   # no cap

    if not args.skip_uncapped:
        target = paths.json(f"e3_stratified_uncapped_wang1d_{args.budget}B")
        if target.exists() and not args.force:
            print(f"uncapped stratified evaluation already at {target} "
                  f"(--force to re-run)", flush=True)
        else:
            print("=" * 60, "\nuncapped stratified evaluation (Gap 2)\n",
                  "=" * 60, flush=True)
            t = time.time()
            e3 = run_e3(store, manifest, cfg, model_name="wang1d",
                        budget=args.budget, seed=0, paths=None, verbose=True)
            save_json(e3, target)
            print(f"  overall macro-F1 {e3['overall']['macro_f1']:.4f}, "
                  f"weighted-F1 {e3['overall']['weighted_f1']:.4f}, "
                  f"gap {e3['overall']['weighted_f1'] - e3['overall']['macro_f1']:+.4f}")
            print(f"  took {(time.time() - t) / 60:.1f} min", flush=True)

    if not args.skip_granularity:
        print("\n" + "=" * 60, "\nStep 10(iii) granularity\n", "=" * 60, flush=True)
        flow_path = Path(spec["flows"])
        if not flow_path.exists():
            print(f"  directional store missing; build it first with:")
            print(f"    scripts/build_store.py --corpus {args.corpus} "
                  f"--root {spec['root']} \\")
            print(f"        --out {flow_path} --directional")
            print("  skipping the granularity ablation; it is reported as not run.")
        else:
            flows = Store(flow_path, lazy_bytes=False)
            split_file = Path(str(flow_path) + ".split.json")
            flow_manifest = (SplitManifest.load(str(split_file))
                             if split_file.exists() else make_split(flows, seed=0))
            print(split_summary(flows, flow_manifest), flush=True)
            t = time.time()
            gran = ablate_granularity(store, flows, manifest, flow_manifest, cfg,
                                      seeds=(0, 1, 2), budget=args.budget,
                                      paths=paths, verbose=True)
            print(f"\n{summarise({'variants': gran['variants']}, 'Step 10(iii) granularity')}")
            print(f"  verdict: {gran['verdict']}  (delta {gran['delta_macro_f1']:+.4f}, "
                  f"seed noise {gran['seed_noise']:.4f})")
            print(f"  took {(time.time() - t) / 60:.1f} min", flush=True)
            flows.close()

    store.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

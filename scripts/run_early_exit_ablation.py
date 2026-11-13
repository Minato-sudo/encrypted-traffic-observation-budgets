#!/usr/bin/env python3
"""Step 10(iv) on its own, with per-epoch logging.

Split out of run_study.py because this is the most expensive stage in the
study -- the adaptive encoder does no downsampling, so each run costs roughly
twenty times a 1D baseline run -- and because a silent stall in it is
otherwise invisible for hours. Every epoch prints, so progress is checkable
at a glance, and the store is loaded into RAM rather than re-read per call.
"""
from __future__ import annotations

import dataclasses
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import torch

from nettc.ablations import ablate_early_exit, summarise
from nettc.experiments import ExperimentPaths
from nettc.split import SplitManifest, cap_per_class
from nettc.store import Store
from nettc.train import TrainConfig


def main() -> int:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--store", default="stores/ustc.h5")
    ap.add_argument("--out", default="results/ustc")
    ap.add_argument("--ceiling", type=int, default=1024)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--max-per-class", type=int, default=5000)
    args = ap.parse_args()

    print(f"cuda available: {torch.cuda.is_available()}", flush=True)
    store = Store(args.store, lazy_bytes=False)
    manifest = SplitManifest.load(args.store + ".split.json")
    if args.max_per_class:
        manifest = cap_per_class(store, manifest, args.max_per_class, seed=0)

    cfg = TrainConfig(epochs=args.epochs, patience=max(4, args.epochs // 5),
                      batch_size=args.batch_size, pad_mode="embed", verbose=True)
    print(f"device {cfg.resolved_device()}  ceiling {args.ceiling}B  "
          f"seeds {args.seeds}  batch {args.batch_size}", flush=True)

    t0 = time.time()
    out = ablate_early_exit(store, manifest, cfg, seeds=tuple(args.seeds),
                            ceiling=args.ceiling,
                            paths=ExperimentPaths(Path(args.out)), verbose=True)
    print(f"\n{summarise(out, 'Step 10(iv) early-exit supervision')}")
    print(f"\ntotal {(time.time() - t0) / 60:.0f} min", flush=True)
    store.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

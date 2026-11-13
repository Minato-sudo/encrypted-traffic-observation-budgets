#!/usr/bin/env python3
"""Preprocess a corpus of captures into the HDF5 session store (Phase 2 Step 3).

    python3 scripts/build_store.py --corpus ustc --root data/USTC-TFC2016 \
        --out stores/ustc.h5 --checksums

Run this once per corpus. Every budget in the sweep is then served by slicing
the store, so preprocessing never runs again and cannot drift between budgets.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from nettc.datasets import load_corpus, manifest, REGISTRY
from nettc.dissect import STRATUM_NAMES
from nettc.encryption import CIPHER_NAMES, cipher_is_deprecated
from nettc.sessionize import sessionize
from nettc.split import SPLIT_NAMES, make_split, summarise
from nettc.store import Store


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--corpus", required=True, choices=sorted(REGISTRY))
    ap.add_argument("--root", required=True, help="directory holding the capture files")
    ap.add_argument("--out", required=True, help="path for the .h5 store")
    ap.add_argument("--split-out", default=None, help="path for the split manifest JSON")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--keep-payloadless", action="store_true",
                    help="retain sessions with no application-layer payload")
    ap.add_argument("--no-dedupe", action="store_true")
    ap.add_argument("--checksums", action="store_true",
                    help="record a SHA-256 per capture file (Step 2 verification)")
    ap.add_argument("--limit", type=int, default=0, help="process only N captures (smoke test)")
    ap.add_argument("--directional", action="store_true",
                    help="key each direction separately, giving unidirectional flows "
                         "instead of bidirectional sessions (the Step 10(iii) ablation)")
    args = ap.parse_args()

    corpus = load_corpus(args.corpus, args.root)
    captures = corpus.captures[:args.limit] if args.limit else corpus.captures
    print(corpus.summary())
    print(f"\npreprocessing {len(captures)} captures -> {args.out}")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if args.checksums:
        print("checksumming captures ...")
        sums = manifest(captures)
        Path(str(out_path) + ".checksums.json").write_text(json.dumps(sums, indent=2))
        print(f"  wrote {len(sums)} checksums")

    t0 = time.time()
    stats = sessionize(
        captures, out_path, corpus.label_names, corpus.label_of,
        drop_payloadless=not args.keep_payloadless,
        deduplicate=not args.no_dedupe,
        directional=args.directional,
        progress=lambda msg: print(f"  {msg}", flush=True),
    )
    print(f"\npreprocessed in {time.time() - t0:.1f}s")
    for key, value in stats.as_dict().items():
        if key == "stratum_counts":
            value = {STRATUM_NAMES[int(k)]: v for k, v in value.items()}
        print(f"  {key}: {value}")

    # lazy: the summary below reads only the scalar columns, and the byte
    # matrix for a multi-gigabyte corpus does not belong in RAM to print stats.
    store = Store(out_path, lazy_bytes=True)
    import numpy as np
    print(f"\nstore: {store.n} sessions, {store.n_classes} classes")
    print(f"  true_length   med {int(np.median(store.true_length))} "
          f"p90 {int(np.quantile(store.true_length, 0.9))} max {store.true_length.max()}")
    print(f"  packet_count  med {int(np.median(store.packet_count))} "
          f"p90 {int(np.quantile(store.packet_count, 0.9))} max {store.packet_count.max()}")
    for budget in (784, 1024, 1500):
        below = float((store.prefix_len < budget).mean())
        print(f"  at a {budget}-byte budget, {below:6.1%} of sessions are padded rather "
              f"than truncated")

    enc = store.stratum
    print("  encryption strata: " + ", ".join(
        f"{STRATUM_NAMES[s]} {float((enc == s).mean()):.1%}" for s in range(len(STRATUM_NAMES))))
    tagged = store.cipher_suite[store.cipher_suite > 0]
    if tagged.size:
        dep = float(np.mean([cipher_is_deprecated(int(c)) for c in tagged]))
        print(f"  TLS handshakes parsed: {tagged.size} ({dep:.1%} using a deprecated suite)")

    split = make_split(store, seed=args.seed)
    split_path = Path(args.split_out or (str(out_path) + ".split.json"))
    split.save(split_path)
    print(f"\n{summarise(store, split)}")
    print(f"\nsplit manifest -> {split_path}")
    store.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

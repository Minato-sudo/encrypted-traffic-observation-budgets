#!/usr/bin/env python3
"""Write the corpus-level descriptive statistics the report's Gap 3 needs.

These come from the session store rather than from a model run: how much of the
corpus is actually encrypted, how much of each fixed budget is padding, and how
much of the TLS that is there is obsolete. build_store.py prints them once at
preprocessing time; this writes them where make_report.py can read them, so the
report never depends on a log file that may have scrolled away.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from nettc.dissect import STRATUM_NAMES
from nettc.encryption import cipher_is_deprecated
from nettc.store import Store

# The budgets the reviewed papers actually use, plus the two our sweep adds at
# either end, so the padding fractions line up with the comparison table.
BUDGETS = (64, 128, 256, 512, 784, 1024, 1500)


def collect(store: Store) -> dict:
    enc = store.stratum
    tagged = store.cipher_suite[store.cipher_suite > 0]
    out = {
        "n_sessions": int(store.n),
        "n_classes": int(store.n_classes),
        "class_names": list(store.label_names),
        "class_counts": {name: int((store.label == i).sum())
                         for i, name in enumerate(store.label_names)},
        "true_length": {
            "median": int(np.median(store.true_length)),
            "p90": int(np.quantile(store.true_length, 0.9)),
        },
        "packet_count": {
            "median": int(np.median(store.packet_count)),
            "p90": int(np.quantile(store.packet_count, 0.9)),
        },
        "padded_fraction_at_budget": {
            f"{b}B": float((store.prefix_len < b).mean()) for b in BUDGETS
        },
        "strata": {
            STRATUM_NAMES[s]: float((enc == s).mean()) for s in range(len(STRATUM_NAMES))
        },
        "tls": {"handshakes_parsed": int(tagged.size)},
    }
    # How many capture files each class occupies. This decides what kind of
    # split that class got: several files allow a group-wise split with whole
    # files held out, a single file forces a temporal one where train and test
    # come from the same capture. Per-class scores are not comparable across
    # that line, so the report needs the counts to say which is which.
    labels, srcs = store.label, store.src_file
    out["class_capture_files"] = {
        name: int(np.unique(srcs[labels == i]).size)
        for i, name in enumerate(store.label_names)
    }
    out["single_capture_classes"] = sorted(
        n for n, c in out["class_capture_files"].items() if c == 1)
    # Captures that survived preprocessing but contributed no unique session.
    # Five-tuple sessionizing merges a conversation that the corpus ships split
    # across two files, and content dedup drops a session a second capture
    # repeats, so the number of capture groups actually available to a
    # group-wise split is smaller than the number of files on disk.
    registered = len(store.src_file_names)
    used = set(int(v) for v in np.unique(srcs))
    out["captures"] = {
        "registered": registered,
        "contributing_sessions": len(used),
        "contributed_nothing": sorted(
            name for i, name in enumerate(store.src_file_names) if i not in used),
    }
    # Every stratum except plaintext and unencrypted_binary leaves open the
    # possibility of encryption; only those two rule it out. Reporting the
    # complement is what Gap 3 asks for, so compute it here rather than leaving
    # the reader to add five percentages.
    known_clear = out["strata"].get("plaintext", 0.0) + \
        out["strata"].get("unencrypted_binary", 0.0)
    out["definitely_not_encrypted_fraction"] = float(known_clear)
    out["confirmed_tls_fraction"] = float(out["strata"].get("encrypted_tls", 0.0))
    if tagged.size:
        out["tls"]["deprecated_fraction"] = float(
            np.mean([cipher_is_deprecated(int(c)) for c in tagged]))
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--store", required=True)
    ap.add_argument("--corpus", default=None, choices=("ustc", "iscx"),
                    help="corpus kind, to check the realised classes against the "
                         "taxonomy the scheme defines")
    ap.add_argument("--split", default=None,
                    help="split manifest (default: <store>.split.json)")
    ap.add_argument("--out", required=True, help="results directory")
    args = ap.parse_args()

    store = Store(Path(args.store), lazy_bytes=True)
    stats = collect(store)

    # Which split each class got is decided per class by make_split, and it
    # changes what a per-class score means: a temporal split draws train and
    # test from the same capture, a group split holds whole captures out. The
    # report needs this to avoid comparing the two as if they were one scale.
    # ISCX's label set is derived from the captures that are present, so a
    # category absent from the download is silently absent from the class list
    # rather than reported as missing. Check the realised names against the
    # taxonomy the scheme defines, so the report can say which are missing.
    if args.corpus == "iscx":
        from nettc.datasets import ISCX_CATEGORY_PATTERNS
        expected = [f"{prefix}{cat}" for cat, _ in ISCX_CATEGORY_PATTERNS
                    for prefix in ("", "vpn_")]
        realised = set(stats["class_names"])
        stats["taxonomy"] = {
            "scheme": "category, vpn-aware",
            "expected": sorted(expected),
            "missing": sorted(c for c in expected if c not in realised),
        }

    split_path = Path(args.split or (str(args.store) + ".split.json"))
    if split_path.exists():
        strategy = json.loads(split_path.read_text()).get("strategy", {})
        stats["split_strategy"] = strategy
        kinds: dict[str, int] = {}
        for value in strategy.values():
            kind = value.split(" (")[0]
            kinds[kind] = kinds.get(kind, 0) + 1
        stats["split_strategy_counts"] = kinds
        stats["temporally_split_classes"] = sorted(
            name for name, value in strategy.items()
            if value.startswith("temporal"))
    out = Path(args.out) / "corpus_stats.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(stats, indent=2))
    print(f"wrote {out}")
    print(f"  {stats['n_sessions']} sessions, {stats['n_classes']} classes")
    print(f"  definitely not encrypted: {stats['definitely_not_encrypted_fraction']:.1%}")
    print(f"  confirmed TLS: {stats['confirmed_tls_fraction']:.1%}")
    if stats.get("taxonomy", {}).get("missing"):
        print(f"  classes absent from the download: "
              f"{', '.join(stats['taxonomy']['missing'])}")
    if "split_strategy_counts" in stats:
        print(f"  split strategy: {stats['split_strategy_counts']}")
    if "deprecated_fraction" in stats["tls"]:
        print(f"  deprecated TLS suites: {stats['tls']['deprecated_fraction']:.1%} "
              f"of {stats['tls']['handshakes_parsed']} handshakes")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

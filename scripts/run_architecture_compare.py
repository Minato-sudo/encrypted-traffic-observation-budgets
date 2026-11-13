"""Is the budget saturation a property of the data or of the convolution?

The three reproduced baselines are convolutional and date from 2017-2020,
while every study reviewed in Phase 1 is a transformer, a graph network or an
ensemble from 2025-2026. That gap matters for the study's central claim. If
accuracy stops improving at 256 bytes only because a stack of local
convolutions cannot use more, the finding is about the model. If a model that
attends globally at every layer stops at the same place, the finding is about
the traffic -- and it then applies to the architectures the reviewed papers
actually use.

This script compares two budget sweeps that differ in nothing but the
backbone: same store, same split manifest, same budgets, same seeds, same
schedule, same padding representation. It reports where each one saturates and
whether they agree, and draws both curves on one axis.

The transformer is trained from scratch. The published transformers are
pre-trained on large unlabelled corpora this study does not use, so this is a
comparison of architectures at matched supervision, not a reproduction of
their headline numbers -- which is what makes the *shape* of the two curves
comparable even though their heights are not comparable to anything published.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from nettc import plots as P
from nettc.metrics import save_json

DISPLAY = {"wang1d": "1D CNN (Wang et al. 2017)",
           "wang2d": "2D byte image (Wang et al. 2017)",
           "deeppacket": "Deep Packet (2020)",
           "transformer": "Byte transformer (this work)"}


def load(path: Path):
    return json.loads(path.read_text()) if path.exists() else None


def sweep(results: Path, model: str):
    d = load(results / f"e2_budget_sweep_{model}.json")
    if d is None:
        raise SystemExit(f"no sweep for {model!r} in {results}")
    return d


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--reference", required=True,
                    help="results dir holding the reference sweep")
    ap.add_argument("--reference-model", default="wang1d")
    ap.add_argument("--contender", required=True,
                    help="results dir holding the contender sweep")
    ap.add_argument("--contender-model", default="transformer")
    ap.add_argument("--corpus-name", default=None)
    ap.add_argument("--out", default=None,
                    help="where to write the JSON and figure (default: --reference)")
    args = ap.parse_args()

    ref_dir, con_dir = Path(args.reference), Path(args.contender)
    out = Path(args.out) if args.out else ref_dir
    ref = sweep(ref_dir, args.reference_model)
    con = sweep(con_dir, args.contender_model)
    corpus = args.corpus_name or ref_dir.name

    rows = {}
    for axis, key in (("bytes", "bytes_saturation"), ("packets", "packets_saturation")):
        a, b = ref.get(key) or {}, con.get(key) or {}
        rows[axis] = {
            args.reference_model: a,
            args.contender_model: b,
            "agree": bool(a.get("at") and a.get("at") == b.get("at")),
        }

    # Peak macro-F1 per model, for the height comparison that is *not* the
    # point but which a reader will look for anyway.
    peaks = {}
    for name, d in ((args.reference_model, ref), (args.contender_model, con)):
        cells = d.get("bytes") or {}
        if cells:
            best = max(cells.items(), key=lambda kv: kv[1]["macro_f1"]["mean"])
            peaks[name] = {"budget": best[0],
                           "macro_f1": best[1]["macro_f1"]["mean"],
                           "std": best[1]["macro_f1"]["std"]}

    result = {"corpus": corpus, "reference": args.reference_model,
              "contender": args.contender_model, "saturation": rows,
              "peak": peaks}

    figures = out / "figures"
    curves = {DISPLAY.get(args.reference_model, args.reference_model): ref["bytes"],
              DISPLAY.get(args.contender_model, args.contender_model): con["bytes"]}
    path = P.budget_profile(
        curves, figures / "architecture_budget_profile.png", axis="bytes",
        paper_budgets=(ref.get("paper_budgets") or {}).get("bytes"),
        title=f"{corpus}: budget saturation, two architecture families")
    result["figure"] = str(path)
    save_json(result, out / "architecture_comparison.json")

    print(f"{'axis':<9}{args.reference_model:>16}{args.contender_model:>16}   agree")
    for axis, cell in rows.items():
        print(f"{axis:<9}{cell[args.reference_model].get('at','n/a'):>16}"
              f"{cell[args.contender_model].get('at','n/a'):>16}   {cell['agree']}")
    print()
    for name, pk in peaks.items():
        print(f"  peak {name:<14} {pk['macro_f1']:.4f} ± {pk['std']:.4f} at {pk['budget']}")
    print(f"\nwrote {out / 'architecture_comparison.json'} and {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

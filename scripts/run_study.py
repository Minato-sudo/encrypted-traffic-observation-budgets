#!/usr/bin/env python3
"""Run the study end to end against a prepared store (Phase 2 Steps 7-12).

    python3 scripts/run_study.py --store stores/ustc.h5 --out results/ustc

Order matters. E1 runs first and carries the leakage gate; if the gate fails,
nothing else runs, because a model that can classify from fields we claimed to
anonymise makes every later number meaningless. Pass --ignore-gate only to
inspect a known-broken pipeline, never to report a result.
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import numpy as np

from nettc import ablations as A
from nettc import experiments as X
from nettc import literature as L
from nettc import plots as P
from nettc.metrics import save_json
from nettc.split import SplitManifest, TEST, TRAIN, VAL, make_split
from nettc.store import Store
from nettc.train import TrainConfig


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--store", required=True)
    ap.add_argument("--split", default=None, help="split manifest (default: <store>.split.json)")
    ap.add_argument("--out", required=True, help="directory for JSON results and figures")
    ap.add_argument("--model", default="wang1d", help="backbone for the sweep")
    ap.add_argument("--pad-mode", default="embed",
                    choices=("zero", "embed", "mask", "convmask"),
                    help="input representation for the sweep; 'zero' reproduces the "
                         "scalar byte/255 input the byte-image baselines use")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--max-per-class", type=int, default=0,
                    help="cap each class to N sessions per split, applied after "
                         "splitting. Controls runtime and the corpus's native "
                         "imbalance; changes the class priors, so it is recorded "
                         "in the run config and must be reported.")
    ap.add_argument("--lazy", action="store_true",
                    help="keep the byte matrix on disk; needed for the larger corpus")
    ap.add_argument("--stages", nargs="+",
                    default=["e1", "e2", "e3", "e4", "padding", "ablations",
                             "ports", "compare"],
                    choices=["e1", "e2", "e3", "e4", "padding", "ablations",
                             "ports", "compare"])
    ap.add_argument("--corpus-name", default=None,
                    help="corpus name for the literature comparison, e.g. USTC-TFC2016")
    ap.add_argument("--stratify-budget", type=int, default=784)
    ap.add_argument("--adaptive-ceiling", type=int, default=1024)
    ap.add_argument("--ignore-gate", action="store_true")
    ap.add_argument("--force", action="store_true",
                    help="re-run stages whose results are already on disk. By "
                         "default a stage with an existing result file is "
                         "skipped, so an interrupted run resumes by re-issuing "
                         "the same command.")
    ap.add_argument("--compare-representations", action="store_true",
                    help="also sweep with the scalar input, for the representation contrast")
    args = ap.parse_args()

    store = Store(args.store, lazy_bytes=args.lazy)
    split_path = Path(args.split or (str(args.store) + ".split.json"))
    manifest = (SplitManifest.load(split_path) if split_path.exists()
                else make_split(store, seed=0))
    if args.max_per_class:
        from nettc.split import cap_per_class, summarise as split_summary
        manifest = cap_per_class(store, manifest, args.max_per_class, seed=0)
        print(f"\ncapped to {args.max_per_class} sessions per class per split")
        print(split_summary(store, manifest))

    out = Path(args.out)
    paths = X.ExperimentPaths(out)
    figures = out / "figures"
    seeds = tuple(args.seeds)
    chance = float(np.bincount(store.label[manifest.indices(TRAIN)]).max()
                   / max(manifest.indices(TRAIN).size, 1))

    cfg = TrainConfig(epochs=args.epochs, batch_size=args.batch_size,
                      pad_mode=args.pad_mode, patience=max(4, args.epochs // 5))
    print(f"store {args.store}: {store.n} sessions, {store.n_classes} classes")
    print(f"device {cfg.resolved_device()}  seeds {seeds}  model {args.model}  "
          f"pad_mode {args.pad_mode}")

    results: dict = {}
    t0 = time.time()

    # Stage-level resume. Each stage writes one JSON when it completes, so a
    # run killed part-way loses only the stage in flight; re-issuing the same
    # command picks up from there.
    stage_outputs = {
        "e1": "e1_baseline_and_leakage",
        "e2": f"e2_budget_sweep_{args.model}",
        "e3": f"e3_stratified_{args.model}_{args.stratify_budget}B",
        "e4": "e4_adaptive_frontier",
        "padding": f"padding_controls_{args.model}_{args.adaptive_ceiling}B",
        "ports": f"port_dependence_{args.model}",
    }

    def done(stage: str) -> bool:
        name = stage_outputs.get(stage)
        if not name or args.force:
            return False
        path = paths.json(name)
        if path.exists():
            results[stage] = json.loads(path.read_text())
            print(f"  skipping {stage}: {path.name} already exists "
                  f"(--force to re-run)")
            return True
        return False

    # The budget sweep is an input to E4's frontier plot and to the literature
    # comparison, not only to its own stage. A resumed run that executes just
    # the later stages still needs it, so load it before anything runs.
    if "e2" not in results:
        prior = paths.json(f"e2_budget_sweep_{args.model}")
        if prior.exists():
            results["e2"] = json.loads(prior.read_text())
            print(f"  loaded the existing budget sweep from {prior.name}")

    if "ablations" in args.stages and not args.force:
        have = all((paths.json(n)).exists() for n in
                   ("ablation_representation", "ablation_layer_scope",
                    "ablation_early_exit"))
        if have:
            print("  skipping ablations: all three result files already exist")
            args.stages = [s_ for s_ in args.stages if s_ != "ablations"]

    if "e1" in args.stages and not done("e1"):
        print("\n================ E1: baselines and leakage gate ================")
        results["e1"] = X.run_e1(store, manifest, cfg, seeds=seeds, paths=paths)
        if not results["e1"]["leakage"]["gate_passed"] and not args.ignore_gate:
            print("\nHALTING: the leakage gate failed. Fix preprocessing before "
                  "running anything else. (--ignore-gate overrides, for diagnosis only.)")
            store.close()
            return 2

    if "e2" in args.stages and not done("e2"):
        print("\n================ E2: budget sweep ================")
        e2 = X.run_e2(store, manifest, cfg, model_name=args.model, seeds=seeds, paths=paths)
        results["e2"] = e2
        curves = {f"{args.pad_mode} input": e2["bytes"]}
        packet_curves = {f"{args.pad_mode} input": e2["packets"]}
        if args.compare_representations and args.pad_mode != "zero":
            alt = X.run_e2(store, manifest, dataclasses.replace(cfg, pad_mode="zero"),
                           model_name=args.model, seeds=seeds, paths=None)
            save_json(alt, paths.json(f"e2_budget_sweep_{args.model}_zero"))
            results["e2_zero"] = alt
            curves["scalar byte/255"] = alt["bytes"]
            packet_curves["scalar byte/255"] = alt["packets"]
        P.budget_profile(curves, figures / "budget_profile_bytes.png", axis="bytes",
                         paper_budgets=X.PAPER_BUDGETS, chance=chance,
                         saturation=e2.get("bytes_saturation"))
        P.budget_profile(packet_curves, figures / "budget_profile_packets.png",
                         axis="packets", paper_budgets=X.PAPER_BUDGETS, chance=chance,
                         saturation=e2.get("packets_saturation"))
        print(f"\n  bytes saturation:   {e2['bytes_saturation']}")
        print(f"  packets saturation: {e2['packets_saturation']}")

    if "e3" in args.stages and not done("e3"):
        print("\n================ E3: stratified evaluation ================")
        e3 = X.run_e3(store, manifest, cfg, model_name=args.model,
                      budget=args.stratify_budget, seed=seeds[0], paths=paths)
        results["e3"] = e3
        P.per_class_against_support(e3["overall"], figures / "per_class_support.png")
        for group, title in (("flow_length", "Macro-F1 by true flow length"),
                             ("class_support", "Macro-F1 by class support"),
                             ("encryption", "Macro-F1 by encryption stratum")):
            if e3["strata"].get(group):
                P.stratified_bars(e3["strata"][group], figures / f"stratified_{group}.png",
                                  title=title)

    if "padding" in args.stages and not done("padding"):
        print("\n================ padding controls ================")
        pc = X.run_padding_controls(store, manifest, cfg, model_name=args.model,
                                    budget=args.adaptive_ceiling, seeds=seeds, paths=paths)
        results["padding"] = pc
        P.accuracy_vs_padding(pc["controls"]["zero"]["accuracy_vs_padding_fraction"],
                              figures / "accuracy_vs_padding.png")
        if "mask_attribution" in pc:
            P.padding_attribution(pc["mask_attribution"],
                                  figures / "padding_attribution.png")

    if "e4" in args.stages and not done("e4"):
        print("\n================ E4: adaptive early exit ================")
        e4 = X.run_e4(store, manifest, cfg, e2=results.get("e2"), seeds=seeds,
                      ceiling=args.adaptive_ceiling, paths=paths)
        results["e4"] = e4
        if "comparison" in e4:
            c = e4["comparison"]
            # Prefer this model's own fixed-depth heads: same weights, same
            # training run. The sweep is only drawn when it is available and
            # comparable, and its absence must not take the figure down.
            fixed = [{"cost": h["cost"], "macro_f1": h["macro_f1"],
                      "label": h["label"]} for h in c.get("fixed_heads", [])]
            if not fixed and "e2" in results:
                fixed = [{"cost": float(cell["n_bytes"] or cell["width"]),
                          "macro_f1": cell["macro_f1"]["mean"], "label": cell["budget"]}
                         for cell in sorted(results["e2"]["bytes"].values(),
                                            key=lambda d: d["n_bytes"] or d["width"])]
            if fixed:
                P.accuracy_per_byte_frontier(fixed, c["adaptive_points"],
                                             figures / "accuracy_per_byte_frontier.png",
                                             frontier=c["pareto_frontier"])

    if "ablations" in args.stages:
        print("\n================ Step 10 ablations ================")
        rep = A.ablate_representation(store, manifest, cfg, seeds=seeds, paths=paths)
        scope = A.ablate_layer_scope(store, manifest, cfg, seeds=seeds, paths=paths)
        exit_ab = A.ablate_early_exit(store, manifest, cfg, seeds=seeds,
                                      ceiling=args.adaptive_ceiling, paths=paths)
        results["ablations"] = {"representation": rep, "layer_scope": scope,
                                "early_exit": exit_ab}
        for name, ab in (("Step 10(i) representation", rep),
                         ("Step 10(ii) layer scope", scope),
                         ("Step 10(iv) early-exit supervision", exit_ab)):
            print("\n" + A.summarise(ab, name))
        print("\nStep 10(iii) granularity needs a second store built with "
              "--directional; see scripts/build_store.py.")

    if "ports" in args.stages and not done("ports"):
        print("\n================ port dependence ================")
        pd = X.run_port_dependence(store, manifest, cfg, model_name=args.model,
                                   seeds=seeds, paths=paths)
        results["ports"] = pd
        print(f"\n  largest macro-F1 drop from masking ports: "
              f"{pd['summary']['largest_drop']:+.4f}")
        # The ports-only probe comes from E1, which on a resumed run is on
        # disk rather than in this run's results. The figure is still worth
        # drawing without it, so a missing E1 is not an error.
        e1_for_probe = results.get("e1")
        if e1_for_probe is None:
            prior_e1 = paths.json(stage_outputs["e1"])
            if prior_e1.exists():
                e1_for_probe = json.loads(prior_e1.read_text())
        probe = None
        if e1_for_probe:
            probe = (e1_for_probe.get("leakage", {}).get("probes", {})
                     .get("ports", {}).get("accuracy", {}).get("mean"))
        P.port_dependence(pd, figures / "port_dependence.png", probe=probe)

    if "compare" in args.stages and "e2" in results:
        print("\n================ comparison with existing research ================")
        corpus = args.corpus_name or Path(args.store).stem
        budgets = results["e2"]["bytes"]
        best = max(budgets.values(), key=lambda c: c["macro_f1"]["mean"])
        table = L.comparison_table(best, corpus, store.n_classes,
                                   our_label=f"this work ({args.model})")
        print("\n" + table)
        (out / "comparison_with_literature.txt").write_text(table)
        results["comparison"] = {"corpus": corpus, "our_best_budget": best["budget"],
                                 "our_macro_f1": best["macro_f1"]["mean"]}

    save_json({"store": str(args.store), "model": args.model, "pad_mode": args.pad_mode,
               "seeds": list(seeds), "seconds": time.time() - t0,
               "stages": args.stages, "epochs": args.epochs,
               "max_per_class": args.max_per_class,
               "corpus_name": args.corpus_name,
               "n_train": int(manifest.indices(TRAIN).size),
               "n_test": int(manifest.indices(TEST).size)},
              paths.json("run_config"))
    print(f"\ndone in {time.time() - t0:.0f}s. JSON in {out}, figures in {figures}")
    store.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

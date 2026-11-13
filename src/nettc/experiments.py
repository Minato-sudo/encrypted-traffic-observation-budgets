"""The four experiments of Phase 2 Step 7, plus the Step 11 controls.

E1  reproduce the byte-image baseline and run the leakage control.
E2  sweep byte budgets and packet budgets, architecture and training fixed.
E3  stratify every test result by flow length, class support and encryption.
E4  evaluate the adaptive early-exit policy against the best fixed budget.

The leakage control is a gate, not a diagnostic. It trains the same
architecture on nothing but the MAC and IP bytes that anonymisation was
supposed to destroy. If that model classifies appreciably better than
chance, anonymisation did not work and no downstream number means anything,
so :func:`run_e1` reports ``gate_passed=False`` and the caller is expected to
stop. Publishing the probe result next to the headline metrics is the
reporting standard the review proposed in Gap 4, where no reviewed paper
offered a positive control.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Sequence

import numpy as np

from .metrics import (compare_budgets, evaluate, length_bins, mcnemar,
                      pareto_frontier, per_class_table, save_json, seed_summary,
                      stratified, support_bins)
from .slicer import (BudgetSpec, length_balance_plan,
                     length_balanced_indices, resolve_packet_width)
from .split import SplitManifest, TEST, TRAIN, VAL
from .models import DEFAULT_EXITS
from .train import (TrainConfig, SlicedView, early_exit_sweep, evaluate_model,
                    predict, train_model)
from .dissect import STRATUM_NAMES

# The budgets swept, chosen in Step 7 to coincide with the choices made by
# the reviewed papers so their decisions can be located on our curve.
BYTE_BUDGETS: tuple[int, ...] = (64, 128, 256, 512, 784, 1024, 1500)
PACKET_BUDGETS: tuple[int, ...] = (1, 2, 3, 5, 8, 15, 16, 20)

# Where each reviewed paper sits, for annotating the budget profile.
PAPER_BUDGETS = {
    "bytes": {784: "P10 TransECA-Net", 1024: "P8 MFF"},
    "packets": {5: "P3 OHCE / P6 TrafficFormer / P7 MLETC", 15: "P4 MH-Net",
                16: "P2 CLET", 20: "P1 S2F-GNN"},
}

DEFAULT_SEEDS = (0, 1, 2)
EXIT_THRESHOLDS = (0.0, 0.3, 0.5, 0.7, 0.8, 0.9, 0.95, 0.99, 1.01)


@dataclass
class ExperimentPaths:
    root: Path

    def __post_init__(self):
        self.root = Path(self.root)
        self.root.mkdir(parents=True, exist_ok=True)

    def json(self, name: str) -> Path:
        return self.root / f"{name}.json"


def _chance_accuracy(store, idx: np.ndarray) -> float:
    """Majority-class rate: the bar the leakage probe must fail to clear."""
    counts = np.bincount(store.label[idx], minlength=store.n_classes)
    return float(counts.max() / max(counts.sum(), 1))


# ------------------------------------------------------------------------ E1

def run_e1(store, manifest: SplitManifest, cfg: TrainConfig,
           seeds: Sequence[int] = DEFAULT_SEEDS,
           leakage_margin: float = 0.10,
           paths: Optional[ExperimentPaths] = None, verbose: bool = True) -> dict:
    """Baseline reproduction plus the leakage gate.

    ``leakage_margin`` is how far above the majority-class rate the probe may
    score before we call anonymisation failed. It is deliberately tight: the
    identifier fields in these corpora are near-perfectly predictive when
    present, so a working anonymisation should leave the probe at chance, not
    merely below the baseline.
    """
    tr, va, te = manifest.indices(TRAIN), manifest.indices(VAL), manifest.indices(TEST)
    out: dict = {"baselines": {}, "leakage": {}}
    say = print if verbose else (lambda *_a, **_k: None)

    # --- baselines at their published budgets
    for name, spec in (("wang2d", BudgetSpec(n_bytes=784)),
                       ("wang1d", BudgetSpec(n_bytes=784)),
                       ("deeppacket", BudgetSpec(n_bytes=1500))):
        per_seed = []
        for seed in seeds:
            model, hist, _tr, _va = train_model(store, tr, va, spec, name, cfg, seed=seed)
            res, pred, _te = evaluate_model(model, store, te, spec, cfg)
            per_seed.append({"seed": seed, "result": res.to_dict(),
                             "history": hist.__dict__})
            say(f"  {name:<11} {spec.label:>6} seed {seed}  {res.summary_line()}")
        out["baselines"][name] = {
            "budget": spec.label,
            "accuracy": seed_summary([s["result"]["accuracy"] for s in per_seed]),
            "macro_f1": seed_summary([s["result"]["macro_f1"] for s in per_seed]),
            "weighted_f1": seed_summary([s["result"]["weighted_f1"] for s in per_seed]),
            "per_seed": per_seed,
        }

    # --- leakage controls (Step 11i)
    #
    # Three probes, one of which is the gate. All train the same architecture
    # on a restricted view of the input, so a score is directly comparable to
    # the majority-class rate.
    chance = _chance_accuracy(store, te)
    probe_spec = BudgetSpec(n_bytes=48)   # width comes from the source, not the spec
    probes: dict[str, dict] = {}

    for source, role in (("id_positions", "gate"),
                         ("ident", "shortcut_strength"),
                         ("ports", "retained_field_risk")):
        accs, macros = [], []
        for seed in seeds:
            model, _h, _tr, _va = train_model(store, tr, va, probe_spec, "wang1d", cfg,
                                              seed=seed, source=source)
            res, _pred, _te = evaluate_model(model, store, te, probe_spec, cfg,
                                             source=source)
            accs.append(res.accuracy)
            macros.append(res.macro_f1)
            say(f"  probe[{source:<13}] seed {seed}  acc {res.accuracy:.4f} "
                f"macroF1 {res.macro_f1:.4f}  (majority-class rate {chance:.4f})")
        summary = seed_summary(accs)
        probes[source] = {
            "role": role,
            "accuracy": summary,
            "macro_f1": seed_summary(macros),
            "excess_over_chance": float(summary["mean"] - chance),
        }

    gate = probes["id_positions"]
    out["leakage"] = {
        "majority_class_rate": chance,
        "margin": leakage_margin,
        "probes": probes,
        "gate_passed": bool(gate["excess_over_chance"] <= leakage_margin),
        "notes": {
            "id_positions": "MAC and IP byte positions read from the stored, "
                            "anonymised prefix. This is the halt condition: it "
                            "must not beat the majority class.",
            "ident": "the same positions before anonymisation. Reports how much "
                     "identifier signal anonymisation destroyed; high is expected "
                     "and is not a failure.",
            "ports": "the retained port bytes. Not a halt condition, but a high "
                     "score is a leakage finding that must be reported beside "
                     "the headline metrics.",
        },
    }
    if not out["leakage"]["gate_passed"]:
        say(f"\n  GATE FAILED: a model trained on the anonymised identifier "
            f"positions reaches {gate['accuracy']['mean']:.4f} against a "
            f"{chance:.4f} majority-class rate.\n"
            f"  The zeroing did not land where it was supposed to. Downstream "
            f"results are not trustworthy; fix preprocessing before proceeding.")
    else:
        say(f"\n  gate passed: anonymised identifier positions give "
            f"{gate['accuracy']['mean']:.4f} vs chance {chance:.4f} "
            f"(excess {gate['excess_over_chance']:+.4f} <= {leakage_margin})")
        say(f"  for context, the same positions before anonymisation give "
            f"{probes['ident']['accuracy']['mean']:.4f}, and the retained port "
            f"bytes alone give {probes['ports']['accuracy']['mean']:.4f}")

    if paths:
        save_json(out, paths.json("e1_baseline_and_leakage"))
    return out


# ------------------------------------------------------------------------ E2

def run_e2(store, manifest: SplitManifest, cfg: TrainConfig,
           model_name: str = "wang1d",
           byte_budgets: Sequence[int] = BYTE_BUDGETS,
           packet_budgets: Sequence[int] = PACKET_BUDGETS,
           seeds: Sequence[int] = DEFAULT_SEEDS,
           paths: Optional[ExperimentPaths] = None, verbose: bool = True) -> dict:
    """The budget sweep: the study's primary result.

    Only the budget changes. Same architecture, same optimiser, same schedule,
    same seeds, same split. Both axes are swept because the literature
    disagrees on both: by packet count (5 to 20) and by byte count (784 to
    1024).
    """
    tr, va, te = manifest.indices(TRAIN), manifest.indices(VAL), manifest.indices(TEST)
    say = print if verbose else (lambda *_a, **_k: None)
    out: dict = {"model": model_name, "bytes": {}, "packets": {},
                 "paper_budgets": PAPER_BUDGETS}

    for axis, budgets in (("bytes", byte_budgets), ("packets", packet_budgets)):
        say(f"\n  -- {axis} axis --")
        for b in budgets:
            spec = (BudgetSpec(n_bytes=b) if axis == "bytes"
                    else resolve_packet_width(store, BudgetSpec(n_packets=b), tr))
            per_seed, preds_by_seed = [], {}
            for seed in seeds:
                model, hist, tr_view, _va = train_model(store, tr, va, spec, model_name,
                                                        cfg, seed=seed)
                res, pred, te_view = evaluate_model(model, store, te, spec, cfg)
                per_seed.append({"seed": seed, "result": res.to_dict(),
                                 "epochs_run": hist.epochs_run,
                                 "seconds": hist.seconds, "params": hist.params})
                preds_by_seed[seed] = pred
                say(f"  {spec.label:>6} (w={spec.tensor_width():>5}) seed {seed}  "
                    f"{res.summary_line()}")
            cell = {
                "budget": spec.label,
                "width": spec.tensor_width(),
                "n_packets": spec.n_packets,
                "n_bytes": spec.n_bytes,
                "accuracy": seed_summary([s["result"]["accuracy"] for s in per_seed]),
                "macro_f1": seed_summary([s["result"]["macro_f1"] for s in per_seed]),
                "weighted_f1": seed_summary([s["result"]["weighted_f1"] for s in per_seed]),
                "concealment_gap": seed_summary(
                    [s["result"]["concealment_gap"] for s in per_seed]),
                "mean_padding_fraction": per_seed[0]["result"]["meta"]["mean_padding_fraction"],
                "per_seed": per_seed,
                "predictions": {str(k): v.tolist() for k, v in preds_by_seed.items()},
            }
            out[axis][spec.label] = cell

    # Where does accuracy stop improving?
    #
    # Not "the first step that is inside seed noise" -- that was the first
    # version and it is wrong, because a curve can plateau and then rise
    # again. On synthetic data with a cue planted past byte 800 it reported
    # saturation at 256 bytes on a curve that went on to reach 1.000. The
    # honest definition is the smallest budget that *no larger* budget
    # improves on by more than seed noise.
    for axis in ("bytes", "packets"):
        cells = list(out[axis].values())
        saturation = None
        for i, cand in enumerate(cells):
            later = cells[i + 1:]
            if not later:
                continue
            gains = [compare_budgets(c, cand, "macro_f1") for c in later]
            if all(g["verdict"] != "a_better" for g in gains):
                best_later = max(later, key=lambda c: c["macro_f1"]["mean"])
                saturation = {
                    "at": cand["budget"],
                    "macro_f1": cand["macro_f1"]["mean"],
                    "best_larger_budget": best_later["budget"],
                    "best_larger_macro_f1": best_later["macro_f1"]["mean"],
                    "largest_remaining_gain": max(g["diff"] for g in gains),
                    "seed_noise": max(g["seed_noise"] for g in gains),
                }
                break
        out[axis + "_saturation"] = saturation
        if saturation is None and cells:
            out[axis + "_saturation"] = {
                "at": None,
                "note": "accuracy still rising at the largest budget swept; "
                        "the curve has not saturated inside this range",
            }

    if paths:
        save_json(out, paths.json(f"e2_budget_sweep_{model_name}"))
    return out


# ------------------------------------------------------------------------ E3

def run_e3(store, manifest: SplitManifest, cfg: TrainConfig,
           model_name: str = "wang1d", budget: int = 784, seed: int = 0,
           paths: Optional[ExperimentPaths] = None, verbose: bool = True) -> dict:
    """Stratified evaluation: flow length, class support, encryption status.

    This is where aggregate reporting is taken apart. Gap 2 of the review is
    that minority and short-lived classes fail while the headline metric
    conceals it; Gap 3 is that these corpora are largely unencrypted, so one
    averaged number does not describe encrypted traffic.
    """
    tr, va, te = manifest.indices(TRAIN), manifest.indices(VAL), manifest.indices(TEST)
    say = print if verbose else (lambda *_a, **_k: None)
    spec = BudgetSpec(n_bytes=budget)

    model, _hist, _tr, _va = train_model(store, tr, va, spec, model_name, cfg, seed=seed)
    res, pred, te_view = evaluate_model(model, store, te, spec, cfg)
    y = te_view.y

    out: dict = {"model": model_name, "budget": spec.label, "seed": seed,
                 "overall": res.to_dict(), "strata": {}}
    say(f"\n  overall  {res.summary_line()}")
    say("\n" + per_class_table(res))

    lb, lnames = length_bins(store.true_length[te])
    out["strata"]["flow_length"] = {
        k: v.to_dict() for k, v in
        stratified(y, pred, lb, lnames, store.label_names).items()}

    sb, snames = support_bins(store)
    out["strata"]["class_support"] = {
        k: v.to_dict() for k, v in
        stratified(y, pred, sb[te], snames, store.label_names).items()}

    out["strata"]["encryption"] = {
        k: v.to_dict() for k, v in
        stratified(y, pred, store.stratum[te], list(STRATUM_NAMES),
                   store.label_names).items()}

    for group in ("flow_length", "class_support", "encryption"):
        say(f"\n  -- by {group} --")
        for name, cell in out["strata"][group].items():
            flag = "  (underpowered)" if cell["meta"].get("underpowered") else ""
            say(f"  {name:<16} n={cell['n']:>6}  acc {cell['accuracy']:.4f}  "
                f"macroF1 {cell['macro_f1']:.4f}{flag}")

    # Classes that collapse entirely are the review's central complaint; name them.
    out["failed_classes"] = [c["name"] for c in res.per_class
                             if c["support"] > 0 and c["f1"] < 0.05]
    if out["failed_classes"]:
        say(f"\n  classes at F1 < 0.05: {', '.join(out['failed_classes'])}")

    if paths:
        save_json(out, paths.json(f"e3_stratified_{model_name}_{budget}B"))
    return out


# ------------------------------------------------------------------------ E4

def run_e4(store, manifest: SplitManifest, cfg: TrainConfig,
           e2: Optional[dict] = None, seeds: Sequence[int] = DEFAULT_SEEDS,
           thresholds: Sequence[float] = EXIT_THRESHOLDS,
           ceiling: int = 1024,
           paths: Optional[ExperimentPaths] = None, verbose: bool = True) -> dict:
    """Adaptive early-exit policy against the best fixed budget.

    Step 12 says plainly that if the adaptive policy fails to beat the best
    fixed budget, that is reported as a negative result and the budget
    profile stands on its own. So this returns the comparison either way and
    makes no attempt to tune the policy into a win.
    """
    tr, va, te = manifest.indices(TRAIN), manifest.indices(VAL), manifest.indices(TEST)
    say = print if verbose else (lambda *_a, **_k: None)
    spec = BudgetSpec(n_bytes=ceiling)
    # Build the model with only the exits the ceiling can reach. Otherwise a
    # head deeper than the input exists but never produces logits, and any
    # code indexing heads by position walks off the end of the list.
    exits = tuple(e for e in DEFAULT_EXITS if e <= ceiling)
    if not exits:
        raise ValueError(f"a {ceiling}-byte ceiling reaches no exit; the shallowest "
                         f"is {min(DEFAULT_EXITS)} bytes")
    out: dict = {"ceiling": ceiling, "exits": list(exits), "per_seed": [], "frontier": []}

    rows_by_seed = []
    for seed in seeds:
        model, hist, _tr, _va = train_model(store, tr, va, spec, "adaptive", cfg, seed=seed,
                                            model_kwargs={"exits": exits})
        rows = early_exit_sweep(model, store, te, spec, cfg, thresholds)
        rows_by_seed.append(rows)
        # Each head on its own, as the fixed-budget control for its own depth.
        heads = []
        for k, exit_bytes in enumerate(model.exits):
            res, _pred, _te = evaluate_model(model, store, te, spec, cfg, head=k)
            heads.append({"exit_bytes": int(exit_bytes), "result": res.to_dict()})
            say(f"  seed {seed} head@{exit_bytes:>5}B  {res.summary_line()}")
        out["per_seed"].append({"seed": seed, "thresholds": rows, "heads": heads,
                                "params": hist.params, "seconds": hist.seconds})
        for r in rows:
            say(f"  seed {seed} tau={r['threshold']:.2f}  acc {r['accuracy']:.4f}  "
                f"mean bytes {r['mean_bytes_read']:.1f}")

    # Average the policy across seeds at each threshold.
    for i, tau in enumerate(thresholds):
        out["frontier"].append({
            "threshold": float(tau),
            "accuracy": seed_summary([rs[i]["accuracy"] for rs in rows_by_seed]),
            "macro_f1": seed_summary([rs[i]["macro_f1"] for rs in rows_by_seed]),
            "mean_bytes_read": seed_summary([rs[i]["mean_bytes_read"] for rs in rows_by_seed]),
        })

    # --- the fair comparison: the policy against this model's own fixed heads
    #
    # Step 6(iv) asks for fixed-budget variants of our own encoder "because
    # only the budget changes". The strictest version of that is the adaptive
    # model's own heads: identical weights, identical training, the single
    # difference being whether the exit policy runs. Comparing instead
    # against the E2 sweep would change the ceiling, the training run and
    # possibly the input representation all at once, and the first version of
    # this code did exactly that -- it called a policy that matched full-depth
    # accuracy at a third of the bytes a failure, because the sweep's best
    # budget was larger than the adaptive model's ceiling.
    head_points = []
    for k, exit_bytes in enumerate(out["per_seed"][0]["heads"]):
        macro = [ps["heads"][k]["result"]["macro_f1"] for ps in out["per_seed"]]
        cost = float(exit_bytes["exit_bytes"])
        head_points.append({"label": f"head@{int(cost)}B", "cost": cost,
                            **{"macro_f1": float(np.mean(macro)),
                               "std": float(np.std(macro, ddof=1) if len(macro) > 1 else 0.0)}})

    adaptive = [{"label": f"tau={r['threshold']:.2f}",
                 "cost": r["mean_bytes_read"]["mean"],
                 "macro_f1": r["macro_f1"]["mean"], "std": r["macro_f1"]["std"]}
                for r in out["frontier"]]

    best_head = max(head_points, key=lambda d: d["macro_f1"])
    tolerance = max(best_head["std"], 1e-9)
    dominating = [a for a in adaptive
                  if a["macro_f1"] >= best_head["macro_f1"] - tolerance
                  and a["cost"] < best_head["cost"]]
    combined = head_points + adaptive
    front_idx = pareto_frontier([(d["cost"], d["macro_f1"]) for d in combined])
    cheapest = min(dominating, key=lambda d: d["cost"]) if dominating else None

    out["comparison"] = {
        "basis": "the adaptive model's own fixed-depth heads: same weights, same "
                 "training run, the only difference is whether the exit policy runs",
        "fixed_heads": head_points,
        "best_fixed_head": best_head,
        "adaptive_points": adaptive,
        "adaptive_dominates": bool(dominating),
        "dominating_points": dominating,
        "pareto_frontier": [combined[i] for i in sorted(front_idx,
                                                        key=lambda i: combined[i]["cost"])],
        "byte_saving": (None if cheapest is None else
                        {"at_threshold": cheapest["label"],
                         "macro_f1": cheapest["macro_f1"],
                         "mean_bytes_read": cheapest["cost"],
                         "against_fixed_bytes": best_head["cost"],
                         "fraction_of_fixed": cheapest["cost"] / max(best_head["cost"], 1e-9)}),
        "verdict": (
            f"the adaptive policy matches the best fixed head's macro-F1 "
            f"({best_head['macro_f1']:.4f} at {best_head['cost']:.0f} bytes) while reading "
            f"{cheapest['cost']:.0f} bytes on average, "
            f"{cheapest['cost'] / best_head['cost']:.0%} of the fixed cost"
            if cheapest else
            "the adaptive policy does not beat the best fixed budget "
            "(reported as a negative result, per Step 12)"),
    }
    say(f"\n  {out['comparison']['verdict']}")

    # The E2 sweep is kept only as context, and only when it is actually
    # comparable: a different ceiling or representation makes it a different
    # experiment, not a baseline.
    if e2 is not None:
        comparable = all(c["n_bytes"] is None or c["n_bytes"] <= ceiling
                         for c in e2["bytes"].values())
        sweep = [{"label": c["budget"], "cost": float(c["n_bytes"] or c["width"]),
                  "macro_f1": c["macro_f1"]["mean"], "std": c["macro_f1"]["std"]}
                 for c in e2["bytes"].values()]
        out["sweep_context"] = {
            "points": sweep,
            "within_adaptive_ceiling": bool(comparable),
            "caveat": ("budgets larger than the adaptive model's "
                       f"{ceiling}-byte ceiling are not a like-for-like comparison and "
                       "are reported here only as context"),
        }

    if paths:
        save_json(out, paths.json("e4_adaptive_frontier"))
    return out


# ------------------------------------------------------- port dependence

def run_port_dependence(store, manifest: SplitManifest, cfg: TrainConfig,
                        model_name: str = "wang1d",
                        budgets: Sequence[int] = (64, 128, 256, 784, 1500),
                        seeds: Sequence[int] = DEFAULT_SEEDS,
                        paths: Optional[ExperimentPaths] = None,
                        verbose: bool = True) -> dict:
    """How much of the sweep is port lookup rather than traffic content?

    Phase 2 Step 3c retained ports and deferred the consequence to Step 11.
    The Step 11 probe answers it bluntly on USTC-TFC2016: the four port bytes
    alone reach 0.751 accuracy against a 0.136 majority-class rate. That is
    three quarters of the gain over chance that a model reading 784 bytes
    achieves, from four bytes that a port-based classifier of the 1990s would
    also have used.

    This re-runs the budget sweep with every copy of those four bytes zeroed,
    in every packet. The difference is the part of the curve that was port
    lookup. Reporting the masked curve beside the unmasked one is the only way
    the budget question can be asked about traffic content at all.
    """
    tr, va, te = manifest.indices(TRAIN), manifest.indices(VAL), manifest.indices(TEST)
    say = print if verbose else (lambda *_a, **_k: None)
    out: dict = {"model": model_name, "budgets": {}, "chance": _chance_accuracy(store, te)}

    for b in budgets:
        spec = BudgetSpec(n_bytes=b)
        cell: dict = {}
        for source, label in (("bytes", "with ports"), ("no_ports", "ports masked")):
            accs, macros = [], []
            for seed in seeds:
                model, _h, _t, _v = train_model(store, tr, va, spec, model_name, cfg,
                                                seed=seed, source=source)
                res, _p, _te = evaluate_model(model, store, te, spec, cfg, source=source)
                accs.append(res.accuracy)
                macros.append(res.macro_f1)
                say(f"  {b:>5}B {label:<14} seed {seed}  {res.summary_line()}")
            cell[source] = {"accuracy": seed_summary(accs), "macro_f1": seed_summary(macros)}
        drop = cell["bytes"]["macro_f1"]["mean"] - cell["no_ports"]["macro_f1"]["mean"]
        noise = max(cell["bytes"]["macro_f1"]["std"], cell["no_ports"]["macro_f1"]["std"])
        cell["macro_f1_lost_to_port_masking"] = float(drop)
        cell["seed_noise"] = float(noise)
        cell["conclusive"] = bool(abs(drop) > noise)
        out["budgets"][f"{b}B"] = cell
        say(f"  {b:>5}B  macro-F1 falls {drop:+.4f} when ports are masked "
            f"(seed noise {noise:.4f})")

    worst = max(out["budgets"].values(), key=lambda c: c["macro_f1_lost_to_port_masking"])
    out["summary"] = {
        "largest_drop": worst["macro_f1_lost_to_port_masking"],
        "interpretation": (
            "the difference between the two curves is the part of the budget "
            "profile that was port lookup rather than traffic content; the "
            "masked curve is the one that answers the research question"),
    }
    if paths:
        save_json(out, paths.json(f"port_dependence_{model_name}"))
    return out


# ------------------------------------------------- padding controls (Step 11ii)

def run_padding_controls(store, manifest: SplitManifest, cfg: TrainConfig,
                         model_name: str = "wang1d", budget: int = 1024,
                         seeds: Sequence[int] = DEFAULT_SEEDS,
                         paths: Optional[ExperimentPaths] = None,
                         verbose: bool = True) -> dict:
    """Is the budget curve measuring traffic, or measuring padding?

    The central methodological risk of the study. At a 1024-byte budget most
    sessions in these corpora are padded rather than truncated, so three
    mitigations are run against the default zero padding: a dedicated PAD
    symbol the model embeds separately, a mask that excludes padded positions
    from pooling, and a length-balanced resample. Accuracy is also reported
    against padding fraction directly.
    """
    import dataclasses
    tr, va, te = manifest.indices(TRAIN), manifest.indices(VAL), manifest.indices(TEST)
    say = print if verbose else (lambda *_a, **_k: None)
    spec = BudgetSpec(n_bytes=budget)
    out: dict = {"budget": spec.label, "model": model_name, "controls": {}}

    # "embed" is not a mitigation; it is the attribution control. Without it,
    # a gain from "mask" cannot be separated from the gain of simply replacing
    # a scalar input with a learned embedding.
    variants = [("zero", tr, va, te), ("embed", tr, va, te),
                ("mask", tr, va, te), ("convmask", tr, va, te)]

    # Derive the bins and the retained class set from the training split and
    # reuse them for validation and test. Balancing each split independently
    # gives them different label sets, and the resulting train/test mismatch
    # looks exactly like the dramatic collapse this control exists to detect.
    edges, keep = length_balance_plan(store, tr, n_bins=8)
    bal_report: dict = {}
    bal_tr = length_balanced_indices(store, tr, seed=0, edges=edges, classes=keep,
                                     report=bal_report)
    bal_va = length_balanced_indices(store, va, seed=0, edges=edges, classes=keep)
    bal_te = length_balanced_indices(store, te, seed=0, edges=edges, classes=keep)
    out["length_balance"] = dict(bal_report)
    out["length_balance"].update({
        "n_train": int(bal_tr.size), "n_val": int(bal_va.size),
        "n_test": int(bal_te.size),
        "chance_within_balanced_set": (1.0 / len(keep)) if keep else None,
    })
    if bal_tr.size and bal_te.size and bal_va.size:
        variants.append(("length_balanced", bal_tr, bal_va, bal_te))
        say(f"  length balance: {len(keep)} of {store.n_classes} classes retained, "
            f"{bal_tr.size} train / {bal_te.size} test; chance within that set is "
            f"{1.0/len(keep):.4f}")
    else:
        say("  length-balanced control could not be formed on this split; "
            "reported as not run rather than silently omitted")

    for pad_mode, t_idx, v_idx, e_idx in variants:
        mode = "zero" if pad_mode == "length_balanced" else pad_mode
        vcfg = dataclasses.replace(cfg, pad_mode=mode)
        per_seed, bands = [], None
        for seed in seeds:
            model, _h, _tr, _va = train_model(store, t_idx, v_idx, spec, model_name,
                                              vcfg, seed=seed)
            res, pred, te_view = evaluate_model(model, store, e_idx, spec, vcfg)
            per_seed.append(res.to_dict())
            if seed == seeds[0]:
                bands = _accuracy_vs_padding(te_view, pred)
            say(f"  {pad_mode:<16} seed {seed}  {res.summary_line()}")
        out["controls"][pad_mode] = {
            "n_train": int(t_idx.size), "n_test": int(e_idx.size),
            "accuracy": seed_summary([r["accuracy"] for r in per_seed]),
            "macro_f1": seed_summary([r["macro_f1"] for r in per_seed]),
            "accuracy_vs_padding_fraction": bands,
        }

    base = out["controls"]["zero"]["macro_f1"]["mean"]
    out["deltas_vs_zero_padding"] = {
        k: round(v["macro_f1"]["mean"] - base, 5)
        for k, v in out["controls"].items() if k != "zero"}
    if "length_balanced" in out["controls"]:
        out["deltas_vs_zero_padding"]["length_balanced_caveat"] = (
            "scored on a subset of classes with equalised length distributions, "
            "so its macro-F1 is not on the same scale as the others; compare it "
            "to chance within that subset, not to the zero-padding baseline")
    say(f"\n  macro-F1 change against zero padding: {out['deltas_vs_zero_padding']}")

    # Split the "mask" gain into the part that is the representation and the
    # part that is genuinely about padding.
    if "embed" in out["controls"] and "mask" in out["controls"]:
        embed_m = out["controls"]["embed"]["macro_f1"]["mean"]
        mask_m = out["controls"]["mask"]["macro_f1"]["mean"]
        noise = max(out["controls"]["zero"]["macro_f1"]["std"],
                    out["controls"]["mask"]["macro_f1"]["std"])
        out["mask_attribution"] = {
            "representation_effect": round(embed_m - base, 5),
            "padding_symbol_effect": round(mask_m - embed_m, 5),
            "seed_noise": round(float(noise), 5),
            "padding_symbol_conclusive": bool(abs(mask_m - embed_m) > noise),
        }
        say(f"  of which representation {out['mask_attribution']['representation_effect']:+.4f} "
            f"and padding symbol {out['mask_attribution']['padding_symbol_effect']:+.4f} "
            f"(seed noise {noise:.4f})")

    if paths:
        save_json(out, paths.json(f"padding_controls_{model_name}_{budget}B"))
    return out


def _accuracy_vs_padding(view: SlicedView, pred: np.ndarray,
                         n_bands: int = 5) -> list[dict]:
    """Accuracy as a function of how much of the input was padding."""
    pf = view.padding_fraction
    edges = np.linspace(0.0, 1.0, n_bands + 1)
    rows = []
    correct = pred == view.y
    for i in range(n_bands):
        lo, hi = edges[i], edges[i + 1]
        m = (pf >= lo) & (pf < hi if i < n_bands - 1 else pf <= hi)
        if not m.any():
            continue
        rows.append({"padding_fraction": f"[{lo:.1f},{hi:.1f}]", "n": int(m.sum()),
                     "accuracy": float(correct[m].mean())})
    return rows

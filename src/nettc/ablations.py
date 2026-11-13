"""The ablations of Phase 2 Step 10, plus the comparison of Step 12.

Four ablations, each removing or replacing one component while holding
everything else fixed:

(i)   representation  -- the 2D byte-image form against the 1D sequence, at a
                         matched 784-byte budget, which is the only budget at
                         which the 28x28 reshape is defined;
(ii)  layer scope     -- all layers against headers only and payload only;
(iii) granularity     -- bidirectional session against unidirectional flow,
                         which needs a second store and so is driven from
                         scripts rather than from here;
(iv)  early exit      -- the confidence policy against its fixed-depth
                         control, and auxiliary-head supervision against a
                         single terminal head.

Each returns the same shape of record: a dict of variant name to seed-summary,
plus deltas against the named reference and a verdict that refuses to call a
difference real when it is inside seed noise.
"""

from __future__ import annotations

import dataclasses
from typing import Optional, Sequence

import numpy as np

from .metrics import save_json, seed_summary
from .models import DEFAULT_EXITS
from .slicer import BudgetSpec
from .split import TEST, TRAIN, VAL
from .train import TrainConfig, evaluate_model, train_model


def _run_variants(store, manifest, cfg: TrainConfig, variants: Sequence[dict],
                  seeds: Sequence[int], reference: str, say) -> dict:
    """Train and evaluate each variant, then difference them against a reference."""
    tr, va, te = manifest.indices(TRAIN), manifest.indices(VAL), manifest.indices(TEST)
    out: dict = {"reference": reference, "variants": {}}

    for v in variants:
        vcfg = dataclasses.replace(cfg, **v.get("config", {}))
        accs, macros = [], []
        for seed in seeds:
            model, hist, _tr, _va = train_model(
                store, tr, va, v["spec"], v["model"], vcfg, seed=seed,
                model_kwargs=v.get("model_kwargs"), source=v.get("source", "bytes"))
            res, _pred, _te = evaluate_model(model, store, te, v["spec"], vcfg,
                                             source=v.get("source", "bytes"))
            accs.append(res.accuracy)
            macros.append(res.macro_f1)
            say(f"  {v['name']:<26} seed {seed}  {res.summary_line()}")
        out["variants"][v["name"]] = {
            "accuracy": seed_summary(accs),
            "macro_f1": seed_summary(macros),
            "params": hist.params,
            "description": v.get("description", ""),
        }

    base = out["variants"].get(reference)
    if base:
        for name, cell in out["variants"].items():
            if name == reference:
                continue
            diff = cell["macro_f1"]["mean"] - base["macro_f1"]["mean"]
            noise = max(cell["macro_f1"]["std"], base["macro_f1"]["std"])
            cell["delta_macro_f1"] = float(diff)
            cell["seed_noise"] = float(noise)
            cell["verdict"] = ("inconclusive" if abs(diff) <= noise else
                               ("better" if diff > 0 else "worse"))
    return out


def ablate_representation(store, manifest, cfg: TrainConfig,
                          seeds: Sequence[int] = (0, 1, 2), budget: int = 784,
                          paths=None, verbose: bool = True) -> dict:
    """Step 10(i): the 2D byte image against the 1D sequence, matched budget.

    784 bytes because that is the only budget where the 28x28 reshape of the
    Wang byte-image baseline is defined. Both forms see exactly the same
    bytes, so any difference is the geometry imposed on them, not the amount
    of traffic observed.
    """
    say = print if verbose else (lambda *_a, **_k: None)
    spec = BudgetSpec(n_bytes=budget)
    variants = [
        {"name": "1D sequence", "model": "wang1d", "spec": spec,
         "description": "the prefix as a byte sequence; the primary representation"},
        {"name": "2D byte image 28x28", "model": "wang2d", "spec": spec,
         "description": "Wang et al. ICOIN 2017: the same 784 bytes reshaped to a square"},
        {"name": "1D Deep Packet", "model": "deeppacket", "spec": spec,
         "description": "Lotfollahi et al. 2020, at the same budget"},
    ]
    say(f"\n  -- Step 10(i) representation, matched at {budget} bytes --")
    out = _run_variants(store, manifest, cfg, variants, seeds, "1D sequence", say)
    out["budget"] = spec.label
    if paths:
        save_json(out, paths.json("ablation_representation"))
    return out


def ablate_layer_scope(store, manifest, cfg: TrainConfig,
                       seeds: Sequence[int] = (0, 1, 2), budget: int = 784,
                       paths=None, verbose: bool = True) -> dict:
    """Step 10(ii): all layers against headers only and payload only.

    P2 reports its best configuration uses zero payload bytes, and P5 finds
    encrypted payload alone supports roughly 0.12-0.30 accuracy. If headers
    alone match the full prefix here, the budget being swept is in effect a
    budget over headers, which changes how the whole curve should be read.
    """
    say = print if verbose else (lambda *_a, **_k: None)
    spec = BudgetSpec(n_bytes=budget)
    variants = [
        {"name": "all layers", "model": "wang1d", "spec": spec, "source": "bytes",
         "description": "the prefix as preprocessed"},
        {"name": "headers only", "model": "wang1d", "spec": spec, "source": "headers",
         "description": "link, network and transport headers of every packet, concatenated"},
        {"name": "payload only", "model": "wang1d", "spec": spec, "source": "payload",
         "description": "application-layer bytes only"},
    ]
    say(f"\n  -- Step 10(ii) layer scope at {budget} bytes --")
    out = _run_variants(store, manifest, cfg, variants, seeds, "all layers", say)
    out["budget"] = spec.label

    # How much of the budget each scope can actually fill: a scope that runs
    # out of bytes is not a fair test of whether those bytes matter.
    from .slicer import layer_scoped
    te = manifest.indices(TEST)
    for scope in ("headers", "payload"):
        _packed, lengths = layer_scoped(store, te, scope)
        out["variants"][f"{scope} only"]["mean_bytes_available"] = float(lengths.mean())
        out["variants"][f"{scope} only"]["fraction_of_budget_filled"] = float(
            np.minimum(lengths, budget).mean() / budget)
    if paths:
        save_json(out, paths.json("ablation_layer_scope"))
    return out


def ablate_early_exit(store, manifest, cfg: TrainConfig,
                      seeds: Sequence[int] = (0, 1, 2), ceiling: int = 1024,
                      paths=None, verbose: bool = True) -> dict:
    """Step 10(iv): what the early-exit machinery actually contributes.

    Two questions, each isolating one component. Does auxiliary-head
    supervision help or hurt the deepest head, against a model trained with a
    single terminal head? And does the confidence policy beat simply always
    using the deepest head?
    """
    say = print if verbose else (lambda *_a, **_k: None)
    spec = BudgetSpec(n_bytes=ceiling)
    exits = tuple(e for e in DEFAULT_EXITS if e <= ceiling)
    variants = [
        {"name": "all heads supervised", "model": "adaptive", "spec": spec,
         "model_kwargs": {"exits": exits},
         "description": "auxiliary classification head at every budget"},
        {"name": "terminal head only", "model": "adaptive", "spec": spec,
         "model_kwargs": {"exits": (exits[-1],)},
         "description": "the same encoder, supervised only at the deepest budget"},
    ]
    say(f"\n  -- Step 10(iv) early-exit supervision at a {ceiling}-byte ceiling --")
    out = _run_variants(store, manifest, cfg, variants, seeds,
                        "all heads supervised", say)
    out["ceiling"] = ceiling
    out["note"] = ("evaluate_model reads the deepest head, so this compares like "
                   "with like: the question is whether supervising the shallow "
                   "heads costs the deep one anything")
    if paths:
        save_json(out, paths.json("ablation_early_exit"))
    return out


def ablate_granularity(session_store, flow_store, session_manifest, flow_manifest,
                       cfg: TrainConfig, seeds: Sequence[int] = (0, 1, 2),
                       budget: int = 784, paths=None, verbose: bool = True) -> dict:
    """Step 10(iii): bidirectional session against unidirectional flow.

    Needs two stores, because the unit of analysis is fixed at preprocessing
    time. Build the second with ``build_store.py --directional``. The two are
    not directly comparable on sample count -- one session becomes two flows --
    so class counts are reported alongside.
    """
    say = print if verbose else (lambda *_a, **_k: None)
    spec = BudgetSpec(n_bytes=budget)
    out: dict = {"budget": spec.label, "variants": {}}

    for name, store, manifest in (("bidirectional session", session_store, session_manifest),
                                  ("unidirectional flow", flow_store, flow_manifest)):
        tr, va, te = manifest.indices(TRAIN), manifest.indices(VAL), manifest.indices(TEST)
        accs, macros = [], []
        for seed in seeds:
            model, hist, _tr, _va = train_model(store, tr, va, spec, "wang1d", cfg, seed=seed)
            res, _pred, _te = evaluate_model(model, store, te, spec, cfg)
            accs.append(res.accuracy)
            macros.append(res.macro_f1)
            say(f"  {name:<24} seed {seed}  {res.summary_line()}")
        out["variants"][name] = {
            "accuracy": seed_summary(accs), "macro_f1": seed_summary(macros),
            "n_units": int(store.n), "n_test": int(te.size),
            "median_true_length": float(np.median(store.true_length)),
        }
    a = out["variants"]["bidirectional session"]["macro_f1"]
    b = out["variants"]["unidirectional flow"]["macro_f1"]
    noise = max(a["std"], b["std"])
    out["delta_macro_f1"] = float(b["mean"] - a["mean"])
    out["seed_noise"] = float(noise)
    out["verdict"] = ("inconclusive" if abs(out["delta_macro_f1"]) <= noise else
                      ("flow better" if out["delta_macro_f1"] > 0 else "session better"))
    out["caveat"] = ("one session becomes up to two flows, so the two settings differ "
                     "in sample count as well as in granularity; the comparison is "
                     "between settings, not between equally-sized datasets")
    if paths:
        save_json(out, paths.json("ablation_granularity"))
    return out


def summarise(ablation: dict, title: str) -> str:
    """A table fit to paste into the report."""
    rows = [f"{title}", f"{'variant':<28}{'macro-F1':>12}{'sd':>8}{'delta':>9}  verdict"]
    for name, cell in ablation["variants"].items():
        delta = cell.get("delta_macro_f1")
        rows.append(
            f"{name:<28}{cell['macro_f1']['mean']:>12.4f}{cell['macro_f1']['std']:>8.4f}"
            f"{'' if delta is None else f'{delta:>+9.4f}'}"
            f"{'  (reference)' if delta is None else '  ' + cell.get('verdict', '')}")
    return "\n".join(rows)

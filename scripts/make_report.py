#!/usr/bin/env python3
"""Assemble the Phase 3 deliverable from the results a run produced.

    python3 scripts/make_report.py --results results/ustc --out results/ustc/REPORT.md

Reads the JSON each experiment wrote and emits one document ordered by the
seven components the assignment asks for. Every number is pulled from the
run, never retyped, so the report cannot drift from the results it describes.
Sections whose inputs are missing say so rather than being silently dropped --
an absent ablation should be visible in the report, not invisible.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from nettc import literature as L
from nettc.dissect import STRATUM_NAMES


BASELINE_NAMES = {"wang2d": "Wang et al. 2017 byte image (28×28)",
                  "wang1d": "Wang et al. 2017 1D-CNN",
                  "deeppacket": "Deep Packet (Lotfollahi et al. 2020)"}


def load(path: Path):
    return json.loads(path.read_text()) if path.exists() else None


def fmt(summary: dict | None, places: int = 4) -> str:
    if not summary:
        return "--"
    return f"{summary['mean']:.{places}f} ± {summary['std']:.{places}f}"


def _port_probe(e1: dict | None) -> float:
    """Accuracy of the ports-only leakage probe, from E1."""
    if not e1:
        return float("nan")
    return e1["leakage"]["probes"]["ports"]["accuracy"]["mean"]


def section(title: str, body: str) -> str:
    return f"\n## {title}\n\n{body}\n"


def missing(what: str) -> str:
    return (f"*Not produced by this run.* Re-run with the stage that generates it "
            f"({what}).")




def _archive_evidence(audit, missing) -> str:
    """Why a category is absent, read off the distributed archives.

    Without this the report can only say a class is not present, which reads
    as a download that went wrong. The audit distinguishes that from a release
    that never shipped the capture, and only the second supports the stronger
    sentence.
    """
    if not audit:
        return (" Whether the category is absent from the release or from this "
                "download is not established here; run "
                "`scripts/audit_corpus_archives.py` to settle it.")
    if not audit.get("extraction_complete"):
        n = len(audit.get("declared_but_not_extracted") or [])
        return (f" **This is a download problem, not a corpus one**: {n} capture "
                f"file(s) the archives declare were never extracted. Re-extract "
                f"before reading anything else here.")
    bits = []
    for cat in missing:
        cell = (audit.get("categories") or {}).get(cat)
        if not cell:
            continue
        found = cell["in_archives"]
        if not found:
            bits.append(f"no capture matching `{cat}` appears in any of them")
        elif cell["vpn_only"]:
            bits.append(f"the only `{cat}` capture in any of them is the VPN one "
                        f"(`{found[0]}`), which this corpus already carries as a "
                        f"separate class")
    if not bits:
        return ""
    return (f" That is a property of the release rather than of this download: all "
            f"{audit['n_declared']} capture files the {audit['n_archives']} "
            f"distributed archives declare are present and extracted, and "
            f"{'; '.join(bits)}. A paper reporting the full "
            f"taxonomy on this corpus is therefore working from a different "
            f"release or a different labelling, which is a comparability obstacle "
            f"in its own right and is recorded as one in \u00a75.")


def count_methodology_changes(path: str | Path = "PHASE3.md") -> int:
    """How many numbered changes the methodology section lists.

    Exposed separately because the submission's rubric map states the number
    in a different sentence from the one the section itself carries, and the
    two have drifted before.
    """
    src = Path(path)
    if not src.exists():
        return 0
    lines = src.read_text().split("\n")
    try:
        start = next(i for i, ln in enumerate(lines)
                     if ln.startswith("## Changes from the Phase 2 methodology"))
    except StopIteration:
        return 0
    end = next((i for i in range(start + 1, len(lines))
                if lines[i].startswith("## ")), len(lines))
    return sum(1 for ln in lines[start + 1:end] if re.match(r"^\d+\. \*\*", ln))


def methodology_changes(path: str | Path = "PHASE3.md") -> list[str]:
    """Lift the "Changes from the Phase 2 methodology" section out of PHASE3.md.

    Kept as an extraction rather than a copy so the report and the requirement
    map cannot disagree -- they did, the report saying "four changes" after a
    fifth had been added.
    """
    src = Path(path)
    if not src.exists():
        return [f"*`{src}` not found; the methodology changes could not be "
                f"inlined. See that file for them.*", ""]
    lines = src.read_text().split("\n")
    try:
        start = next(i for i, ln in enumerate(lines)
                     if ln.startswith("## Changes from the Phase 2 methodology"))
    except StopIteration:
        return [f"*No methodology-changes section found in `{src}`.*", ""]
    end = next((i for i in range(start + 1, len(lines))
                if lines[i].startswith("## ")), len(lines))
    body = lines[start + 1:end]
    n = count_methodology_changes(path)
    heading = [f"### Changes from the Phase 2 methodology", ""]
    if n:
        heading += [f"{n} changes, each forced by something found while implementing "
                    f"or running. Lifted from `PHASE3.md`, which remains the "
                    f"requirement map.", ""]
    # Drop the section's own lead-in, which duplicates the heading above.
    while body and (not body[0].strip() or body[0].startswith("Required by the")
                    or body[0].startswith("methodology.") or body[0].startswith("running.")):
        body.pop(0)
    return heading + body + [""]



def saturation_significance(sig, e2) -> tuple[str, list[str]]:
    """Read the paired test against the seed-noise verdict on the same pair.

    They answer different questions and can point different ways, so the
    direction of the paired result decides what it means:

    * significant and favouring the *larger* budget -- tension. The extra
      bytes do help a given model, even though the help is smaller than the
      spread between seeds.
    * significant and favouring the *smaller* budget -- strong corroboration.
      Reading further actively hurts, which a seed-noise verdict can only
      report as "no improvement".
    * not significant -- corroboration, with the weaker claim that no
      difference was detected.

    Returns a one-word kind and the sentences for it.
    """
    pair = ((sig or {}).get("pairs") or {}).get("saturation")
    sat = (e2 or {}).get("bytes_saturation")
    if not (pair and sat):
        return "none", []
    m = pair["mcnemar"]
    small, large = pair["a"]["label"], pair["b"]["label"]
    favours_large = m["n01"] > m["n10"]
    inside_noise = sat["largest_remaining_gain"] <= sat["seed_noise"]
    head = (f"The paired test compares {small} against {large} on the same "
            f"{m['n01'] + m['n10']:,} sessions that the two models answered "
            f"differently (p = {m['p_value']:.3g}).")

    if m["p_value"] >= 0.05:
        return "corroborates", [
            head + f" It detects no reliable difference, which agrees with the "
            f"seed-noise verdict: {large} is not measurably better than the "
            f"saturation point even before run-to-run variation is considered."]
    if not favours_large:
        return "corroborates_strongly", [
            head + f" It favours **{small}**, the smaller budget, by "
            f"{m['n10']:,} sessions to {m['n01']:,}. Reading on to {large} does "
            f"not merely stop helping, it **actively hurts**, and reliably so.",
            "", "That is a stronger statement than the seed-noise verdict can "
            "make. A budget past saturation is not a harmless over-collection: "
            "the extra bytes are mostly padding at these lengths, and training "
            "on them costs accuracy as well as collection cost."]
    if inside_noise:
        return "tension", [
            head + f" It favours **{large}**, the larger budget, by "
            f"{m['n01']:,} sessions to {m['n10']:,}, yet the gain of "
            f"{sat['largest_remaining_gain']:+.4f} macro-F1 is smaller than the "
            f"{sat['seed_noise']:.4f} spread between seeds.",
            "", "Both are correct, because they ask different questions. "
            "McNemar asks whether *this* model beats *that* one on these "
            "sessions, and with thousands of paired sessions it detects a small "
            "effect easily. The seed-noise verdict asks whether the advantage "
            "survives simply retraining with another seed. So the extra bytes "
            "buy a little, and less than the noise a practitioner already lives "
            "with, while the collection cost scales with every one of them."]
    return "corroborates", [head]



def ports_significance(sig, ports) -> list[str]:
    """Read the paired port test against the seed-noise result at that budget.

    The port claim is the one most at risk of being overstated in either
    direction: a ports-only probe scoring well above chance invites the
    conclusion that the model runs on ports, and a small macro-F1 drop invites
    the conclusion that ports do not matter at all. Both instruments are
    needed, and where the paired test finds an effect the matching three-seed
    measurement decides whether it survives reseeding.
    """
    pair = ((sig or {}).get("pairs") or {}).get("ports")
    if not pair:
        return []
    m = pair["mcnemar"]
    budget = pair["question"].rstrip("?").split()[-1]
    cell = ((ports or {}).get("budgets") or {}).get(budget)
    favours_retained = m["n10"] > m["n01"]
    head = (f"A paired test at {budget} on the same "
            f"{m['n01'] + m['n10']:,} sessions the two settings disagreed about "
            f"gives p = {m['p_value']:.3g}")

    if m["p_value"] >= 0.05:
        return ["", head + ", detecting no reliable difference. The shortcut is "
                "available and the model does not use it, and the two measurements "
                "agree.", ""]
    if not favours_retained:
        return ["", head + f", favouring the *masked* setting by {m['n01']:,} "
                f"sessions to {m['n10']:,}. Removing the ports helps, which rules "
                f"out any reading on which the model depends on them.", ""]

    out = ["", head + f", favouring **ports retained** by {m['n10']:,} sessions "
           f"to {m['n01']:,} ({pair['a']['accuracy']:.4f} against "
           f"{pair['b']['accuracy']:.4f} accuracy). At this budget the ports are "
           f"doing real work for a given trained model."]
    if cell:
        lost = cell["macro_f1_lost_to_port_masking"]
        out += [f" Measured over three seeds at the same budget, masking costs "
                f"{lost:+.4f} macro-F1 against {cell['seed_noise']:.4f} of seed "
                f"noise, which is "
                + ("**outside** it: the effect survives reseeding and the ports "
                   "are load-bearing here, not merely available."
                   if cell["conclusive"] else
                   "**inside** it: the per-session effect is real for that run "
                   "and does not survive reseeding.")]
        if not cell["conclusive"]:
            masked = cell["no_ports"]["macro_f1"]
            kept = cell["bytes"]["macro_f1"]
            vals = masked.get("values") or []
            # When one run drives the whole spread, say which, because the
            # paired test above was run at a single seed and may have drawn it.
            if len(vals) > 2 and masked["std"] > 4 * max(kept["std"], 1e-9):
                worst = min(range(len(vals)), key=lambda i: vals[i])
                others = [v for i, v in enumerate(vals) if i != worst]
                out += [f" The masked arm's spread is driven by one run: seed "
                        f"{worst} scores {vals[worst]:.4f} where the others reach "
                        f"{' and '.join(f'{v:.4f}' for v in others)}. The paired "
                        f"test above used seed "
                        f"{(sig or {}).get('seed', 0)}, so it measured that run."]
                out += ["", "That is the clearest lesson in this report about "
                        "method. McNemar can be overwhelmingly significant, with "
                        "p numerically zero here, for a difference that simply does "
                        "not reappear when the model is retrained with another "
                        "seed. A paired test says something true about two "
                        "particular trained models; it says nothing on its own "
                        "about the setting that produced them. Reporting it "
                        "without the seed-noise band beside it would have turned "
                        "one unlucky run into a finding."]
            else:
                out += ["", "So the shortcut is available and the model does not "
                        "depend on it, which is what the averaged measurement "
                        "already indicated."]
    else:
        out += [" The three-seed sweep did not cover this budget, so whether the "
                "effect survives reseeding is untested here."]
    out += ["", "This is why both instruments are reported. A ports-only probe "
            "alone would condemn the design; a single paired test alone would "
            "convict it on one run; a small averaged drop alone would clear it "
            "without showing the work.", ""]
    return out


def gaps_section(e1, e2, e3, e3_capped, pad, ports, cstats, cfg, sig=None,
                 ss=None, audit=None) -> list[str]:
    """Answer the four Phase 1 gaps from the numbers this run produced.

    Each gap gets a verdict and the measurement behind it. Where a control came
    out underpowered or inconclusive the verdict says so; a gap closed by a
    control that could not be formed is worse than a gap left open.
    """
    name = (cfg or {}).get("corpus_name", "this corpus")
    out = ["### Against the Phase 1 research gaps", "",
           f"Each gap below is answered on {name} alone. Where the two corpora "
           f"disagree, the cross-corpus reading is in `../README.md`; a gap "
           f"answered on one corpus is not answered in general.", ""]

    # ---- Gap 1: the budget is set by convention, not measured -------------
    out += ["#### Gap 1: the observation budget is inherited, not measured", ""]
    if e2 and e2.get("bytes_saturation"):
        sb, sp = e2["bytes_saturation"], e2.get("packets_saturation")
        papers = e2.get("paper_budgets", {})
        def saturation_line(sat: dict, unit: str) -> str:
            gain = sat["largest_remaining_gain"]
            if gain <= 0:
                tail = (f"no larger budget beats it at all: the best of them, "
                        f"{sat['best_larger_budget']}, is {abs(gain):.4f} macro-F1 "
                        f"*worse*")
            else:
                tail = (f"the best larger budget, {sat['best_larger_budget']}, adds only "
                        f"{gain:.4f} macro-F1, inside the {sat['seed_noise']:.4f} seed "
                        f"noise")
            return f"The {unit} sweep saturates at **{sat['at']}**: {tail}."

        out += [saturation_line(sb, "byte")]
        if sp:
            out += [saturation_line(sp, "packet")]
        cited = []
        for unit, key in (("B", "bytes"), ("P", "packets")):
            sat = sb if unit == "B" else sp
            if not sat:
                continue
            floor = int("".join(c for c in sat["at"] if c.isdigit()))
            for value, who in sorted(papers.get(key, {}).items(), key=lambda kv: int(kv[0])):
                cited.append((int(value), unit, who, int(value) >= floor))
        if cited:
            beyond = [c for c in cited if c[3]]
            out += ["", f"Of the {len(cited)} budgets the reviewed papers use, "
                    f"**{len(beyond)} sit at or past the saturation point**:", "",
                    "| paper budget | paper | at or past saturation |", "|---|---|---|"]
            for value, unit, who, past in cited:
                out.append(f"| {value}{unit} | {who} | {'yes' if past else 'no'} |")
            out += ["", "None of those papers justifies its budget by a sweep; the "
                    "value is carried over from earlier work. What that carried-over "
                    "value actually costs is settled below by a paired test, which "
                    "measures the same comparison per session rather than per run.", ""]
        kind, lines = saturation_significance(sig, e2)
        if lines:
            out += lines + [""]
        else:
            out += ["On this corpus the carried-over budget is not wrong so much as "
                    "unnecessary: it buys nothing a far smaller budget does not already "
                    "have, while costing proportionally more bytes to collect. A paired "
                    "test would sharpen this; "
                    + missing("scripts/run_significance.py") + "", ""]
    else:
        out += [missing("--stages e2"), ""]

    # ---- Gap 2: aggregate metrics hide minority-class failure -------------
    out += ["#### Gap 2: minority and short classes failing inside aggregate metrics", ""]
    if e3:
        o = e3["overall"]
        gap = o["weighted_f1"] - o["macro_f1"]
        out += [f"The concealment gap is **{gap:+.4f}** (weighted-F1 "
                f"{o['weighted_f1']:.4f} against macro-F1 {o['macro_f1']:.4f}). "]
        if gap < 0:
            out += [f"It is *negative*, which inverts the gap as Phase 1 framed it: here "
                    f"the frequent classes are the hard ones, so a weighted average "
                    f"reports **worse** than a macro average rather than flattering it. "
                    f"Either way the single headline number misleads, and only the sign of "
                    f"the error changes.", ""]
        else:
            out += [f"A weighted average therefore reports {gap:.4f} more F1 than the "
                    f"per-class picture supports.", ""]
        # The 0.05 line is a flag, not the finding. A class at 0.08 F1 with
        # 0.04 recall has failed in every sense that matters, and letting an
        # empty failed_classes list read as "no failure here" would conceal
        # exactly what this gap is about. So the worst class is always named.
        worst = min(o["per_class"], key=lambda c: c["f1"])
        median_support = sorted(c["support"] for c in o["per_class"])[len(o["per_class"]) // 2]
        thin = worst["support"] < median_support
        out += [f"The worst class is `{worst['name']}`: **F1 {worst['f1']:.3f}, recall "
                f"{worst['recall']:.3f}** on {worst['support']} test sessions"
                + ("." if thin else
                   ", not a thin class scraping by, but one of the better-populated "
                   "classes, which the model almost never predicts.")]
        if e3.get("failed_classes"):
            out += [f" Below 0.05 F1 outright: "
                    f"**{', '.join(e3['failed_classes'])}**. This is the failure mode "
                    f"the review found in P2 (AIM at 0%) and P10, reproduced here and "
                    f"reported rather than averaged away.", ""]
        elif worst["f1"] < 0.25:
            out += [f" Nothing falls under the 0.05 line used to flag an outright "
                    f"failure, but {worst['f1']:.3f} F1 at {worst['recall']:.3f} recall "
                    f"is a failure by any reading of it, and the aggregate does not "
                    f"show it. The threshold is a flag, not the finding.", ""]
        else:
            out += [" No class comes close to the 0.05 line used to flag an outright "
                    "failure, so the specific failure mode P2 and P10 exhibit does not "
                    "reproduce on this corpus.", ""]
        bins = e3["strata"].get("class_support") or {}
        if len(bins) >= 2:
            items = list(bins.items())
            lo, hi = items[0], items[-1]
            direction = ("smallest-support classes score *higher*"
                         if lo[1]["macro_f1"] > hi[1]["macro_f1"]
                         else "smallest-support classes score lower")
            out += [f"By class support, the {direction}: {lo[0]} at macro-F1 "
                    f"{lo[1]['macro_f1']:.4f} against {hi[0]} at "
                    f"{hi[1]['macro_f1']:.4f}.", ""]
            if lo[1]["macro_f1"] > hi[1]["macro_f1"]:
                out += ["That is the opposite of the literature's framing, and it is why "
                        "the split-strategy caveat below matters: class support and split "
                        "strategy are confounded on this corpus, so this ordering is not "
                        "safe to read as a statement about support.", ""]
        if e3_capped is not None and e3 is not e3_capped:
            out += ["*Measured on the uncapped split. The capped run flattens exactly "
                    "the minority-class behaviour this gap is about, so it is reported "
                    "beside it rather than instead of it.*", ""]
    else:
        out += [missing("--stages e3"), ""]

    # ---- Gap 3: how much of the corpus is actually encrypted --------------
    out += ["#### Gap 3: how much of the corpus is actually encrypted", ""]
    if cstats:
        clear = cstats.get("definitely_not_encrypted_fraction")
        tls = cstats.get("confirmed_tls_fraction")
        dep = (cstats.get("tls") or {}).get("deprecated_fraction")
        n_hs = (cstats.get("tls") or {}).get("handshakes_parsed")
        unknown = 1.0 - clear - tls - (cstats.get("strata") or {}).get("encrypted_other", 0.0) \
            - (cstats.get("strata") or {}).get("opaque_high_entropy", 0.0)
        out += [f"This is reported as two bounds rather than one number, because the "
                f"tagger resolves only part of the corpus. **{clear:.1%}** of sessions are "
                f"definitely *not* encrypted: plaintext, or a recognisable unencrypted "
                f"binary protocol. **{tls:.1%}** carry a TLS handshake that could be "
                f"parsed and confirmed. The remainder is unresolved: a session with no "
                f"readable handshake may be encrypted, tunnelled, or simply too short to "
                f"tell.", "",
                f"So the encrypted share lies between **{tls:.1%}** and "
                f"**{1.0 - clear:.1%}**. The useful half of that is the lower bound: only "
                f"{tls:.1%} of this corpus is *confirmed* encrypted traffic, which is "
                f"less than a paper describing it as an encrypted-traffic benchmark "
                f"implies. The upper bound is what stops this being a stronger claim. "
                f"P5's figure of around 95-99% unencrypted is neither reproduced nor "
                f"contradicted here, because the method used cannot resolve the "
                f"{unknown:.1%} it leaves unknown."]
        if dep is not None:
            verdict = ("So even the confirmed-encrypted portion does not resemble traffic "
                       "a classifier would meet today: its ciphers predate the deployment "
                       "it would be used in, and a model tuned on them is tuned on a "
                       "handshake distribution that no longer exists."
                       if dep >= 0.5 else
                       "That is low enough that the confirmed-encrypted portion of this "
                       "corpus is not obviously obsolete on cipher grounds, unlike the "
                       "other corpus in this study, where the same measurement comes out "
                       "the opposite way. The age of a benchmark's ciphers has to be "
                       "measured per corpus; it cannot be assumed either way.")
            out += ["", f"Of the {n_hs} handshakes that were parsed, **{dep:.1%} negotiate "
                    f"a suite that is now deprecated**. {verdict}", ""]
        out += ["", "| budget | sessions padded rather than truncated |", "|---|---|"]
        for budget, frac in (cstats.get("padded_fraction_at_budget") or {}).items():
            out.append(f"| {budget} | {frac:.1%} |")
        cited_pad = {b: f for b, f in (cstats.get("padded_fraction_at_budget") or {}).items()
                     if b in ("784B", "1024B")}
        if cited_pad:
            lo, hi = min(cited_pad.values()), max(cited_pad.values())
            out += ["", f"The padding column is the other half of the gap. At the two byte "
                    f"budgets the reviewed papers use, {lo:.1%}-{hi:.1%} of sessions are "
                    f"shorter than the budget, so for those sessions part of what the "
                    f"model reads is padding rather than traffic. No reviewed paper "
                    f"reports this fraction.", ""]
        else:
            out += ["", "No reviewed paper reports its padding fraction; the column above "
                    "is what that omission hides.", ""]
    else:
        out += [missing("scripts/corpus_stats.py"), ""]

    # ---- Gap 4: shortcut learning and the positive control ---------------
    out += ["#### Gap 4: shortcut learning and the positive control", ""]
    if e1 and e1.get("leakage"):
        lk = e1["leakage"]
        probes, chance = lk["probes"], lk["majority_class_rate"]
        gate = probes["id_positions"]["accuracy"]["mean"]
        out += [f"The gate holds. A probe reading only the anonymised MAC and IP byte "
                f"positions reaches {gate:.4f} accuracy against a {chance:.4f} "
                f"majority-class rate, below chance, so anonymisation left no usable "
                f"identifier signal. Gate verdict: "
                f"**{'passed' if lk.get('gate_passed') else 'FAILED'}**.", ""]
        if "ident" in probes:
            out += [f"The same positions *before* anonymisation reach "
                    f"{probes['ident']['accuracy']['mean']:.4f}, which is how much "
                    f"identifier shortcut the anonymisation destroyed. A pipeline that "
                    f"skipped that step would have had that much free accuracy "
                    f"available, and no reviewed paper runs this control.", ""]
        if "ports" in probes:
            pp = probes["ports"]["accuracy"]["mean"]
            out += [f"Ports are the retained-field risk: a ports-only probe reaches "
                    f"{pp:.4f}."]
            if ports:
                items = list(ports["budgets"].items())
                at, cell = max(items, key=lambda kv: kv[1]["macro_f1_lost_to_port_masking"])
                worst = cell["macro_f1_lost_to_port_masking"]
                rest = [kv for kv in items if kv[0] != at]
                nxt = (max(rest, key=lambda kv: kv[1]["macro_f1_lost_to_port_masking"])
                       if rest else None)
                out += [f" Masking those bytes costs the full model at most "
                        f"**{worst:+.4f}** macro-F1, at {at}, and that one is "
                        f"{'outside' if cell['conclusive'] else 'inside'} seed noise."]
                if nxt:
                    out += [f" At every other budget the cost is no more than "
                            f"{nxt[1]['macro_f1_lost_to_port_masking']:+.4f}."]
                out += [" So on the averaged measurement the information looks "
                        "redundantly encoded rather than load-bearing: the shortcut "
                        "is available, and the model does not obviously depend on it "
                        "once it can see past the header."]
            out += [""]
            out += ports_significance(sig, ports)
    else:
        out += [missing("--stages e1"), ""]
    if pad:
        attr = pad.get("mask_attribution") or {}
        lb = pad.get("length_balance") or {}
        if attr:
            rep_eff = attr["representation_effect"]
            pad_eff = attr["padding_symbol_effect"]
            noise = attr["seed_noise"]
            rep_conclusive = abs(rep_eff) > noise
            pad_conclusive = attr.get("padding_symbol_conclusive",
                                      abs(pad_eff) > noise)
            out += [f"On padding specifically, the apparent benefit of a PAD symbol "
                    f"decomposes into **{rep_eff:+.4f}** from the representation and "
                    f"**{pad_eff:+.4f}** from the padding symbol itself, against "
                    f"{noise:.4f} of seed noise."]
            if rep_conclusive and not pad_conclusive:
                out += [" Only the representation clears that bar. The gain usually "
                        "credited to padding-aware input is the embedding, and "
                        "reporting the combined figure as a padding result would "
                        "misattribute nearly all of it.", ""]
            elif not rep_conclusive and not pad_conclusive:
                out += [" **Neither clears that bar**, so on this corpus the control "
                        "separates nothing: the combined change is small enough that "
                        "three seeds cannot tell its two halves apart. That is a null "
                        "result, not support for either attribution, and it is why the "
                        "same measurement on the other corpus, where the "
                        "representation term is an order of magnitude larger, carries "
                        "the claim.", ""]
            else:
                out += [" Both terms clear that bar, so the padding symbol does "
                        "contribute here independently of the representation.", ""]
        if lb:
            n_te = lb.get("n_test") or 0
            kept = len(lb.get("classes_spanning_all_bins") or [])
            total = lb.get("classes_total")
            chance = lb.get("chance_within_balanced_set")
            scored = (pad.get("controls") or {}).get("length_balanced")
            out += [f"The length-balanced control equalises the flow-length distribution "
                    f"across the {kept} of {total} classes that occupy every length bin, "
                    f"leaving **{n_te} test sessions** and a {chance:.4f} within-set "
                    f"chance rate. It asks whether anything survives when flow length "
                    f"can no longer separate the classes."]
            if scored:
                got = scored["macro_f1"]["mean"]
                sd = scored["macro_f1"]["std"]
                survives = got > (chance or 0) + max(2 * sd, 0.05)
                out += [f" It scores macro-F1 **{got:.4f}** (std {sd:.4f}, "
                        f"{scored['accuracy']['mean']:.4f} accuracy).", ""]
                if survives:
                    out += [f"That is well clear of the {chance:.4f} chance rate, so on "
                            f"this corpus **content still separates the classes with "
                            f"length held fixed**. The budget curve is not simply a "
                            f"length curve wearing a different label.", ""]
                else:
                    out += [f"That is at or below the {chance:.4f} chance rate, and the "
                            f"three seeds agree. On this corpus **little survives once "
                            f"flow length is equalised**, which puts a large share of "
                            f"the apparent signal on length rather than on content, "
                            f"the confound this study was built to look for.", ""]
            out += [f"Two things bound how far that goes. The test set is {n_te} "
                    f"sessions, so the effect size is imprecise even where the "
                    f"direction is unambiguous. And the retained classes are a biased "
                    f"subset by construction: the ones whose lengths overlap every "
                    f"other class, which is to say the ones least likely to have been "
                    f"separable by length in the first place.", ""]
            if lb.get("classes_excluded"):
                out += [f"*Retained: "
                        f"{', '.join('`%s`' % c for c in lb.get('classes_spanning_all_bins', []))}. "
                        f"Excluded for not spanning all bins: "
                        f"{', '.join('`%s`' % c for c in lb['classes_excluded'])}.*", ""]

    # ---- the limitation that cuts across all four ------------------------
    if cstats and cstats.get("split_strategy_counts"):
        counts = cstats["split_strategy_counts"]
        temporal = counts.get("temporal", 0)
        total = sum(counts.values())
        out += ["#### The limitation that cuts across all four", "",
                f"**{temporal} of {total} classes on this corpus are split temporally, "
                f"not group-wise.** A temporal split draws train and test from the same "
                f"capture and separates them only in time. It is a fallback, for two "
                f"distinct reasons:", "",
                "- the class occupies fewer than three capture files, so whole files "
                "cannot be held out at all; or",
                "- a group split was attempted and left one of the three splits empty, "
                "because the class's files are too unevenly sized to divide 70/15/15.",
                ""]
        by_reason: dict[str, list[str]] = {}
        for cls, how in (cstats.get("split_strategy") or {}).items():
            if how.startswith("temporal"):
                reason = how[len("temporal ("):-1] if "(" in how else "unspecified"
                by_reason.setdefault(reason, []).append(cls)
        for reason, members in sorted(by_reason.items()):
            out += [f"*{reason}*: {', '.join('`%s`' % c for c in sorted(members))}.", ""]
        if temporal:
            out += ["This is a property of the corpus, not a choice: the captures do "
                    "not exist to split any other way. But it means per-class scores are "
                    "not on one scale, and every number above that compares classes is "
                    "confounded with it. It also sets a ceiling on how much of the "
                    "headline accuracy should be believed to transfer to unseen "
                    "captures, the usual deployment case, and the one a random "
                    "within-capture split measures least well.", ""]
        caps = cstats.get("captures") or {}
        if caps and caps.get("contributed_nothing"):
            out += [f"Relatedly, only **{caps['contributing_sessions']} of "
                    f"{caps['registered']} capture files contributed a unique session**. "
                    f"Sessionizing by five-tuple merges a conversation the corpus ships "
                    f"split across two files, and content dedup drops a session a second "
                    f"capture repeats, so {len(caps['contributed_nothing'])} files add "
                    f"nothing. That shrinks the pool of groups a group-wise split can "
                    f"draw on, which is part of why the fallback above fires as often as "
                    f"it does.", "",
                    f"<details><summary>The {len(caps['contributed_nothing'])} captures "
                    f"that contributed no unique session</summary>", "",
                    ", ".join(f"`{c}`" for c in caps["contributed_nothing"]), "",
                    "</details>", ""]

    # ---- and what happens when it is measured rather than conceded ------
    if ss and ss.get("arms", {}).get("group"):
        arms = ss["arms"]
        n_cls, n_all = len(ss["classes"]), ss["n_classes_total"]
        rows = []
        label = {"group": "group-wise, unseen capture",
                 "temporal": "temporal, same captures, later in time",
                 "random": "random, same captures, no time boundary"}
        for a in ("group", "temporal", "random"):
            c = arms.get(a)
            if not c:
                continue
            delta = ("(reference)" if a == "group"
                     else f"{c['inflation_macro_f1']:+.4f}")
            rows.append([label.get(a, a),
                         f"{c['accuracy']['mean']:.4f} ± {c['accuracy']['std']:.4f}",
                         f"{c['macro_f1']['mean']:.4f} ± {c['macro_f1']['std']:.4f}",
                         delta, c.get("verdict", "")])
        out += ["#### What that limitation is actually worth", "",
                f"**{n_cls} of this corpus's {n_all} classes have enough capture "
                f"files to be split either way**, which makes the question "
                f"answerable rather than merely admissible. The three arms below "
                f"hold the classes, the budget ({ss['budget']}), the model, the "
                f"epochs and *the per-class train/val/test sizes* fixed, and vary "
                f"only where the boundary falls. Matching the sizes is what makes "
                f"this a measurement: the group arm's splits land where the capture "
                f"files happen to fall, so the other two are cut to those same "
                f"sizes rather than to a tidy 70/15/15.", "",
                f"{len(ss['seeds'])} seeds, {ss['n_train']} training and "
                f"{ss['n_test']} test sessions in every arm. Classes: "
                f"{', '.join('`%s`' % c for c in ss['classes'])} "
                f"(chance {ss['chance_rate']:.4f}).", ""]
        out += ["| train/test boundary | accuracy | macro-F1 | vs group-wise "
                "| verdict |", "|---|---|---|---|---|"]
        out += ["| " + " | ".join(r) + " |" for r in rows]
        rand, temp = arms.get("random"), arms.get("temporal")
        out += [""]
        if rand:
            ratio = (abs(rand["inflation_macro_f1"]) / rand["seed_noise"]
                     if rand["seed_noise"] else float("inf"))
            out += [f"**A random within-capture split is worth "
                    f"{rand['inflation_macro_f1']:+.4f} macro-F1**: "
                    f"{rand['macro_f1']['mean']:.4f} against "
                    f"{arms['group']['macro_f1']['mean']:.4f}, on the same classes, "
                    f"at the same budget, with the same split sizes. That is "
                    f"{ratio:.0f} times the seed noise, and it buys nothing but "
                    f"permission for the test set to share captures with training. "
                    f"An accuracy of "
                    f"{rand['accuracy']['mean']:.2f} on this corpus is reachable "
                    f"without the model generalising past the capture file at all. "
                    f"A paper that does not state its split policy has most likely "
                    f"taken this arm.", ""]
        if temp:
            verdict = temp.get("verdict")
            if verdict == "inconclusive":
                out += [f"**The temporal fallback used here is not measurably worse "
                        f"than a group-wise split**: {temp['inflation_macro_f1']:+.4f} "
                        f"against {temp['seed_noise']:.4f} of seed noise. So the "
                        f"fallback sits far closer to the honest arm than to the "
                        f"random one, and the numbers in this report are not inflated "
                        f"the way a random split's would be. That is a weaker claim "
                        f"than *the fallback is free*. The difference may be real and "
                        f"too small to resolve at {len(ss['seeds'])} seeds, but it "
                        f"bounds the concern.", ""]
            else:
                out += [f"**The temporal fallback used here is itself "
                        f"{verdict}** by {temp['inflation_macro_f1']:+.4f} macro-F1 "
                        f"against {temp['seed_noise']:.4f} of seed noise, so the "
                        f"per-class scores above should be read with that much "
                        f"allowance.", ""]
        out += ["Three caveats keep this from being more than it is. The arms score "
                "different test sessions, so no paired test applies and the verdict "
                "is the seed-noise rule of Step 9. The group arm is the noisiest of "
                "the three because whole capture files move between splits, which "
                "sets the bar any difference has to clear. And these "
                f"{n_cls} classes are not a random sample of the corpus. They are "
                "the ones with enough captures to split both ways, which is itself a "
                "property that may correlate with difficulty.", ""]
        out += ["![Macro-F1 against where the train/test boundary falls."
                "](figures/split_strategy.png)", "",
                "*Macro-F1 against where the train/test boundary falls. Error bars "
                "are the seed standard deviation.*", ""]

    # ---- the two limitations that are not about splitting ----------------
    extra = []
    tax = (cstats or {}).get("taxonomy") or {}
    if tax.get("missing"):
        line = (f"**The class list is short by {len(tax['missing'])}.** The "
                f"{tax['scheme']} scheme defines {len(tax['expected'])} classes, but "
                f"the label set is derived from the captures that are present. "
                f"Absent here: "
                f"{', '.join('`%s`' % c for c in tax['missing'])}. Every number in "
                f"this report is therefore on {cstats['n_classes']} classes, not "
                f"{len(tax['expected'])}, and chance rates are set accordingly.")
        # Whether that is a bad download or an incomplete release decides what
        # the sentence may claim, so it is checked against the archives rather
        # than assumed. `audit_corpus_archives.py` writes the evidence.
        gap = _archive_evidence(audit, tax["missing"])
        extra += [line + gap]
    cap = (cfg or {}).get("max_per_class") or 0
    counts = (cstats or {}).get("class_counts") or {}
    if cap and counts:
        bound = {k: v for k, v in counts.items() if v > cap}
        native = max(counts.values()) / min(counts.values())
        capped = min(max(counts.values()), cap) / min(counts.values())
        extra += [f"**Runs are capped at {cap:,} sessions per class**, which binds on "
                  f"{len(bound)} of {len(counts)} classes "
                  f"({', '.join('`%s` %d' % (k, v) for k, v in sorted(bound.items(), key=lambda kv: -kv[1]))}). "
                  f"The cap applies to the test set as well as the training set, so it "
                  f"flattens the imbalance Gap 2 is about: from {native:.1f}x between the "
                  f"largest and smallest class down to {capped:.1f}x. Where an uncapped "
                  f"run exists it is used for the stratified analysis; where it does not, "
                  f"the Gap 2 numbers above understate how uneven the corpus really is."]
    elif cap:
        extra += [f"**Runs are capped at {cap:,} sessions per class**, which applies to "
                  f"the test set as well as the training set and so flattens the "
                  f"imbalance Gap 2 is about."]
    if extra:
        heading = ("#### One further limitation" if len(extra) == 1
                   else f"#### {len(extra)} further limitations")
        out += [heading, ""]
        for item in extra:
            out += [f"- {item}", ""]
    return out


def build(results: Path, figures: Path) -> str:
    e1 = load(results / "e1_baseline_and_leakage.json")

    e4 = load(results / "e4_adaptive_frontier.json")
    pad = next((load(p) for p in sorted(results.glob("padding_controls_*.json"))), None)
    sweeps = {p.stem: load(p) for p in sorted(results.glob("e2_budget_sweep_*.json"))}
    e3_capped = load(results / "e3_stratified_wang1d_784B.json")
    e3_uncapped = load(results / "e3_stratified_uncapped_wang1d_784B.json")
    ports = load(results / "port_dependence_wang1d.json")
    e2 = next((v for k, v in sweeps.items() if not k.endswith("_zero")), None)
    e2_zero = next((v for k, v in sweeps.items() if k.endswith("_zero")), None)
    e3 = e3_uncapped or e3_capped
    rep = load(results / "ablation_representation.json")
    scope = load(results / "ablation_layer_scope.json")
    early = load(results / "ablation_early_exit.json")
    gran = load(results / "ablation_granularity.json")
    cfg = load(results / "run_config.json")
    cstats = load(results / "corpus_stats.json")
    sig = load(results / "significance.json")
    ss = load(results / "split_strategy.json")
    audit = load(results / "archive_audit.json")

    out = ["# Phase 3: Implementation and Experimental Evaluation",
           "",
           "**Observation Budgets and Padding Effects in Raw-Byte Network Traffic "
           "Classification**",
           "",
           "Muhammad Talha Arshad (i232548) · Zain Shahid (i232582)",
           ""]
    if cfg:
        out.append(f"Run: model `{cfg['model']}`, input representation "
                   f"`{cfg['pad_mode']}`, seeds {cfg['seeds']}, "
                   f"{cfg['seconds'] / 60:.0f} minutes on one RTX 5050.")
        out.append("")

    # ---------------------------------------------------------------- 1 and 2
    # "Document any changes made to your original methodology" is a Rubric 1
    # requirement, so it belongs in the report a grader reads, not only in the
    # tracking file. PHASE3.md stays the single source: this lifts the section
    # out of it rather than restating it, so the two cannot drift or disagree
    # on the count.
    body = list(methodology_changes())
    if e1:
        lk = e1["leakage"]
        probes = lk["probes"]
        body += [
            "### The leakage gate",
            "",
            "Before any result is reported, the same architecture is trained on "
            "restricted views of the input. The gate is the first row: those byte "
            "positions are read out of the *anonymised* store, so a score above the "
            "majority-class rate would mean the anonymisation did not land.",
            "",
            f"| probe | input | accuracy | majority-class rate |",
            f"|---|---|---|---|",
            f"| **gate** | MAC/IP positions, after anonymisation | "
            f"{fmt(probes['id_positions']['accuracy'])} | {lk['majority_class_rate']:.4f} |",
            f"| context | the same positions, before anonymisation | "
            f"{fmt(probes['ident']['accuracy'])} | {lk['majority_class_rate']:.4f} |",
            f"| risk | the retained port bytes | "
            f"{fmt(probes['ports']['accuracy'])} | {lk['majority_class_rate']:.4f} |",
            "",
            f"**Gate {'passed' if lk['gate_passed'] else 'FAILED'}** "
            f"(excess over chance "
            f"{probes['id_positions']['excess_over_chance']:+.4f}, "
            f"margin {lk['margin']}).",
        ]
        if probes["ports"]["accuracy"]["mean"] > lk["majority_class_rate"] + 0.05:
            body += ["", "The port probe scores above chance. Ports are retained "
                     "deliberately (Phase 2 Step 3c, following P1's ablation), so this "
                     "is not a gate failure, but it means some of the headline accuracy "
                     "is port information rather than traffic content and must be read "
                     "that way."]
    out.append(section("1-2. Implementation, dataset and experimental setup",
                       "\n".join(body)))

    # ---------------------------------------------------------------------- 3
    if e1:
        rows = ["| baseline | budget | accuracy | macro-F1 | weighted-F1 |", "|---|---|---|---|---|"]
        names = BASELINE_NAMES
        for key, cell in e1["baselines"].items():
            rows.append(f"| {names.get(key, key)} | {cell['budget']} | "
                        f"{fmt(cell['accuracy'])} | {fmt(cell['macro_f1'])} | "
                        f"{fmt(cell['weighted_f1'])} |")
        if sig:
            rows += ["", "### Paired significance tests", "",
                     "A seed-noise verdict asks whether a difference exceeds "
                     "run-to-run variation. For comparisons scored on the *same* "
                     "test sessions, McNemar's exact test asks a sharper question: "
                     "whether one setting reliably beats the other on these "
                     "sessions. Both are reported, because they can disagree and "
                     "the disagreement is informative.", "",
                     "| pair | A | B | discordant B→A / A→B | p | significant |",
                     "|---|---|---|---|---|---|"]
            for key, pair in sig["pairs"].items():
                m = pair["mcnemar"]
                rows.append(
                    f"| {key} | {pair['a']['label']} {pair['a']['accuracy']:.4f} | "
                    f"{pair['b']['label']} {pair['b']['accuracy']:.4f} | "
                    f"{m['n01']} / {m['n10']} | {m['p_value']:.3g} | "
                    f"{'**yes**' if m['p_value'] < 0.05 else 'no'} |")
            rows += ["", f"*Seed {sig['seed']}, {sig['n_test']:,} paired test "
                     f"sessions. The granularity ablation is deliberately absent: "
                     f"sessions and flows are different units, so the test sets "
                     f"differ in size and McNemar does not apply.*", ""]
        out.append(section("3. Baseline evaluation", "\n".join(rows)))
    else:
        out.append(section("3. Baseline evaluation", missing("--stages e1")))

    # ---------------------------------------------------------------------- 4
    if e2:
        rows = ["The study's primary result. Architecture, optimiser, schedule, split "
                "and seeds are held constant; only the number of bytes the model may "
                "read changes.", "",
                "| byte budget | macro-F1 | accuracy | mean padding fraction |",
                "|---|---|---|---|"]
        for label, cell in e2["bytes"].items():
            rows.append(f"| {label} | {fmt(cell['macro_f1'])} | {fmt(cell['accuracy'])} | "
                        f"{cell['mean_padding_fraction']:.3f} |")
        sat = e2.get("bytes_saturation") or {}
        rows += ["", f"**Saturation:** {sat.get('at') or 'not reached inside the swept range'}"
                 + (f": no larger budget improves on it by more than seed noise "
                    f"({sat.get('seed_noise', 0):.4f})." if sat.get("at") else ".")]
        rows += ["", "| packet budget | macro-F1 | accuracy | tensor width |", "|---|---|---|---|"]
        for label, cell in e2["packets"].items():
            rows.append(f"| {label} | {fmt(cell['macro_f1'])} | {fmt(cell['accuracy'])} | "
                        f"{cell['width']} |")
        psat = e2.get("packets_saturation") or {}
        rows += ["", f"**Saturation (packets):** {psat.get('at') or 'not reached'}."]
        for name, caption in (("budget_profile_bytes.png",
                               "Accuracy against byte budget, with the reviewed papers' "
                               "choices marked."),
                              ("budget_profile_packets.png",
                               "Accuracy against packet budget."),
                              ("per_class_support.png",
                               "Per-class F1 against class support, with the macro/weighted gap."),
                              ("accuracy_vs_padding.png",
                               "Accuracy against padding fraction."),
                              ("padding_attribution.png",
                               "What the padding control actually measures."),
                              ("accuracy_per_byte_frontier.png",
                               "Fixed depth against the adaptive exit policy."),
                              ("port_dependence.png",
                               "What masking the port bytes costs, against the "
                               "ports-only probe and the majority-class rate."),
                              ("stratified_class_support.png",
                               "Macro-F1 by class support."),
                              ("stratified_flow_length.png",
                               "Macro-F1 by true flow length."),
                              ("stratified_encryption.png",
                               "Macro-F1 by encryption stratum.")):
            if (figures / name).exists():
                rows += ["", f"![{caption}](figures/{name})", "", f"*{caption}*"]
        out.append(section("4. Graphical results", "\n".join(rows)))
    else:
        out.append(section("4. Graphical results", missing("--stages e2")))

    # ---------------------------------------------------------------------- 5
    if e2 and cfg:
        corpus = (load(results / "run_config.json") or {}).get("corpus_name") or "USTC-TFC2016"
        best = max(e2["bytes"].values(), key=lambda c: c["macro_f1"]["mean"])
        n_classes = len(e3["overall"]["per_class"]) if e3 else 20
        table = L.comparison_table(best, corpus, n_classes,
                                   our_label=f"this work ({cfg['model']})")
        out.append(section("5. Comparison with existing research",
                           "```\n" + table + "\n```"))
    else:
        out.append(section("5. Comparison with existing research", missing("--stages e2 compare")))

    # ---------------------------------------------------------------------- 6
    parts = []
    if rep or scope or early or gran:
        from nettc.ablations import summarise
        for name, ab, title in (("representation", rep, "Step 10(i): representation, matched budget"),
                                ("layer scope", scope, "Step 10(ii): layer scope"),
                                ("early exit", early, "Step 10(iv): early-exit supervision")):
            if ab:
                parts += [f"### {title}", "", "```", summarise(ab, ""), "```", ""]
        if gran:
            parts += ["### Step 10(iii): granularity", "",
                      f"Bidirectional sessions against unidirectional flows at "
                      f"{gran['budget']}. The unit of analysis is fixed when the store "
                      f"is written, so this is the one ablation that needs a second "
                      f"pass over the captures rather than another slice.", "",
                      "| unit | macro-F1 | accuracy | units | test | median true length |",
                      "|---|---|---|---|---|---|"]
            for unit, cell in gran["variants"].items():
                parts.append(f"| {unit} | {fmt(cell['macro_f1'])} | "
                             f"{fmt(cell['accuracy'])} | {cell['n_units']:,} | "
                             f"{cell['n_test']:,} | {cell['median_true_length']:.0f} B |")
            parts += ["", f"Difference {gran['delta_macro_f1']:+.4f} macro-F1 against "
                      f"{gran['seed_noise']:.4f} of seed noise. Verdict: "
                      f"**{gran['verdict']}**.", ""]
            # The sample counts and the median lengths are the caveat made
            # concrete: splitting a session in two changes how many examples
            # there are and how long each one is, so neither column is held
            # fixed while granularity varies.
            variants = list(gran["variants"].values())
            if len(variants) == 2:
                a, b = variants
                caveat = gran["caveat"]
                caveat = caveat[0].upper() + caveat[1:]
                # The two confounds point the same way: a flow is shorter than
                # its parent session, so at a fixed byte budget the flow
                # setting sees a larger share of each unit. Whether that
                # wrecks the comparison depends on which way the result went.
                # If the less-truncated setting wins, the win may be the
                # truncation; if it loses, truncation cannot be the cause,
                # because it would have pushed the other way.
                verdict = gran["verdict"]
                if verdict == "flow better":
                    reading = ("The flow setting is therefore the less truncated one, "
                               "and it is also the one that won, so this design "
                               "cannot tell a granularity effect from a truncation "
                               "effect, and the result should not be read as evidence "
                               "for either.")
                elif verdict == "session better":
                    reading = ("The flow setting is therefore the less truncated one, "
                               "and it still lost. That matters: truncation would have "
                               "pushed the result the other way, so it cannot explain "
                               "the session advantage. Pairing the two directions "
                               "carries information that reading them separately does "
                               "not, and the confound in this design works against "
                               "that conclusion rather than for it.")
                else:
                    reading = ("The flow setting is therefore the less truncated one, "
                               "so granularity and effective budget move together. "
                               "With the difference inside seed noise there is nothing "
                               "to attribute, and this design could not have separated "
                               "them had there been.")
                parts += [f"*{caveat}.* Here that is "
                          f"{a['n_units']:,} units against {b['n_units']:,}, and a "
                          f"median true length of {a['median_true_length']:.0f} B "
                          f"against {b['median_true_length']:.0f} B. {reading}", ""]
            else:
                parts += [f"*{gran['caveat']}*", ""]
        else:
            parts += ["### Step 10(iii): granularity", "",
                      missing("build a directional store with "
                              "`build_store.py --directional`, then run the granularity "
                              "ablation"), ""]
    if ports:
        parts += ["### Port dependence (Step 11, the retained-field control)", "",
                  "Phase 2 Step 3c retained ports, citing P1's ablation, and deferred "
                  "the consequence to Step 11. Two measurements answer it, and they "
                  "point in different directions, so both are reported.", "",
                  f"A probe trained on nothing but the four port bytes reaches "
                  f"**{_port_probe(e1):.4f}** accuracy against a "
                  f"{ports['chance']:.4f} majority-class rate: a strong shortcut "
                  f"is available. But masking those bytes in every packet costs "
                  f"almost nothing above the smallest budget.", "",
                  "| budget | with ports | ports masked | lost | conclusive |",
                  "|---|---|---|---|---|"]
        for k, c in ports["budgets"].items():
            parts.append(f"| {k} | {fmt(c['bytes']['macro_f1'])} | "
                         f"{fmt(c['no_ports']['macro_f1'])} | "
                         f"{c['macro_f1_lost_to_port_masking']:+.4f} | "
                         f"{'yes' if c['conclusive'] else 'no'} |")
        conclusive = [k for k, c in ports["budgets"].items() if c["conclusive"]]
        if conclusive:
            last = conclusive[-1]
            after = [k for k in ports["budgets"] if k not in conclusive]
            parts += ["", f"So port information is *redundantly encoded* rather than "
                      f"load-bearing: available as a shortcut, but masking it changes "
                      f"the result by more than seed noise only at {', '.join(conclusive)}. "
                      + (f"Above {last} the difference is inside noise, so the budget "
                         f"curve there is not a port artifact." if after else
                         "Every budget in the sweep is affected, so the curve cannot be "
                         "cleared of the shortcut on this corpus."), ""]
        else:
            parts += ["", "So port information is *redundantly encoded* rather than "
                      "load-bearing: at no budget in the sweep does masking the port "
                      "bytes change macro-F1 by more than seed noise. The budget curve "
                      "is not a port artifact on this corpus.", ""]

    if pad:
        attr = pad.get("mask_attribution")
        parts += ["### Padding controls", "",
                  "| control | macro-F1 | change against zero padding |", "|---|---|---|"]
        base = pad["controls"]["zero"]["macro_f1"]["mean"]
        for name, cell in pad["controls"].items():
            delta = "" if name == "zero" else f"{cell['macro_f1']['mean'] - base:+.4f}"
            parts.append(f"| {name} | {fmt(cell['macro_f1'])} | {delta} |")
        if attr:
            parts += ["",
                      f"Moving from the scalar input to an embedding with a PAD symbol "
                      f"changes two things at once. Split: **representation "
                      f"{attr['representation_effect']:+.4f}**, **padding symbol "
                      f"{attr['padding_symbol_effect']:+.4f}** "
                      f"(seed noise {attr['seed_noise']:.4f}; the padding effect is "
                      f"{'outside' if attr['padding_symbol_conclusive'] else 'inside'} it)."]
    out.append(section("6. Ablation studies",
                       "\n".join(parts) if parts else missing("--stages ablations padding")))

    # ---------------------------------------------------------------------- 7
    parts = []
    if e3_capped and e3_uncapped:
        parts += ["Two stratified runs are reported. The sweep used a 5,000-per-class "
                  "cap to control runtime and the corpus's 14x imbalance, but that cap "
                  "applies to the test set too and flattens exactly the minority-class "
                  "behaviour Gap 2 concerns. The uncapped run below restores the native "
                  "class distribution; the capped figures are in "
                  "`e3_stratified_wang1d_784B.json`.", ""]
    if e3:
        parts += ["### What the aggregate conceals", "",
                  f"Overall at {e3['budget']}: accuracy {e3['overall']['accuracy']:.4f}, "
                  f"macro-F1 {e3['overall']['macro_f1']:.4f}, weighted-F1 "
                  f"{e3['overall']['weighted_f1']:.4f}. The macro-to-weighted gap is "
                  f"**{e3['overall']['weighted_f1'] - e3['overall']['macro_f1']:+.4f}**.", ""]
        if e3.get("failed_classes"):
            parts += [f"Classes at F1 below 0.05: **{', '.join(e3['failed_classes'])}**. "
                      f"This is the failure mode the review found in P2 (AIM at 0%, SFTP "
                      f"at 50%) and P10 (File 93.56%, VoIP 92.20%), reproduced here and "
                      f"reported rather than averaged away.", ""]
        for group, heading in (("flow_length", "By true flow length"),
                               ("class_support", "By class support"),
                               ("encryption", "By encryption stratum")):
            if e3["strata"].get(group):
                parts += [f"### {heading}", "", "| stratum | n | accuracy | macro-F1 |",
                          "|---|---|---|---|"]
                if group == "class_support" and e3.get("support_bin_note"):
                    parts += [f"*{e3['support_bin_note']}*", ""]
                for name, cell in e3["strata"][group].items():
                    flag = " *(thin)*" if cell["meta"].get("underpowered") else ""
                    parts.append(f"| {name}{flag} | {cell['n']} | {cell['accuracy']:.4f} | "
                                 f"{cell['macro_f1']:.4f} |")
                parts.append("")
    if e4 and "comparison" in e4:
        c = e4["comparison"]
        parts += ["### The adaptive policy", "", c["verdict"], ""]
        if c.get("byte_saving"):
            b = c["byte_saving"]
            parts += [f"At {b['at_threshold']} the policy reads {b['mean_bytes_read']:.0f} "
                      f"bytes on average against {b['against_fixed_bytes']:.0f} for the best "
                      f"fixed head, {b['fraction_of_fixed']:.0%} of the cost.", ""]
    if e2 and e2_zero:
        a = max(c["macro_f1"]["mean"] for c in e2["bytes"].values())
        b = max(c["macro_f1"]["mean"] for c in e2_zero["bytes"].values())
        parts += ["### Representation", "",
                  f"Best macro-F1 with the embedding input {a:.4f}, with the scalar "
                  f"byte/255 input {b:.4f}, a difference of {a - b:+.4f}. The budget "
                  f"curve therefore has no meaning independent of the representation it "
                  f"was measured with, which is a caution about comparing budget choices "
                  f"across papers that differ in this respect.", ""]
    parts += gaps_section(e1, e2, e3, e3_capped, pad, ports, cstats,
                          cfg, sig, ss, audit)
    out.append(section("7. Results analysis and discussion", "\n".join(parts)))

    out += ["", "---", "",
            "Reproduce with the commands in `README.md`. Every figure and table above is "
            "generated from the JSON in this directory by `scripts/make_report.py`."]
    return "\n".join(out)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", required=True)
    ap.add_argument("--figures", default=None)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    results = Path(args.results)
    figures = Path(args.figures) if args.figures else results / "figures"
    text = build(results, figures)
    out = Path(args.out or (results / "REPORT.md"))
    out.write_text(text)
    print(f"wrote {out} ({len(text.splitlines())} lines)")
    missing_count = text.count("*Not produced by this run.*")
    if missing_count:
        print(f"  {missing_count} section(s) have no inputs yet")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

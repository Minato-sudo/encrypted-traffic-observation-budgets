"""Figures for the four Step 12 outputs.

These are static publication figures, so they commit to the light surface
rather than carrying a dark variant. Three categorical hues are used at most,
which is the all-pairs-validated subset of the reference palette; every
series is direct-labelled, which is also the required relief for the one hue
that sits below 3:1 contrast on this surface.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional, Sequence

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FuncFormatter

# Categorical slots 1-3 of the reference palette, validated all-pairs in light
# mode (worst CVD dE 9.2, worst normal-vision dE 24.0).
SERIES = ("#2a78d6", "#eb6834", "#1baf7a")
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
INK_MUTED = "#8a8882"
GRID = "#e4e3df"

plt.rcParams.update({
    "figure.facecolor": SURFACE,
    "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE,
    "axes.edgecolor": GRID,
    "axes.labelcolor": INK_2,
    "axes.titlecolor": INK,
    "text.color": INK,
    "xtick.color": INK_2,
    "ytick.color": INK_2,
    "font.size": 9,
    "axes.titlesize": 11,
    "axes.titleweight": "semibold",
    "axes.grid": True,
    "grid.color": GRID,
    "grid.linewidth": 0.8,
    "legend.frameon": False,
    "figure.dpi": 140,
})


def _clean(ax, ylabel: str = "", xlabel: str = "") -> None:
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.set_axisbelow(True)
    ax.grid(axis="x", visible=False)
    if ylabel:
        ax.set_ylabel(ylabel)
    if xlabel:
        ax.set_xlabel(xlabel)


def _save(fig, path: Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    return path


# ------------------------------------------------------------- 1. budget profile

def budget_profile(curves: dict, path: str | Path, axis: str = "bytes",
                   paper_budgets: Optional[dict] = None,
                   metric: str = "macro_f1", chance: Optional[float] = None,
                   title: Optional[str] = None,
                   saturation: Optional[dict] = None) -> Path:
    """The study's headline figure: accuracy against observation budget.

    ``curves`` maps a series name to the per-budget cells produced by
    :func:`nettc.experiments.run_e2`. The budgets chosen by the reviewed
    papers are marked on a second axis above the plot so their decisions can
    be read off our curve, which is the point of sweeping these particular
    values.

    Budgets are spaced evenly by swept value rather than by magnitude. The
    question here is *which budget*, not what a byte costs -- and on a
    continuous axis the 15- and 16-packet budgets, which two different papers
    actually chose, overprint each other. Cost is on a true linear axis in
    :func:`accuracy_per_byte_frontier`, where it is the thing being measured.
    """
    fig, ax = plt.subplots(figsize=(7.4, 4.4))
    xkey = "n_bytes" if axis == "bytes" else "n_packets"

    reference = sorted(next(iter(curves.values())).values(),
                       key=lambda c: c[xkey] or c["width"])
    budgets = [c[xkey] or c["width"] for c in reference]
    pos = {b: i for i, b in enumerate(budgets)}

    ends = []
    lo, hi = np.inf, -np.inf
    for i, (name, cells) in enumerate(curves.items()):
        rows = sorted(cells.values(), key=lambda c: c[xkey] or c["width"])
        xs = np.array([pos[c[xkey] or c["width"]] for c in rows], dtype=float)
        ys = np.array([c[metric]["mean"] for c in rows])
        sd = np.array([c[metric]["std"] for c in rows])
        colour = SERIES[i % len(SERIES)]
        ax.fill_between(xs, ys - sd, ys + sd, color=colour, alpha=0.13, linewidth=0)
        ax.plot(xs, ys, color=colour, linewidth=2.0, marker="o", markersize=5,
                markeredgecolor=SURFACE, markeredgewidth=1.2, label=name, zorder=3)
        ends.append([xs[-1], ys[-1], name])
        lo = min(lo, float((ys - sd).min()))
        hi = max(hi, float((ys + sd).max()))

    chance_drawn = False

    if saturation and saturation.get("at") is not None:
        label = saturation["at"]
        value = next((b for b, c in zip(budgets, reference) if c["budget"] == label), None)
        if value is not None:
            ax.axvline(pos[value], color=INK_MUTED, linewidth=1.0,
                       linestyle=(0, (2, 3)), zorder=1)
            ax.annotate("saturates", xy=(pos[value], 0.03),
                        xycoords=("data", "axes fraction"), xytext=(4, 0),
                        textcoords="offset points", fontsize=8, color=INK_MUTED,
                        ha="left", va="bottom",
                        bbox=dict(facecolor=SURFACE, edgecolor="none", pad=1.5))

    # Scale to the data. Forcing the axis down to chance compresses the whole
    # curve into the top tenth of the figure, which hides the shape that is
    # the actual result.
    span = max(hi - lo, 1e-6)
    y0, y1 = lo - 0.18 * span, hi + 0.12 * span
    if chance is not None and chance >= y0:
        y0 = min(y0, chance - 0.05 * span)
    ax.set_ylim(y0, y1)

    # Series that converge would otherwise print their labels on top of each
    # other; separate them vertically when they are closer than a tenth of
    # the visible range.
    ends.sort(key=lambda e: e[1])
    gap = 0.10 * (y1 - y0)
    for j in range(1, len(ends)):
        if ends[j][1] - ends[j - 1][1] < gap:
            ends[j][1] = ends[j - 1][1] + gap
    for x_end, y_label, name in ends:
        ax.annotate(f" {name}", (x_end, y_label), color=INK_2, fontsize=9,
                    va="center", ha="left", annotation_clip=False)

    ax.set_xticks(range(len(budgets)), [f"{b}" for b in budgets])
    unit = "bytes observed per session" if axis == "bytes" else "packets observed per session"
    _clean(ax, ylabel=metric.replace("_", "-"), xlabel=unit)
    ax.set_xlim(-0.4, len(budgets) - 0.6)

    # The reviewed papers' budgets go on their own axis above the plot, so the
    # labels cannot collide with the data or with each other.
    if chance is not None:
        if chance >= ax.get_ylim()[0]:
            ax.axhline(chance, color=INK_MUTED, linewidth=1.0,
                       linestyle=(0, (4, 3)), zorder=1)
            ax.annotate("chance", xy=(0.996, chance),
                        xycoords=("axes fraction", "data"), color=INK_MUTED,
                        fontsize=8, va="bottom", ha="right")
            chance_drawn = True
        else:
            ax.annotate(f"majority-class rate {chance:.3f}, below this view",
                        xy=(0.004, 0.02), xycoords="axes fraction",
                        color=INK_MUTED, fontsize=8, va="bottom", ha="left")

    marks = {b: lab for b, lab in (paper_budgets or {}).get(axis, {}).items() if b in pos}
    if marks:
        for b in marks:
            ax.axvline(pos[b], color=GRID, linewidth=1.0, zorder=0)
        top = ax.secondary_xaxis("top")
        top.set_xticks([pos[b] for b in sorted(marks)],
                       [marks[b].split()[0] for b in sorted(marks)])
        top.tick_params(colors=INK_MUTED, labelsize=8, length=3)
        top.spines["top"].set_visible(False)
        key = "   ".join(f"{marks[b].split()[0]} {' '.join(marks[b].split()[1:])}"
                         for b in sorted(marks))
        fig.text(0.5, -0.015, key, ha="center", fontsize=7.5, color=INK_MUTED)

    ax.set_title(title or f"Accuracy against observation budget ({axis} axis)",
                 pad=18 if marks else 8)
    return _save(fig, path)


# -------------------------------------------------------- 2. padding diagnostics

def accuracy_vs_padding(rows: Sequence[dict], path: str | Path,
                        title: Optional[str] = None) -> Path:
    """Accuracy as a function of how much of the input was padding.

    If this slopes, the budget curve is partly a padding curve, which is the
    confound the whole study is built to detect.
    """
    fig, ax = plt.subplots(figsize=(5.6, 3.6))
    labels = [r["padding_fraction"] for r in rows]
    vals = [r["accuracy"] for r in rows]
    ns = [r["n"] for r in rows]
    x = np.arange(len(rows))
    ax.bar(x, vals, width=0.62, color=SERIES[0], linewidth=0)
    for xi, v, n in zip(x, vals, ns):
        ax.annotate(f"{v:.3f}\nn={n}", (xi, v), xytext=(0, 4), textcoords="offset points",
                    ha="center", va="bottom", fontsize=8, color=INK_2)
    ax.set_xticks(x, labels)
    ax.set_ylim(0, min(1.05, max(vals) * 1.28))
    _clean(ax, ylabel="accuracy", xlabel="fraction of the input that was padding")
    ax.set_title(title or "Accuracy against padding fraction")
    return _save(fig, path)


def padding_attribution(attribution: dict, path: str | Path,
                        title: Optional[str] = None) -> Path:
    """Split the gain from marking padding into representation and padding.

    Moving from a scalar byte input to an embedding with a PAD symbol changes
    two things at once. This figure keeps them apart, against the seed-noise
    band, so a representation gain is not reported as a padding finding.
    """
    fig, ax = plt.subplots(figsize=(5.6, 3.2))
    names = ["input representation\n(scalar -> embedding)", "padding symbol\n(0x00 -> PAD)"]
    vals = [attribution["representation_effect"], attribution["padding_symbol_effect"]]
    noise = attribution["seed_noise"]
    y = np.arange(len(names))
    ax.axvspan(-noise, noise, color=GRID, zorder=0)
    ax.barh(y, vals, height=0.46, color=[SERIES[0], SERIES[1]], linewidth=0, zorder=2)
    for yi, v in zip(y, vals):
        ax.annotate(f"{v:+.3f}", (v, yi), xytext=(6 if v >= 0 else -6, 0),
                    textcoords="offset points", va="center",
                    ha="left" if v >= 0 else "right", fontsize=9, color=INK_2)
    ax.annotate(f"seed noise\n+/-{noise:.3f}", xy=(0, 1.0), xycoords=("data", "axes fraction"),
                xytext=(0, 4), textcoords="offset points", ha="center", va="bottom",
                fontsize=7.5, color=INK_MUTED, linespacing=1.3)
    ax.set_yticks(y, names, fontsize=8)
    ax.axvline(0, color=INK_MUTED, linewidth=1.0)
    _clean(ax, xlabel="change in macro-F1")
    ax.grid(axis="y", visible=False)
    ax.margins(x=0.22)
    ax.set_title(title or "What the padding control is actually measuring", pad=22)
    return _save(fig, path)


def port_dependence(result: dict, path: str | Path, probe: Optional[float] = None,
                    title: Optional[str] = None) -> Path:
    """Macro-F1 with the port bytes against macro-F1 with them masked out.

    Step 3c retains ports, citing P1's ablation. Two measurements decide
    whether that was safe and they can disagree: a ports-only probe may score
    well above chance while masking the ports costs the full model almost
    nothing. That pattern means the information is redundantly encoded rather
    than load-bearing, so the figure plots the paired budgets and marks where
    the difference exceeds seed noise.
    """
    budgets = list(result["budgets"].items())
    fig, ax = plt.subplots(figsize=(6.0, 3.6))
    x = np.arange(len(budgets))
    width = 0.38
    with_ports = [c["bytes"]["macro_f1"]["mean"] for _, c in budgets]
    masked = [c["no_ports"]["macro_f1"]["mean"] for _, c in budgets]

    ax.bar(x - width / 2, with_ports, width, label="ports retained",
           color=SERIES[0], linewidth=0)
    ax.bar(x + width / 2, masked, width, label="ports masked",
           color=SERIES[1], linewidth=0)

    for xi, (_, cell) in zip(x, budgets):
        lost = cell["macro_f1_lost_to_port_masking"]
        ax.annotate(f"{lost:+.3f}" + ("" if cell["conclusive"] else "\nn.s."),
                    (xi, max(with_ports[xi], masked[xi])), xytext=(0, 5),
                    textcoords="offset points", ha="center", va="bottom",
                    fontsize=7.5, color=INK_2, linespacing=1.2)

    # The two reference lines are labelled in the legend rather than inline:
    # either one can land behind a bar, and which one does depends on the
    # corpus, so there is no inline position that is safe for both.
    if probe is not None:
        ax.axhline(probe, color=SERIES[2], linewidth=1.2, linestyle="--",
                   zorder=1, label=f"ports-only probe ({probe:.3f})")
    chance = result.get("chance")
    if chance is not None:
        ax.axhline(chance, color=INK_MUTED, linewidth=1.0, zorder=1,
                   label=f"majority class ({chance:.3f})")

    ax.set_xticks(x, [b for b, _ in budgets])
    ax.set_ylim(0, 1.16)
    _clean(ax, ylabel="macro-F1", xlabel="observation budget")
    ax.legend(frameon=False, fontsize=8, loc="upper center", ncol=2,
              bbox_to_anchor=(0.5, -0.22))
    ax.set_title(title or "What masking the port bytes costs")
    return _save(fig, path)



def split_strategy(result: dict, path: str | Path,
                   title: Optional[str] = None) -> Path:
    """Macro-F1 against where the train/test boundary falls.

    The three arms score the same classes at the same budget with the same
    per-class split sizes, and differ only in the boundary: whole capture
    files, a timestamp inside those captures, or nothing at all. Ordered
    left to right by how much capture and time information the test set
    shares with training, so the bars read as a dose-response rather than as
    three unrelated numbers. Error bars are the seed standard deviation,
    which the group arm needs: assigning whole files makes it the noisiest
    of the three.
    """
    arms = [a for a in ("group", "temporal", "random") if a in result["arms"]]
    labels = {"group": "group-wise\n(unseen capture)",
              "temporal": "temporal\n(same captures)",
              "random": "random\n(no boundary)"}
    means = [result["arms"][a]["macro_f1"]["mean"] for a in arms]
    sds = [result["arms"][a]["macro_f1"]["std"] for a in arms]

    fig, ax = plt.subplots(figsize=(5.6, 3.6))
    x = np.arange(len(arms))
    # The reference arm is the honest one, so it carries the accent colour and
    # the other two are shown as what they add to it.
    colours = [SERIES[2] if a == "group" else SERIES[1] for a in arms]
    ax.bar(x, means, 0.56, color=colours, linewidth=0)
    ax.errorbar(x, means, yerr=sds, fmt="none", ecolor=INK_2,
                elinewidth=1.0, capsize=4)

    for xi, a in zip(x, arms):
        cell = result["arms"][a]
        if a == "group":
            continue
        gain = cell["inflation_macro_f1"]
        tag = f"{gain:+.4f}" + ("" if cell["verdict"] != "inconclusive" else "\nn.s.")
        ax.annotate(tag, (xi, means[int(xi)] + sds[int(xi)]), xytext=(0, 6),
                    textcoords="offset points", ha="center", va="bottom",
                    fontsize=8, color=INK_2, linespacing=1.2)

    chance = result.get("chance_rate")
    if chance is not None:
        ax.axhline(chance, color=INK_MUTED, linewidth=1.0, zorder=1,
                   label=f"chance on {len(result['classes'])} classes "
                         f"({chance:.3f})")
        # Upper left is the only corner all three bars leave free: the group
        # arm is the shortest and the random arm the tallest.
        ax.legend(frameon=False, fontsize=8, loc="upper left")

    ax.set_xticks(x, [labels.get(a, a) for a in arms])
    ax.set_ylim(0, max(m + s for m, s in zip(means, sds)) * 1.28)
    _clean(ax, ylabel="macro-F1", xlabel="train/test boundary")
    ax.set_title(title or "What the split strategy is worth")
    return _save(fig, path)


# ------------------------------------------------------------- 3. cost frontier

def accuracy_per_byte_frontier(fixed: Sequence[dict], adaptive: Sequence[dict],
                               path: str | Path, frontier: Optional[Sequence[dict]] = None,
                               title: Optional[str] = None,
                               callout: Optional[dict] = None) -> Path:
    """Fixed depth against the adaptive early-exit policy, on cost and accuracy.

    The x axis is bytes genuinely read per flow -- for the adaptive policy the
    mean over the test set, capped by each session's real length. Unlike the
    budget profile this axis is linear in cost, because here cost *is* the
    measurement.

    Both series converge on the same top-right point by construction (the
    policy with its threshold raised to 1 is the deepest fixed head), so they
    are identified by legend rather than by end-of-line labels, which would
    overprint each other there. The annotated point is the finding: the
    cheapest threshold that matches the best fixed head.
    """
    fig, ax = plt.subplots(figsize=(6.6, 4.2))
    fixed = sorted(fixed, key=lambda p: p["cost"])
    adaptive = sorted(adaptive, key=lambda p: p["cost"])

    for pts, colour, name, marker in ((fixed, SERIES[0], "fixed depth", "o"),
                                      (adaptive, SERIES[1], "adaptive early exit", "D")):
        ax.plot([p["cost"] for p in pts], [p["macro_f1"] for p in pts], color=colour,
                linewidth=1.8, marker=marker, markersize=6, markeredgecolor=SURFACE,
                markeredgewidth=1.2, label=name, zorder=3)

    if callout is None and fixed and adaptive:
        best = max(fixed, key=lambda p: p["macro_f1"])
        matching = [a for a in adaptive
                    if a["macro_f1"] >= best["macro_f1"] - max(best.get("std", 0.0), 1e-9)
                    and a["cost"] < best["cost"]]
        if matching:
            cheap = min(matching, key=lambda p: p["cost"])
            callout = {"point": cheap, "against": best}

    if callout:
        cheap, best = callout["point"], callout["against"]
        # A guide joining the two points that carry the comparison.
        ax.plot([cheap["cost"], best["cost"]], [cheap["macro_f1"]] * 2,
                color=INK_MUTED, linewidth=1.0, linestyle=(0, (3, 3)), zorder=2)
        ax.scatter([cheap["cost"]], [cheap["macro_f1"]], s=150, facecolor="none",
                   edgecolor=SERIES[1], linewidth=1.6, zorder=4)
        saving = 1.0 - cheap["cost"] / max(best["cost"], 1e-9)
        ax.annotate(
            f"{cheap['label']}: same macro-F1 as {best['label']}\n"
            f"for {cheap['cost']:.0f} bytes instead of {best['cost']:.0f}"
            f"  ({saving:.0%} fewer)",
            xy=(cheap["cost"], cheap["macro_f1"]), xytext=(22, -62),
            textcoords="offset points", fontsize=8.5, color=INK_2, linespacing=1.4,
                bbox=dict(facecolor=SURFACE, edgecolor="none", pad=2.0),
            arrowprops=dict(arrowstyle="-", color=INK_MUTED, linewidth=0.9,
                            shrinkA=0, shrinkB=9))

    _clean(ax, ylabel="macro-F1", xlabel="mean bytes read per flow")
    ax.set_title(title or "Accuracy per byte read")
    ax.legend(loc="lower right", fontsize=8.5)
    ax.margins(x=0.1, y=0.12)
    return _save(fig, path)


# ------------------------------------------- 4. what the aggregate metric hides

def per_class_against_support(result: dict, path: str | Path,
                              label_below: float = 0.5,
                              title: Optional[str] = None) -> Path:
    """Per-class F1 against class support, with the failing classes named.

    Gap 2 of the review: minority classes fail and aggregate metrics conceal
    it. Naming them on the figure is the cheapest possible remedy.
    """
    fig, ax = plt.subplots(figsize=(6.2, 4.0))
    rows = [c for c in result["per_class"] if c["support"] > 0]
    xs = np.array([c["support"] for c in rows], dtype=float)
    ys = np.array([c["f1"] for c in rows])
    ax.scatter(xs, ys, s=46, color=SERIES[0], edgecolor=SURFACE, linewidth=1.2,
               alpha=0.85, zorder=3)
    macro, weighted = result["macro_f1"], result["weighted_f1"]

    # The two aggregates coincide on a balanced corpus. Drawing both lines and
    # both labels then overprints them into an unreadable smudge, so collapse
    # to one line when the gap they exist to show is not there.
    if abs(weighted - macro) < 0.01:
        ax.axhline(macro, color=SERIES[1], linewidth=1.6, zorder=2)
        ax.annotate(f" macro-F1 = weighted-F1 = {macro:.3f}",
                    xy=(1.0, macro), xycoords=("axes fraction", "data"), color=INK_2,
                    fontsize=9, va="center", ha="left", annotation_clip=False)
    else:
        for value, colour, style, name in (
                (macro, SERIES[1], "-", "macro-F1"),
                (weighted, SERIES[2], (0, (4, 3)), "weighted-F1")):
            ax.axhline(value, color=colour, linewidth=1.6, linestyle=style, zorder=2)
            ax.annotate(f" {name} {value:.3f}", xy=(1.0, value),
                        xycoords=("axes fraction", "data"), color=INK_2, fontsize=9,
                        va="center", ha="left", annotation_clip=False)
        ax.annotate(f"gap {weighted - macro:+.3f}\nis what the aggregate hides",
                    xy=(0.02, (macro + weighted) / 2), xycoords=("axes fraction", "data"),
                    fontsize=8, color=INK_MUTED, va="center", ha="left")
    for c, x, y in zip(rows, xs, ys):
        if y < label_below:
            ax.annotate(c["name"], (x, y), xytext=(4, -9), textcoords="offset points",
                        fontsize=8, color=INK_2)
    if len(set(xs.tolist())) > 1:
        ax.set_xscale("log")
    ax.set_ylim(-0.04, 1.04)
    coincident = len(rows) - len({(round(x, 3), round(y, 3)) for x, y in zip(xs, ys)})
    if coincident:
        ax.annotate(f"{len(rows)} classes, {coincident + 1} coincident at the same point",
                    xy=(0.02, 0.04), xycoords="axes fraction", fontsize=8, color=INK_MUTED)
    _clean(ax, ylabel="per-class F1", xlabel="sessions available for the class (log scale)")
    ax.set_title(title or "Per-class F1 against class support")
    return _save(fig, path)


def stratified_bars(strata: dict, path: str | Path, metric: str = "macro_f1",
                    title: Optional[str] = None) -> Path:
    """One group of bars per stratum, so no stratum hides inside an average."""
    fig, ax = plt.subplots(figsize=(7.0, 3.8))
    names = list(strata.keys())
    vals = [strata[k][metric] for k in names]
    ns = [strata[k]["n"] for k in names]
    x = np.arange(len(names))
    under = [strata[k].get("meta", {}).get("underpowered", False) for k in names]
    colours = [INK_MUTED if u else SERIES[0] for u in under]
    ax.bar(x, vals, width=0.6, color=colours, linewidth=0)
    for xi, v, n, u in zip(x, vals, ns, under):
        ax.annotate(f"{v:.3f}\nn={n}" + ("\n(thin)" if u else ""), (xi, v),
                    xytext=(0, 4), textcoords="offset points", ha="center",
                    va="bottom", fontsize=8, color=INK_2)
    ax.set_xticks(x, names, fontsize=8)
    ax.set_ylim(0, min(1.12, max(vals) * 1.34) if vals else 1.0)
    _clean(ax, ylabel=metric.replace("_", "-"))
    ax.set_title(title or "Results by stratum")
    return _save(fig, path)


def load(path: str | Path) -> dict:
    return json.loads(Path(path).read_text())

"""Poster/standee figures: same data, text sized for display viewing.

plots.py sets many annotation sizes explicitly (fontsize=8, 7.5, ...), so
raising rcParams only scales part of the text and leaves callouts unreadable
while pushing tick labels into each other. Shrinking the canvas instead keeps
every label in its designed relative position and makes all of it larger
relative to the figure, which is what matters once the SVG is scaled to a
poster.
"""
import json, pathlib, sys
sys.path.insert(0, "src")
import matplotlib.pyplot as plt

SCALE = 0.78
_orig = plt.subplots
def subplots(*a, **kw):
    if "figsize" in kw and kw["figsize"]:
        w, h = kw["figsize"]
        kw["figsize"] = (w * SCALE, h * SCALE)
    return _orig(*a, **kw)
plt.subplots = subplots

from nettc import plots as P, experiments as X

ROOT = pathlib.Path("/home/minato/Documents/CNET_Project")
OUT = ROOT / "poster" / "figures"

def load(c, name):
    p = ROOT / "results" / c / name
    return json.loads(p.read_text()) if p.exists() else None

BLANK = " "
for c in ("ustc", "iscx"):
    st = load(c, "corpus_stats.json") or {}
    chance = st.get("majority_class_rate") or (1.0 / st["n_classes"] if st.get("n_classes") else None)
    e2, e2z = load(c, "e2_budget_sweep_wang1d.json"), load(c, "e2_budget_sweep_wang1d_zero.json")
    curves = {"learned byte embedding": e2["bytes"]}
    if e2z:
        curves["scalar byte/255"] = e2z["bytes"]
    P.budget_profile(curves, OUT / f"{c}_budget_profile_bytes.pdf", axis="bytes",
                     paper_budgets=X.PAPER_BUDGETS, chance=None,
                     saturation=e2.get("bytes_saturation"), title=BLANK)

    e4 = load(c, "e4_adaptive_frontier.json") or {}
    comp = e4.get("comparison") or {}
    fixed = [{"cost": h["cost"], "macro_f1": h["macro_f1"], "label": h["label"]}
             for h in comp.get("fixed_heads", [])]
    if fixed and comp.get("adaptive_points"):
        P.accuracy_per_byte_frontier(fixed, comp["adaptive_points"],
                                     OUT / f"{c}_accuracy_per_byte_frontier.pdf",
                                     frontier=comp.get("pareto_frontier"), title=BLANK)

ss = load("iscx", "split_strategy.json")
if ss:
    P.split_strategy(ss, OUT / "iscx_split_strategy.pdf", title=BLANK)

pc = load("iscx", "padding_controls_wang1d_1024B.json")
if pc and "mask_attribution" in pc:
    P.padding_attribution(pc["mask_attribution"], OUT / "iscx_padding_attribution.pdf", title=BLANK)

print("regenerated poster figure PDFs")

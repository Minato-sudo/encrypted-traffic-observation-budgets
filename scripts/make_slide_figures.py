"""Phase 6 slide figures: same data as the paper, text sized for a screen.

Same approach as make_poster_figures.py. plots.py sets several annotation
sizes explicitly, so raising rcParams would scale only part of the text;
shrinking the canvas keeps every label in its designed relative position and
makes all of it larger once the SVG is scaled to fill a slide.
"""
import json, pathlib, shutil, sys
sys.path.insert(0, "src")
import matplotlib.pyplot as plt

SCALE = 0.72
_orig = plt.subplots
def subplots(*a, **kw):
    if "figsize" in kw and kw["figsize"]:
        w, h = kw["figsize"]
        kw["figsize"] = (w * SCALE, h * SCALE)
    return _orig(*a, **kw)
plt.subplots = subplots

from nettc import plots as P, experiments as X

ROOT = pathlib.Path("/home/minato/Documents/CNET_Project")
OUT = ROOT / "slides" / "figures"
OUT.mkdir(parents=True, exist_ok=True)
BLANK = " "          # the slide carries the title, so the axes do not

def load(c, name):
    p = ROOT / "results" / c / name
    return json.loads(p.read_text()) if p.exists() else None

# ---- 1. budget response, both corpora ------------------------------------
for c in ("ustc", "iscx"):
    e2 = load(c, "e2_budget_sweep_wang1d.json")
    e2z = load(c, "e2_budget_sweep_wang1d_zero.json")
    curves = {"learned byte embedding": e2["bytes"]}
    if e2z:
        curves["scalar byte/255"] = e2z["bytes"]
    P.budget_profile(curves, OUT / f"{c}_budget_profile_bytes.svg", axis="bytes",
                     paper_budgets=X.PAPER_BUDGETS, chance=None,
                     saturation=e2.get("bytes_saturation"), title=BLANK)

# ---- 2. the architecture control: CNN against byte transformer ----------
ref = load("iscx", "e2_budget_sweep_wang1d.json")
con = load("iscx_transformer", "e2_budget_sweep_transformer.json")
if ref and con:
    P.budget_profile({"Wang 1D CNN": ref["bytes"], "byte transformer": con["bytes"]},
                     OUT / "architecture_budget_profile.svg", axis="bytes",
                     paper_budgets=X.PAPER_BUDGETS, chance=None, title=BLANK)

# ---- 3. accuracy per byte read, the adaptive frontier -------------------
for c in ("ustc", "iscx"):
    comp = (load(c, "e4_adaptive_frontier.json") or {}).get("comparison") or {}
    fixed = [{"cost": h["cost"], "macro_f1": h["macro_f1"], "label": h["label"]}
             for h in comp.get("fixed_heads", [])]
    if fixed and comp.get("adaptive_points"):
        P.accuracy_per_byte_frontier(fixed, comp["adaptive_points"],
                                     OUT / f"{c}_accuracy_per_byte_frontier.svg",
                                     frontier=comp.get("pareto_frontier"), title=BLANK)

# ---- 4. padding attribution --------------------------------------------
pc = load("iscx", "padding_controls_wang1d_1024B.json")
if pc and "mask_attribution" in pc:
    P.padding_attribution(pc["mask_attribution"],
                          OUT / "iscx_padding_attribution.svg", title=BLANK)

# ---- 5. what the split strategy is worth --------------------------------
ss = load("iscx", "split_strategy.json")
if ss:
    P.split_strategy(ss, OUT / "iscx_split_strategy.svg", title=BLANK)

# ---- 7. assets shared with the Phase 5 artifacts ------------------------
for asset in ("pipeline.svg", "qr_repo.svg", "logo_nuces.png"):
    src = ROOT / "poster" / "figures" / asset
    if src.exists():
        shutil.copy2(src, OUT / asset)

for f in sorted(OUT.iterdir()):
    if f.suffix == ".pdf":
        f.unlink()               # _save writes a PDF beside every SVG
print("slide figures in", OUT)
for f in sorted(OUT.iterdir()):
    print("  ", f.name)

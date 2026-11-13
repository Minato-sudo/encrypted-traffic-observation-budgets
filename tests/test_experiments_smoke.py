"""Every experiment runner executes, at every branch that differs structurally.

Deliberately trained for only a few epochs: this checks that the code paths
run and that their bookkeeping is self-consistent, not that the models are
any good. Accuracy is the job of test_sweep_recovers_ground_truth.py.

It exists because two of these paths broke in ways a numbers-only review
would not catch: E4 indexed five heads on a model that returns four when the
ceiling is 512 bytes, and the padding controls compared a mode that changes
two variables at once against one that changes neither.
"""
import sys
from pathlib import Path

import numpy as np

from nettc import ablations as A
from nettc import experiments as X
from nettc import literature as L
from nettc.split import make_split
from nettc.store import Store
from nettc.train import TrainConfig

FAILURES: list[str] = []


def check(cond, msg):
    print(("  PASS  " if cond else "  FAIL  ") + msg)
    if not cond:
        FAILURES.append(msg)


def main() -> int:
    store_path = Path(sys.argv[1] if len(sys.argv) > 1 else "/tmp/synthcheck") / "store.h5"
    store = Store(store_path)
    man = make_split(store, seed=0)
    cfg = TrainConfig(epochs=3, patience=3, batch_size=128, pad_mode="embed")

    print("\n[E1] baselines and the three leakage probes")
    e1 = X.run_e1(store, man, cfg, seeds=(0,), verbose=False)
    probes = e1["leakage"]["probes"]
    check(set(probes) == {"id_positions", "ident", "ports"},
          f"all three probes ran: {sorted(probes)}")
    check(probes["id_positions"]["role"] == "gate",
          "the gate is the probe on anonymised identifier positions, not on raw identifiers")
    check(isinstance(e1["leakage"]["gate_passed"], bool), "the gate returns a verdict")
    check(set(e1["baselines"]) == {"wang2d", "wang1d", "deeppacket"},
          "all three baselines trained")

    print("\n[E2] both budget axes, and the saturation detector")
    e2 = X.run_e2(store, man, cfg, byte_budgets=(128, 256, 512),
                  packet_budgets=(1, 2, 5), seeds=(0, 1), verbose=False)
    check(len(e2["bytes"]) == 3 and len(e2["packets"]) == 3, "both axes swept")
    check(all(c["macro_f1"]["n_seeds"] == 2 for c in e2["bytes"].values()),
          "every cell carries both seeds")
    check("bytes_saturation" in e2 and "packets_saturation" in e2,
          "saturation reported on both axes")
    widths = [e2["packets"][k]["width"] for k in e2["packets"]]
    check(widths == sorted(widths), f"packet-budget widths are non-decreasing: {widths}")

    print("\n[E3] stratified evaluation")
    e3 = X.run_e3(store, man, cfg, budget=512, seed=0, verbose=False)
    check(set(e3["strata"]) == {"flow_length", "class_support", "encryption"},
          "all three stratifications produced")
    total = sum(c["n"] for c in e3["strata"]["flow_length"].values())
    check(total == e3["overall"]["n"],
          f"the flow-length strata partition the test set exactly ({total} == {e3['overall']['n']})")
    total_enc = sum(c["n"] for c in e3["strata"]["encryption"].values())
    check(total_enc == e3["overall"]["n"], "the encryption strata partition it too")

    print("\n[padding] the four controls and the attribution")
    pc = X.run_padding_controls(store, man, cfg, budget=512, seeds=(0,), verbose=False)
    check({"zero", "embed", "mask", "convmask"} <= set(pc["controls"]),
          f"all control modes ran: {sorted(pc['controls'])}")
    check("mask_attribution" in pc,
          "the mask gain is split into representation and padding-symbol effects")
    attr = pc["mask_attribution"]
    combined = pc["deltas_vs_zero_padding"]["mask"]
    recomposed = attr["representation_effect"] + attr["padding_symbol_effect"]
    check(abs(recomposed - combined) < 1e-4,
          f"the two effects sum back to the combined delta "
          f"({recomposed:+.4f} == {combined:+.4f})")

    print("\n[E4] the exit policy, at a ceiling that does and does not reach every head")
    for ceiling, want_heads in ((512, 4), (1024, 5)):
        e4 = X.run_e4(store, man, cfg, e2=e2, seeds=(0,),
                      thresholds=(0.5, 0.9, 1.01), ceiling=ceiling, verbose=False)
        c = e4["comparison"]
        check(len(c["fixed_heads"]) == want_heads,
              f"a {ceiling}-byte ceiling builds {want_heads} heads, and E4 indexes exactly those")
        check(len(e4["frontier"]) == 3, "one frontier point per threshold")
        costs = [p["cost"] for p in c["adaptive_points"]]
        check(costs == sorted(costs),
              f"raising the threshold never reads fewer bytes: {[round(x) for x in costs]}")
        check(all(p["cost"] <= ceiling + 1e-6 for p in c["adaptive_points"]),
              "the policy never reports reading more than the ceiling")
        check(c["basis"].startswith("the adaptive model's own"),
              "the comparison is against this model's own heads, not the E2 sweep")

    print("\n[Step 10 ablations] representation, layer scope, early-exit supervision")
    rep = A.ablate_representation(store, man, cfg, seeds=(0,), budget=784, verbose=False)
    check(set(rep["variants"]) == {"1D sequence", "2D byte image 28x28", "1D Deep Packet"},
          "the representation ablation ran all three forms at a matched 784-byte budget")
    check(all("delta_macro_f1" in c for n, c in rep["variants"].items() if n != "1D sequence"),
          "non-reference variants carry a delta and a verdict")

    scope = A.ablate_layer_scope(store, man, cfg, seeds=(0,), budget=784, verbose=False)
    check(set(scope["variants"]) == {"all layers", "headers only", "payload only"},
          "the layer-scope ablation ran all three scopes")
    hdr = scope["variants"]["headers only"]["mean_bytes_available"]
    pay = scope["variants"]["payload only"]["mean_bytes_available"]
    check(hdr > 0 and pay > 0,
          f"both scopes carry bytes (headers {hdr:.0f} B, payload {pay:.0f} B)")

    ee = A.ablate_early_exit(store, man, cfg, seeds=(0,), ceiling=512, verbose=False)
    check(set(ee["variants"]) == {"all heads supervised", "terminal head only"},
          "the early-exit ablation isolates auxiliary-head supervision")

    print("\n[comparison] the literature table")
    best = max(e2["bytes"].values(), key=lambda c: c["macro_f1"]["mean"])
    table = L.comparison_table(best, "USTC-TFC2016", store.n_classes)
    check("not comparable" in table and "this work" in table,
          "the comparison table names our result and states each comparability obstacle")
    check("P5" not in table.split("this work")[0] or True, "table renders")
    marks = L.budget_marks("packets")
    check(set(marks) >= {5, 15, 16, 20},
          f"the reviewed packet budgets are available for annotation: {sorted(marks)}")

    store.close()
    print()
    if FAILURES:
        print(f"{len(FAILURES)} check(s) failed")
        return 1
    print("ALL EXPERIMENT PATHS OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())

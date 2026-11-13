"""Does the budget sweep recover a cue whose depth we planted ourselves?

The sweep is the study's primary instrument, so it needs a positive control.
This builds a corpus where the answer is known by construction:

* the retained destination port gives away one bit (class parity), visible
  in the first 64 bytes;
* a weak cue (coarse family, 3 values) sits in packet 0 at byte 78, so with
  parity it identifies the class modulo 6 -- about 0.875 accuracy over the
  eight classes, two of which stay ambiguous;
* a strong cue (exact class) sits in packet 4, landing between bytes 780 and
  1000, so only a budget past ~1000 can resolve the remaining pair;
* ``min_packets=6``, so nearly every flow actually reaches the late cue.

A correct sweep must therefore trace chance -> ~0.875 -> 1.000 as the budget
grows past 78 and then past 1000 bytes. The cue positions are verified
directly in the store by ``test_store_contains_planted_cues``, so a failure
here is a failure of the sweep, not of the generator.

The run is repeated under two input representations, because which one is
used decides whether the late cue is reachable at all. That contrast is a
finding in its own right and is asserted here so it cannot regress silently.
"""
import dataclasses
import sys
from pathlib import Path

import numpy as np

from nettc.experiments import ExperimentPaths, run_e1, run_e2
from nettc.sessionize import sessionize
from nettc.split import make_split
from nettc.store import Store
from nettc.synth import SyntheticCorpus
from nettc.train import TrainConfig

FAILURES: list[str] = []


def check(cond: bool, msg: str) -> None:
    print(("  PASS  " if cond else "  FAIL  ") + msg)
    if not cond:
        FAILURES.append(msg)


def test_store_contains_planted_cues(store) -> None:
    """The generator's cues must be physically present at the expected offsets."""
    print("\n[store] planted cues")
    b = store.get_bytes(np.arange(store.n))
    po = store.pkt_offsets

    weak_ok = True
    for cls in range(store.n_classes):
        m = store.label == cls
        want = 0xA0 + (cls % 3)
        weak_ok &= bool((b[m, 78] == want).mean() > 0.95)
    check(weak_ok, "the weak cue is at byte 78 in >95% of every class's sessions")

    strong_ok = True
    for cls in range(store.n_classes):
        idx = np.flatnonzero(store.label == cls)[:300]
        start = po[idx, 3] + 54 + 8          # end of packet 4, past headers, payload offset 8
        hit = [bool(st + 2 <= store.byte_ceiling and b[i, st] == 0xE0 and b[i, st + 1] == cls)
               for i, st in zip(idx, start)]
        strong_ok &= bool(np.mean(hit) > 0.95)
    check(strong_ok, "the strong cue is where the packet offsets say it should be, "
                     "in >95% of sessions")

    mean_late = float((po[:, 3] + 62).mean())
    check(780 <= mean_late <= 1100,
          f"the strong cue lands between bytes 780 and 1100 (mean {mean_late:.0f}), "
          f"so only budgets past ~1000 can use it")


def main() -> int:
    out = Path(sys.argv[1] if len(sys.argv) > 1 else "/tmp/synth_gt")
    seeds = (0, 1)
    budgets = (64, 128, 256, 512, 784, 1024, 1500)

    if not (out / "store.h5").exists():
        corpus = SyntheticCorpus(n_classes=8, sessions_per_class=500, strong_cue_packet=4,
                                 length_mu=2.1, length_sigma=0.5, min_packets=6, seed=7)
        caps = corpus.write(out)
        names = corpus.class_names()
        stats = sessionize(caps, out / "store.h5", names, lambda p, g: names.index(g))
        print(f"sessions written: {stats.sessions_written}")

    store = Store(out / "store.h5")
    chance = 1.0 / store.n_classes
    print(f"flows reaching packet 5: {(store.packet_count >= 5).mean():.3f}")
    test_store_contains_planted_cues(store)

    man = make_split(store, seed=0)
    base = TrainConfig(epochs=40, patience=8, batch_size=128)
    paths = ExperimentPaths(out / "results")

    print("\n[E1] leakage probes")
    e1 = run_e1(store, man, dataclasses.replace(base, pad_mode="embed"),
                seeds=seeds, paths=paths, verbose=False)
    probes = e1["leakage"]["probes"]
    check(e1["leakage"]["gate_passed"],
          f"gate: anonymised identifier positions give "
          f"{probes['id_positions']['accuracy']['mean']:.3f} vs chance {chance:.3f}")
    check(probes["ident"]["accuracy"]["mean"] > 0.5,
          f"the same positions before anonymisation give "
          f"{probes['ident']['accuracy']['mean']:.3f}, so the gate tests something real "
          f"rather than a vacuously zeroed input")
    check(probes["ports"]["accuracy"]["mean"] > chance + 0.05,
          f"the retained port bytes alone give {probes['ports']['accuracy']['mean']:.3f}, "
          f"recovering the parity bit the generator put in the destination port")

    curves = {}
    for mode in ("embed", "zero"):
        cfg = dataclasses.replace(base, pad_mode=mode)
        e2 = run_e2(store, man, cfg, model_name="wang1d", byte_budgets=budgets,
                    packet_budgets=(1, 2, 5, 20), seeds=seeds, paths=None, verbose=False)
        ys = [e2["bytes"][f"{b}B"]["macro_f1"]["mean"] for b in budgets]
        curves[mode] = ys
        print(f"\n[E2] pad_mode={mode}: macro-F1 by byte budget")
        for b, y, sd in zip(budgets, ys,
                            [e2["bytes"][f"{b}B"]["macro_f1"]["std"] for b in budgets]):
            print(f"  {b:>5}B  {y:.4f} +/- {sd:.4f}")
        print(f"  saturation: {e2['bytes_saturation']}")
        if mode == "embed":
            e2_embed = e2

    print("\n[ground truth] the sweep must trace chance -> ~0.875 -> 1.000")
    ys = curves["embed"]
    check(ys[0] < 0.30, f"64B is at chance ({ys[0]:.3f}): no cue is visible before byte 78")
    check(ys[1] > ys[0] + 0.30, f"128B jumps to {ys[1]:.3f}: the byte-78 cue is picked up")
    check(0.75 <= ys[1] <= 0.95,
          f"128B lands near the predicted modulo-6 ceiling of 0.875 (got {ys[1]:.3f})")
    check(max(ys[-2:]) > ys[1] + 0.05,
          f"the largest budgets exceed the modulo-6 ceiling "
          f"({max(ys[-2:]):.3f} vs {ys[1]:.3f}): the late cue resolves the last pair")
    check(max(ys) > 0.95, f"the sweep reaches {max(ys):.3f}, essentially solving the task")
    check(int(np.argmax(ys)) >= len(budgets) - 2,
          f"the best budget is one of the two largest (it is {budgets[int(np.argmax(ys))]}B), "
          f"consistent with a cue planted past byte 780")
    check(e2_embed["bytes_saturation"].get("at") is None,
          "the curve is correctly reported as still rising at the largest budget swept")

    print("\n[representation] the scalar baseline input cannot reach the late cue")
    zs = curves["zero"]
    check(max(zs) < 0.60,
          f"scalar byte/255 input plateaus at {max(zs):.3f}: it cannot match exact byte "
          f"values, because class 3 and class 4 differ by 0.004 after scaling")
    check(max(ys) - max(zs) > 0.3,
          f"the embedding input beats it by {max(ys) - max(zs):.3f} macro-F1, so the shape "
          f"of a budget curve depends on the representation and not only on the budget")

    store.close()
    print()
    if FAILURES:
        print(f"GROUND TRUTH NOT RECOVERED -- {len(FAILURES)} check(s) failed")
        return 1
    print("GROUND TRUTH RECOVERED -- all checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())

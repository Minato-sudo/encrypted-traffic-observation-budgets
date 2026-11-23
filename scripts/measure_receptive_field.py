"""Measure the early-exit encoder's receptive field empirically.

The paper's exit-honesty argument rests on two properties of the encoder:
a perturbed byte must reach *no* earlier position, and it must reach exactly
as far forward as the dilation schedule predicts. The analytic figure is

    1 + 2 * (kernel_size - 1) * sum(dilations) = 1 + 2 * 4 * 63 = 505 bytes

for six blocks of two kernel-5 convolutions at dilations 1, 2, 4, 8, 16, 32.
This script perturbs one byte in the middle of the sequence and reports how
far the change actually travels in each direction, so the 505 in the paper is
a measurement rather than arithmetic on paper.

    PYTHONPATH=src python3 scripts/measure_receptive_field.py
"""
import sys
sys.path.insert(0, "src")
import torch
from nettc.models import build_model

POS, LENGTH, TOL = 100, 1024, 1e-6

torch.manual_seed(0)
model = build_model("adaptive", n_classes=8).eval()

x = torch.randint(0, 256, (1, LENGTH), dtype=torch.uint8)
x_pert = x.clone()
x_pert[0, POS] = (int(x[0, POS]) + 128) % 256          # a byte that certainly differs

with torch.no_grad():
    delta = (model.encode(x) - model.encode(x_pert)).abs().amax(dim=1)[0]

moved = (delta > TOL).nonzero().flatten()
first, last = int(moved.min()), int(moved.max())

print(f"perturbed byte          : position {POS}")
print(f"positions changed       : {first} to {last}")
print(f"backward reach          : {max(0, POS - first)} bytes   (must be 0)")
print(f"forward  reach          : {last - POS + 1} bytes")
print(f"analytic receptive field: 1 + 2*(5-1)*(1+2+4+8+16+32) = 505 bytes")
ok = first == POS and (last - POS + 1) == 505
print(f"\n{'PASS' if ok else 'FAIL'}  causal, and the reach matches the analysis")
raise SystemExit(0 if ok else 1)

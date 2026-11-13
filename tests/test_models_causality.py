import torch, numpy as np
from nettc.models import build_model, count_parameters, DEFAULT_EXITS

torch.manual_seed(0)
m = build_model("adaptive", n_classes=8).eval()
print("adaptive params:", count_parameters(m))

x_full = torch.randint(0, 256, (6, 1024), dtype=torch.uint8)

# Causality test 1: head k's logits must not change when bytes beyond its
# budget are replaced with noise.
with torch.no_grad():
    a = m(x_full)
x_pert = x_full.clone()
x_pert[:, 256:] = torch.randint(0, 256, (6, 1024-256), dtype=torch.uint8)
with torch.no_grad():
    b = m(x_pert)
for k, bud in enumerate(DEFAULT_EXITS):
    d = (a[k]-b[k]).abs().max().item()
    flag = "OK " if (d < 1e-5) == (bud <= 256) else "FAIL"
    print(f"{flag} head@{bud:>5}B  max|delta| after perturbing bytes>=256: {d:.3e}")

# Causality test 2: head k fed a TRUNCATED input must match head k fed the
# full input. This is what makes 'bytes read' honest.
print()
for bud in (64, 128, 256, 512):
    with torch.no_grad():
        trunc = m(x_full[:, :bud])
    k = DEFAULT_EXITS.index(bud)
    d = (a[k] - trunc[-1]).abs().max().item()
    print(f"{'OK ' if d < 1e-4 else 'FAIL'} head@{bud:>5}B  truncated-input vs full-input max|delta|: {d:.3e}")

# Other models: shape checks and masked pooling path
print()
for name, kw, width in [("wang2d", {"n_bytes":784}, 784), ("wang1d", {}, 512),
                        ("deeppacket", {}, 1500), ("transformer", {}, 512)]:
    for pad in ("zero", "mask", "convmask"):
        mm = build_model(name, 8, pad_mode=pad, **kw).eval()
        xx = (torch.randint(0,257,(4,width),dtype=torch.int16) if pad=="mask"
              else torch.randint(0,256,(4,width),dtype=torch.uint8))
        vl = torch.tensor([width, width//2, 10, width])
        with torch.no_grad():
            out = mm(xx, vl if pad=="convmask" else None)
        print(f"OK  {name:<11} pad={pad:<9} out {tuple(out.shape)} params {count_parameters(mm):>9,}")

# Early-exit policy
print()
m2 = build_model("adaptive", 8, pad_mode="convmask").eval()
vl = torch.tensor([40., 300., 900., 1024., 70., 2000.])
pred, used, eidx = m2.early_exit(x_full, vl, threshold=0.5)
print("early_exit pred      :", pred.tolist())
print("early_exit bytes_read:", used.tolist())
print("early_exit exit_index:", eidx.tolist())
print("bytes_read never exceeds true length:", bool((used <= torch.minimum(vl, torch.tensor(1024.))).all()))

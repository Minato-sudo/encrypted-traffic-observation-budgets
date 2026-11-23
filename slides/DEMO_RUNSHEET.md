# Phase 6: live demonstration run sheet

Everything on this sheet has been run on this machine and the outputs below are
what it actually printed. Nothing here trains a model, so nothing can stall in
front of the camera: **the whole sequence takes under 20 seconds of compute, and
about 1 minute 20 seconds to present.**

Presented by **Muhammad Talha Arshad (i232548)** on slide 11, which is the last
slide of his half. He runs the terminal, then hands over to Zain for the results
on slide 12, so the demonstration closes the first half of the talk.

---

## Before recording

```bash
cd /home/minato/Documents/CNET_Project
export PY=/home/minato/.local/share/pipx/venvs/jupyterlab/bin/python3
clear
```

Terminal setup, which matters as much as the commands:

- **Font size at least 16pt.** Anything smaller is unreadable after YouTube
  re-encodes it. Check by playing a 10-second test upload at 1080p.
- **Light-on-dark, high contrast.** The deck is light, so a dark terminal makes
  the cut between slides and terminal obvious to the viewer.
- Window **maximised**, and nothing else on screen. No file manager, no chat,
  no notification popups. Turn on Do Not Disturb.
- Run each command once **before** recording so that Python import time does not
  appear as a pause on camera.

---

## The sequence

### 1. The receptive field is 505 bytes, measured (about 1 s)

```bash
PYTHONPATH=src $PY scripts/measure_receptive_field.py
```

```
perturbed byte          : position 100
positions changed       : 100 to 604
backward reach          : 0 bytes   (must be 0)
forward  reach          : 505 bytes
analytic receptive field: 1 + 2*(5-1)*(1+2+4+8+16+32) = 505 bytes

PASS  causal, and the reach matches the analysis
```

**Say:** one byte changed at position 100; positions 100 to 604 move, so forward
reach is 505 bytes, exactly the analytic figure, and backward reach is zero.

### 2. Exit honesty: a head on a truncated input matches the full input (about 1 s)

```bash
PYTHONPATH=src $PY tests/test_models_causality.py
```

The block to point at is the second one:

```
OK  head@   64B  truncated-input vs full-input max|delta|: 0.000e+00
OK  head@  128B  truncated-input vs full-input max|delta|: 0.000e+00
OK  head@  256B  truncated-input vs full-input max|delta|: 0.000e+00
OK  head@  512B  truncated-input vs full-input max|delta|: 0.000e+00
```

**Say:** head *k* fed only its own prefix gives logits identical to head *k* fed
the whole sequence. That is what makes "bytes read" honest rather than nominal.

> If asked about the *first* block, where head@512B shows a non-zero delta after
> bytes past 256 are perturbed: that is correct and expected. A 512-byte head is
> *supposed* to see bytes 256 to 512. The test asserts zero delta only for the
> heads whose budget ends before the perturbation.

### 3. The one line the guarantee rests on

```bash
sed -n '170,189p' src/nettc/models.py
```

Point at the docstring and at `mean = h.mean(dim=1, keepdim=True)`.

**Say:** this normalises over channels at each position independently. BatchNorm
and GroupNorm both average over the length axis, which would let a byte at
position 900 shift the activation at position 10. Swapping this one layer leaves
every accuracy number looking plausible while making the byte saving a fiction,
which is exactly why it is guarded by a regression test.

Optionally also show the block it sits in:

```bash
sed -n '191,208p' src/nettc/models.py
```

### 4. The budget sweep, read from its own JSON

```bash
jq -r '.bytes | to_entries[] | "\(.key)  \(.value.macro_f1.mean)"' \
   results/ustc/e2_budget_sweep_wang1d.json
```

```
64B  0.9187809992407985
128B  0.942645241626114
256B  0.9652004811000955
512B  0.973062942429153
784B  0.9563971547319187
1024B  0.9539869958907042
1500B  0.9367464205538943
```

**Say:** 0.9731 at 512 bytes, then 0.9564 at 784 and 0.9367 at 1500. The curve
turns over. These are the numbers on the results slide, not a retyping of them.

### 5. And that the fall is significant

```bash
jq -c '.pairs.saturation.mcnemar' results/ustc/significance.json
```

```
{"n01":94,"n10":2488,"p_value":0.0,"n_discordant":2582}
```

**Say:** `n10` is 2488, the sessions correct at 512 bytes and wrong at 784.
`n01` is 94, the other direction. McNemar's exact *p* underflows to zero in
double precision, which we report as below ten to the minus four.

### 6. The report is generated, not written

```bash
sed -n '126,142p' results/ustc/REPORT.md
```

```
| byte budget | macro-F1 | accuracy | mean padding fraction |
|---|---|---|---|
| 64B | 0.9188 ± 0.0047 | 0.9560 ± 0.0067 | 0.000 |
...
| 512B | 0.9731 ± 0.0006 | 0.9726 ± 0.0112 | 0.242 |
| 784B | 0.9564 ± 0.0018 | 0.9407 ± 0.0154 | 0.307 |
| 1500B | 0.9367 ± 0.0142 | 0.9205 ± 0.0088 | 0.398 |

**Saturation:** 512B: no larger budget improves on it by more than seed noise (0.0142).
```

**Say:** `make_report.py` produces every table in here from the JSON we just
looked at, including the saturation verdict and the seed-noise threshold it is
tested against. No number in the paper, the poster or this deck is typed by
hand.

---

## Optional extras, only if you are recording the longer version

These are **outside** the 15-minute timing. Skip both unless you have decided to
run long.

### The ground-truth test (about 10 s, CPU)

```bash
PYTHONPATH=src $PY tests/test_pipeline_endtoend.py /tmp/synthcheck
```

Runs the whole pipeline on synthetic captures with known labels. Worth showing
if a marker is likely to ask "how do you know the pipeline is right?".

### The project tree

```bash
ls src/nettc/
```

Fifteen modules, matching the table on slide 10.

---

## Do not do these on camera

- **Do not start a training run.** The USTC study is 40 epochs over 353,496
  sessions. It will not finish during the video and a half-rendered progress bar
  looks worse than no demo at all.
- **Do not open the 55 GB `data/` directory.** It is not in the repository and
  listing it proves nothing.
- **Do not run `study.sh`.** Same reason as the first point.
- **Do not alt-tab to a browser** to show the GitHub page. Slide 23 already
  carries the URL and the QR code, and a live page load is a risk with no payoff.

---

## If something fails on camera

Keep going and say what happened. A line such as "that import is slow on this
machine, the result is in `results/ustc/REPORT.md` and I will show it there" is
a better recovery than a cut, and markers generally read it as familiarity with
the code rather than as a fault. Then re-record that segment afterwards if the
pause was long.

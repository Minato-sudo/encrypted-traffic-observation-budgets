# How Much Traffic Is Enough?

Implementation of *Observation Budgets and Padding Effects in Raw-Byte Network
Traffic Classification* (Topic 15, FAST-NUCES Islamabad).

Muhammad Talha Arshad (i232548), Zain Shahid (i232582)

The Phase 1 literature review found that all twelve reviewed
studies fix an observation budget (between 5 and 20 packets, or between 784
and 1024 bytes) and that **none of those values is derived from
measurement**. P7 states plainly that its budget was chosen to fit a model's
512-token ceiling. P8's own statistics imply that at a 1024-byte budget
roughly 87% of ISCX-VPN sessions and 70% of USTC-TFC sessions are *padded*
rather than truncated, which raises the possibility that a large body of
reported accuracy is partly measuring padding.

This repository treats the budget and the padding fraction as measured
variables rather than preprocessing constants.

## What it does

Preprocessing runs **once**, at a generous 2048-byte ceiling, into an HDF5
session store. Every budget in the sweep is then served by slicing that store,
so architecture, optimiser, schedule, seed and split are identical across
budgets and the only thing that changes is how many bytes the model may see.

Each session record keeps the four independent variables the study needs:
true byte length, packet count, padding fraction, and encryption stratum,
plus the per-packet offsets that let a *packet* budget be cut from the same
bytes, and the identifier bytes anonymisation removed, for the leakage
control.

## Layout

```
src/nettc/
  pcapio.py       streaming pcap and pcapng readers
  dissect.py      five-tuple keying, anonymisation, field offsets
  encryption.py   TLS cipher-suite tagging and encryption strata (after P5)
  sessionize.py   two-pass session assembly, cleaning, SHA-256 dedupe
  store.py        the HDF5 session store
  split.py        group-wise 70/15/15 with a temporal fallback
  slicer.py       byte and packet budget slicing, padding controls
  models.py       Wang 2D/1D, Deep Packet, and the adaptive early-exit CNN
  train.py        one training loop for every model and budget
  metrics.py      per-class, stratified, McNemar, cost frontier
  experiments.py  E1-E4 and the padding controls
  plots.py        the Step 12 figures
  datasets.py     USTC-TFC2016 and ISCX VPN-nonVPN adapters
  synth.py        synthetic captures with known ground truth
scripts/
  get_datasets.sh fetch and verify the corpora
  build_store.py  captures -> session store
  run_study.py    store -> results and figures
tests/            verification, including a ground-truth positive control
```

## Running it

There is no project virtualenv. The working interpreter is the pipx-managed
jupyterlab environment, which is what `python3` resolves to on `PATH` and
which holds torch with CUDA.

```bash
PY=/home/minato/.local/share/pipx/venvs/jupyterlab/bin/python3
```

Verify the pipeline before anything else. All four scripts run on generated
captures and need no corpus:

```bash
PYTHONPATH=src $PY tests/test_pipeline_endtoend.py /tmp/synthcheck
PYTHONPATH=src $PY tests/test_models_causality.py
PYTHONPATH=src $PY tests/test_split_and_slicer.py /tmp/synthcheck
PYTHONPATH=src $PY tests/test_sweep_recovers_ground_truth.py /tmp/synth_gt
```

Then, with a corpus in place:

```bash
bash scripts/get_datasets.sh check
PYTHONPATH=src $PY scripts/build_store.py --corpus ustc --root data/USTC-TFC2016 \
    --out stores/ustc.h5 --checksums
PYTHONPATH=src $PY scripts/run_study.py --store stores/ustc.h5 --out results/ustc \
    --compare-representations
```

`run_study.py` halts if the leakage gate fails.

Both corpora are driven by one pauseable runner, which is resumable at stage
granularity: each stage writes one JSON and a re-run skips whatever is already
there:

```bash
bash scripts/study.sh ustc            # run or resume
bash scripts/study.sh iscx --status   # what is done, what is left
bash scripts/study.sh iscx --pause    # freeze; --continue to unfreeze
bash scripts/study.sh iscx --stop     # stop cleanly before suspend or reboot
```

The write-ups are generated, never hand-edited. `corpus_stats.py` records what
the store says about the corpus itself (encryption strata, padding fraction per
budget, per-class capture counts and the split strategy each class ended up
with); `make_report.py` turns one results directory into that corpus's
`REPORT.md`; `make_cross_corpus.py` writes the side-by-side block at the end of
this file:

```bash
PYTHONPATH=src $PY scripts/corpus_stats.py --store stores/ustc.h5 \
    --corpus ustc --out results/ustc
PYTHONPATH=src $PY scripts/make_report.py --results results/ustc \
    --out results/ustc/REPORT.md
PYTHONPATH=src $PY scripts/make_cross_corpus.py \
    --results results/ustc results/iscx
```

Two analyses sit outside `study.sh` because they are cross-cutting rather than
per-stage. `run_significance.py` runs McNemar's exact test on the two genuinely
paired comparisons; `run_split_strategy.py` answers the study's largest
limitation by holding the classes, budget, model, epochs, seeds and per-class
split sizes fixed and varying only where the train/test boundary falls. It
refuses to run on a corpus with too few group-splittable classes, which USTC is:

`audit_corpus_archives.py` settles a question the class list cannot: whether a
missing category is a bad download or an incomplete release. It reads the
distributed archives directly and compares them against what was extracted. On
ISCX all 140 declared captures are present, and the only p2p capture in the
whole distribution is the VPN one, so the 11-class list is the corpus's
ceiling, not a download error:

```bash
PYTHONPATH=src $PY scripts/audit_corpus_archives.py --archives data \
    --extracted data/ISCX-VPN-nonVPN --out results/iscx --corpus iscx
```

```bash
PYTHONPATH=src $PY scripts/run_significance.py --corpus iscx
PYTHONPATH=src $PY scripts/run_split_strategy.py --corpus iscx \
    --seeds 0 1 2 3 4 5 6 7
```

`run_architecture_compare.py` asks whether the budget curve belongs to the data
or to the model reading it. Every baseline here is a 2017-2020 convolutional
network while every reviewed paper is a transformer or a graph network, so the
whole E2 sweep was re-run on ISCX with `models.ByteTransformer`: same store,
same split manifest, same schedule, same seeds, only the model changed. The two
families are not separable (0.5245 ± 0.0208 against 0.5195 ± 0.0379) and
saturate at the same packet budget, so the finding is not an artefact of
convolution. The transformer is trained from scratch and its absolute scores
are not comparable to the pre-trained models it imitates:

```bash
PYTHONPATH=src $PY scripts/run_study.py --store stores/iscx.h5 \
    --out results/iscx_transformer --model transformer --stages e2 \
    --epochs 15 --max-per-class 5000 --pad-mode embed --corpus-name iscx
PYTHONPATH=src $PY scripts/run_architecture_compare.py \
    --reference results/iscx --reference-model wang1d \
    --contender results/iscx_transformer --contender-model transformer \
    --corpus-name "ISCX VPN-nonVPN" --out results/iscx
```

The sweep writes to its own results directory on purpose: `run_study.py`
rewrites `run_config.json` unconditionally, so pointing it at `results/iscx`
would overwrite the main study's metadata with the transformer's.

The single submission document is assembled from both corpora by
`make_submission.py`, which answers the seven required components once with the
two corpora side by side and carries each per-corpus report as an appendix.
`render_html.py` turns it into one self-contained print-ready page (every figure
embedded as a data URI), which prints to PDF from any browser:

```bash
PYTHONPATH=src $PY scripts/make_submission.py \
    --results results/ustc results/iscx --out Phase3_Implementation.md
$PY scripts/render_html.py --markdown Phase3_Implementation.md --base .
```

`render_html.py` needs the pure-Python `markdown` package (`pip install
markdown`); nothing else in the project does.

## Three things the implementation established

**The early-exit heads are causally wired, and it is tested.** Head *k*
returns bit-identical logits whether it is fed `exits[k]` bytes or the full
2048, so the accuracy-per-byte frontier measures bytes genuinely read rather
than bytes read and then masked. This broke on the first attempt because
`GroupNorm` computes statistics across the length axis; `CausalBlock` now uses
`ChannelNorm1d`, which normalises within a position. Substituting GroupNorm or
BatchNorm back in silently destroys the property the contribution rests on.
`tests/test_models_causality.py` is the regression check.

**The padding control needs an attribution control of its own.** Moving from a
scalar `byte/255` input to an embedding with a dedicated PAD symbol changes
two things at once. On the synthetic corpus that combination is worth +0.519
macro-F1, of which the representation accounts for +0.519 and the padding
symbol for **+0.000**, inside seed noise. Reporting the combined figure as a
padding result would have been wrong by the whole effect. `pad_mode="embed"`
exists only to separate them.

**The input representation decides whether depth pays off at all.** On a corpus
with a cue planted past byte 780, an embedding input traces chance → 0.837 at
128 bytes → 1.000 at 1500 bytes, recovering the planted structure exactly. The
scalar `byte/255` input used by the byte-image baselines plateaus at 0.492 and
never finds the late cue: it cannot match exact byte values, because after
scaling, class 3 and class 4 differ by 0.004. A budget curve therefore has no
meaning independent of the representation it was measured with, and budget
figures quoted across papers that differ in representation are not comparable.

## On the corpora

Following P5, every session is tagged with its TLS cipher suite before
training and every result can be split into encrypted and plaintext strata.
P5 measured ISCXVPN2016 at 98.9% unencrypted and USTC-TFC2016 at 94.7%
unencrypted. If that reproduces here, then a single averaged number does not
describe encrypted traffic classification, and `run_e3` reports the strata
separately so it cannot be quoted as though it did.

<!-- BEGIN cross-corpus (generated by scripts/make_cross_corpus.py) -->

## What the two corpora say together

Each corpus has its own report under `results/`. This is the comparison those reports defer to. Every row is generated from the JSON in both directories.

|  | **USTC-TFC2016** | **ISCX VPN-nonVPN** |
| --- | --- | --- |
| sessions | 353,496 | 31,855 |
| classes | 20 | 11 |
| byte budget where the sweep saturates | **512B** | **256B** |
| packet budget where the sweep saturates | **5P** | **2P** |
| best macro-F1 over the whole sweep | 0.9731 | 0.5245 |
| macro-F1 at the smallest budget tried | 0.9188 | 0.4643 |
| padded rather than truncated at 1024 B | 60.3% | 87.9% |
| definitely not encrypted | 41.3% | 27.6% |
| confirmed TLS | 7.8% | 7.4% |
| of those handshakes, deprecated suites | 95.2% | 13.5% |
| ports-only probe against chance | 0.7509 vs 0.1359 | 0.3019 vs 0.2258 |
| largest macro-F1 lost to masking ports | +0.0428 at 64B | +0.0040 at 784B |
| concealment gap (weighted − macro F1) | +0.0133 | -0.1313 |
| classes below 0.05 F1 | none | none |
| classes split temporally, not group-wise | 19 of 20 | 5 of 11 |
| macro-F1 with flow length equalised | 0.7907 (chance 0.1429, n=168) | 0.1106 (chance 0.1667, n=84) |
| session against unidirectional flow | +0.0045 (inconclusive) | -0.0976 (session better) |
| padding: representation effect | +0.0138 | +0.2408 |
| padding: PAD-symbol effect | +0.0065 | +0.0086 |

### What replicates

**Every reviewed budget sits at or past saturation, on both corpora.** The saturation point differs, in that the harder corpus saturates *earlier*, not later, but no paper budget falls below it in either case. This is the study's main result and it is the one that replicates most cleanly.

**The padding gain is the representation, not the padding symbol, where the effect is large enough to tell.** On USTC-TFC2016 the whole effect is +0.0203 against 0.0176 of seed noise, too small to split; on ISCX VPN-nonVPN the representation is worth +0.2408 and the padding symbol +0.0086, against 0.0373 of seed noise. So the attribution is carried by ISCX VPN-nonVPN, where the effect is big enough to decompose. Elsewhere the control returns a null result: that is not evidence the padding symbol helps, it is the absence of evidence either way, and it is reported as such rather than pooled into an average that would look conclusive.

**Most of what the model reads at a literature budget is padding, and nobody reports it.** The ISCX figure matches P8's implied ~87% from entirely independent preprocessing, which is the closest thing here to an external check on the pipeline.

**The leakage gate passes on both**, so anonymisation left no usable identifier signal, and on both the ports shortcut exists but is redundantly encoded rather than load-bearing.

### What does not replicate

**The concealment gap changes sign.** On USTC a weighted average flatters the per-class picture, which is the direction Phase 1 anticipated. On ISCX it is strongly negative: the frequent classes are the hard ones, so the weighted average reports *worse* than the macro average. The headline number misleads either way, but a paper that assumes the sign can correct in the wrong direction.

**The ports shortcut is far stronger on USTC.** A ports-only probe is close to the full model there and much weaker on ISCX. Neither is load-bearing, but the risk a reviewer should check is corpus-dependent.

**Equalising flow length destroys the signal on one corpus and not the other.** USTC-TFC2016 scores 0.7907 against a 0.1429 chance rate on 168 test sessions; ISCX VPN-nonVPN scores 0.1106 against a 0.1667 chance rate on 84 test sessions. So on USTC-TFC2016 content still separates the classes with length held fixed, while on ISCX VPN-nonVPN very little does, which would put much of that corpus's apparent accuracy on flow length rather than on content. This is the sharpest disagreement between the two, and it is a reason not to generalise a padding or budget result from one corpus to the other. Both figures rest on a test set in the low hundreds, so the direction is what should be read here, not the magnitude.

**The unit of analysis matters on one corpus and not measurably on the other.** On USTC-TFC2016 the difference is +0.0045 macro-F1 against 0.0304 of seed noise (inconclusive); on ISCX VPN-nonVPN the difference is -0.0976 macro-F1 against 0.0275 of seed noise (session better). Only ISCX VPN-nonVPN separates them, and it does so in the direction the confound argues against: flows are shorter than their parent sessions, so at a fixed budget the flow setting sees a larger share of each unit and still loses. Pairing the two directions of a conversation carries information that reading them separately does not, at least where the classes are hard enough for it to show. On the easier corpus everything is near ceiling and the question does not arise.

### The disagreement that is probably not about traffic

The two corpora differ in measured difficulty by a wide margin, and they also differ in how their classes can be split: USTC-TFC2016 splits 19 of 20 classes temporally; ISCX VPN-nonVPN splits 5 of 11 classes temporally. A temporal split draws train and test from the same capture; a group split holds whole captures out. The first is the easier problem.

Those two differences point the same way, and this study cannot separate them: the corpus with more temporal splits is also the one that scores higher. So the difficulty gap between the corpora should **not** be read as a statement about the traffic. It is a reason to be sceptical of any cross-corpus accuracy comparison, including the ones in the reviewed papers, that does not say how its splits were formed.

This is a limitation of the present study, not a finding against the literature, and it is measurable within one corpus even though it is not measurable between them.

On the 6 ISCX VPN-nonVPN classes that have enough capture files to be split either way, that experiment has now been run: the classes, the budget, the model, the epochs, the seeds and the per-class split sizes are held fixed and only the train/test boundary moves. A **random within-capture split scores 0.9378 macro-F1 against 0.5596 for a group-wise one**: +0.3782, against 0.0510 of seed noise. A paper that does not state its split policy is most likely reporting the first number. The temporal fallback used here is not measurably worse than group-wise (+0.0431 against 0.0510), so the per-corpus numbers above are not inflated that way, but the cross-corpus comparison stays confounded, because the two corpora differ in how many classes can be split group-wise at all.

<!-- END cross-corpus -->

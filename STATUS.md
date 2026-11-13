# Implementation status: Phase 3 pipeline

Research topic: *Observation Budgets and Padding Effects in Raw-Byte Network
Traffic Classification*. This directory implements the methodology in
`papers/Phase2_Methodology.pdf`; step numbers below refer to that document.
See `README.md` for what the project is and how to run it.

**Last updated:** 2026-11-11, both corpora run end to end; reports generated;
the E2 sweep replicated on ISCX under a second architecture.

## State

The pipeline is complete, validated on synthetic ground truth, and **run to
completion on both corpora**. Results live in `results/ustc/REPORT.md` and
`results/iscx/REPORT.md`, with the side-by-side reading at the end of
`README.md`. Every table in those files is generated from the JSON beside it, and
nothing in them is hand-typed.

| Corpus | Sessions | Classes | Store | Stages |
|---|---|---|---|---|
| USTC-TFC2016 | 353,496 | 20 | `stores/ustc.h5` | all, uncapped, 40 epochs |
| ISCX VPN-nonVPN | 31,855 | 11 | `stores/iscx.h5` | all, capped at 5,000/class, 15 epochs |

| Module | Covers | State |
|---|---|---|
| `pcapio.py` | streaming pcap + pcapng | verified |
| `dissect.py` | five-tuple keying, anonymisation, field offsets (3a, 3c) | verified |
| `encryption.py` | TLS cipher tagging, strata (3b, after P5) | verified |
| `sessionize.py` | two-pass assembly, cleaning, dedupe (3a-3e) | verified |
| `store.py` | HDF5 session store (3e) | verified |
| `split.py` | group-wise 70/15/15 + temporal fallback (3f) | verified |
| `slicer.py` | byte/packet budgets, padding controls, probe sources (4, 11ii) | verified |
| `models.py` | Wang 2D/1D, Deep Packet, adaptive early-exit (5, 6), byte transformer | verified incl. causality |
| `train.py` | one loop for every model and budget, OOM backoff | verified |
| `metrics.py` | per-class, stratified, McNemar, frontier (8, 9) | verified |
| `experiments.py` | E1-E4 and padding controls (7, 11) | run on both corpora |
| `plots.py` | the four Step 12 figures | rendered and inspected |
| `datasets.py` | USTC-TFC2016 and ISCX adapters (2) | run on both corpora |
| `ablations.py` | Step 10 ablations (i), (ii), (iii), (iv) | all four run on both corpora |
| `literature.py` | 17 published results from P1-P12, comparability notes (12) | verified |
| `corpus_stats.py` | corpus-level descriptives for Gap 3 and the split caveat | run on both |
| `make_report.py` | one results directory → that corpus's `REPORT.md` | run on both |
| `make_cross_corpus.py` | both directories → the README's comparison block | run |
| `study.sh` | pauseable, stage-resumable runner for either corpus | run on both |
| `run_significance.py` | McNemar's exact test on the paired comparisons | run on both |
| `run_split_strategy.py` | group vs temporal vs random split, everything else fixed | run on ISCX; USTC cannot host it |
| `audit_corpus_archives.py` | distributed archives vs extracted vs labelled | run on ISCX: 140/140 extracted, no non-VPN p2p in the release |
| `run_architecture_compare.py` | one corpus's E2 sweep under two model families | run on ISCX: CNN and transformer not separable |
| `references.py` | full citations for P1-P12, both corpora and the tooling | 19 entries |
| `make_submission.py` | both results directories → the single Phase 3 document | run |
| `render_html.py` | that document → one print-ready, self-contained page | run |

Step 10(iii) (session against unidirectional flow) needed a second pass over
the captures, because the unit of analysis is fixed when the store is written.
Both directional stores exist (`stores/ustc_flows.h5`, `stores/iscx_flows.h5`)
and the ablation has run on both. **Every item in the Phase 2 methodology now
has a result on real data.**

## Verification

```bash
PY=/home/minato/.local/share/pipx/venvs/jupyterlab/bin/python3
PYTHONPATH=src $PY tests/test_pipeline_endtoend.py /tmp/synthcheck        # ~10s
PYTHONPATH=src $PY tests/test_models_causality.py                         # ~5s
PYTHONPATH=src $PY tests/test_split_and_slicer.py /tmp/synthcheck         # ~5s
PYTHONPATH=src $PY tests/test_experiments_smoke.py /tmp/synthcheck        # ~4min, GPU
PYTHONPATH=src $PY tests/test_sweep_recovers_ground_truth.py /tmp/synth_gt # ~6min, GPU
```

The last is the important one: it plants a cue at a known byte depth and
requires the sweep to find it. The fourth runs every experiment path at low
epochs, which is where the structural bugs showed up.

## Things that must not regress

1. **`CausalBlock` uses `ChannelNorm1d`, never GroupNorm or BatchNorm.** Both
   compute statistics across the length axis, which lets a byte at position
   900 shift the activation at position 10 and silently destroys the
   causality the early-exit heads rely on. Without causality, "bytes read" on
   the frontier is a fiction. `tests/test_models_causality.py` guards it.

2. **The padding control needs `pad_mode="embed"` alongside `mask`.** Moving
   from the scalar input to an embedding with a PAD symbol changes two things
   at once; `embed` holds the representation change and drops the padding
   change, so the two are separable. This replicated on both corpora: the
   representation accounts for nearly all of the combined effect and the
   padding symbol for almost none of it. Dropping `embed` would restore a
   wrong conclusion.

3. **The leakage gate trains on `id_positions`, not `ident`.** `id_positions`
   reads the identifier byte positions out of the *anonymised* store and must
   sit at chance, which is the halt condition. `ident` reads the same
   positions *before* anonymisation and is expected to score high; it is
   context, not a gate. A `ports` probe runs beside them because Step 3c
   deliberately retains ports.

4. **E4 compares the policy against the adaptive model's own heads**, not
   against the E2 sweep. The sweep's best budget may use a different ceiling
   and representation; comparing against it once turned a 66% byte saving
   into a reported failure.

5. **Saturation is "the smallest budget no larger budget beats"**, not "the
   first step inside seed noise". The latter reported saturation at 256 bytes
   on a curve that went on to reach 1.000.

6. **`AdaptiveByteCNN.forward` returns one logit tensor per *reachable* exit,
   not one per head.** It stops at the first head whose budget covers the
   whole input, so at a 512-byte ceiling a model with five exits returns
   four. `run_e4` builds the model with only the exits the ceiling reaches;
   anything else indexing heads by position should call `reachable_exits`.

7. **The length-balanced control derives its bins and class set from the
   training split and reuses them.** Derived independently per split they
   disagree: on ISCX, train kept six classes and test kept six with only
   three in common. A model trained on one label set and scored against
   another collapses to chance, which reads exactly like the finding the
   control exists to detect. `length_balance_plan` is the fix and
   `tests/test_split_and_slicer.py` asserts every split shares one class set.
   This one was caught after it had already produced a "macro-F1 0.0985,
   collapse to chance" result that was about to be written up.

8. **Per-class scores are not comparable across split strategies.** A class
   with fewer than three capture files cannot have whole files held out, so it
   falls back to a temporal split with train and test from the same capture,
   the easier problem. 19 of 20 USTC classes and 5 of 11 ISCX classes are in
   that position. Any per-class or per-stratum comparison has to carry the
   caveat, and the corpora's difficulty gap cannot be separated from it.
   `run_split_strategy.py` measures what the strategy is worth *within* ISCX,
   on the six classes that can be split both ways: a random within-capture
   split is worth +0.3782 macro-F1 over a group-wise one, while the temporal
   fallback is not measurably worse than group-wise. So the per-corpus numbers
   survive; the cross-corpus comparison does not become safe.

## What the synthetic validation established

On a corpus with a weak cue at byte 78 and a strong cue past byte 780:

* the sweep traces chance → 0.837 at 128 B → 1.000 at 1500 B, recovering the
  planted structure exactly;
* the scalar `byte/255` input used by the byte-image baselines plateaus at
  0.492 and never finds the late cue, because after scaling, class 3 and
  class 4 differ by 0.004. **A budget curve has no meaning independent of the
  representation it was measured with**;
* the adaptive early-exit policy matches its best fixed head's macro-F1
  exactly (0.9528) while reading 350 bytes rather than 1024, which is 34% of the
  cost, with 450 of 600 test flows exiting at the 128-byte head.

These are demonstrations that the instruments work on data engineered to have
a known answer. They are **not** evidence about real traffic; for that, see
the two reports.

## What is left

1. **The length-balanced padding control answers, but imprecisely.**
   Equalising flow length leaves 168 test sessions on USTC and 84 on ISCX, and
   retains a biased subset of classes, the ones whose lengths overlap
   everything else, which is to say the ones least likely to be separable by
   length. Within that, the direction is unambiguous and the two corpora
   disagree: USTC holds 0.7907 macro-F1 against a 0.1429 chance rate, ISCX
   falls to 0.1106 against 0.1667. So content still separates USTC's classes
   with length held fixed and barely separates ISCX's. The reports give the
   direction and decline the magnitude. Tightening it needs either a corpus
   with more overlap between the per-class length distributions or a
   different way to hold length fixed.
2. **The split-strategy confound (invariant 8) cannot be settled here.**
   It needs a corpus where the same classes can be split both ways, so split
   strategy varies with everything else held fixed.
3. **The ISCX class list is short by one.** The category scheme defines 12
   classes; the non-VPN `p2p` captures are not in the release that was
   downloaded, so every ISCX number is on 11 classes. Comparisons with papers
   reporting the full taxonomy are not like-for-like.
4. **The ISCX per-application label scheme is heuristic** and cannot be fully
   correct, because ISCX filenames are not systematic, so `ftps_down_1a` and
   `vpn_ftps_A` land in different classes. The runs here use the category
   scheme for that reason. `build_store.py` prints the class map before
   preprocessing; read it before committing to a long run.

# Pausing and resuming a study run

**State as of 11 Nov 2026: both corpora complete.** Every stage on both
USTC-TFC2016 and ISCX VPN-nonVPN has a result on disk, so there is nothing to
resume. This file stays as the operating guide for re-running a stage or
starting a new corpus.

Check at any time:

```bash
bash scripts/study.sh ustc --status
bash scripts/study.sh iscx --status
```

That prints which stages have results, whether a run is live, and whether a
live run is frozen.

## Running or resuming

```bash
bash scripts/study.sh ustc      # run or resume
bash scripts/study.sh iscx
```

That is all. It prints what is already done, skips it, and runs the rest. Safe
to re-run any number of times. Resume is at **stage** granularity: each stage
writes one JSON when it finishes and a re-run skips whatever is already there.

```bash
bash scripts/study.sh iscx --watch    # follow the log
```

## Short break, laptop stays awake ,  soft pause

```bash
bash scripts/study.sh iscx --pause      # freeze
bash scripts/study.sh iscx --continue   # carry on exactly where it was
```

Nothing is lost and no work repeats. The process keeps its GPU memory reserved
while frozen, so the GPU is unusable for anything else, and this does **not**
survive suspend or reboot.

## Going out with the laptop ,  clean stop

Suspending the machine usually destroys the CUDA context, and the process will
crash on wake rather than continue. So before closing the lid:

```bash
bash scripts/study.sh iscx --stop
```

Then later, `bash scripts/study.sh iscx`. Completed stages are detected and
skipped.

## What a stop costs

Killing during a stage discards that stage's progress and it restarts from the
beginning. So the cost of stopping is however far into the current stage you
are. Stopping between stages costs nothing.

| stage | restart cost if interrupted |
|---|---|
| E2 byte and packet sweeps | up to ~45 min |
| E4 adaptive early exit | up to ~45 min per seed |
| Step 10 ablations | up to ~25 min each |
| padding controls | up to ~30 min |
| port dependence | up to ~2 min per run |

## Re-running one stage on purpose

`run_study.py` skips a stage whose JSON exists, so forcing a re-run is
explicit. Each corpus has its own hyperparameters; match them or the stage's
numbers will not be comparable with the rest of that corpus's report:

```bash
PY=/home/minato/.local/share/pipx/venvs/jupyterlab/bin/python3

# USTC: uncapped, 40 epochs
PYTHONPATH=src $PY -u scripts/run_study.py \
    --store stores/ustc.h5 --out results/ustc --corpus-name USTC-TFC2016 \
    --model wang1d --pad-mode embed --seeds 0 1 2 --epochs 40 \
    --batch-size 512 --max-per-class 0 --lazy \
    --compare-representations --stages padding --force

# ISCX: capped at 5,000 per class, 15 epochs
PYTHONPATH=src $PY -u scripts/run_study.py \
    --store stores/iscx.h5 --out results/iscx --corpus-name iscx \
    --model wang1d --pad-mode embed --seeds 0 1 2 --epochs 15 \
    --batch-size 512 --max-per-class 5000 --lazy \
    --compare-representations --stages padding --force
```

Then regenerate that corpus's write-up, since the reports are generated and
never hand-edited:

```bash
PYTHONPATH=src $PY scripts/corpus_stats.py --store stores/iscx.h5 \
    --corpus iscx --out results/iscx
PYTHONPATH=src $PY scripts/make_report.py --results results/iscx \
    --out results/iscx/REPORT.md
PYTHONPATH=src $PY scripts/make_cross_corpus.py --results results/ustc results/iscx
```

## What is already safe

Everything with a file in `results/<corpus>/` is durable and will not be
recomputed. Preprocessing is done once and never repeats ,  rebuilding
`stores/ustc.h5` takes about 75 seconds, `stores/iscx.h5` about 160.

## One thing worth knowing about process detection

`pgrep -f scripts/run_study` matches any shell whose command text merely
mentions the script, including the one you are typing. A dead run reported as
alive is the worst failure mode this tooling has, and it happened twice during
development ,  once leading to results being read off a JSON file four minutes
after the process that would have updated it had already crashed.
`scripts/study.sh` matches on the executable and the output directory instead.
Prefer `--status` over checking by hand.

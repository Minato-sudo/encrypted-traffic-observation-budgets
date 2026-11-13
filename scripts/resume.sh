#!/usr/bin/env bash
# SUPERSEDED by scripts/study.sh, which does the same thing for either corpus
# and detects a live run by executable rather than by command-line text. Use:
#
#   bash scripts/study.sh ustc [--status|--pause|--continue|--stop|--watch]
#
# Kept because it is what the 23 Oct USTC run was driven by, and the log paths
# it writes are referenced in that run's notes. Do not extend it.
#
# Resume the USTC-TFC2016 study from wherever it stopped.
#
#   bash scripts/resume.sh            # run the remaining stages
#   bash scripts/resume.sh --status   # just show what is done and what is left
#
# Completed stages are detected from results/ustc/ and skipped, so this is
# safe to re-run as many times as you like.
set -euo pipefail
cd "$(dirname "$0")/.."

PY=/home/minato/.local/share/pipx/venvs/jupyterlab/bin/python3
OUT=results/ustc
LOG=/tmp/run_ustc_resume.log

declare -A STAGES=(
    [e1]="e1_baseline_and_leakage.json"
    [e2]="e2_budget_sweep_wang1d.json"
    [e3]="e3_stratified_wang1d_784B.json"
    [e4]="e4_adaptive_frontier.json"
    [padding]="padding_controls_wang1d_1024B.json"
    [ports]="port_dependence_wang1d.json"
)

status() {
    printf '\n\033[1mstage status\033[0m\n'
    for s in e1 e2 e3 padding e4 ports; do
        if [ -f "$OUT/${STAGES[$s]}" ]; then
            printf '  %-10s done\n' "$s"
        else
            printf '  %-10s PENDING\n' "$s"
        fi
    done
    local n=0
    for f in ablation_representation ablation_layer_scope ablation_early_exit; do
        [ -f "$OUT/$f.json" ] && n=$((n+1))
    done
    printf '  %-10s %s\n' "ablations" "$n/3 done"
    # Bracket idiom: the pattern cannot match the grep process itself, and
    # restricting to a python command avoids matching any shell that merely
    # mentions the script name.
    if ps -eo args --no-headers | grep -q "[p]ython3.*run_study\.py"; then
        printf '\n  a run is currently active (tail -f %s)\n' "$LOG"
    else
        printf '\n  nothing running\n'
    fi
}

if [ "${1:-}" = "--status" ]; then status; exit 0; fi
if [ "${1:-}" = "--stop" ]; then
    if pkill -f "run_study\.py"; then echo "stopped"; else echo "nothing to stop"; fi
    exit 0
fi

status
printf '\n\033[1mresuming\033[0m: ports first (cheapest to interrupt), then ablations\n'
printf 'log: %s\n\n' "$LOG"

# run_study.py executes stages in a fixed internal order, so passing them in
# the order we want is not enough. Two sequential invocations guarantee that
# port dependence (~1 h, the most important outstanding result) completes
# before the ablations (~5 h) begin, which matters if the run is interrupted.
# Deliberately no "set -e": these are independent stage groups, and a failure
# in one should not cancel the others. An unattended five-hour job that
# silently skips its remaining work because an earlier stage tripped on a
# reporting bug is the worst of both worlds.
nohup bash -c "
  export PYTHONPATH=src
  '$PY' -u scripts/run_study.py \
      --store stores/ustc.h5 --out '$OUT' --corpus-name USTC-TFC2016 \
      --model wang1d --pad-mode embed --seeds 0 1 2 --epochs 15 \
      --batch-size 512 --max-per-class 5000 --lazy \
      --stages ports compare
  '$PY' -u scripts/run_study.py \
      --store stores/ustc.h5 --out '$OUT' --corpus-name USTC-TFC2016 \
      --model wang1d --pad-mode embed --seeds 0 1 2 --epochs 15 \
      --batch-size 512 --max-per-class 5000 --lazy \
      --stages ablations compare
" > "$LOG" 2>&1 &

echo "started, pid $!"
echo "  watch:  tail -f $LOG"
echo "  stop:   bash scripts/resume.sh --stop"

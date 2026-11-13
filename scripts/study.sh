#!/usr/bin/env bash
# Run, pause, resume or inspect the study for a corpus.
#
#   bash scripts/study.sh ustc              run or resume
#   bash scripts/study.sh iscx              run or resume
#   bash scripts/study.sh iscx --status     what is done, what is left
#   bash scripts/study.sh iscx --pause      freeze (laptop stays on)
#   bash scripts/study.sh iscx --continue   unfreeze
#   bash scripts/study.sh iscx --stop       stop cleanly (before suspend/reboot)
#   bash scripts/study.sh iscx --watch      follow the log
#
# Everything is resumable at stage granularity: each stage writes one JSON when
# it finishes, and re-running skips whatever is already there. Stopping between
# stages costs nothing; stopping inside one repeats that stage only.
set -uo pipefail
cd "$(dirname "$0")/.."

PY=/home/minato/.local/share/pipx/venvs/jupyterlab/bin/python3
CORPUS="${1:-}"; shift || true
ACTION="${1:---run}"

case "$CORPUS" in
    ustc) ROOT=data/USTC-TFC2016;    STORE=stores/ustc.h5;  OUT=results/ustc;  KIND=ustc
          NAME="USTC-TFC2016" ;;
    iscx) ROOT=data/ISCX-VPN-nonVPN; STORE=stores/iscx.h5;  OUT=results/iscx;  KIND=iscx
          NAME="ISCX VPN-nonVPN" ;;
    *) echo "usage: $0 {ustc|iscx} [--status|--pause|--continue|--stop|--watch]" >&2; exit 2 ;;
esac
LOG=/tmp/study_${CORPUS}.log
MODEL=wang1d

# Match on the executable, not on the command line. Any shell whose command
# text merely mentions these script names -- including the one running this
# function -- would otherwise register as a live run, and a dead run reported
# as alive is the worst failure mode this script has.
procs() {
    ps -eo pid=,stat=,comm=,args= | awk -v out="$OUT" -v store="$STORE" '
        $3 ~ /^python3?(\.[0-9]+)?$/ &&
        $0 ~ /(build_store|run_study|run_early_exit)\.py/ &&
        ($0 ~ out || $0 ~ store) { print $1, $2 }'
}
pids()  { procs | awk '{print $1}'; }

say() { printf '\n\033[1m%s\033[0m\n' "$*"; }

status() {
    say "corpus: $CORPUS"
    if [ ! -d "$ROOT" ] || [ -z "$(find "$ROOT" -iname '*.pcap' -o -iname '*.pcapng' 2>/dev/null | head -1)" ]; then
        printf '  captures   MISSING  (expected under %s)\n' "$ROOT"
    else
        printf '  captures   %s files, %s\n' \
            "$(find "$ROOT" \( -iname '*.pcap' -o -iname '*.pcapng' \) | wc -l)" \
            "$(du -sh "$ROOT" | cut -f1)"
    fi
    [ -f "$STORE" ] && printf '  store      built (%s)\n' "$(du -h "$STORE" | cut -f1)" \
                    || printf '  store      PENDING\n'
    printf '\n  stages\n'
    for pair in "e1:e1_baseline_and_leakage" "e2:e2_budget_sweep_${MODEL}" \
                "e3:e3_stratified_${MODEL}_784B" "padding:padding_controls_${MODEL}_1024B" \
                "e4:e4_adaptive_frontier" "ports:port_dependence_${MODEL}" \
                "ablations(i):ablation_representation" "ablations(ii):ablation_layer_scope" \
                "ablations(iii):ablation_granularity" "ablations(iv):ablation_early_exit"; do
        s=${pair%%:*}; f=${pair#*:}
        [ -f "$OUT/$f.json" ] && printf '    %-14s done\n' "$s" || printf '    %-14s PENDING\n' "$s"
    done
    [ -f "$OUT/REPORT.md" ] && printf '    %-14s built\n' "report" || printf '    %-14s PENDING\n' "report"

    printf '\n'
    local p; p=$(pids)
    if [ -n "$p" ]; then
        local st; st=$(procs | awk '{print $2}' | head -1)
        case "$st" in
            T*) printf '  PAUSED (pid %s), resume with: %s %s --continue\n' "$p" "$0" "$CORPUS" ;;
            *)  printf '  running (pid %s), follow with: %s %s --watch\n' "$p" "$0" "$CORPUS" ;;
        esac
    else
        printf '  nothing running\n'
    fi
}

case "$ACTION" in
  --status) status; exit 0 ;;
  --watch)  tail -f "$LOG"; exit 0 ;;
  --pause)
      p=$(pids); [ -z "$p" ] && { echo "nothing running"; exit 0; }
      kill -STOP $p && echo "paused (pid $p). GPU memory stays reserved."
      echo "resume: $0 $CORPUS --continue"
      echo "NOTE: a pause does not survive suspend or reboot. Before closing the"
      echo "      lid, use --stop instead."
      exit 0 ;;
  --continue)
      p=$(pids); [ -z "$p" ] && { echo "nothing to continue; use: $0 $CORPUS"; exit 0; }
      kill -CONT $p && echo "continued (pid $p)"; exit 0 ;;
  --stop)
      p=$(pids); [ -z "$p" ] && { echo "nothing running"; exit 0; }
      kill -CONT $p 2>/dev/null   # a stopped process cannot handle TERM
      kill -TERM $p; sleep 3
      p=$(pids); [ -n "$p" ] && { kill -KILL $p; echo "killed"; } || echo "stopped"
      echo "resume later with: $0 $CORPUS"
      exit 0 ;;
esac

# ---------------------------------------------------------------- run / resume
if [ -n "$(pids)" ]; then
    echo "already running (pid $(pids)). Use --status, --watch, --pause or --stop."
    exit 1
fi
if [ ! -d "$ROOT" ] || [ -z "$(find "$ROOT" \( -iname '*.pcap' -o -iname '*.pcapng' \) 2>/dev/null | head -1)" ]; then
    echo "no captures under $ROOT, see scripts/get_datasets.sh $CORPUS" >&2
    exit 1
fi

status
say "starting, log: $LOG"
mkdir -p "$OUT" stores

# Stages are chained as separate invocations so their order is guaranteed and a
# failure in one cannot cancel the rest. No "set -e" here, deliberately.
nohup bash -c "
  export PYTHONPATH=src
  if [ ! -f '$STORE' ]; then
      echo '### building session store'
      '$PY' -u scripts/build_store.py --corpus $KIND --root '$ROOT' \\
          --out '$STORE' --checksums
  else
      echo '### store already built, skipping preprocessing'
  fi
  for group in 'e1 e2 e3 padding' 'ports' 'e4' 'ablations' 'compare'; do
      echo \"### stage group: \$group\"
      '$PY' -u scripts/run_study.py \\
          --store '$STORE' --out '$OUT' --corpus-name '$NAME' \\
          --model $MODEL --pad-mode embed --seeds 0 1 2 --epochs 15 \\
          --batch-size 512 --max-per-class 5000 --lazy \\
          --compare-representations --stages \$group
  done
  echo '### assembling report'
  '$PY' -u scripts/make_report.py --results '$OUT' --out '$OUT/REPORT.md'
  echo '### done'
" > "$LOG" 2>&1 &

echo "started (pid $!)"
echo "  status: $0 $CORPUS --status"
echo "  pause:  $0 $CORPUS --pause      (short break, laptop on)"
echo "  stop:   $0 $CORPUS --stop       (before suspend or reboot)"

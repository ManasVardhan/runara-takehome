#!/usr/bin/env bash
# Full experiment matrix. Each config = one server lifetime, then all
# (output_len x concurrency) cells against it, then (for PROFILE_CONFIGS) a short
# torch-profiler capture at concurrency 1 and 16.
# Usage: ./sweep.sh [configs...]   e.g. ./sweep.sh baseline draft_model:4
#   default: baseline + draft_model K=1,2,4,8 + eagle3 K=1,2,4,8 + ngram K=4
# Env: TEMP (default 0), LENS, CONCS, PROFILE_CONFIGS
set -uo pipefail
cd "$(dirname "$0")"
RAW="${RAW:-results/raw}"; mkdir -p "$RAW" results/logs
LENS="${LENS:-32,128,512}"; CONCS="${CONCS:-1,4,16}"; TEMP="${TEMP:-0}"
PROFILE_CONFIGS="${PROFILE_CONFIGS:-baseline draft_model:4 eagle3:4}"
CONFIGS=("$@")
if [[ ${#CONFIGS[@]} -eq 0 ]]; then
  CONFIGS=(baseline eagle3:1 eagle3:2 eagle3:4 eagle3:8
           draft_model:1 draft_model:2 draft_model:4 draft_model:8 ngram:4)
fi
source /root/venv/bin/activate
SUFFIX="${LABEL_SUFFIX:-}"; [[ $TEMP != 0 ]] && SUFFIX="${SUFFIX}_t${TEMP}"
for cfg in "${CONFIGS[@]}"; do
  echo "=== $cfg $(date -u +%H:%M:%S)"
  if [[ $cfg == baseline ]]; then
    method=none; k=0; label="baseline$SUFFIX"; spec=none
    ./run_baseline.sh || { echo "FAILED $cfg"; continue; }
  else
    method=${cfg%%:*}; k=${cfg##*:}; label="${method}_k${k}$SUFFIX"
    ./run_speculative.sh --method "$method" --num-speculative-tokens "$k" || { echo "FAILED $cfg"; continue; }
    spec=$(cat "results/logs/spec_config_${method}_k${k}.json")
  fi
  rm -f "$RAW/$label.jsonl" "$RAW/$label.jsonl.requests.jsonl"
  python benchmark/benchmark.py --label "$label" --method "$method" --k "$k" --temperature "$TEMP" \
    --spec-config "$spec" --output-lens "$LENS" --concurrency "$CONCS" --out "$RAW/$label.jsonl" \
    | tee "results/logs/bench_$label.log" || echo "BENCH FAILED $cfg"
  # Any Triton JIT compile after warmup would stall a measured cell: record them.
  wdone=$(grep -oE "WARMUP_DONE [0-9-]+ [0-9:]+" "results/logs/bench_$label.log" | cut -d" " -f2-)
  slog="results/logs/server_${method}_k${k}.log"; [[ $method == none ]] && slog=results/logs/server_baseline.log
  grep "JIT compilation during inference" "$slog" \
    | awk -v w="$wdone" '{ for (i=1;i<=NF;i++) if ($i ~ /^[0-9][0-9]-[0-9][0-9]$/) { t=$i" "$(i+1); break } } t > w' \
    > "results/logs/jit_after_warmup_$label.txt"
  echo "JIT_AFTER_WARMUP $label $(wc -l < "results/logs/jit_after_warmup_$label.txt")"
  if [[ $TEMP == 0 && " $PROFILE_CONFIGS " == *" $cfg "* ]]; then
    for c in 1 16; do
      python benchmark/profile_load.py --label "$label" --concurrency "$c" \
        --trace-dir /workspace/traces_raw || echo "PROFILE FAILED $cfg c=$c"
    done
  fi
done
source ./common.sh >/dev/null 2>&1 && stop_server
echo SWEEP_DONE

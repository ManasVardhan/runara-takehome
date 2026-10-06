#!/usr/bin/env bash
# Start vLLM with speculative decoding disabled. Usage: ./run_baseline.sh
source "$(dirname "$0")/common.sh"
stop_server
LOG="$LOG_DIR/server_baseline.log"
nohup vllm serve "$TARGET_MODEL" "${ENGINE_ARGS[@]}" > "$LOG" 2>&1 &
wait_for_server $!
if curl -s "http://localhost:$PORT/metrics" | grep -q "spec_decode_num_drafts"; then
  echo "baseline server unexpectedly has speculation on" >&2; exit 1
fi

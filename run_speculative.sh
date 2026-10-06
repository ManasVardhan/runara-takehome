#!/usr/bin/env bash
# Start vLLM with speculative decoding enabled.
# Usage: ./run_speculative.sh --method draft_model|eagle3|ngram --num-speculative-tokens K
source "$(dirname "$0")/common.sh"
METHOD=draft_model; K=4
while [[ $# -gt 0 ]]; do
  case $1 in
    --method) METHOD=$2; shift 2 ;;
    --num-speculative-tokens) K=$2; shift 2 ;;
    *) echo "unknown arg $1" >&2; exit 2 ;;
  esac
done
case $METHOD in
  draft_model) SPEC="{\"method\":\"draft_model\",\"model\":\"${DRAFT_MODEL:-Qwen/Qwen3-0.6B}\",\"num_speculative_tokens\":$K,\"draft_tensor_parallel_size\":1}" ;;
  eagle3)      SPEC="{\"method\":\"eagle3\",\"model\":\"${EAGLE_MODEL:-AngelSlim/Qwen3-a3B_eagle3}\",\"num_speculative_tokens\":$K}" ;;
  ngram)       SPEC="{\"method\":\"ngram\",\"num_speculative_tokens\":$K,\"prompt_lookup_min\":2,\"prompt_lookup_max\":5}" ;;
  *) echo "bad method $METHOD" >&2; exit 2 ;;
esac
echo "speculative_config=$SPEC"
stop_server
echo "$SPEC" > "$LOG_DIR/spec_config_${METHOD}_k${K}.json"
LOG="$LOG_DIR/server_${METHOD}_k${K}.log"
nohup vllm serve "$TARGET_MODEL" "${ENGINE_ARGS[@]}" --speculative-config "$SPEC" > "$LOG" 2>&1 &
wait_for_server $!
# The spec-decode counters only exist when speculation is really on.
curl -s "http://localhost:$PORT/metrics" | grep -q "spec_decode_num_drafts" \
  || { echo "spec metrics missing: speculation not active" >&2; exit 1; }

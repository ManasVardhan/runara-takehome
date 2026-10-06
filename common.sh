#!/usr/bin/env bash
# Shared engine configuration for baseline and speculative runs.
# Everything except the speculative config is identical across runs.
set -euo pipefail
export HF_HOME="${HF_HOME:-/workspace/hf}"
# Emit record_function ranges ("gpu_model_runner: forward/draft/sample/...") so
# torch-profiler traces can split step time by phase. No-op unless profiling.
export VLLM_CUSTOM_SCOPES_FOR_PROFILING=1
VENV="${VENV:-/root/venv}"
# shellcheck disable=SC1091
source "$VENV/bin/activate"

TARGET_MODEL="${TARGET_MODEL:-Qwen/Qwen3-30B-A3B}"
PORT="${PORT:-8000}"
LOG_DIR="${LOG_DIR:-$(cd "$(dirname "$0")" && pwd)/results/logs}"
TRACE_DIR="${TRACE_DIR:-/workspace/traces_raw}"
mkdir -p "$LOG_DIR" "$TRACE_DIR"

ENGINE_ARGS=(
  --dtype bfloat16
  --tensor-parallel-size 1
  --max-model-len 4096
  --max-num-seqs 64
  --gpu-memory-utilization 0.90
  --no-enable-prefix-caching   # prompts repeat across cells; caching would fake TTFT
  --seed 0
  --port "$PORT"
  # Torch profiler is armed but idle until POST /start_profile (benchmark/profile_load.py).
  --profiler-config "{\"profiler\":\"torch\",\"torch_profiler_dir\":\"$TRACE_DIR\",\"torch_profiler_with_stack\":false}"
)

wait_for_server() {
  local pid=$1
  for _ in $(seq 1 180); do
    if ! kill -0 "$pid" 2>/dev/null; then echo "server died; see log" >&2; return 1; fi
    if curl -sf "http://localhost:$PORT/health" >/dev/null; then echo "server ready"; return 0; fi
    if ! kill -0 "$pid" 2>/dev/null; then echo "server died; see log" >&2; return 1; fi
    sleep 5
  done
  echo "server did not become ready in 15 min" >&2; return 1
}

gpu_mem_used_mib() { nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1; }

# Stop every vLLM process (API server and the EngineCore child) and wait until the
# port is closed and GPU memory is released, so the next config can never be
# benchmarked against a leftover server.
stop_server() {
  pkill -f "vllm serve" 2>/dev/null || true
  pkill -f "EngineCore" 2>/dev/null || true
  for _ in $(seq 1 30); do
    pgrep -f "vllm serve|EngineCore" >/dev/null || break; sleep 2
  done
  pkill -9 -f "vllm serve|EngineCore" 2>/dev/null || true
  for _ in $(seq 1 60); do
    if ! curl -sf "http://localhost:$PORT/health" >/dev/null && [[ $(gpu_mem_used_mib) -lt 2000 ]]; then
      return 0
    fi
    sleep 2
  done
  echo "GPU memory or port not released after stop" >&2; return 1
}

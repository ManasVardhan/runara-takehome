# Speculative decoding on vLLM: Qwen3-30B-A3B on one H100

Baseline vs. speculative decoding (separate draft model, EAGLE-3, n-gram) in vLLM's V1 engine,
benchmarked across speculation length K, output length and concurrency, profiled with the torch
profiler, and explained against the engine source.

- **Results summary:** [`results/summary.md`](results/summary.md) (assignment table format), every cell in [`results/all_cells.md`](results/all_cells.md) / `.csv`
- **Analysis (Parts 2, 4, 5, 6):** [`analysis.md`](analysis.md)
- **Plan and requirement traceability:** [`docs/PLAN.md`](docs/PLAN.md)

## Setup

| | |
|---|---|
| Engine | vLLM **0.31.0** (V1 engine), torch 2.13.0+cu130, transformers 5.17.0 |
| Hardware | 1x NVIDIA H100 80GB HBM3 (SXM, 700 W), driver 580.126.09, RunPod Secure Cloud |
| Target model | [`Qwen/Qwen3-30B-A3B`](https://huggingface.co/Qwen/Qwen3-30B-A3B), bf16, MoE (128 experts, 8 active; 30.5B total / 3.3B active params) |
| Draft models | [`Qwen/Qwen3-0.6B`](https://huggingface.co/Qwen/Qwen3-0.6B) (`method=draft_model`), [`AngelSlim/Qwen3-a3B_eagle3`](https://huggingface.co/AngelSlim/Qwen3-a3B_eagle3) (`method=eagle3`), prompt n-gram lookup (`method=ngram`) |
| Parallelism | TP=1, PP=1 (whole model on one GPU; no TP noise) |
| Engine config | `--dtype bfloat16 --max-model-len 4096 --max-num-seqs 64 --gpu-memory-utilization 0.90 --no-enable-prefix-caching --seed 0`; CUDA graphs and torch.compile on (vLLM defaults); chunked prefill on (default) |
| Sampling | temperature 0 (greedy) for the main matrix; one slice at temperature 0.7 (with Qwen's `generation_config.json` defaults top_k=20, top_p=0.95, which vLLM applies unless overridden); `ignore_eos=true`, `max_tokens` = output length; Qwen3 chat template with thinking mode on (the model's default) |
| Workload | 24 fixed real prompts (`benchmark/prompts.jsonl`: code, math, explanation, writing, summarization/extraction); output lengths 32 / 128 / 512; closed-loop concurrency 1 / 4 / 16 with 24 / 24 / 72 requests per cell (whole multiples of the prompt set) |

Full environment dump: [`results/env.txt`](results/env.txt).

## Install

On a machine with an 80 GB NVIDIA GPU (driver supporting CUDA 13):

```bash
uv venv /root/venv --python 3.12 && source /root/venv/bin/activate
uv pip install vllm==0.31.0 pandas matplotlib aiohttp nvidia-ml-py
export HF_HOME=/workspace/hf HF_HUB_ENABLE_HF_TRANSFER=1
hf download Qwen/Qwen3-30B-A3B && hf download Qwen/Qwen3-0.6B && hf download AngelSlim/Qwen3-a3B_eagle3
```

Scripts assume the venv at `/root/venv` (override with `VENV=...`) and the HF cache at `$HF_HOME`.

## Reproduce

```bash
# Baseline server (speculative decoding off), then benchmark it
./run_baseline.sh
python benchmark/benchmark.py --label baseline --out results/raw/baseline.jsonl \
    --output-lens 32,128,512 --concurrency 1,4,16

# Speculative server; K is exposed directly
./run_speculative.sh --method eagle3 --num-speculative-tokens 4      # or draft_model / ngram
python benchmark/benchmark.py --label eagle3_k4 --method eagle3 --k 4 --out results/raw/eagle3_k4.jsonl --concurrency 4

# Whole matrix (server restart per config), temperature slice, noise replicate
./run_all.sh            # = ./sweep.sh ; TEMP=0.7 ./sweep.sh ... ; LABEL_SUFFIX=_rep ./sweep.sh ...

# Lossless check: greedy outputs with speculation vs. baseline
python benchmark/lossless_check.py --baseline results/raw/baseline.jsonl.requests.jsonl \
    --spec results/raw/eagle3_k*.jsonl.requests.jsonl results/raw/draft_model_k*.jsonl.requests.jsonl

# Profiling: servers start with --profiler-config (torch profiler armed, idle);
# sweep.sh captures traces after benchmarking; analyze them:
python benchmark/analyze_trace.py results/traces/* --out results/profile_summary.json

# Tables and plots
python benchmark/make_report.py
```

## Layout

```
common.sh               shared engine args, server start/stop (waits for port + GPU memory release)
run_baseline.sh         vLLM server, speculation off (fails if spec metrics appear)
run_speculative.sh      vLLM server with --speculative-config {method, model, num_speculative_tokens}
sweep.sh, run_all.sh    experiment matrix drivers
benchmark/benchmark.py  closed-loop streaming client: TTFT/TPOT/ITL/E2E, tok/s, req/s,
                        spec-decode counter deltas from /metrics, NVML GPU util/memory, KV usage
benchmark/profile_load.py   /start_profile -> fixed load -> /stop_profile, collects traces
benchmark/analyze_trace.py  per-phase / per-kernel-family GPU time, step time vs tokens per step
benchmark/lossless_check.py greedy output equality vs baseline
benchmark/make_report.py    tables (summary.md, all_cells.md/csv) and plots
results/raw/            one JSONL row per cell + per-request records (incl. generated text)
results/logs/           server logs and the exact speculative configs used
```

## Metric definitions

- **TTFT**: request send to first streamed token (client side, includes queueing).
- **TPOT**: `(E2E - TTFT) / (output_tokens - 1)`, per request. Under speculation one streamed chunk can carry several tokens, so the per-chunk gap (`chunk_itl_mean_ms`) is reported separately and is not a per-token number.
- **E2E**: request send to last token.
- Per-request latency is reported as p50/p90/p99 across requests in a cell (p99 over 24-72 samples is effectively the max).
- **Server throughput**: total output tokens / cell wall time (`output_tok_s`) and completed requests / wall time (`req_s`).
- **Acceptance rate**: accepted draft tokens / proposed draft tokens (deltas of `vllm:spec_decode_num_accepted_tokens_total` and `..._num_draft_tokens_total`).
- **Tokens per verification step** (`mean_accepted_len`): `1 + accepted / drafts`; each target verification step emits the accepted draft tokens plus one token from the target itself.
- **Per-position acceptance**: `num_accepted_tokens_per_pos[i] / drafts`.
- **GPU**: NVML utilization (fraction of time any kernel runs, not SM efficiency) and memory used, sampled every 100 ms; vLLM's `kv_cache_usage_perc` gauge sampled every 500 ms (NVML memory is flat because vLLM preallocates the KV cache).

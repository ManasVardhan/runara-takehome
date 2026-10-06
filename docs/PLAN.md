# Plan: Speculative decoding in vLLM on Qwen3-30B-A3B

> This is the plan as written before any GPU work. What changed during execution:
> - **Main draft.** EAGLE-3 turned out to be the useful draft. The Qwen3-0.6B `draft_model` became the
>   contrast case (the best guesser, but CPU launch-bound).
> - **Part 6 topic.** The deep dive became drafter execution and CUDA graphs, because that is what the
>   data pointed to. Verification and KV-cache handling are covered in Part 2.
> - **Temperature slice.** It showed no acceptance drop on this workload (analysis Q3).
> - **Environment record.** It lives in `results/env.txt` (not `env.json`). Engine args are in the README,
>   and the per-run spec config is in `results/logs/spec_config_*.json` and every result row.
> - **Reruns.** A first full run was partly contaminated by mid-cell Triton JIT compiles. The warmup was
>   extended, every config was rerun, and the first run is kept as a noise replicate.
> - **Actual GPU time.** About 2.7 H100-hours (about $9.60), against a 5.2-hour budget.

## Fixed decisions

| Item | Choice | Why |
|---|---|---|
| Engine | vLLM (V1 engine), pinned version recorded in results | Built-in `vllm bench serve`, spec-decode Prometheus counters, readable `v1/spec_decode` + rejection sampler source |
| Hardware | 1x H100 SXM 80GB (RunPod, Secure Cloud) | Model fits in bf16 at TP=1, so no TP noise |
| Target | `Qwen/Qwen3-30B-A3B`, bf16, TP=1 | Named in the assignment; MoE with 3.3B active params |
| Draft (primary) | `Qwen/Qwen3-0.6B` via `method=draft_model` | Classic separate draft model; same family, same tokenizer and vocab (required for lossless verification); ~5x fewer active params than target |
| Draft (secondary) | EAGLE-3 head `AngelSlim/Qwen3-a3B_eagle3` via `method=eagle3` | Trained on target hidden states, so expected higher acceptance at lower draft cost; contrasts "draft quality" for Part 5 Q3 |
| Draft (control) | `method=ngram` | No model; low acceptance on open-ended chat, so it shows when spec decoding hurts |
| Sampling | temperature=0 (greedy) for the main matrix; one temperature=0.7 slice | Greedy makes outputs comparable token-for-token (lossless check) and is the best case for acceptance; the 0.7 slice shows the drop |
| Prompts | Fixed, real chat prompts (code, math, explanation, rewrite, summarization) in `benchmark/prompts.jsonl` | Random-token prompts would make acceptance rates meaningless |
| Output lengths | 32, 128, 512 tokens, `ignore_eos` so length is controlled | Assignment asks for 3 lengths incl. 256+ |
| Concurrency | 1, 4, 16 | Assignment's suggested values; H100 can take more but budget can't |
| K | 1, 2, 4, 8 | Assignment's values |

## Budget (hard constraint: $25 credit at $3.49/hr, about 7 GPU-hours)

| Phase | GPU time |
|---|---|
| Install + download (overlaps with writing scripts) | 0.5 h |
| Smoke tests (every script, every method, 1 tiny cell each) | 0.5 h |
| Main sweep: baseline + draft_model K=1,2,4,8 (5 servers x 9 cells) | ~1.5 h |
| EAGLE-3 K=1,2,4,8 + ngram K=4 (conc 1 and 16 cells only if time is short) | ~1.0 h |
| Temperature slice + lossless check | 0.2 h |
| Profiling (torch profiler traces, nvidia-smi) | 0.5 h |
| Buffer | 1.0 h |

Total ~5.2 h, about $18. Pod is stopped as soon as GPU work ends; writing happens off-GPU.

## Requirements traceability (PDF part → how it is covered)

- **P1 baseline**: `run_baseline.sh` starts vLLM with spec decoding off; `benchmark.py` records TTFT, TPOT, ITL, E2E, tok/s, req/s; GPU util + memory sampled via NVML during each run; `results/env.json` records model, vLLM version, GPU, driver, dtype, TP, engine args, sampling params.
- **P2 enable spec**: `run_speculative.sh --method {draft_model,eagle3,ngram} --num-speculative-tokens K`. Understanding is shown by (a) `analysis.md` walk-through tied to vLLM source lines (proposer, rejection sampler, KV slot handling, scheduler), (b) per-position acceptance counters from `/metrics`, (c) a lossless check: greedy outputs with spec on are identical to baseline outputs.
- **P3 benchmark**: full matrix above; per-request latency (TTFT/TPOT/E2E percentiles) reported separately from server throughput (output tok/s, req/s); acceptance rate and mean accepted tokens per verification step from `/metrics` deltas.
- **P4 profile**: torch profiler traces of the running server for baseline vs spec at conc 1 and 16; per-step breakdown of draft time vs target verification time; GPU util from NVML. Identify at least one bottleneck with evidence.
- **P5 explain**: answers to the 5 questions, each citing table rows.
- **P6 internals**: one deep dive (draft-token verification + KV-cache consistency in the V1 rejection sampler and scheduler), with file and function references from the installed vLLM version.
- **Deliverables**: `README.md`, `benchmark/`, `results/`, `analysis.md`, scripts to reproduce, summary table in the PDF's format.

## Smoke-test gate (before any sweep)

1. Server comes up for baseline and for each spec method at K=4; `/metrics` exposes the spec-decode counters.
2. `benchmark.py` on 2 prompts x 1 length x conc 1 writes a valid JSON row with every field non-null.
3. Acceptance counters move between before/after scrapes; computed rate is in [0, 1].
4. Lossless check: greedy outputs for 4 prompts match baseline exactly.
5. Profiler start/stop produces a trace file.
Only then launch the sweep.

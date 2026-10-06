# Analysis: speculative decoding for Qwen3-30B-A3B in vLLM 0.31

Setup, install and reproduction steps are in the [README](README.md). Every number below comes from
`results/` (main greedy matrix in `results/raw/`, tables in `results/summary.md`,
`results/step_cost.md`, `results/all_cells.md`). "TPOT" is per-request time per output token (p50 over
requests); "tok/s" is server output throughput for the whole cell. Speedups are against the baseline
cell with the same output length, concurrency and temperature.

## TL;DR

1. **EAGLE-3 helps everywhere except at K=8 on long outputs.** At concurrency 1 it cuts TPOT by
   1.22-1.48x. At concurrency 16 it raises server throughput by 1.30-1.67x. Best K is 1-4 (Q4).
2. **The separate Qwen3-0.6B draft is the better guesser but the worse draft.** It has 73-93%
   acceptance against EAGLE-3's 58-71% at K=1. Even so, it makes single-request decoding slower (TPOT
   0.29-0.94x for 128/512-token outputs). Each 0.6B draft pass costs about 3.7-6.3 ms, close to a full
   target decode step (4.4 ms). The profiler shows why: the drafter only runs under piecewise CUDA graphs
   and is CPU launch-bound, with GPU busy 25% of the time against 91-96% for the baseline and EAGLE-3.
3. **The single-request conclusion flips under concurrency.** At concurrency 16 the same 0.6B draft gives
   1.28-1.68x TPOT speedup at K=1-4. Its per-pass overhead is roughly fixed, so a batch amortizes it.
   The MoE target also makes verification relatively cheaper at large batch, because the experts are
   already being loaded.
4. **Acceptance rate alone predicts nothing; acceptance x draft cost does.** Tokens per verification step
   rise with K while acceptance falls, and draft cost grows linearly in K. Speedup peaks at K=2-4 and
   then declines. K=8 is worse than K=4 in every EAGLE-3 cell.
5. **Speculation never hurts correctness.** Greedy outputs at concurrency 1 are byte-identical to the
   baseline for all 576 requests from EAGLE-3 and draft_model. Two n-gram requests flipped a near-tie
   token after 600+ characters, which is a numerics effect (below). It does cost KV memory: the draft
   model shrinks the KV cache from 141,664 to 58,496 tokens.

## Part 1: Baseline

Qwen3-30B-A3B, bf16, vLLM 0.31.0 V1, 1x H100 80GB SXM, TP=1, CUDA graphs and torch.compile on, prefix
caching off, chunked prefill on, `max_num_seqs=64`, `max_model_len=4096`, greedy, `ignore_eos`. Full
details are in the README and `results/env.txt`.

| Output len | Conc | TTFT p50 (ms) | TPOT p50 (ms) | E2E p50 (ms) | Server tok/s | Req/s | GPU util (NVML mean) | Peak KV usage |
|---|---|---|---|---|---|---|---|---|
| 32 | 1 | 27.5 | 4.46 | 166 | 191 | 5.97 | 91% | 0.1% |
| 32 | 4 | 62.8 | 7.39 | 292 | 430 | 13.43 | 74% | 0.2% |
| 32 | 16 | 135.4 | 11.64 | 500 | 983 | 30.71 | 70% | 0.3% |
| 128 | 1 | 28.7 | 4.48 | 597 | 214 | 1.67 | 93% | 0.1% |
| 128 | 4 | 66.3 | 7.74 | 1045 | 486 | 3.79 | 90% | 0.5% |
| 128 | 16 | 128.6 | 12.44 | 1707 | 1127 | 8.80 | 92% | 1.7% |
| 512 | 1 | 31.3 | 4.50 | 2331 | 219 | 0.43 | 99% | 0.4% |
| 512 | 4 | 66.8 | 7.85 | 4073 | 502 | 0.98 | 98% | 1.5% |
| 512 | 16 | 128.2 | 13.12 | 6847 | 1128 | 2.20 | 98% | 6.2% |

From 1 to 16 concurrent requests, per-request latency (TTFT, TPOT, E2E) gets worse while server
throughput rises about 5x: the usual batching trade-off. Short cells show lower NVML utilization because
prefill, request turnover and HTTP overhead take a larger share of their wall time.

(Exact values for every cell, including p90/p99, req/s and GPU util, are in `results/all_cells.md`.)

GPU memory sits at about 72 GiB in every run. vLLM preallocates the KV cache up to
`gpu_memory_utilization=0.9`, so NVML memory carries no information; we report vLLM's
`kv_cache_usage_perc` instead. The baseline KV pool holds 141,664 tokens.

The baseline already shows the property that drives everything later. **Going from 1 to 16 concurrent
requests raises the per-step cost from 4.37 ms to 12.41 ms** (profiler, `results/profile_summary.json`).
A dense model would be almost flat over that range. Each token routes to 8 of 128 experts, so a larger
batch touches more distinct expert weights. `fused_moe_kernel` takes 45% of GPU kernel time at batch 1
and 79% at batch 16.

## Part 2: How the engine runs speculation

Speculation is configured with `--speculative-config '{"method": ..., "model": ..., "num_speculative_tokens": K}'`
(`run_speculative.sh --method {draft_model,eagle3,ngram} --num-speculative-tokens K`). File and line
references are to the installed `vllm` 0.31.0 package and were checked against that install.
Longer notes are in `docs/internals_notes.md`.

### Draft model choice

| Draft | Why it fits Qwen3-30B-A3B | What it costs |
|---|---|---|
| `Qwen/Qwen3-0.6B` (`draft_model`) | Same family, same tokenizer, identical 151,936-token vocabulary, which vLLM requires for exact verification. Similar training data, so it agrees with the target often | A 28-layer transformer runs once per proposed token, and its KV cache shares the pool with the target's |
| `AngelSlim/Qwen3-a3B_eagle3` (`eagle3`) | Trained on this target's own hidden states: its hidden size of 2048 matches, and its 32k draft vocabulary maps back to the full one. It reuses target features instead of recomputing them | One decoder layer plus an LM head over the reduced vocabulary, per proposed token |
| prompt n-gram (`ngram`) | Zero-parameter control that copies continuations of n-grams found in the prompt | CPU matching. It also disables async scheduling, and its variable-length drafts break the uniform-decode CUDA-graph shape |

The 0.6B model is the textbook "small sibling" draft. We picked EAGLE-3 as the main candidate because a
draft that has seen the target's hidden states should keep acceptance high while costing a single layer.
The data bears that out (Q3).

### One engine step

1. **Draft proposes K tokens.** In `v1/spec_decode/llm_base_proposer.py`, `propose()` runs one drafter
   pass over the tokens the target just processed. It then loops
   `for token_index in range(self.num_speculative_tokens - 1)` (line 689), feeding back the last draft
   token. That is K sequential drafter passes per step. `draft_model.py` and `eagle.py` are thin
   subclasses; the difference is whether the target's hidden states are fed in. `ngram_proposer.py`
   matches on the CPU and may propose fewer than K tokens, or none.
2. **Scheduler books the drafts.** A decoding request's token count includes its pending drafts
   (`v1/request.py`, `num_tokens_with_spec`), so the scheduler gives it 1+K tokens in the step
   (`v1/core/sched/scheduler.py`, around line 677). It first reserves KV slots for them through
   `allocate_slots(..., num_lookahead_tokens=self.num_lookahead_tokens)` (scheduler.py:750;
   `kv_cache_manager.py:539`).
3. **Target verifies everything in one forward pass** over K+1 query positions per request.
   `_calc_spec_decode_metadata` in `v1/worker/gpu_model_runner.py` selects K logits that check the
   drafts plus one bonus logit. The profiler confirms this: each speculative step has exactly one target
   forward, annotated `generation_16(80)` for 16 requests x (4+1) tokens at K=4.
4. **Accept or reject.** `v1/sample/rejection_sampler.py`, `rejection_greedy_sample_kernel` (line 726),
   walks the drafts left to right. It emits the target's argmax at each position and stops at the first
   mismatch; if every draft matched, it appends the bonus token. So each step emits 1 to K+1 tokens, all
   exactly what the target would have produced alone. With temperature > 0, a draft token is accepted
   with probability min(1, p_target/p_draft); otherwise a token is resampled from the residual
   max(p - q, 0), which preserves the target distribution.
5. **KV and state stay consistent.** `update_from_output` computes
   `num_rejected = num_draft - num_accepted` and rolls back with
   `request.num_computed_tokens -= num_rejected` (scheduler.py:2067). Rejected positions are not erased.
   They are marked as not computed and overwritten when those positions are scheduled again. Prefix
   caching only commits KV up to `request.num_tokens`, so unverified drafts are never cached. The
   drafter's own KV is handled the same way (`prepare_inputs_padded` reduces its `seq_lens` by the number
   of rejected tokens).
6. **Decoding continues from the accepted sequence.** The next step drafts from the last emitted token,
   so no model state ever derives from a rejected token.

### Evidence from our runs

- **Lossless output** (`results/lossless_check.json`). At concurrency 1 with greedy decoding,
  **576/576** outputs from the eight EAGLE-3 and draft_model configs are byte-identical to the baseline,
  at every output length up to 512 tokens. A missed rollback or a stale KV entry would change the text.
  - n-gram matched 70/72. Its two divergences come 600-850 characters in, at near-ties ("use the formula
    for distance" vs "set up equations for the distance").
  - n-gram is the one method whose variable-length drafts change the target's kernel shapes (no
    uniform-decode FULL graph), so a bf16 rounding difference can flip a near-tie argmax.
  - At concurrency 16, exact-match is not a valid test. **Two identical baseline runs agree on only 51/72
    outputs**, because batch composition changes reduction order. The concurrency-1 result is the clean one.
- **Per-position acceptance** (`results/acceptance_by_position.png`, from
  `vllm:spec_decode_num_accepted_tokens_per_pos`) decays monotonically along the draft. Position i counts
  only if positions 0..i-1 were accepted, which is the stop-at-first-mismatch rule.
- **KV cost of a separate draft** (from the server startup logs):

  | Config | KV pool (tokens) | vs baseline |
  |---|---|---|
  | baseline | 141,664 | 1.00 |
  | ngram | 138,848 | 0.98 |
  | eagle3 | 129,584 | 0.91 |
  | draft_model | 58,496 | 0.41 |

  The 0.6B drafter's KV (28 layers x 8 KV heads) is larger per token than the target's (48 layers x 4 KV
  heads), and both share one pool. Peak KV usage for the same workload doubles: 13.5% vs 6.2% at
  512 tokens, concurrency 16.

## Part 3: Results

The main table is in `results/summary.md`; all 100 cells are in `results/all_cells.md` / `.csv`.
Below is the assignment's summary format for 128-token outputs, followed by the full K sweep.

| Configuration | K | Conc | TTFT p50 | TPOT p50 | Server tok/s | Acceptance | Tokens/step |
|---|---|---|---|---|---|---|---|
| Baseline | - | 1 | 28.7 | 4.48 | 214 | - | - |
| EAGLE-3 | 1 | 1 | 30.7 | 3.52 | 269 | 0.59 | 1.59 |
| EAGLE-3 | 2 | 1 | 33.8 | 3.63 | 258 | 0.45 | 1.90 |
| EAGLE-3 | 4 | 1 | 25.5 | 3.67 | 260 | 0.28 | 2.13 |
| EAGLE-3 | 8 | 1 | 25.9 | 4.49 | 217 | 0.15 | 2.22 |
| Draft 0.6B | 1 | 1 | 28.1 | 4.76 | 199 | 0.76 | 1.76 |
| Draft 0.6B | 2 | 1 | 43.6 | 6.21 | 154 | 0.67 | 2.34 |
| Draft 0.6B | 4 | 1 | 68.3 | 8.63 | 113 | 0.52 | 3.10 |
| Draft 0.6B | 8 | 1 | 118.3 | 13.99 | 70 | 0.35 | 3.79 |
| n-gram | 4 | 1 | 26.5 | 5.76 | 171 | 0.32* | 2.30* |
| Baseline | - | 4 | 66.3 | 7.74 | 486 | - | - |
| EAGLE-3 (best K at conc 1 = 1) | 1 | 4 | 47.2 | 5.77 | 634 | 0.61 | 1.61 |
| EAGLE-3 K=2 | 2 | 4 | 47.1 | 5.60 | 653 | 0.45 | 1.90 |
| Baseline | - | 16 | 128.6 | 12.44 | 1127 | - | - |
| EAGLE-3 (best K at conc 1 = 1) | 1 | 16 | 72.5 | 9.46 | 1463 | 0.59 | 1.59 |
| EAGLE-3 K=2 / K=4 | 2 / 4 | 16 | 68.8 / 60.0 | 8.83 / 8.83 | 1616 / 1609 | 0.45 / 0.28 | 1.90 / 2.14 |
| Draft 0.6B K=2 | 2 | 16 | 65.9 | 8.83 | 1589 | 0.67 | 2.34 |

\* n-gram acceptance and tokens/step count only steps where an n-gram matched. Steps with no match emit
one token and are not counted (`v1/spec_decode/metrics.py`; scheduler.py only calls `observe_draft` for
non-empty drafts), so these figures overstate n-gram's per-step benefit. Judge n-gram by TPOT and tok/s.

**Per-request latency vs server throughput.** These are separate columns and they can disagree:
- n-gram at concurrency 16 and 32 tokens: tok/s 1.12x but TPOT p50 0.97x. A few highly repetitive
  requests finish fast and raise the aggregate, while the median request does not improve.
- At concurrency 16, TTFT drops with speculation (128.6 ms baseline vs 60-73 ms EAGLE-3). This is a
  closed-loop queueing effect: requests finish sooner, so new requests wait less for a slot. Prefill
  itself does not get faster.

**Plots:** `results/speedup_eagle3.png`, `results/speedup_draft_model.png`, `results/speedup_ngram.png`
(per-request E2E speedup and server tok/s speedup vs K, per output length and concurrency) and
`results/acceptance_by_position.png`.

**Run-to-run noise** (`results/noise.md`). We ran every cell twice: a full first run, and the
128/512-token cells again as part of the final run.
- Median absolute difference is **0.9% for TPOT, 1.2% for throughput, 0.0% for acceptance**.
- GPU-bound configs (baseline, EAGLE-3) reproduce within about 2%.
- The CPU-bound draft_model configs were 12-17% faster in the second run. That is consistent with their
  sensitivity to host CPU state (Part 4).
- We therefore treat differences under about 3% as noise for GPU-bound configs and about 15% for
  draft_model. No conclusion below rests on a difference that small.
- The first run's 32-token cells were discarded. Triton JIT compiles (new `fused_moe` shapes) landed in
  them, which `vllm`'s JIT monitor flagged in the server log. The final run warms up every concurrency
  level and records zero post-warmup JIT compiles for every config (`results/logs/jit_after_warmup_*.txt`).

## Part 4: Profiling: where the time goes

**Tools.**
- The torch profiler, armed at server start with `--profiler-config`, captured traces via
  `/start_profile` and `/stop_profile` for baseline, EAGLE-3 K=4 and draft_model K=4 at concurrency 1
  and 16 (`benchmark/profile_load.py`, traces in `results/traces/`).
- `benchmark/analyze_trace.py` groups GPU kernels by CUPTI correlation id; one id corresponds to one
  launch call. A launch of 20 or more kernels is a CUDA-graph replay of a whole forward pass: the target
  if it contains `fused_moe` kernels (only the target is MoE), otherwise the drafter. It also reads
  vLLM's per-step GPU annotation `execute_context_X_generation_R(T)`, which gives step time vs tokens
  per step.
- NVML sampling during benchmarks, and `top` on the EngineCore process (`results/cpu_samples/`).

| Trace | GPU busy | Target fwd | Draft fwd (graphs) | Eager kernels | Eager launches | Step GPU time |
|---|---|---|---|---|---|---|
| baseline, conc 1 | 91% | 92% | - | 8% | 2,639 / 127 steps | 4.37 ms (1 token) |
| baseline, conc 16 | 92% | 96% | - | 5% | 3,497 / 127 steps | 12.41 ms (16 tokens) |
| eagle3 K=4, conc 1 | 96% | 85% | 8% | 7% | 2,178 / 62 steps | 6.60 ms (5 tokens) |
| eagle3 K=4, conc 16 | 97% | 90% | 4% | 6% | 2,920 / 68 steps | 17.23 ms (80 tokens) |
| draft_model K=4, conc 1 | **25%** | 52% | **0** | **48%** | **58,430 / 40 steps** | 8.36 ms (5 tokens) |
| draft_model K=4, conc 16 | **45%** | 68% | **0** | 32% | **71,962 / 49 steps** | 18.80 ms (80 tokens) |

### Bottleneck 1 (the main one): the separate draft model is CPU launch-bound

How we know:
1. **Its GPU is mostly idle.** It is busy only 25% of the traced window at concurrency 1, against 91-96%
   for baseline and EAGLE-3.
2. **None of its kernels are graph replays.** Zero launches of 20 or more kernels lack MoE kernels;
   there are about 1,460 eager launches per step against about 21 for the baseline. The source explains
   why: `llm_base_proposer.py:424` says the drafter "Only supports PIECEWISE cudagraphs (via
   mixed_mode)". Piecewise capture splits the model at every attention op, so each of the K passes of a
   28-layer model becomes dozens of small CPU-issued launches. The EAGLE-3 server log shows its drafter
   capturing FULL decode graphs ("Capturing decode CUDA graphs (FULL): 14"); the draft_model server log
   has no drafter capture at all.
3. **The engine process is CPU-saturated.** During the draft_model K=8 benchmark the EngineCore process
   ran at 90-110% CPU (one core) while NVML GPU utilization was 44-55%. During EAGLE-3 K=4 the GPU was
   at 100% and the CPU at 50-90%. Mean NVML utilization over whole runs: draft_model K=1/2/4/8 =
   87/77/65/51%, against 89-96% for baseline and EAGLE-3.
4. **Per-draft-token cost is far above what the FLOPs predict.** From TPOT x tokens/step
   (`results/step_cost.md`), each extra draft token adds 3.7-6.3 ms per step at concurrency 1. That is
   comparable to an entire target decode step (4.4 ms), for a model with 0.6B parameters against 3.3B
   active in the target. EAGLE-3 adds 0.7-1.2 ms per draft token.
5. **Run-to-run variance is CPU-shaped.** Only the draft_model cells moved (+12-17%) between runs.
   GPU-bound configs reproduced within about 2%.

The pattern is a step that costs more than the work it contains. **The GPU waits on Python and
kernel-launch overhead in the drafter loop.** Full CUDA-graph capture of the drafter, or a fused
multi-token drafter like EAGLE-3, removes it.

### Bottleneck 2: verification cost on an MoE target

Even with a free draft, verifying K+1 tokens is not free for this model.

| Tokens verified per step | 1 request | 16 requests |
|---|---|---|
| 1 per request (baseline) | 4.37 ms | 12.41 ms |
| 5 per request (K=4) | 6.60 ms (1.51x) | 17.23 ms (1.39x) |

- **At 1 request** the extra 4 tokens route to new experts, so the step loads more expert weights:
  50% more time for 5x the tokens.
- **At 16 requests** most of the 128 experts are already touched by 16 tokens x 8 experts, so 5x more
  tokens costs only 39% more.
- `fused_moe_kernel` is 59% of GPU time in the EAGLE-3 concurrency-1 trace and 80% at concurrency 16.
- vLLM also logged that no tuned fused-MoE config exists for this shape on H100
  ("Using default MoE config. Performance might be sub-optimal", `E=128,N=768`). That makes MoE kernels
  a larger share of the step than they need to be, both for the baseline and for verification. Tuning
  them (vLLM's `benchmark_moe.py`) is the obvious next optimization, but it is outside this
  assignment's budget.

### Smaller overheads

- **Eager work** (sampling, rejection sampling, input preparation) is 5-8% of GPU kernel time in every
  graph-based config.
- **n-gram's step** costs more than its verification work. vLLM disables async scheduling for the
  CPU n-gram proposer (`config/vllm.py`), and variable-length drafts lose uniform-decode FULL graphs.
  Its TPOT is 0.78-0.89x baseline at concurrency 1, even though "drafting is free".

## Part 5: Explaining the results

### Q1. When does speculative decoding help?

- **EAGLE-3 at K=1-4 helps everywhere we measured:**

  | Concurrency | Metric | Speedup |
  |---|---|---|
  | 1 | TPOT | 1.22-1.48x |
  | 4 | throughput | 1.28-1.52x |
  | 16 | throughput | 1.30-1.67x |

  It helps most on short outputs (32 tokens: up to 1.48x TPOT at concurrency 1, 1.67x throughput at
  concurrency 16). Early reasoning text is formulaic: acceptance is 0.71 at 32 tokens against 0.58 at
  512 tokens for K=1.
- **The 0.6B draft helps only under load.** It gives 1.28-1.68x TPOT at concurrency 16 for K=1-4, and
  1.15-1.39x throughput at concurrency 4 for K=1.
- **Mechanism.** Speedup = tokens per step / (step cost / baseline step cost). EAGLE-3 K=1 at
  128 tokens yields 1.59 tokens/step at 1.25x the cost of a baseline step, so 1.27x.

### Q2. When does it hurt?

| Config | Regime | TPOT vs baseline | Why |
|---|---|---|---|
| draft_model K=2/4/8 | conc 1, all lengths | 0.29-0.87x | Draft cost 3.7-6.3 ms per draft token (launch-bound) exceeds the value of the extra accepted tokens. At K=8 a step costs 11.9x a baseline step but yields 3.5-6.5 tokens |
| draft_model K=1 | conc 1, 128/512 tokens | 0.93-0.94x | 1.73-1.76 tokens/step at 1.86x step cost |
| draft_model K=8 | conc 16, 128/512 tokens | 0.82-0.83x | 8 sequential launch-bound draft passes even with batch amortization |
| ngram K=4 | conc 1, all; conc 4/16, 128/512 tokens | 0.78-0.97x | Our prompts are open-ended generation, so the prompt rarely contains the continuation (true acceptance is far below the 0.21-0.58 counter value). Async scheduling and FULL decode graphs are lost |
| eagle3 K=8 | conc 1, 128/512 tokens | 0.99-1.00x | 85% of drafts at positions 2-8 are rejected (acceptance 0.14-0.15), and verifying 9 tokens is not free on an MoE target |

In every losing case the extra per-step cost (draft passes, verification of rejected tokens, lost
graphs) exceeds what acceptance returns.

### Q3. How important is acceptance rate?

Necessary but not sufficient. The chain, with our numbers (128 tokens, concurrency 1, greedy):

```
draft quality -> acceptance -> tokens per target step -> latency/throughput
0.6B  K=1:     0.76      ->  1.76 tokens/step   x  step cost 1.87x  ->  TPOT 0.94x (slower)
EAGLE K=1:     0.59      ->  1.59 tokens/step   x  step cost 1.25x  ->  TPOT 1.27x (faster)
0.6B  K=4:     0.52      ->  3.10 tokens/step   x  step cost 5.98x  ->  TPOT 0.52x
EAGLE K=4:     0.28      ->  2.13 tokens/step   x  step cost 1.74x  ->  TPOT 1.22x
```

- **Higher acceptance means more tokens per step.** Tokens per step is 1 + accepted/drafts, and
  per-position acceptance decays geometrically along the draft (`acceptance_by_position.png`).
- **The gain is divided by what a step costs, and draft quality is bought with draft size.** The 0.6B
  model agrees with the target more often because it is a full language model, and the same depth makes
  each guess expensive.
- **EAGLE-3 takes the other end of that trade:** weaker guesses (one layer), almost free to produce,
  conditioned on the target's own hidden states.
- **Break-even.** A configuration wins only when tokens/step > step cost / baseline step cost. For
  EAGLE-3 the cost multiplier is 1.25-2.2x, so 1.6-2.2 tokens/step are enough. For the 0.6B draft at
  concurrency 1 the multiplier is 1.9-12x and 1.7-3.8 tokens/step never catch up.
- **Temperature.** At temperature 0.7 (with Qwen's default top-k 20 / top-p 0.95), outputs differ from
  greedy in 23/24 prompts, yet acceptance is unchanged (0.283 vs 0.282 for EAGLE-3 K=4; 0.527 vs 0.525
  for draft K=4), and so is TPOT. With greedy drafts, a token is accepted with probability
  p_target(draft). Qwen3's reasoning text is so peaked that this probability is close to the greedy
  match rate. Lower-confidence workloads would see a drop.

### Q4. What happens as K increases?

It does not keep improving.
- **Tokens per step saturate.** For EAGLE-3 at 128 tokens: 1.59 → 1.90 → 2.13 → 2.22 for K = 1, 2, 4, 8.
- **Acceptance collapses.** 0.59 → 0.45 → 0.28 → 0.15, because each extra position is accepted only if
  every earlier one was.
- **Cost keeps growing.** K sequential draft passes, plus K+1 verified positions per request, which on
  an MoE target load more experts. Step cost goes 1.25x → 1.54x → 1.74x → 2.22x.
- **The optimum is small.**
  - EAGLE-3: K=1 for single requests on 128/512-token outputs, K=2 at concurrency 4, K=2-4 at
    concurrency 16, and K=4 for 32-token outputs at every concurrency.
  - 0.6B draft: K=1 at low concurrency, K=2 at concurrency 16.
- **K=8 is worse than K=4 in every EAGLE-3 cell, and worse than baseline for the 0.6B draft at every
  concurrency on 128/512 tokens.**
- **The best K depends on the regime**, which is why vLLM offers per-batch-size K
  (`num_speculative_tokens_per_batch_size`, `v1/spec_decode/dynamic/`). We did not enable it.

### Q5. What happens under concurrency?

**Single-request conclusions do not carry over, in either direction.**

1. **The 0.6B draft flips from a loss to a win.** K=2 at 128 tokens has TPOT 0.72x at concurrency 1 and
   1.41x at concurrency 16.
   - Its cost multiplier falls from 3.25x to 1.66x. The drafter's launch overhead is mostly per pass,
     not per request, so a batch of 16 shares it.
   - Meanwhile the baseline step itself gets 2.8x more expensive (4.4 → 12.4 ms), because the MoE loads
     more experts.
2. **EAGLE-3's gain grows rather than shrinks.** Throughput rises from 1.21x (K=4, concurrency 1) to
   1.43x (concurrency 16) at 128 tokens, and from 1.38x to 1.67x at 32 tokens.
   - The classic expectation is that speculation stops paying once batching makes decode compute-bound.
   - At 16 requests on an H100 we are still memory-bound. MoE verification is relatively cheaper there:
     5 tokens per request costs 1.39x at batch 16 against 1.51x at batch 1, because the experts are
     already loaded.
   - We expect gains to shrink at much higher batch sizes or with a dense target; we did not test past 16.
3. **The best K moves up with concurrency.** For EAGLE-3 at 128 tokens the best K is 1 at concurrency 1
   and 2-4 at concurrency 16.

**How vLLM's continuous batching interacts with speculation:**
- **Draft tokens are just scheduled tokens.** Each decoding request takes 1+K slots from the per-step
  token budget (scheduler.py around lines 582/825). Rejected drafts consume budget and KV slots exactly
  like accepted ones.
- **Nothing turns speculation off under load.** vLLM 0.31 has no batch-size cutoff, only the opt-in
  dynamic K.
- **The target runs FULL CUDA graphs** for uniform batches of exactly 1+K tokens per request
  (`gpu_model_runner.py:883`, `uniform_decode_query_len = 1 + self.num_spec_tokens`), with capture sizes
  rounded to multiples of K+1. New requests get placeholder drafts to keep that shape. n-gram's
  variable-length drafts break it.
- **Async scheduling** (on for EAGLE-3 and draft_model, off for n-gram) overlaps the next step's
  scheduling with the current GPU work.
- **A separate draft model shares the KV pool**, cutting capacity to 41% of baseline. At our peak load
  (16 requests x 512 tokens, 13.5% usage) this never bound. In production, a separate draft model would
  make vLLM queue or preempt requests at about 2.4x lower concurrency than the baseline or EAGLE-3.

## Part 6: Internals deep dive: draft execution and CUDA graphs

We chose drafter execution and its CUDA-graph mode because it decides most of the results above: why the
better guesser loses at concurrency 1, why it recovers at concurrency 16, and why K has an optimum.

**What the code does**
- **Drafting is a Python loop of K forward passes** (`llm_base_proposer.py:689`). Draft cost is linear in
  K whether or not tokens are accepted.
- **The drafter is limited to piecewise CUDA graphs** (`llm_base_proposer.py:424`: "Only supports
  PIECEWISE cudagraphs (via mixed_mode)").
  - Piecewise mode splits the compiled graph at every op in `splitting_ops`, which includes
    `vllm::unified_attention_with_output` (see the `compilation_config` in any server log).
  - For a 28-layer drafter that means roughly 28 graph segments plus 28 attention launches per draft
    pass, each issued by the CPU, so about 60 or more launches x K passes per step.
  - The target, by contrast, replays one FULL graph per step
    (`gpu_model_runner.py:883`, uniform decode of 1+K tokens).
- **EAGLE-3 has one layer**, so piecewise or not, a draft pass is a handful of launches. Its server log
  also shows FULL decode-graph capture for the drafter.
- **The draft model's KV lives in the target's pool** (same `KVCacheConfig` group and block table;
  `uses_draft_kv_cache`). That is where the 0.41x KV capacity comes from.

**How it shows up in the benchmark**
- **Trace:** 58,430 eager kernel launches in 40 steps (about 1,460 per step), zero drafter graph
  replays, GPU busy 25%.
- **CPU sample:** EngineCore pinned at about 100% of a core while the GPU idles at about 50%.
- **Step cost:** 3.7-6.3 ms per draft token, roughly constant per pass. That is why cost grows linearly
  with K (`step_cost.md`) and why batching amortizes it (Q5).
- **Variance:** the configs that depend on CPU launch rate are the ones that moved between runs.

**Implication.** On this target, the choice between a small sibling model and an EAGLE-style head is less
about guess quality and more about whether the drafter can run as one fused, graph-captured launch. A
0.6B drafter with full CUDA-graph capture would cut most of the 3.7-6.3 ms per pass. Even then its
28-layer depth would keep it well above EAGLE-3's cost.

## Limitations

- **One GPU type, one target, 24 prompts.** These are mostly chat-style reasoning text, since Qwen3's
  thinking mode is on by default. Code completion or document-grounded tasks would favor n-gram.
- **Closed-loop load at fixed concurrency**, not an open-loop arrival process. TTFT at concurrency 16
  includes queueing.
- **`ignore_eos` fixes output lengths.** After a natural EOS, the model keeps generating, and that text
  can be more repetitive than a real answer.
- **The MoE kernels ran untuned** (vLLM's default config). Tuned kernels would make the target step
  cheaper, which shifts every break-even point.
- **Repeats.** Each final cell was run once in the final run. A full first run (128/512-token cells)
  plus replicates provide the noise estimate; percentiles are over 24-72 requests per cell.
- **Trace classification is heuristic.** Splitting kernels into target, draft and eager uses
  launch-group size and MoE kernel names. It is unambiguous for these three configs (counts match engine
  steps exactly) but is not general.

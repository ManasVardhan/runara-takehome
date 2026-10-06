## Part 2: What the engine does when speculation is on

Configured with `--speculative-config '{"method": ..., "model": ..., "num_speculative_tokens": K}'`
(`run_speculative.sh`). K is the only thing that changes between runs of one method. All paths below
are inside the installed `vllm` 0.31.0 package; line numbers were checked against that install.

### Draft model choice

| Draft | Why it is a sensible draft for Qwen3-30B-A3B | What it costs |
|---|---|---|
| `Qwen/Qwen3-0.6B` (`draft_model`) | Same family and tokenizer, identical 151,936-token vocabulary (vLLM requires matching vocabularies for exact verification); trained on similar data, so it agrees with the target often (73-93% per-token acceptance at K=1 in our runs) | A full 28-layer transformer run once per proposed token, with its own KV cache in the shared pool |
| `AngelSlim/Qwen3-a3B_eagle3` (`eagle3`) | Trained specifically on this target's hidden states (hidden size 2048 matches the target; draft vocabulary reduced to 32k tokens and mapped back), so it reuses the target's own features instead of recomputing them | One decoder layer per proposed token, plus an LM head over the reduced vocabulary |
| prompt n-gram (`ngram`) | Zero-parameter control: proposes the continuation of the longest matching n-gram from the prompt | CPU-side matching; nothing on the GPU |

The 0.6B model is the textbook "small model from the same family" choice, and it is the better
*guesser*. EAGLE-3 is the better *draft* in our measurements, because its guesses are nearly free.
That tension is the main result of this report (Part 5, Q3).

### One engine step, end to end

1. **Draft proposes K tokens.** `v1/spec_decode/llm_base_proposer.py` `propose()` runs one pass of the
   drafter over the tokens the target just processed, then loops
   `for token_index in range(self.num_speculative_tokens - 1)` (line 689), feeding back the previous draft
   token each time. Drafting is autoregressive: K sequential drafter passes per engine step.
   `draft_model.py` and `eagle.py` are thin subclasses; the difference is whether the target's hidden
   states are fed to the drafter (`pass_hidden_states_to_model`). n-gram proposals come from
   `ngram_proposer.py` on the CPU and can be shorter than K or empty.
2. **Scheduler books the draft tokens for verification.** A decoding request's token count includes its
   pending drafts (`v1/request.py`, `num_tokens_with_spec`), so the scheduler schedules 1 + K tokens for it
   (`v1/core/sched/scheduler.py` ~line 677) and records them in `scheduled_spec_decode_tokens`. Before that it
   allocates KV slots for them: `allocate_slots(..., num_lookahead_tokens=self.num_lookahead_tokens)`
   (scheduler.py:750; `kv_cache_manager.py:539` adds the lookahead to the slots needed).
3. **Target verifies all K in one forward pass.** The target runs once over K + 1 query positions per
   request; `_calc_spec_decode_metadata` in `v1/worker/gpu_model_runner.py` picks out the K target logits
   that check each draft and one "bonus" logit after the last draft. This is why speculation can win at
   all: decode is memory-bound, so one pass over 1 + K tokens costs far less than 1 + K passes. Our profile
   shows exactly how much less for this MoE target (Part 4).
4. **Accept or reject.** `v1/sample/rejection_sampler.py`. Greedy (our main matrix):
   `rejection_greedy_sample_kernel` (line 726) walks the drafts left to right, emits the target's argmax at
   each position, and stops at the first position where the draft differs. If every draft matched, the
   bonus token is appended. So each step emits between 1 and K + 1 tokens, all of them exactly what the
   target alone would have produced. With temperature > 0 the random path accepts a draft token with
   probability min(1, p_target / p_draft) and otherwise resamples from the residual max(p - q, 0), which
   preserves the target's sampling distribution.
5. **Keep KV and state consistent.** After the step, `update_from_output` computes
   `num_rejected = num_draft - num_accepted` and rolls the request back with
   `request.num_computed_tokens -= num_rejected` (scheduler.py:2067). The rejected positions' KV entries are
   not erased; they are simply treated as not computed and are overwritten when those positions are
   scheduled again. Prefix caching never commits them (`kv_cache_manager.py` only caches up to
   `request.num_tokens`, excluding unverified drafts). The drafter's own KV is handled the same way:
   `prepare_inputs_padded` shortens its `seq_lens` by the number of rejected tokens.
6. **Continue from the accepted sequence.** The accepted tokens plus the corrected or bonus token are
   appended; the next step drafts from that last token, so neither model ever carries state derived from
   a rejected token.

### Evidence that this is what happened in our runs

- **Lossless output.** With greedy decoding, every speculative configuration produced text identical to the
  baseline for every prompt and output length at concurrency 1 (`results/lossless_check.json`). Step 4 is
  doing exact verification, and step 5 never let rejected-token state leak into later tokens; a stale KV
  entry or a missed rollback would change the text.
- **Per-position acceptance** (`results/acceptance_by_position.png`, from
  `vllm:spec_decode_num_accepted_tokens_per_pos`) decays along the draft: position i is only reached if
  positions 0..i-1 were accepted, exactly the left-to-right stop rule of step 4.
- **Step structure in the profiler trace.** In every speculative trace the number of target forward passes
  equals the number of engine steps, and each step's GPU annotation reports `generation_<reqs>(<reqs x (K+1)>)`
  tokens (for example 16 requests x 5 tokens = 80 at K=4). That is one verification pass over K + 1
  positions per request (step 3).
- **KV-cache cost of a separate draft model.** Peak `kv_cache_usage_perc` for the same workload is about 2x
  higher with `draft_model` than with the baseline (for example 0.13 vs 0.06 at 512 tokens, concurrency 16).
  The 0.6B drafter's KV (28 layers x 8 KV heads) lives in the same block pool as the target's
  (48 layers x 4 KV heads), so each token needs about 2.2x the KV memory. EAGLE-3 (1 layer) adds about 2%.

## Part 6: Internals deep dive: how a step's cost is built, and why the 0.6B draft loses

The aspect investigated is **draft execution and CUDA graphs**, together with how verification batches
into the target's forward pass. It decides almost every result in this report.

### What the code says

- **Drafting is K sequential passes** (`llm_base_proposer.py:689`). Draft cost therefore grows linearly
  with K regardless of acceptance; an accepted-or-not draft token costs the same to produce.
- **The drafter only supports piecewise CUDA graphs.** `llm_base_proposer.py:424`: "Only supports
  PIECEWISE cudagraphs (via mixed_mode)". Piecewise capture splits the model at every attention op
  (the `splitting_ops` list in the server's compilation config includes
  `vllm::unified_attention_with_output`). Each draft pass of a 28-layer model is therefore dozens of
  separate graph replays and eager attention launches issued by the CPU, instead of one replay.
- **The target uses FULL CUDA graphs for uniform decode batches** of exactly 1 + K tokens per request
  (`gpu_model_runner.py:883`: `self.uniform_decode_query_len = 1 + self.num_spec_tokens`), with capture
  sizes rounded to multiples of K + 1. New requests are padded with placeholder drafts to keep that shape.
  One verification pass is one graph replay.
- **Nothing turns speculation off under load.** There is no batch-size cutoff in 0.31.0; draft tokens
  count fully against the token budget (scheduler.py ~582/825). The only adaptation is opt-in dynamic K
  (`num_speculative_tokens_per_batch_size`, `v1/spec_decode/dynamic/`), which we did not enable.

### What the trace says (`results/profile_summary.json`)

`benchmark/analyze_trace.py` groups kernels by CUPTI correlation id (one id per launch call). A launch with
20 or more kernels is a CUDA-graph replay of a model forward; it is the target if it contains `fused_moe`
kernels (only the target is MoE), otherwise the drafter.

FILL: table from profile_summary (launch counts, busy fraction, target/draft/eager split) for baseline,
eagle3 K=4, draft_model K=4 at conc 1 and 16.

### How it shapes the benchmark

FILL after run 2: connect the eager-launch-bound draft to the per-draft-token step cost (step_cost.md),
the concurrency flip, and K behavior.

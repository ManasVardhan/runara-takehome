#!/usr/bin/env python3
"""Closed-loop streaming benchmark against a running vLLM OpenAI server.

For every (output_len, concurrency) cell it:
  1. scrapes /metrics (spec-decode counters, KV usage) before the cell,
  2. samples GPU util / memory with NVML in a background thread,
  3. sends N streaming chat requests with at most `concurrency` in flight,
  4. scrapes /metrics again and records the deltas.

Per-request latency (TTFT, TPOT, ITL, E2E) and server throughput (output tok/s,
req/s) are reported as separate fields. One JSON row per cell is appended to
--out; per-request records go to <out>.requests.jsonl.
"""
import argparse
import asyncio
import json
import os
import re
import statistics
import threading
import time

import aiohttp

# Spec-decode counters exported by the vLLM V1 engine (names verified against
# the installed version in vllm/v1/spec_decode/metrics.py).
SPEC_COUNTERS = {
    "drafts": "vllm:spec_decode_num_drafts_total",
    "draft_tokens": "vllm:spec_decode_num_draft_tokens_total",
    "accepted_tokens": "vllm:spec_decode_num_accepted_tokens_total",
}
PER_POS = "vllm:spec_decode_num_accepted_tokens_per_pos_total"
KV_GAUGE = "vllm:kv_cache_usage_perc"
RUNNING_GAUGE = "vllm:num_requests_running"
METRIC_LINE = re.compile(r"^([a-zA-Z_:][a-zA-Z0-9_:]*)(\{[^}]*\})?\s+([0-9.eE+-]+|NaN)$")


def parse_metrics(text):
    """Return {name: total_over_labels} and {position: value} for per-pos counter."""
    totals, per_pos = {}, {}
    for line in text.splitlines():
        if not line or line.startswith("#"):
            continue
        m = METRIC_LINE.match(line.strip())
        if not m:
            continue
        name, labels, value = m.group(1), m.group(2) or "", float(m.group(3))
        totals[name] = totals.get(name, 0.0) + value
        if name == PER_POS:
            pos = re.search(r'position="(\d+)"', labels)
            if pos:
                per_pos[int(pos.group(1))] = per_pos.get(int(pos.group(1)), 0.0) + value
    return totals, per_pos


async def scrape(session, base_url):
    async with session.get(f"{base_url}/metrics") as r:
        return parse_metrics(await r.text())


class GpuSampler(threading.Thread):
    """Samples GPU utilization and memory every `interval` seconds via NVML."""

    def __init__(self, interval=0.1):
        super().__init__(daemon=True)
        self.interval, self.samples, self._halt = interval, [], threading.Event()

    def run(self):
        try:
            import pynvml
            pynvml.nvmlInit()
            h = pynvml.nvmlDeviceGetHandleByIndex(int(os.environ.get("GPU_INDEX", "0")))
        except Exception as e:  # sampler is best effort; never kill the benchmark
            print(f"[gpu-sampler] disabled: {e}")
            return
        while not self._halt.is_set():
            u = pynvml.nvmlDeviceGetUtilizationRates(h)
            mem = pynvml.nvmlDeviceGetMemoryInfo(h)
            self.samples.append((u.gpu, u.memory, mem.used / 2**30, mem.total / 2**30))
            time.sleep(self.interval)

    def stop(self):
        self._halt.set()
        self.join(timeout=2)

    def summary(self):
        if not self.samples:
            return {}
        sm, membw, used, total = zip(*self.samples)
        return {
            "gpu_util_mean": statistics.mean(sm),
            "gpu_mem_bw_util_mean": statistics.mean(membw),
            "gpu_mem_used_gib_max": max(used),
            "gpu_mem_total_gib": total[0],
        }


async def one_request(session, args, prompt, output_len):
    body = {
        "model": args.model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": output_len,
        "temperature": args.temperature,
        "ignore_eos": True,  # vLLM extension: makes output length exactly output_len
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    if args.temperature > 0:
        body["seed"] = args.seed
    t0 = time.perf_counter()
    ttft, chunk_times, text, usage = None, [], [], None
    async with session.post(f"{args.base_url}/v1/chat/completions", json=body) as r:
        r.raise_for_status()
        async for raw in r.content:
            line = raw.decode().strip()
            if not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if data == "[DONE]":
                break
            obj = json.loads(data)
            if obj.get("usage"):
                usage = obj["usage"]
            for ch in obj.get("choices", []):
                delta = ch.get("delta", {})
                piece = (delta.get("content") or "") + (delta.get("reasoning_content") or "")
                if piece:
                    now = time.perf_counter()
                    if ttft is None:
                        ttft = now - t0
                    chunk_times.append(now)
                    text.append(piece)
    e2e = time.perf_counter() - t0
    n_out = usage["completion_tokens"] if usage else None
    # TPOT excludes the first token; with speculation one chunk can carry several
    # tokens, so per-chunk ITL is bursty and TPOT is the fair per-token number.
    tpot = (e2e - ttft) / (n_out - 1) if (n_out and n_out > 1 and ttft is not None) else None
    itls = [b - a for a, b in zip(chunk_times, chunk_times[1:])]
    return {
        "ttft_s": ttft, "tpot_s": tpot, "e2e_s": e2e, "output_tokens": n_out,
        "prompt_tokens": usage["prompt_tokens"] if usage else None,
        "num_chunks": len(chunk_times), "itl_mean_s": statistics.mean(itls) if itls else None,
        "text": "".join(text),
    }


def ms(xs, p):
    v = pct(xs, p)
    return None if v is None else v * 1e3


def safe_mean(xs):
    xs = [x for x in xs if x is not None]
    return statistics.mean(xs) if xs else None


def pct(xs, p):
    xs = sorted(x for x in xs if x is not None)
    if not xs:
        return None
    k = (len(xs) - 1) * p / 100
    lo, hi = int(k), min(int(k) + 1, len(xs) - 1)
    return xs[lo] + (xs[hi] - xs[lo]) * (k - lo)


async def run_cell(args, prompts, output_len, conc):
    # Whole multiples of the prompt set, so every cell sees the same prompt mix.
    want = max(args.min_requests, args.requests_per_conc * conc)
    n = len(prompts) * -(-want // len(prompts))
    cell_prompts = [prompts[i % len(prompts)] for i in range(n)]
    timeout = aiohttp.ClientTimeout(total=None, sock_read=600)
    async with aiohttp.ClientSession(timeout=timeout) as s:
        before, pos_before = await scrape(s, args.base_url)
        sampler = GpuSampler()
        sampler.start()
        sem = asyncio.Semaphore(conc)

        async def guarded(i, p):
            async with sem:
                return {"req_idx": i, **await one_request(s, args, p, output_len)}

        kv_samples, done = [], asyncio.Event()

        async def poll_kv():
            while not done.is_set():
                try:
                    tot, _ = await scrape(s, args.base_url)
                    kv_samples.append((tot.get(KV_GAUGE), tot.get(RUNNING_GAUGE)))
                except Exception:
                    pass
                await asyncio.sleep(0.5)

        poller = asyncio.create_task(poll_kv())
        t0 = time.perf_counter()
        results = await asyncio.gather(*(guarded(i, p) for i, p in enumerate(cell_prompts)), return_exceptions=True)
        wall = time.perf_counter() - t0
        done.set()
        await poller
        sampler.stop()
        after, pos_after = await scrape(s, args.base_url)
    errors = [repr(r) for r in results if isinstance(r, BaseException)]
    reqs = [r for r in results if not isinstance(r, BaseException)]
    kv = [x for x, _ in kv_samples if x is not None]
    running = [y for _, y in kv_samples if y is not None]

    d = {k: after.get(v, 0.0) - before.get(v, 0.0) for k, v in SPEC_COUNTERS.items()}
    per_pos = {p: pos_after.get(p, 0.0) - pos_before.get(p, 0.0) for p in sorted(pos_after)}
    total_out = sum(r["output_tokens"] or 0 for r in reqs)
    row = {
        "label": args.label, "method": args.method, "k": args.k,
        "output_len": output_len, "concurrency": conc, "num_requests": n,
        "temperature": args.temperature, "max_tokens": output_len, "ignore_eos": True,
        "seed": args.seed if args.temperature > 0 else None, "spec_config": args.spec_config,
        "errors": len(errors), "error_samples": errors[:3],
        # per-request latency (p99 over ~24-72 samples is close to the max; read it as a tail bound)
        **{f"ttft_{p}_ms": ms([r["ttft_s"] for r in reqs], p) for p in (50, 90, 99)},
        **{f"tpot_{p}_ms": ms([r["tpot_s"] for r in reqs], p) for p in (50, 90, 99)},
        **{f"e2e_{p}_ms": ms([r["e2e_s"] for r in reqs], p) for p in (50, 90, 99)},
        "ttft_mean_ms": (safe_mean([r["ttft_s"] for r in reqs]) or 0) * 1e3 if reqs else None,
        "tpot_mean_ms": (safe_mean([r["tpot_s"] for r in reqs]) or 0) * 1e3 if reqs else None,
        # mean gap between streamed chunks; one chunk can hold several tokens under speculation
        "chunk_itl_mean_ms": (safe_mean([r["itl_mean_s"] for r in reqs]) or 0) * 1e3 if reqs else None,
        "per_request_tok_s": safe_mean([
            (r["output_tokens"] - 1) / (r["e2e_s"] - r["ttft_s"]) for r in reqs
            if r["output_tokens"] and r["output_tokens"] > 1 and r["ttft_s"] is not None]),
        # server throughput
        "wall_s": wall, "output_tok_s": total_out / wall, "req_s": len(reqs) / wall,
        "short_outputs": sum(1 for r in reqs if r["output_tokens"] != output_len),
        "kv_cache_usage_max": max(kv) if kv else None,
        "running_reqs_mean": safe_mean(running),
        # speculative decoding
        "spec_drafts": d["drafts"], "spec_draft_tokens": d["draft_tokens"],
        "spec_accepted_tokens": d["accepted_tokens"],
        "acceptance_rate": d["accepted_tokens"] / d["draft_tokens"] if d["draft_tokens"] else None,
        # tokens emitted per target verification step = accepted drafts + 1 bonus/corrected token
        "accepted_per_draft": d["accepted_tokens"] / d["drafts"] if d["drafts"] else None,
        "mean_accepted_len": 1 + d["accepted_tokens"] / d["drafts"] if d["drafts"] else None,
        "acceptance_by_position": {str(p): v / d["drafts"] for p, v in per_pos.items()} if d["drafts"] else None,
        **sampler.summary(),
    }
    return row, reqs


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default="http://localhost:8000")
    ap.add_argument("--model", default="Qwen/Qwen3-30B-A3B")
    ap.add_argument("--prompts", default=os.path.join(os.path.dirname(__file__), "prompts.jsonl"))
    ap.add_argument("--output-lens", default="32,128,512")
    ap.add_argument("--concurrency", default="1,4,16")
    ap.add_argument("--requests-per-conc", type=int, default=4, help="requests = max(min, this * conc)")
    ap.add_argument("--min-requests", type=int, default=24)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--warmup", type=int, default=2)
    ap.add_argument("--label", required=True)
    ap.add_argument("--method", default="none")
    ap.add_argument("--k", type=int, default=0)
    ap.add_argument("--spec-config", default=None, help="recorded verbatim in every row")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    with open(args.prompts) as f:
        prompts = [json.loads(l)["prompt"] for l in f if l.strip()]
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)

    # Warmup. vLLM JIT-compiles some Triton kernels (rejection/resample sampler kernels,
    # top-k/top-p and Gumbel sampling for temperature > 0, fused_moe for new token-count
    # shapes) on first use, which stalls the engine for seconds. Run every concurrency
    # level at the same sampling settings first so no compile lands inside a measured cell.
    # sweep.sh then checks the server log for any JIT compile after WARMUP_DONE.
    async with aiohttp.ClientSession() as s:
        for _ in range(args.warmup):
            for conc in [int(x) for x in args.concurrency.split(",")] + [64]:
                await asyncio.gather(*(one_request(s, args, prompts[i % len(prompts)], 48)
                                       for i in range(conc)))
    print(f"WARMUP_DONE {time.strftime('%m-%d %H:%M:%S', time.gmtime())}", flush=True)

    for output_len in [int(x) for x in args.output_lens.split(",")]:
        for conc in [int(x) for x in args.concurrency.split(",")]:
            row, reqs = await run_cell(args, prompts, output_len, conc)
            with open(args.out, "a") as f:
                f.write(json.dumps(row) + "\n")
            with open(args.out + ".requests.jsonl", "a") as f:
                for r in reqs:
                    f.write(json.dumps({"label": args.label, "output_len": output_len,
                                        "concurrency": conc, "idx": r["req_idx"], **r}) + "\n")
            f = lambda v, nd=2: "-" if v is None else f"{v:.{nd}f}"
            print(f"[{args.label}] len={output_len} conc={conc} n={row['num_requests']} "
                  f"err={row['errors']} ttft_p50={f(row['ttft_50_ms'], 1)}ms "
                  f"tpot_p50={f(row['tpot_50_ms'])}ms tok/s={f(row['output_tok_s'], 1)} "
                  f"acc={f(row['acceptance_rate'], 3)} mal={f(row['mean_accepted_len'])} "
                  f"kv_max={f(row['kv_cache_usage_max'], 3)} short={row['short_outputs']}", flush=True)


if __name__ == "__main__":
    asyncio.run(main())

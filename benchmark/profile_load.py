#!/usr/bin/env python3
"""Capture a torch-profiler trace of the running vLLM server under a fixed load.

The server must be started with --profiler-config (see common.sh). Between
/start_profile and /stop_profile we run `--requests` streaming requests at the
given concurrency, then move the trace files the engine wrote into
results/traces/<label>_c<conc>/.
"""
import argparse
import asyncio
import glob
import os
import shutil
import sys
import time

import aiohttp

sys.path.insert(0, os.path.dirname(__file__))
from benchmark import one_request  # noqa: E402


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default="http://localhost:8000")
    ap.add_argument("--model", default="Qwen/Qwen3-30B-A3B")
    ap.add_argument("--prompts", default=os.path.join(os.path.dirname(__file__), "prompts.jsonl"))
    ap.add_argument("--trace-dir", required=True, help="the server's torch_profiler_dir")
    ap.add_argument("--label", required=True)
    ap.add_argument("--concurrency", type=int, default=1)
    ap.add_argument("--requests", type=int, default=0, help="default: concurrency")
    ap.add_argument("--output-len", type=int, default=128)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out-root", default="results/traces")
    args = ap.parse_args()
    import json
    with open(args.prompts) as f:
        prompts = [json.loads(l)["prompt"] for l in f if l.strip()]
    n = args.requests or args.concurrency
    before = set(glob.glob(os.path.join(args.trace_dir, "**", "*"), recursive=True))
    timeout = aiohttp.ClientTimeout(total=None, sock_read=900)
    async with aiohttp.ClientSession(timeout=timeout) as s:
        await one_request(s, args, prompts[0], 8)  # make sure the engine is warm
        async with s.post(f"{args.base_url}/start_profile") as r:
            r.raise_for_status()
        sem = asyncio.Semaphore(args.concurrency)

        async def go(p):
            async with sem:
                return await one_request(s, args, p, args.output_len)

        t0 = time.perf_counter()
        await asyncio.gather(*(go(prompts[i % len(prompts)]) for i in range(n)))
        wall = time.perf_counter() - t0
        async with s.post(f"{args.base_url}/stop_profile") as r:
            r.raise_for_status()
    # The engine writes traces asynchronously; wait until new files stop growing.
    dest = os.path.join(args.out_root, f"{args.label}_c{args.concurrency}")
    os.makedirs(dest, exist_ok=True)
    last, stable = None, 0
    for _ in range(240):
        new = sorted(set(glob.glob(os.path.join(args.trace_dir, "**", "*"), recursive=True)) - before)
        files = [f for f in new if os.path.isfile(f)]
        sizes = tuple(os.path.getsize(f) for f in files)
        stable = stable + 1 if files and sizes == last else 0
        last = sizes
        if stable >= 3:
            break
        time.sleep(2)
    for f in files:
        shutil.move(f, os.path.join(dest, os.path.basename(f)))
    print(f"[profile] {args.label} conc={args.concurrency} n={n} wall={wall:.2f}s -> {dest}: "
          f"{[os.path.basename(f) for f in files]}", flush=True)


if __name__ == "__main__":
    asyncio.run(main())

#!/usr/bin/env python3
"""Break a vLLM torch-profiler trace down into where engine-step time goes.

vLLM's GPU model runner wraps each phase of an engine step in a
record_function range ("gpu_model_runner: forward", "...: draft", "...: sample",
"...: bookkeep", ...; see vllm/v1/worker/gpu_model_runner.py). Every kernel is
attributed to the phase whose CPU range enclosed its launch (via the CUPTI
correlation id), which works for CUDA-graph launches too. We report:

  * GPU kernel time per phase (target forward vs draft proposal vs sampling)
  * GPU busy fraction of the traced window (idle = CPU/scheduler/sync overhead)
  * kernel-time split by kernel family (MoE experts, attention, GEMM, other)
  * engine steps and mean step time
"""
import argparse
import bisect
import glob
import gzip
import json
import os
from collections import defaultdict

PHASE_PREFIX = "gpu_model_runner: "


def load(path):
    op = gzip.open if path.endswith(".gz") else open
    with op(path, "rt") as f:
        return json.load(f)["traceEvents"]


def family(name):
    n = name.lower()
    if "moe" in n or "expert" in n or "topk" in n:
        return "moe"
    if "attn" in n or "attention" in n or "flash" in n or "fmha" in n:
        return "attention"
    if "gemm" in n or "nvjet" in n or "cutlass" in n or "sm90" in n or "matmul" in n:
        return "gemm"
    return "other"


def union_len(intervals):
    total, end = 0.0, -1.0
    for s, e in sorted(intervals):
        if e <= end:
            continue
        total += e - max(s, end)
        end = e
    return total


def analyze(path):
    ev = [e for e in load(path) if e.get("ph") == "X"]
    kernels = [e for e in ev if e.get("cat") == "kernel"]
    runtime = [e for e in ev if e.get("cat") in ("cuda_runtime", "cuda_driver")]
    phases = [e for e in ev if e.get("cat") == "user_annotation" and e["name"].startswith(PHASE_PREFIX)]
    if not kernels:
        return {"file": path, "error": "no kernel events"}

    # Enclosing phase for each runtime call: same thread, innermost range containing it.
    by_tid = defaultdict(list)
    for p in phases:
        by_tid[(p["pid"], p["tid"])].append((p["ts"], p["ts"] + p["dur"], p["name"][len(PHASE_PREFIX):]))
    for v in by_tid.values():
        v.sort()
    starts = {k: [s for s, _, _ in v] for k, v in by_tid.items()}

    corr_phase = {}
    for r in runtime:
        c = (r.get("args") or {}).get("correlation")
        key = (r["pid"], r["tid"])
        if c is None or key not in by_tid:
            continue
        i = bisect.bisect_right(starts[key], r["ts"]) - 1
        best = None
        while i >= 0:
            s, e, name = by_tid[key][i]
            if s <= r["ts"] <= e:
                best = name if best is None else best  # innermost = latest start
                break
            i -= 1
        if best:
            corr_phase[c] = best

    phase_time = defaultdict(float)
    fam_time = defaultdict(float)
    fam_by_phase = defaultdict(lambda: defaultdict(float))
    for k in kernels:
        ph = corr_phase.get((k.get("args") or {}).get("correlation"), "unattributed")
        fam = family(k["name"])
        phase_time[ph] += k["dur"]
        fam_time[fam] += k["dur"]
        fam_by_phase[ph][fam] += k["dur"]

    t0 = min(k["ts"] for k in kernels)
    t1 = max(k["ts"] + k["dur"] for k in kernels)
    window = t1 - t0
    busy = union_len([(k["ts"], k["ts"] + k["dur"]) for k in kernels])
    kernel_total = sum(k["dur"] for k in kernels)
    steps = sum(1 for p in phases if p["name"] == PHASE_PREFIX + "forward")
    return {
        "file": os.path.basename(path),
        "window_ms": window / 1e3,
        "gpu_busy_frac": busy / window if window else None,
        "kernel_ms_total": kernel_total / 1e3,
        "engine_steps": steps,
        "mean_step_ms": window / steps / 1e3 if steps else None,
        "kernel_ms_by_phase": {k: round(v / 1e3, 2) for k, v in sorted(phase_time.items(), key=lambda x: -x[1])},
        "kernel_frac_by_phase": {k: round(v / kernel_total, 3) for k, v in sorted(phase_time.items(), key=lambda x: -x[1])},
        "kernel_frac_by_family": {k: round(v / kernel_total, 3) for k, v in sorted(fam_time.items(), key=lambda x: -x[1])},
        "family_by_phase_ms": {p: {f: round(v / 1e3, 2) for f, v in d.items()} for p, d in fam_by_phase.items()},
        "top_kernels": top_kernels(kernels),
    }


def top_kernels(kernels, n=8):
    agg = defaultdict(float)
    for k in kernels:
        agg[k["name"][:90]] += k["dur"]
    tot = sum(agg.values())
    return [(name, round(v / tot, 3)) for name, v in sorted(agg.items(), key=lambda x: -x[1])[:n]]


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("dirs", nargs="+", help="trace directories (results/traces/<label>_c<conc>)")
    ap.add_argument("--out", default="results/profile_summary.json")
    args = ap.parse_args()
    out = {}
    for d in args.dirs:
        files = sorted(glob.glob(os.path.join(d, "*.json*")))
        # The engine-core (GPU worker) trace is the one with kernels; skip frontend-only traces.
        for f in files:
            r = analyze(f)
            if "error" not in r:
                out[os.path.basename(d.rstrip("/"))] = r
                break
        else:
            out[os.path.basename(d.rstrip("/"))] = {"error": f"no kernel trace in {files}"}
    with open(args.out, "w") as fh:
        json.dump(out, fh, indent=2)
    for name, r in out.items():
        if "error" in r:
            print(name, r["error"])
            continue
        print(f"{name}: steps={r['engine_steps']} step={r['mean_step_ms']:.2f}ms busy={r['gpu_busy_frac']:.2f} "
              f"phase={r['kernel_frac_by_phase']} family={r['kernel_frac_by_family']}")

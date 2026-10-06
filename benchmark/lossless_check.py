#!/usr/bin/env python3
"""Compare greedy outputs of a speculative run against the baseline run.

Speculative decoding with rejection sampling is lossless: with temperature=0
every emitted token must be the target model's argmax. Exact string equality is
the strongest evidence. Small divergences can still come from bf16 numerics
(a verification forward over K+1 positions uses different kernel shapes than
a 1-token decode step), so we also report the matching-prefix length.
"""
import argparse
import json
import os


def load(path, conc):
    out = {}
    with open(path) as f:
        for line in f:
            r = json.loads(line)
            if r["concurrency"] == conc:
                out[(r["output_len"], r["idx"])] = r["text"]
    return out


def common_prefix(a, b):
    n = 0
    for x, y in zip(a, b):
        if x != y:
            break
        n += 1
    return n


ap = argparse.ArgumentParser()
ap.add_argument("--baseline", required=True)
ap.add_argument("--spec", nargs="+", required=True)
ap.add_argument("--concurrency", type=int, default=1)
ap.add_argument("--out")
args = ap.parse_args()
base = load(args.baseline, args.concurrency)
report = []
for path in args.spec:
    spec = load(path, args.concurrency)
    keys = sorted(set(base) & set(spec))
    exact = sum(base[k] == spec[k] for k in keys)
    prefix = [common_prefix(base[k], spec[k]) / max(1, len(base[k])) for k in keys]
    row = {"run": os.path.basename(path).replace(".jsonl.requests.jsonl", ""),
           "compared": len(keys), "exact_match": exact,
           "mean_matching_prefix_frac": sum(prefix) / len(prefix) if prefix else None}
    report.append(row)
    print(row)
if args.out:
    with open(args.out, "w") as f:
        json.dump(report, f, indent=2)

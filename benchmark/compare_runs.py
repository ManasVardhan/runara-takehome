#!/usr/bin/env python3
"""Run-to-run noise: compare identical cells from two complete sweeps.

Run 1 (results/run1/raw) used a warmup that did not cover every batch shape, so
Triton JIT compiles landed in some of its first cells (output_len=32). Those
cells are excluded; the 128/512 cells reuse already-compiled shapes and are a
genuine repeat of run 2 on a fresh server.
"""
import argparse
import glob
import json
import os

import pandas as pd

ap = argparse.ArgumentParser()
ap.add_argument("--a", default="results/run1/raw")
ap.add_argument("--b", default="results/raw")
ap.add_argument("--exclude-lens", default="32")
ap.add_argument("--out", default="results/noise.md")
args = ap.parse_args()


def load(d):
    rows = []
    for p in glob.glob(os.path.join(d, "*.jsonl")):
        if p.endswith(".requests.jsonl"):
            continue
        rows += [json.loads(l) for l in open(p) if l.strip()]
    return pd.DataFrame(rows)


a, b = load(args.a), load(args.b)
key = ["label", "output_len", "concurrency"]
m = a.merge(b, on=key, suffixes=("_1", "_2"))
m = m[~m.output_len.isin([int(x) for x in args.exclude_lens.split(",")])]
m = m[m.label.str.fullmatch(r"(baseline|eagle3_k\d|draft_model_k\d|ngram_k\d)")]
for c in ["tpot_50_ms", "output_tok_s", "ttft_50_ms", "acceptance_rate"]:
    m[f"{c}_reldiff"] = (m[f"{c}_2"] - m[f"{c}_1"]) / m[f"{c}_1"]
lines = [f"Cells compared: {len(m)} (labels x output_len in (128, 512) x concurrency)\n",
         "| Metric | median abs. rel. diff | 90th pct abs. rel. diff | max abs. rel. diff |",
         "|---|---|---|---|"]
for c in ["tpot_50_ms", "output_tok_s", "ttft_50_ms", "acceptance_rate"]:
    s = m[f"{c}_reldiff"].abs().dropna()
    if s.empty:
        continue
    lines.append(f"| {c} | {s.median():.1%} | {s.quantile(0.9):.1%} | {s.max():.1%} |")
lines += ["", "By concurrency (median abs. rel. diff):", "",
          "| Concurrency | TPOT p50 | output tok/s |", "|---|---|---|"]
for conc, g in m.groupby("concurrency"):
    lines.append(f"| {conc} | {g.tpot_50_ms_reldiff.abs().median():.1%} | {g.output_tok_s_reldiff.abs().median():.1%} |")
worst = m.reindex(m.output_tok_s_reldiff.abs().sort_values(ascending=False).index).head(5)
lines += ["", "Largest throughput differences:", "", "| Label | len | conc | tok/s run 1 | tok/s run 2 | diff |", "|---|---|---|---|---|---|"]
for _, r in worst.iterrows():
    lines.append(f"| {r.label} | {r.output_len} | {r.concurrency} | {r.output_tok_s_1:.0f} | {r.output_tok_s_2:.0f} | {r.output_tok_s_reldiff:+.1%} |")
open(args.out, "w").write("\n".join(lines) + "\n")
print("\n".join(lines))

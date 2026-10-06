#!/usr/bin/env python3
"""Build results tables (markdown + CSV) and plots from results/raw/*.jsonl."""
import argparse
import glob
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--raw", default="results/raw")
ap.add_argument("--out", default="results")
args = ap.parse_args()

rows = []
for path in sorted(glob.glob(os.path.join(args.raw, "*.jsonl"))):
    if path.endswith(".requests.jsonl"):
        continue
    with open(path) as f:
        rows += [json.loads(l) for l in f if l.strip()]
df = pd.DataFrame(rows)
if df.empty:
    raise SystemExit("no results")
# A rerun of a config overwrites its file, but guard against duplicate cells anyway.
df = df.drop_duplicates(subset=["label", "output_len", "concurrency"], keep="last")
df.to_csv(os.path.join(args.out, "all_cells.csv"), index=False)

# Baseline for each cell = the baseline run at the same temperature (and no replicate suffix).
base = df[(df.method == "none") & df.label.str.fullmatch(r"baseline(_t[0-9.]+)?")][
    ["output_len", "concurrency", "temperature", "tpot_50_ms", "output_tok_s", "e2e_50_ms", "ttft_50_ms"]
].rename(columns={"tpot_50_ms": "b_tpot", "output_tok_s": "b_tok_s", "e2e_50_ms": "b_e2e", "ttft_50_ms": "b_ttft"})
df = df.merge(base, on=["output_len", "concurrency", "temperature"], how="left")
df["e2e_speedup"] = df.b_e2e / df.e2e_50_ms        # per-request latency gain
df["throughput_speedup"] = df.output_tok_s / df.b_tok_s  # server throughput gain


def fmt(v, nd=1):
    return "-" if v is None or pd.isna(v) else f"{v:.{nd}f}"


cols = ["label", "temperature", "k", "output_len", "concurrency", "ttft_50_ms", "tpot_50_ms", "e2e_50_ms",
        "per_request_tok_s", "output_tok_s", "req_s", "acceptance_rate", "mean_accepted_len",
        "e2e_speedup", "throughput_speedup", "gpu_util_mean", "gpu_mem_used_gib_max"]
lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
for _, r in df.sort_values(["output_len", "concurrency", "method", "k"]).iterrows():
    vals = []
    for c in cols:
        v = r.get(c)
        if c in ("label",):
            vals.append(str(v))
        elif c in ("k", "output_len", "concurrency"):
            vals.append("-" if pd.isna(v) else str(int(v)))
        elif c in ("acceptance_rate", "e2e_speedup", "throughput_speedup", "mean_accepted_len", "temperature"):
            vals.append(fmt(v, 2))
        else:
            vals.append(fmt(v, 1))
    lines.append("| " + " | ".join(vals) + " |")
with open(os.path.join(args.out, "all_cells.md"), "w") as f:
    f.write("\n".join(lines) + "\n")

# Plots: one figure per method, x = K, lines per concurrency, for each output length.
for method in sorted(m for m in df.method.unique() if m != "none"):
    sub = df[(df.method == method) & (df.temperature == 0)]
    if sub.empty:
        continue
    lens = sorted(sub.output_len.unique())
    fig, axes = plt.subplots(2, len(lens), figsize=(5 * len(lens), 8), squeeze=False)
    for j, L in enumerate(lens):
        for conc in sorted(sub.concurrency.unique()):
            s = sub[(sub.output_len == L) & (sub.concurrency == conc)].sort_values("k")
            axes[0][j].plot(s.k, s.e2e_speedup, marker="o", label=f"conc={conc}")
            axes[1][j].plot(s.k, s.throughput_speedup, marker="o", label=f"conc={conc}")
        for i, name in enumerate(["per-request E2E speedup (p50)", "server output tok/s speedup"]):
            axes[i][j].axhline(1.0, color="grey", ls="--", lw=1)
            axes[i][j].set_title(f"{method}, output_len={L}: {name}")
            axes[i][j].set_xlabel("K (num speculative tokens)")
            axes[i][j].set_xscale("log", base=2)
            axes[i][j].legend()
    fig.tight_layout()
    fig.savefig(os.path.join(args.out, f"speedup_{method}.png"), dpi=120)
    plt.close(fig)

# Acceptance by position (conc=1, longest output) per method/K.
acc = df[(df.method != "none") & df.acceptance_by_position.notna() & (df.concurrency == 1)
         & (df.temperature == 0) & ~df.label.str.contains("_rep")]
if not acc.empty:
    L = acc.output_len.max()
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for _, r in acc[acc.output_len == L].sort_values(["method", "k"]).iterrows():
        pos = sorted(r.acceptance_by_position.items(), key=lambda kv: int(kv[0]))
        ax.plot([int(p) + 1 for p, _ in pos], [v for _, v in pos], marker="o", label=r.label)
    ax.set_xlabel("draft position")
    ax.set_ylabel("P(token at this position accepted)")
    ax.set_title(f"Per-position acceptance (conc=1, output_len={L})")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(args.out, "acceptance_by_position.png"), dpi=120)
    plt.close(fig)
# Summary table in the assignment's format: baseline and every K at conc=1, then
# baseline and the best K (by p50 TPOT at conc=1, same output length) at conc 4 and 16.
def summary(L):
    g = df[(df.temperature == 0) & (df.output_len == L) & ~df.label.str.contains("_rep")]
    out = ["| Configuration | Method | K | Concurrency | TTFT p50 (ms) | TPOT p50 (ms) | Tok/s (server) | "
           "Acceptance | Tokens/step | E2E speedup | Throughput speedup |",
           "|---|---|---|---|---|---|---|---|---|---|---|"]

    def line(r, name):
        out.append(f"| {name} | {r.method} | {'-' if r.k == 0 else int(r.k)} | {int(r.concurrency)} | "
                   f"{fmt(r.ttft_50_ms)} | {fmt(r.tpot_50_ms, 2)} | {fmt(r.output_tok_s)} | "
                   f"{fmt(r.acceptance_rate, 2)} | {fmt(r.mean_accepted_len, 2)} | "
                   f"{fmt(r.e2e_speedup, 2)} | {fmt(r.throughput_speedup, 2)} |")

    c1 = g[g.concurrency == 1].sort_values(["method", "k"])
    for _, r in c1.iterrows():
        line(r, "Baseline" if r.method == "none" else "Speculative")
    for conc in sorted(c for c in g.concurrency.unique() if c != 1):
        for _, r in g[(g.concurrency == conc) & (g.method == "none")].iterrows():
            line(r, "Baseline")
        for m in sorted(x for x in g.method.unique() if x != "none"):
            cand = c1[c1.method == m].dropna(subset=["tpot_50_ms"])
            if cand.empty:
                continue
            best_k = cand.sort_values("tpot_50_ms").iloc[0].k
            for _, r in g[(g.concurrency == conc) & (g.method == m) & (g.k == best_k)].iterrows():
                line(r, f"Speculative (best K at conc 1)")
    return "\n".join(out)


with open(os.path.join(args.out, "summary.md"), "w") as f:
    for L in sorted(df.output_len.unique()):
        f.write(f"### output_len = {int(L)} tokens, temperature 0\n\n{summary(L)}\n\n")
print(f"wrote {len(df)} cells")

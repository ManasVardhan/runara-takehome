### output_len = 32 tokens, temperature 0

| Configuration | Method | K | Concurrency | TTFT p50 (ms) | TPOT p50 (ms) | Tok/s (server) | Acceptance | Tokens/step | E2E speedup | Throughput speedup |
|---|---|---|---|---|---|---|---|---|---|---|
| Baseline | none | - | 1 | 27.5 | 4.46 | 190.9 | - | - | 1.00 | 1.00 |
| Speculative | draft_model | 1 | 1 | 29.1 | 4.22 | 199.3 | 0.93 | 1.93 | 1.04 | 1.04 |
| Speculative | draft_model | 2 | 1 | 42.2 | 5.16 | 157.2 | 0.90 | 2.80 | 0.82 | 0.82 |
| Speculative | draft_model | 4 | 1 | 68.3 | 5.91 | 124.9 | 0.84 | 4.35 | 0.66 | 0.65 |
| Speculative | draft_model | 8 | 1 | 118.5 | 8.19 | 85.1 | 0.69 | 6.50 | 0.45 | 0.45 |
| Speculative | eagle3 | 1 | 1 | 29.6 | 3.25 | 241.7 | 0.71 | 1.71 | 1.28 | 1.27 |
| Speculative | eagle3 | 2 | 1 | 30.3 | 3.18 | 244.7 | 0.58 | 2.16 | 1.29 | 1.28 |
| Speculative | eagle3 | 4 | 1 | 24.2 | 3.01 | 263.7 | 0.40 | 2.60 | 1.42 | 1.38 |
| Speculative | eagle3 | 8 | 1 | 25.7 | 3.48 | 232.9 | 0.24 | 2.88 | 1.24 | 1.22 |
| Speculative | ngram | 4 | 1 | 25.8 | 5.30 | 170.5 | 0.58 | 3.32 | 0.87 | 0.89 |
| Baseline | none | - | 4 | 62.8 | 7.39 | 429.7 | - | - | 1.00 | 1.00 |
| Speculative (best K at conc 1) | draft_model | 1 | 4 | 48.0 | 5.07 | 595.6 | 0.93 | 1.93 | 1.36 | 1.39 |
| Speculative (best K at conc 1) | eagle3 | 4 | 4 | 44.8 | 4.42 | 653.4 | 0.41 | 2.62 | 1.62 | 1.52 |
| Speculative (best K at conc 1) | ngram | 4 | 4 | 34.9 | 7.93 | 435.8 | 0.58 | 3.32 | 1.02 | 1.01 |
| Baseline | none | - | 16 | 135.4 | 11.64 | 982.7 | - | - | 1.00 | 1.00 |
| Speculative (best K at conc 1) | draft_model | 1 | 16 | 95.3 | 7.54 | 1403.5 | 0.93 | 1.93 | 1.47 | 1.43 |
| Speculative (best K at conc 1) | eagle3 | 4 | 16 | 62.9 | 6.76 | 1641.9 | 0.41 | 2.64 | 1.84 | 1.67 |
| Speculative (best K at conc 1) | ngram | 4 | 16 | 51.8 | 11.99 | 1103.7 | 0.56 | 3.25 | 1.14 | 1.12 |

### output_len = 128 tokens, temperature 0

| Configuration | Method | K | Concurrency | TTFT p50 (ms) | TPOT p50 (ms) | Tok/s (server) | Acceptance | Tokens/step | E2E speedup | Throughput speedup |
|---|---|---|---|---|---|---|---|---|---|---|
| Baseline | none | - | 1 | 28.7 | 4.48 | 213.8 | - | - | 1.00 | 1.00 |
| Speculative | draft_model | 1 | 1 | 28.1 | 4.76 | 199.4 | 0.76 | 1.76 | 0.94 | 0.93 |
| Speculative | draft_model | 2 | 1 | 43.6 | 6.21 | 154.4 | 0.67 | 2.34 | 0.72 | 0.72 |
| Speculative | draft_model | 4 | 1 | 68.3 | 8.63 | 112.5 | 0.52 | 3.10 | 0.51 | 0.53 |
| Speculative | draft_model | 8 | 1 | 118.3 | 13.99 | 69.5 | 0.35 | 3.79 | 0.31 | 0.32 |
| Speculative | eagle3 | 1 | 1 | 30.7 | 3.52 | 269.1 | 0.59 | 1.59 | 1.27 | 1.26 |
| Speculative | eagle3 | 2 | 1 | 33.8 | 3.63 | 258.2 | 0.45 | 1.90 | 1.20 | 1.21 |
| Speculative | eagle3 | 4 | 1 | 25.5 | 3.67 | 259.7 | 0.28 | 2.13 | 1.21 | 1.21 |
| Speculative | eagle3 | 8 | 1 | 25.9 | 4.49 | 216.8 | 0.15 | 2.22 | 1.00 | 1.01 |
| Speculative | ngram | 4 | 1 | 26.5 | 5.76 | 170.8 | 0.32 | 2.30 | 0.79 | 0.80 |
| Baseline | none | - | 4 | 66.3 | 7.74 | 485.7 | - | - | 1.00 | 1.00 |
| Speculative (best K at conc 1) | draft_model | 1 | 4 | 37.6 | 6.46 | 559.4 | 0.76 | 1.76 | 1.22 | 1.15 |
| Speculative (best K at conc 1) | eagle3 | 1 | 4 | 47.2 | 5.77 | 633.8 | 0.61 | 1.61 | 1.33 | 1.30 |
| Speculative (best K at conc 1) | ngram | 4 | 4 | 33.1 | 8.76 | 439.4 | 0.32 | 2.27 | 0.91 | 0.90 |
| Baseline | none | - | 16 | 128.6 | 12.44 | 1126.5 | - | - | 1.00 | 1.00 |
| Speculative (best K at conc 1) | draft_model | 1 | 16 | 77.5 | 9.72 | 1457.8 | 0.76 | 1.76 | 1.30 | 1.29 |
| Speculative (best K at conc 1) | eagle3 | 1 | 16 | 72.5 | 9.46 | 1463.4 | 0.59 | 1.59 | 1.32 | 1.30 |
| Speculative (best K at conc 1) | ngram | 4 | 16 | 40.4 | 14.02 | 1053.8 | 0.32 | 2.27 | 0.93 | 0.94 |

### output_len = 512 tokens, temperature 0

| Configuration | Method | K | Concurrency | TTFT p50 (ms) | TPOT p50 (ms) | Tok/s (server) | Acceptance | Tokens/step | E2E speedup | Throughput speedup |
|---|---|---|---|---|---|---|---|---|---|---|
| Baseline | none | - | 1 | 31.3 | 4.50 | 219.3 | - | - | 1.00 | 1.00 |
| Speculative | draft_model | 1 | 1 | 31.1 | 4.83 | 204.6 | 0.73 | 1.73 | 0.93 | 0.93 |
| Speculative | draft_model | 2 | 1 | 46.2 | 6.53 | 155.9 | 0.63 | 2.26 | 0.69 | 0.71 |
| Speculative | draft_model | 4 | 1 | 69.2 | 9.33 | 111.1 | 0.48 | 2.93 | 0.48 | 0.51 |
| Speculative | draft_model | 8 | 1 | 119.6 | 15.73 | 67.2 | 0.31 | 3.48 | 0.29 | 0.31 |
| Speculative | eagle3 | 1 | 1 | 32.2 | 3.55 | 278.0 | 0.58 | 1.58 | 1.26 | 1.27 |
| Speculative | eagle3 | 2 | 1 | 26.4 | 3.58 | 277.6 | 0.44 | 1.87 | 1.26 | 1.27 |
| Speculative | eagle3 | 4 | 1 | 26.5 | 3.69 | 269.4 | 0.27 | 2.08 | 1.22 | 1.23 |
| Speculative | eagle3 | 8 | 1 | 27.1 | 4.56 | 218.4 | 0.14 | 2.15 | 0.99 | 1.00 |
| Speculative | ngram | 4 | 1 | 27.5 | 5.79 | 174.8 | 0.21 | 1.83 | 0.78 | 0.80 |
| Baseline | none | - | 4 | 66.8 | 7.85 | 501.6 | - | - | 1.00 | 1.00 |
| Speculative (best K at conc 1) | draft_model | 1 | 4 | 38.7 | 6.43 | 599.6 | 0.74 | 1.74 | 1.21 | 1.20 |
| Speculative (best K at conc 1) | eagle3 | 1 | 4 | 42.5 | 6.01 | 641.2 | 0.57 | 1.57 | 1.31 | 1.28 |
| Speculative (best K at conc 1) | ngram | 4 | 4 | 36.9 | 9.10 | 424.3 | 0.19 | 1.77 | 0.87 | 0.85 |
| Baseline | none | - | 16 | 128.2 | 13.11 | 1128.0 | - | - | 1.00 | 1.00 |
| Speculative (best K at conc 1) | draft_model | 1 | 16 | 58.4 | 10.21 | 1456.8 | 0.73 | 1.73 | 1.29 | 1.29 |
| Speculative (best K at conc 1) | eagle3 | 1 | 16 | 57.5 | 9.94 | 1485.8 | 0.58 | 1.58 | 1.33 | 1.32 |
| Speculative (best K at conc 1) | ngram | 4 | 16 | 41.2 | 14.73 | 1021.2 | 0.21 | 1.86 | 0.90 | 0.91 |


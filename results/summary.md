### output_len = 32 tokens, temperature 0

| Configuration | Method | K | Concurrency | TTFT p50 (ms) | TPOT p50 (ms) | Tok/s (server) | Acceptance | Tokens/step | E2E speedup | Throughput speedup |
|---|---|---|---|---|---|---|---|---|---|---|
| Baseline | none | - | 1 | 27.3 | 4.50 | 191.7 | - | - | 1.00 | 1.00 |
| Speculative | draft_model | 1 | 1 | 35.5 | 4.50 | 166.7 | 0.93 | 1.93 | 0.94 | 0.87 |
| Speculative | draft_model | 2 | 1 | 55.2 | 5.67 | 128.2 | 0.90 | 2.80 | 0.71 | 0.67 |
| Speculative | draft_model | 4 | 1 | 78.5 | 6.82 | 102.9 | 0.84 | 4.35 | 0.55 | 0.54 |
| Speculative | draft_model | 8 | 1 | 124.0 | 8.49 | 80.6 | 0.69 | 6.50 | 0.43 | 0.42 |
| Speculative | eagle3 | 1 | 1 | 29.4 | 3.25 | 241.4 | 0.71 | 1.71 | 1.28 | 1.26 |
| Speculative | eagle3 | 2 | 1 | 32.2 | 3.05 | 244.6 | 0.58 | 2.16 | 1.29 | 1.28 |
| Speculative | eagle3 | 4 | 1 | 31.6 | 2.93 | 250.0 | 0.40 | 2.60 | 1.33 | 1.30 |
| Speculative | eagle3 | 8 | 1 | 37.5 | 3.49 | 218.4 | 0.24 | 2.88 | 1.17 | 1.14 |
| Baseline | none | - | 4 | 58.1 | 7.42 | 339.2 | - | - | 1.00 | 1.00 |
| Speculative (best K at conc 1) | draft_model | 1 | 4 | 50.3 | 5.42 | 553.2 | 0.93 | 1.93 | 1.27 | 1.63 |
| Speculative (best K at conc 1) | eagle3 | 4 | 4 | 42.2 | 4.58 | 633.4 | 0.41 | 2.62 | 1.48 | 1.87 |
| Baseline | none | - | 16 | 111.5 | 10.79 | 846.5 | - | - | 1.00 | 1.00 |
| Speculative (best K at conc 1) | draft_model | 1 | 16 | 122.7 | 8.15 | 991.7 | 0.93 | 1.93 | 1.17 | 1.17 |
| Speculative (best K at conc 1) | eagle3 | 4 | 16 | 63.2 | 6.94 | 1568.1 | 0.41 | 2.64 | 1.58 | 1.85 |

### output_len = 128 tokens, temperature 0

| Configuration | Method | K | Concurrency | TTFT p50 (ms) | TPOT p50 (ms) | Tok/s (server) | Acceptance | Tokens/step | E2E speedup | Throughput speedup |
|---|---|---|---|---|---|---|---|---|---|---|
| Baseline | none | - | 1 | 28.0 | 4.49 | 213.2 | - | - | 1.00 | 1.00 |
| Speculative | draft_model | 1 | 1 | 34.8 | 4.96 | 193.6 | 0.76 | 1.76 | 0.90 | 0.91 |
| Speculative | draft_model | 2 | 1 | 50.7 | 7.18 | 132.0 | 0.67 | 2.34 | 0.62 | 0.62 |
| Speculative | draft_model | 4 | 1 | 73.4 | 9.79 | 98.1 | 0.52 | 3.10 | 0.46 | 0.46 |
| Speculative | draft_model | 8 | 1 | 124.9 | 14.86 | 65.4 | 0.35 | 3.79 | 0.30 | 0.31 |
| Speculative | eagle3 | 1 | 1 | 28.5 | 3.52 | 268.7 | 0.59 | 1.59 | 1.26 | 1.26 |
| Speculative | eagle3 | 2 | 1 | 30.1 | 3.62 | 260.7 | 0.45 | 1.90 | 1.23 | 1.22 |
| Speculative | eagle3 | 4 | 1 | 31.6 | 3.64 | 257.6 | 0.28 | 2.13 | 1.21 | 1.21 |
| Speculative | eagle3 | 8 | 1 | 40.7 | 4.51 | 211.7 | 0.15 | 2.22 | 0.97 | 0.99 |
| Baseline | none | - | 4 | 65.5 | 7.73 | 486.0 | - | - | 1.00 | 1.00 |
| Speculative (best K at conc 1) | draft_model | 1 | 4 | 45.4 | 6.31 | 588.7 | 0.76 | 1.76 | 1.22 | 1.21 |
| Speculative (best K at conc 1) | eagle3 | 1 | 4 | 43.8 | 5.80 | 636.2 | 0.60 | 1.60 | 1.34 | 1.31 |
| Baseline | none | - | 16 | 133.9 | 12.43 | 1057.0 | - | - | 1.00 | 1.00 |
| Speculative (best K at conc 1) | draft_model | 1 | 16 | 67.0 | 9.59 | 1460.0 | 0.76 | 1.76 | 1.31 | 1.38 |
| Speculative (best K at conc 1) | eagle3 | 1 | 16 | 56.6 | 9.47 | 1504.5 | 0.59 | 1.59 | 1.34 | 1.42 |

### output_len = 512 tokens, temperature 0

| Configuration | Method | K | Concurrency | TTFT p50 (ms) | TPOT p50 (ms) | Tok/s (server) | Acceptance | Tokens/step | E2E speedup | Throughput speedup |
|---|---|---|---|---|---|---|---|---|---|---|
| Baseline | none | - | 1 | 29.1 | 4.52 | 218.9 | - | - | 1.00 | 1.00 |
| Speculative | draft_model | 1 | 1 | 34.1 | 5.13 | 195.4 | 0.73 | 1.73 | 0.88 | 0.89 |
| Speculative | draft_model | 2 | 1 | 50.4 | 7.28 | 139.0 | 0.63 | 2.26 | 0.62 | 0.63 |
| Speculative | draft_model | 4 | 1 | 75.0 | 10.49 | 97.9 | 0.48 | 2.93 | 0.43 | 0.45 |
| Speculative | draft_model | 8 | 1 | 124.3 | 16.96 | 62.2 | 0.31 | 3.48 | 0.27 | 0.28 |
| Speculative | eagle3 | 1 | 1 | 28.1 | 3.56 | 278.3 | 0.58 | 1.58 | 1.27 | 1.27 |
| Speculative | eagle3 | 2 | 1 | 31.4 | 3.70 | 268.6 | 0.44 | 1.87 | 1.22 | 1.23 |
| Speculative | eagle3 | 4 | 1 | 33.5 | 3.76 | 263.9 | 0.27 | 2.08 | 1.19 | 1.21 |
| Speculative | eagle3 | 8 | 1 | 42.8 | 4.58 | 216.3 | 0.14 | 2.15 | 0.97 | 0.99 |
| Baseline | none | - | 4 | 57.2 | 7.89 | 501.5 | - | - | 1.00 | 1.00 |
| Speculative (best K at conc 1) | draft_model | 1 | 4 | 46.7 | 6.41 | 597.2 | 0.74 | 1.74 | 1.23 | 1.19 |
| Speculative (best K at conc 1) | eagle3 | 1 | 4 | 42.4 | 5.99 | 644.0 | 0.58 | 1.58 | 1.31 | 1.28 |
| Baseline | none | - | 16 | 128.6 | 13.14 | 1126.1 | - | - | 1.00 | 1.00 |
| Speculative (best K at conc 1) | draft_model | 1 | 16 | 60.5 | 10.27 | 1456.9 | 0.74 | 1.74 | 1.29 | 1.29 |
| Speculative (best K at conc 1) | eagle3 | 1 | 16 | 57.8 | 9.95 | 1482.4 | 0.58 | 1.58 | 1.32 | 1.32 |


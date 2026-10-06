Cells compared: 60 (labels x output_len in (128, 512) x concurrency)

| Metric | median abs. rel. diff | 90th pct abs. rel. diff | max abs. rel. diff |
|---|---|---|---|
| tpot_50_ms | 0.9% | 9.1% | 13.5% |
| output_tok_s | 1.2% | 11.2% | 17.0% |
| ttft_50_ms | 7.7% | 18.7% | 36.6% |
| acceptance_rate | 0.0% | 0.9% | 5.3% |

By concurrency (median abs. rel. diff):

| Concurrency | TPOT p50 | output tok/s |
|---|---|---|
| 1 | 3.1% | 3.0% |
| 4 | 0.7% | 0.8% |
| 16 | 0.9% | 0.6% |

Largest throughput differences:

| Label | len | conc | tok/s run 1 | tok/s run 2 | diff |
|---|---|---|---|---|---|
| draft_model_k2 | 128 | 1 | 132 | 154 | +17.0% |
| draft_model_k4 | 128 | 1 | 98 | 113 | +14.7% |
| draft_model_k4 | 512 | 1 | 98 | 111 | +13.5% |
| draft_model_k4 | 512 | 4 | 359 | 405 | +12.9% |
| draft_model_k2 | 512 | 4 | 525 | 590 | +12.3% |

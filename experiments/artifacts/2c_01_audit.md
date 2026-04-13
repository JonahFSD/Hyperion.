# 2C-01 Audit: SAE Changes → Event Prediction

**Date:** 2026-04-13  
**Verdict: NULL RESULT — do not pursue**

## What the script reported

AUC = 0.711 on `event_extreme_return` (holdout 2011–2020). Claimed GO.

## Why it's wrong

The model used three feature types: SAE `change_magnitude` + SIC dummies + `prior_cum_return` (current year's returns). The reported AUC is almost entirely driven by `prior_cum_return`.

| Model | AUC |
|---|---|
| `prior_cum_return` only | **0.693** |
| SAE `change_magnitude` only | 0.560 |
| both + SIC controls | 0.711 |

Prior returns alone hit 0.693. SAE adds ~0.018. That's noise.

## Additional code issues found

1. **Broken permutation test** — the script shuffled labels against fixed predictions, guaranteed to return ~0.50. "Beats random p95: True" was trivially true and meaningless.
2. **NaN-contaminated PCA features** — matmul overflow warnings during transform indicated some feature vectors were corrupted. Scale unknown.
3. **Magnitudes parquet not cached** — large intermediates were lost between runs (likely macOS Optimized Storage). PCA recompute takes ~45 min.

## What this means

The hypothesis "SAE fingerprint changes predict corporate events" does not hold beyond the trivial baseline of return persistence. Companies already in distress tend to stay in distress — the SAE adds nothing to that prediction.

**This does not kill the project.** Phase 1 (similarity, retrieval, SIC re-ranking) is unaffected. The product is a similarity/discovery engine, not an early-warning system.

## Do not re-run

The Tier 2 validation (NT filings, AAER, bankruptcies) recommended by the script is not worth pursuing given the baseline result.

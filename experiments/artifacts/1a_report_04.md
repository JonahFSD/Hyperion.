## 4. Bootstrap CIs + Influence Diagnostics

### Observations

**Configuration:** 10,000 bootstrap iterations, 1602 tickers resampled, seed=42, 95% BCa confidence intervals.

**Method CIs:**

| Method | MC | 95% CI | z0 | a | Boot Mean | Boot Std |
|--------|---:|-------:|---:|--:|----------:|---------:|
| SAE | 0.358981 | [0.327674, 0.415299] | 0.6344 | -0.006830 | 0.344215 | 0.023059 |
| SIC | 0.231085 | [0.217848, 0.245870] | 0.1115 | 0.005118 | 0.230303 | 0.007135 |
| SBERT | 0.219345 | [0.210544, 0.228068] | 0.0023 | -0.006054 | 0.219349 | 0.004479 |

**Delta Tests:**

| Comparison | Delta | 95% CI | p-value | t-stat | z0 | a |
|------------|------:|-------:|--------:|-------:|---:|--:|
| SAE - SIC | 0.127896 | [0.096341, 0.185529] | 0.000000 | 5.57 | 0.6014 | -0.006493 |
| SAE - SBERT | 0.139636 | [0.109480, 0.194032] | 0.000000 | 6.24 | 0.6470 | -0.006905 |
| SAE - baseline | 0.198116 | [0.166809, 0.254434] | 0.000000 | 8.59 | 0.6344 | -0.006830 |

**Influence Diagnostics (SAE, leave-one-ticker-out):**

- Conclusion-flipping tickers (removing causes SAE MC < SIC MC): **0**
- Influence distribution: mean=0.000017, std=0.000815, skewness=-1.6403
- Max absolute influence: 0.009832 (CMS-PB)

Top 20 tickers by absolute influence:

| Rank | Ticker | Influence |
|-----:|--------|----------:|
| 1 | CMS-PB | -0.009832 |
| 2 | CMS | -0.009816 |
| 3 | AMAT | 0.008024 |
| 4 | SSY | -0.008009 |
| 5 | KO | -0.006713 |
| 6 | COKE | -0.006707 |
| 7 | IDA | 0.006274 |
| 8 | AVA | 0.005620 |
| 9 | PTSI | -0.005592 |
| 10 | LRCX | 0.005002 |
| 11 | VZ | 0.004883 |
| 12 | T | 0.004883 |
| 13 | WDC | 0.004558 |
| 14 | STX | 0.004555 |
| 15 | HIG | 0.004157 |
| 16 | ERIE | -0.004055 |
| 17 | UHS | -0.004051 |
| 18 | LNC | 0.003956 |
| 19 | ADC | 0.003588 |
| 20 | NNN | 0.003562 |

### Interpretation

*(To be written after reviewing the numbers above.)*

Key questions:
- Is the result broad-based or concentrated? (Check influence distribution skewness and max.)
- What does z0 tell us about bias in the bootstrap? (z0 near 0 = symmetric, large |z0| = skewed.)
- Are there fragilities? (Any conclusion-flipping tickers indicate the result depends on specific companies.)
- Do t-statistics exceed HLZ gold standard of 3.0?

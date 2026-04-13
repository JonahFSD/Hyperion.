## 1. Data Loading and Verification

### Observations

| Dataset | Rows | Year Range |
|---------|-----:|:----------:|
| Pairs (cosine sim + correlation) | 14,920,749 | 1996-2020 |
| Companies (metadata + returns) | 27,888 | — |

| Cluster Method | Years | Year Range |
|----------------|------:|:----------:|
| sae_cd | 25 | 1996-2020 |
| bert | 25 | 1996-2020 |
| sbert | 25 | 1996-2020 |
| palm | 25 | 1996-2020 |
| rolling_cd | 20 | 2001-2020 |
| sic | 28 | 1993-2020 |
| industry | 28 | 1993-2020 |

Population baseline MC (mean of all pairwise correlations): **0.160865**
(ACL paper reports 0.161; acceptable range [0.14, 0.18])

Verification checks: 15/15 passed.

### Interpretation

All data sources loaded and verified against expected ranges. The population baseline MC of 0.1609 is consistent with the ACL paper's reported value of 0.161, confirming we are working with the same data. All 7 cluster pickle files have the expected schema (year + clusters columns). The SAE C-CD clusters cover the full 1996-2020 range (25 years). SIC and Industry clusters extend back to 1993, but only the 1996-2020 overlap will be used for comparison. Rolling CD covers 2001-2020 (20 years) as expected, since it requires a 5-year lookback window.

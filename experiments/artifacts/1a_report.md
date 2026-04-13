# Phase 1A Report: SAE Clustering Validation

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

---

## Step 2: MC Replication (Mean + Median)

### Observations

**MC Replication vs. Published Values**

| Method         |  Mean MC |  Published |    Delta |  Match? |
|----------------|----------|------------|----------|---------|
| population     |   0.1609 |         -- |       -- |      -- |
| sae_cd         |   0.3590 |      0.359 |  -0.0000 |    PASS |
| bert           |   0.1977 |         -- |       -- |      -- |
| sbert          |   0.2193 |         -- |       -- |      -- |
| palm           |   0.2195 |         -- |       -- |      -- |
| rolling_cd     |   0.3848 |         -- |       -- |      -- |
| sic            |   0.2311 |      0.231 |  +0.0001 |    PASS |
| industry       |   0.1868 |         -- |       -- |      -- |

**Mean vs. Median MC by Method**

| Method         |  Mean MC |  Median MC |   Ratio |
|----------------|----------|------------|---------|
| population     |   0.1609 |     0.1727 |   1.074 |
| sae_cd         |   0.3590 |     0.3569 |   0.994 |
| bert           |   0.1977 |     0.2162 |   1.094 |
| sbert          |   0.2193 |     0.2404 |   1.096 |
| palm           |   0.2195 |     0.2401 |   1.094 |
| rolling_cd     |   0.3848 |     0.3838 |   0.997 |
| sic            |   0.2311 |     0.2292 |   0.992 |
| industry       |   0.1868 |     0.2028 |   1.085 |

### Interpretation

_To be filled after running the script with actual data._

Key questions for interpretation:

- Do our mean MC values match the published values within tolerance (1e-3)?
- Are mean and median MC telling the same story across methods?
  If mean >> median for a method, the MC is driven by outlier pairs,
  not broad cluster quality.
- Which methods show the largest mean-median divergence? Does SAE's
  advantage hold under median as well as mean?

---

## 3. Temporal Delta Analysis

### Observations

**Year-by-year deltas (SAE MC minus baseline MC):**

| Year | SAE − SIC | SAE − SBERT |
|------|-----------|-------------|
| 1996 | -0.0464 | -0.0173 |
| 1997 | +0.0493 | +0.0385 |
| 1998 | +0.0730 | +0.0661 |
| 1999 | +0.0264 | +0.0066 |
| 2000 | +0.1793 | +0.1534 |
| 2001 | +0.1159 | +0.1281 |
| 2002 | +0.0715 | +0.0905 |
| 2003 | +0.1507 | +0.1811 |
| 2004 | +0.1628 | +0.1454 |
| 2005 | +0.1335 | +0.1471 |
| 2006 | +0.0341 | +0.0527 |
| 2007 | +0.1274 | +0.1283 |
| 2008 | -0.0006 | +0.0123 |
| 2009 | +0.1357 | +0.1846 |
| 2010 | +0.0821 | +0.1176 |
| 2011 | +0.0717 | +0.0874 |
| 2012 | +0.2412 | +0.2324 |
| 2013 | +0.3436 | +0.3223 |
| 2014 | +0.2017 | +0.2363 |
| 2015 | +0.2205 | +0.2451 |
| 2016 | +0.2059 | +0.2164 |
| 2017 | +0.1856 | +0.2160 |
| 2018 | +0.1728 | +0.2013 |
| 2019 | +0.1039 | +0.1287 |
| 2020 | +0.1557 | +0.1701 |

**Summary statistics:**

| Metric | SAE − SIC | SAE − SBERT |
|--------|-----------|-------------|
| Mean delta | +0.1279 | +0.1396 |
| Min delta | -0.0464 (1996) | -0.0173 (1996) |
| Max delta | +0.3436 (2013) | +0.3223 (2013) |
| Years with delta ≤ 0 | 2 | 1 |

SAE − SIC negative years: [1996, 2008]

SAE − SBERT negative years: [1996]

**OLS trend (delta = intercept + slope × year):**

| Metric | SAE − SIC | SAE − SBERT |
|--------|-----------|-------------|
| Slope (MC/year) | +0.006446 | +0.007489 |
| Intercept | -12.8147 | -14.8991 |
| 95% CI lower | +0.003010 | +0.004375 |
| 95% CI upper | +0.010302 | +0.010887 |
| CI includes zero | False | False |

### Interpretation

SAE underperforms SIC in 2 out of 25 years ([1996, 2008]). These periods warrant investigation.

SAE underperforms SBERT in 1 out of 25 years ([1996]). These are the fairer comparison (both use MST + theta clustering) so negative years here are more concerning than SAE vs SIC.

**SAE − SIC trend:** Slope CI [+0.003010, +0.010302] excludes zero. The SAE advantage is growing at 0.0064 MC units/year. This temporal trend should be investigated in 1B — it could reflect genuine improvement or confounding with time-varying macro factors.
**SAE − SBERT trend:** Slope CI [+0.004375, +0.010887] excludes zero. The SAE advantage is growing at 0.0075 MC units/year. This temporal trend should be investigated in 1B — it could reflect genuine improvement or confounding with time-varying macro factors.

---

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

---

## 5. Theta Sensitivity (Diagnostic)

### Observations

- **Scaler:** StandardScaler fit on all years (1996-2020)
- **Threshold range:** 100 thresholds from -5.2431 to -1.0853 (5th–95th percentile of MST edge weights)
- **Optimal theta:** -2.849210
- **MC at optimal theta:** 0.376384
- **ACL theta:** -2.7
- **MC at ACL theta:** 0.367073
- **Ratio (ACL / optimal):** 0.975260

Note: These MC values do not match Section 2 (MC Replication) because clusters are re-derived here from the cosine similarity matrix, not loaded from ACL pre-computed labels.

### Interpretation

*To be completed after running the script.*

Key questions:
- Is the theta curve flat or sharp near the peak? A flat peak means MC is insensitive to the exact threshold choice; a sharp peak means the result is fragile.
- How does the ratio of MC at ACL theta to optimal MC compare? A ratio > 0.95 means the ACL's choice of -2.7 barely matters — any nearby theta gives similar performance.
- Does the optimal theta found here agree with the ACL's -2.7? Agreement is expected but not guaranteed, because our scaler differs (fit on all years vs. first 75%).

---

## Rolling Temporal Holdout

Window size: 5 years (One business cycle (NBER average expansion+contraction))
Number of windows: 21

### Observations

| Window | SAE MC | SIC MC | SBERT MC | Delta(SIC) | Delta(SBERT) |
|--------|--------|--------|----------|------------|--------------|
| 1996-2000 | 0.2483 | 0.1920 | 0.1988 | +0.0563 | +0.0495 |
| 1997-2001 | 0.2860 | 0.1972 | 0.2075 | +0.0888 | +0.0785 |
| 1998-2002 | 0.3114 | 0.2181 | 0.2224 | +0.0932 | +0.0889 |
| 1999-2003 | 0.3150 | 0.2062 | 0.2030 | +0.1088 | +0.1119 |
| 2000-2004 | 0.3564 | 0.2204 | 0.2167 | +0.1360 | +0.1397 |
| 2001-2005 | 0.3704 | 0.2435 | 0.2319 | +0.1269 | +0.1384 |
| 2002-2006 | 0.3451 | 0.2346 | 0.2218 | +0.1105 | +0.1234 |
| 2003-2007 | 0.3347 | 0.2130 | 0.2038 | +0.1217 | +0.1309 |
| 2004-2008 | 0.3209 | 0.2295 | 0.2238 | +0.0915 | +0.0972 |
| 2005-2009 | 0.3283 | 0.2423 | 0.2233 | +0.0860 | +0.1050 |
| 2006-2010 | 0.3362 | 0.2604 | 0.2371 | +0.0757 | +0.0991 |
| 2007-2011 | 0.3656 | 0.2823 | 0.2595 | +0.0833 | +0.1060 |
| 2008-2012 | 0.3841 | 0.2780 | 0.2572 | +0.1060 | +0.1269 |
| 2009-2013 | 0.4176 | 0.2427 | 0.2287 | +0.1749 | +0.1889 |
| 2010-2014 | 0.4168 | 0.2287 | 0.2176 | +0.1881 | +0.1992 |
| 2011-2015 | 0.4204 | 0.2046 | 0.1957 | +0.2157 | +0.2247 |
| 2012-2016 | 0.4269 | 0.1843 | 0.1764 | +0.2426 | +0.2505 |
| 2013-2017 | 0.4173 | 0.1859 | 0.1701 | +0.2315 | +0.2472 |
| 2014-2018 | 0.4138 | 0.2165 | 0.1908 | +0.1973 | +0.2230 |
| 2015-2019 | 0.3960 | 0.2183 | 0.1945 | +0.1777 | +0.2015 |
| 2016-2020 | 0.4197 | 0.2549 | 0.2332 | +0.1648 | +0.1865 |

**Win rates:**

- SAE > SIC: 21/21 windows
  - Mean delta: +0.1370
  - Worst window: 1996-2000 (delta = +0.0563)
  - Best window: 2012-2016 (delta = +0.2426)
- SAE > SBERT: 21/21 windows
  - Mean delta: +0.1484
  - Worst window: 1996-2000 (delta = +0.0495)
  - Best window: 2012-2016 (delta = +0.2505)

No negative-delta windows for either baseline.

### Interpretation

*Does the SAE advantage hold across market regimes? Which periods are weakest?*

_(Populated after data is available.)_

---

## 7. Verdict

### Hard Tests

| Test | Result | Details |
|------|--------|---------|
| MC replication | PASS | SAE: 0.3590 (pub: 0.359), SIC: 0.2311 (pub: 0.231) |
| SAE > SIC | PASS | p_raw=0.0, p_BY=0.000183, t=5.57 (HLZ>3.0) |
| SAE > SBERT | PASS | p_raw=0.0, p_BY=0.000183, t=6.24 (HLZ>3.0) |
| SAE > baseline | PASS | p_raw=0.0, p_BY=0.000183 |

### Diagnostics

- **Median/Mean MC ratio:** SAE=0.9941, SIC=0.991678, SBERT=1.095767
- **Temporal slope (SAE-SIC):** 0.006446 [0.003010, 0.010302] **(EXCLUDES zero)**
- **Temporal slope (SAE-SBERT):** 0.007489 [0.004375, 0.010887] **(EXCLUDES zero)**
- **Theta sensitivity:** MC(ACL theta) / MC(optimal) = 0.9753
- **Rolling win rate:** SAE vs SIC: 21/21, SAE vs SBERT: 21/21
- **Conclusion-flipping tickers:** 0
- **SAE bootstrap z0:** 0.6344

### Flags

- SAE-SIC advantage is increasing over time (slope=0.006446, CI excludes zero). Investigate in 1B.
- SAE-SBERT advantage is increasing over time (slope=0.007489, CI excludes zero). Investigate in 1B.
- SAE bootstrap bias correction z0 = 0.6344 (|z0| > 0.25 suggests notable bias in bootstrap distribution)

### Overall Verdict

**CONDITIONAL**

All 4 hard tests pass, but 3 diagnostic concern(s) raised. Review flags before proceeding to 1B.

### What This Means for 1B and 1C

1A establishes that the SAE clustering advantage exists and is statistically significant. Two questions remain:

- **1B (Factor Adjustment):** Is the advantage explained by exposure to known risk factors (Fama-French 5), or does company-specific signal remain after factor adjustment?
- **1C (Permutation Test):** Is the advantage an artifact of the clustering algorithm (any features + MST + theta would produce similar MC), or does it require the specific SAE features?

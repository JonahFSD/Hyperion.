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

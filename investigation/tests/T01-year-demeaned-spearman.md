---
id: T01
status: not_run
layer: 2
resolves: [H01, H02]
script: 1a_11
---

## Year-Demeaned Spearman Test

**Resolves:** Is global rho=0.022 confounded by year-regime shifts?

**Method:**
1. For each year, subtract year-mean return correlation from all pairs
2. Recompute global Spearman(cosine, demeaned_correlation)
3. Compare to baseline rho=0.022

**Pass criterion:**
- Year-demeaned rho > 0.022 and CI excludes zero

**Fail:** Signal was purely market-regime driven.

**Data:** 14.9M pairs, cosine + returns for all years.

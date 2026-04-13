---
id: T02
status: not_run
layer: 2
resolves: [H-alive-4, H-alive-6]
script: 1a_11
---

## Top-1% SIC Composition Test

**Resolves:** Is top-1% inversion (E08) from filing-template similarity or real market effect?

**Method:**
1. Identify top 1% pairs by cosine similarity
2. Extract SIC codes for each pair
3. Compute: % same-SIC pairs in top-1% vs baseline rate
4. Year-stratify to detect regime effects

**Pass criterion:**
- Same-SIC rate in top-1% >> baseline (templates isolated by SIC)
- OR year-consistency (signal is time-varying, not template)

**Fail:** Top-1% is truly low-correlation state (h-alive-5: nonlinear metric).

**Data:** Pairs with cosine > 99th percentile.

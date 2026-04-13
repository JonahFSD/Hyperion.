---
id: T04
status: not_run
layer: 2
resolves: [H01, H-alive-8]
script: 1a_11
---

## Within-SIC Retrieval Precision@K

**Resolves:** Does within-industry retrieval work? Product-viability test.

**Method:**
1. For each company-year, rank same-SIC peers by cosine(SAE)
2. For K ∈ {1, 5, 10}, compute avg return correlation of top-K
3. Compare to random same-SIC baseline
4. Repeat for SBERT (industry-agnostic embeddings)

**Pass criterion:**
- SAE top-K same-SIC correlation > random same-SIC baseline
- SAE beats SBERT on same-SIC peers

**Fail:** SAE adds no value for within-industry ranking.

**Data:** Filtered pairs from T03 (same SIC only).

**Expected:** E09 (within-SIC rho=0.117) suggests strong pass.

---
id: T03
status: not_run
layer: 2
resolves: [H01, H-alive-7]
script: 1a_11
---

## Nearest-Neighbor Precision@K Test

**Resolves:** Does SAE retrieval work AT ALL? Make-or-break test.

**Method:**
1. For each company-year, rank all peers by cosine(SAE features)
2. For K ∈ {1, 5, 10, 50}, compute avg return correlation of top-K
3. Compare to random baseline (expected correlation of K random peers)
4. Repeat for SIC and SBERT baselines

**Pass criterion:**
- SAE top-K correlation >> random baseline, CI excludes baseline

**Fail:** Features carry no retrieval signal.

**Data:** ~24k company-years, 12-month forward returns, all peers.

**Key:** Size-stratify results; SAE should beat SIC even in large-cluster regime.

---
id: H04
status: dead
layer: 1
killed_by: [E04, E05]
---

## Mean Correlation Fairly Measures Quality

Equal-weighted Mean Correlation (MC) fairly measures clustering quality across methods with different cluster-size distributions.

**DEAD.** Killed by:
- E04: SAE loses under pair-weighting (0.151 < SIC 0.252)
- E05: Cluster size divergence (SAE shrinking, SIC/SBERT growing)

**Consequence:** Pivot evaluation to precision@K retrieval metrics instead.

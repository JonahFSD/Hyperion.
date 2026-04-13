---
id: E07
layer: 2
source: 1a_10
date: 2026-03-26
---

## Raw Cosine Has Near-Zero Global Predictive Power

Across all 14.9M company-pair comparisons, cosine similarity barely predicts return correlation.

**Numbers:**
- Spearman rho = 0.022

**Challenges:** H01, H02 (features or metric broken?)

**Confound:** Year-regime effect (yearly correlation baseline shifts). T01 designed to isolate.

**Key insight:** Global MC success despite near-zero rho suggests clustering captures structure that correlation-magnitude misses.

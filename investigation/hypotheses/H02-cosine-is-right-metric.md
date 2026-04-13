---
id: H02
status: questionable
layer: 2
tested_by: [T05]
evidence: [E08, E13]
---

## Cosine Similarity Is the Right Distance Metric

Cosine similarity in SAE feature space is the right distance metric for measuring structural similarity.

**Challenges:**
- Top-1% inversion: most similar pairs have below-baseline correlation (E08)
- Hump-shaped ventile relationship: rho rises then falls across similarity quantiles (E13)

**Alternative hypothesis:**
Signal lives in graph topology (nearest-neighbor structure), not raw magnitude.

**Tests:** T05 (topology vs magnitude)

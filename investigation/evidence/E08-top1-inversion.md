---
id: E08
layer: 2
source: 1a_10
date: 2026-03-26
---

## Most Similar Pairs Have Below-Baseline Correlation

The most similar 1% of pairs (by cosine) are LESS correlated than baseline.

**Numbers:**
- Top 1% median correlation: 0.144
- Baseline (all pairs) median: 0.161

**Challenges:** H02 (cosine may not be right metric for similarity)

**Possible causes:**
- Template similarity without business similarity
- Genuine nonlinear relationship (H-alive-5)
- Market regime/year-specific (H-alive-6)

**Tests:** T02 (SIC composition), T01 (year-demeaning).

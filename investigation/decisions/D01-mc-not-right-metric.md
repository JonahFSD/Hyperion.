---
id: D01
date: 2026-03-26
triggered_by: [E04, E05]
consequence: "Pivot evaluation to precision@K retrieval metrics"
---

## Mean Correlation Is Not the Right Metric

Mean Correlation (MC) was used in the ACL paper to measure clustering quality. For Hyperion, it's not a fair evaluation metric.

**Why?**
- E04: SAE loses to SIC under pair-weighting (0.151 < 0.252)
- E05: Cluster size divergence (SAE shrinking, SIC growing)

**The problem:**
SAE's high MC comes from forming tiny clusters, not from capturing robust signal. Pair-weighted MC corrects for this but creates other distortions.

**Solution:**
Switch to **precision@K retrieval metrics** (T03-T04). Tests whether cosine-ranked neighbors actually have higher return correlation than random peers.

**Impact:**
Layer 2 critical path shifts from clustering quality to retrieval accuracy.

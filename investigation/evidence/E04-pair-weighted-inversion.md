---
id: E04
layer: 1
source: 1a_09
date: 2026-03-26
---

## SAE Loses Under Pair-Weighting

When clusters are weighted by pair counts, SAE reverses its advantage.

**Numbers:**
- SAE pair-weighted MC: 0.151
- SIC: 0.252
- SBERT: 0.210

**Kills:** H04 (MC is not fair metric), triggers D01 (pivot to precision@K).

**Interpretation:** SAE's high MC comes from tiny clusters with few pairs, not robust signal.

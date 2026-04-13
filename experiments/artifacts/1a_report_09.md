## Step 9: Cluster Size Confound Diagnostic

### Motivation

SAE clusters shrink over time (2.16 → 1.48 companies/cluster) while
SIC grows (2.71 → 4.34) and SBERT grows (14.1 → 29.1). Smaller clusters
mechanically produce higher MC. This diagnostic controls for cluster size
by weighting each cluster's MC by its pair count instead of equal-weighting.

### Results

**Overall MC (common years only)**

| Method | Equal-Wt MC | Pair-Wt MC | Change |
|--------|------------|------------|--------|
| sae_cd | 0.3590 | 0.1509 | -0.2080 |
| sic | 0.2311 | 0.2520 | +0.0209 |
| sbert | 0.2193 | 0.2102 | -0.0092 |

**Deltas (pair-weighted)**

| Comparison | Equal-Wt Delta | Pair-Wt Delta | % Surviving |
|------------|---------------|---------------|-------------|
| SAE - SIC | +0.1279 | -0.1010 | -79% |
| SAE - SBERT | +0.1396 | -0.0592 | -42% |

**Temporal Trend**

- sae_minus_sic: equal-wt slope=+0.006446 (CI [0.003010, 0.010303]), pair-wt slope=-0.001327 (CI [-0.002848, 0.000049])
- sae_minus_sbert: equal-wt slope=+0.007489 (CI [0.004375, 0.010887]), pair-wt slope=+0.000348 (CI [-0.000424, 0.001100])

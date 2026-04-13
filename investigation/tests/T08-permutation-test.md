---
id: T08
status: not_run
layer: 1
resolves: [algorithmic-artifact]
script: 1c
---

## Feature-Shuffle Permutation Test

**Resolves:** Is clustering advantage an algorithmic artifact, not real signal?

**Method:**
1. Shuffle SAE features row-wise (preserve structure, break company-feature mapping)
2. Rebuild MST + clusters on shuffled features
3. Compute MC on shuffled clusters
4. Repeat 10,000 iterations
5. Empirical p-value: proportion of permutations > observed MC

**Pass criterion:**
- Empirical p < 0.05 (observed MC unlikely under null of random features)

**Fail:** Algorithm inflates correlations regardless of feature-company link.

**Data:** 131K-dim SAE features, ~24k company-years.

**Interpretation:** If pass, clustering procedure is not creating spurious high-correlation clusters.

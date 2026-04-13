---
id: T05
status: not_run
layer: 2
resolves: [H02, H03, H-alive-5, H-alive-10]
script: 1a_11
---

## Topology vs Magnitude Test

**Resolves:** Does MST graph structure add value beyond raw cosine magnitude?

**Method:**
1. For each company-year, identify nearest-neighbor pairs (in MST)
2. Identify non-NN pairs with same cosine magnitude (matched controls)
3. Compare avg return correlation: NN pairs vs magnitude-matched non-NN pairs
4. Bootstrap CIs

**Pass criterion:**
- NN pairs have higher correlation than magnitude-matched controls
- Suggests topology (MST structure) is doing real work

**Fail:** Signal is purely in cosine magnitude, not topology.

**Data:** Pairs with cosine ∈ [50th, 99th] percentiles (avoid extremes).

**Implication:** If pass, graph-based ranking may beat magnitude-based ranking.

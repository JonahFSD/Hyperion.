---
id: D02
date: 2026-03-26
triggered_by: "Flags conflating layers"
consequence: "Layer 2 is critical path; Layer 1 findings don't auto-propagate"
---

## Three-Layer Evaluation Framework

Evaluation has been conflating different claims. Separate into three layers.

**Layer 1 — Upstream (ACL Paper):**
Do the paper's methods (MST clustering, MC metric) work as described?

**Layer 2 — Feature Signal:**
Do SAE features predict return comovement via retrieval? Core Hyperion question.

**Layer 3 — Product Utility:**
Can we build a useful equity research tool from Layer 2 signal?

**Why this matters:**
- ACL paper succeeded (E01, E02, E03 replicate)
- But Layer 1 success ≠ Layer 2 signal (E07: global rho=0.022)
- Layer 2 is the bottleneck; Layer 3 is blocked until Layer 2 resolves

**Implication:**
Tests T01-T05 (retrieval) are critical path. T06-T08 (Layer 1 validation) run in parallel.

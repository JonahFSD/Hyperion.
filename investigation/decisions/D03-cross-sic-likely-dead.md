---
id: D03
date: 2026-03-26
triggered_by: [E10]
consequence: "Focus Hyperion on within-industry retrieval; cross-industry is stretch goal"
---

## Cross-SIC Signal Likely Dead

Evidence E10 shows that SAE has zero predictive power for cross-industry pairs.

**Numbers:**
- Cross-SIC Spearman rho = 0.016 (essentially noise)
- Within-SIC rho = 0.117 (5.6x stronger)

**Implication:**
SAE discovers structural similarity WITHIN industries but fails ACROSS industries. This kills the "find any analog company globally" use case.

**Decision:**
Hyperion's core use case is **within-industry retrieval** (T04). The system asks: "Among companies in my industry, which peers are most similar to this one?"

**Future work:**
Cross-industry analog-finding may require different signals (financials, growth stage, market cap, etc.). Not a SAE problem.

**Consequence:**
Product scope narrows but confidence in within-industry case increases.

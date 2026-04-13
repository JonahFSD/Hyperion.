---
id: H06
status: untested
layer: 1
tested_by: [T07]
---

## SAE Signal Isn't Just Factor Exposure

The SAE clustering advantage is not explained by shared Fama-French factor exposure.

**Motivation:**
If companies in same SAE cluster just have similar betas, the signal is "macro-driven" — SIC already tells you that.

**Test:** T07 (factor adjustment, 1B protocol)

**What we need:**
FF5 regression, residual MC, to isolate company-specific signal.

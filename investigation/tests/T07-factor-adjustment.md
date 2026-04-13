---
id: T07
status: not_run
layer: 1
resolves: [H06, H-alive-2]
script: 1b_01-09
---

## Factor Adjustment: FF5 Regression

**Resolves:** Is SAE signal explained by Fama-French 5-factor exposure?

**Method:**
1. For each company-year, regress 12-month returns on FF5 factors
2. Extract residuals (idiosyncratic return)
3. Recompute Mean Correlation (clusters) on residuals
4. Compare SAE residual MC to SIC residual MC

**Pass criterion:**
- SAE residual MC advantage persists (signal is idiosyncratic, not macro)

**Fail:** SAE advantage disappears (signal is factor-driven, H06 killed).

**Data:** FF5 factors (Fama-French data), 12-month returns, ~24k company-years.

**Expected:** If H01 true, SAE should retain advantage post-FF5.

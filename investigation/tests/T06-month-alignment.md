---
id: T06
status: not_run
layer: 1
resolves: [data-integrity]
script: 1b_00
---

## Month Alignment Validation

**Resolves:** HARD GATE for 1B. Are returns correctly aligned to filings?

**Method:**
1. Spot-check 50 random company-years
2. Pull filing date from SEC
3. Pull 12-month returns from Yahoo Finance starting filing date
4. Verify alignment: no data-bleeding, calendar boundaries correct

**Pass criterion:**
- All 50 spot-checks match expectations (no off-by-one months)
- Return dates align with filing calendar

**Fail:** Factor regression (1B) is built on corrupted returns. Cannot proceed.

**Data:** SEC CIK, filing dates, Yahoo ticker, returns.

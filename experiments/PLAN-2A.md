# PLAN-2A: SAE Signal Backtest

## Context

Phase 1 validation PASSED (13 tests, 3 layers, 25 years, 14.9M pairs). Walk-forward PCA diagnostic PASSED (Spearman 0.985-0.9998, lift preservation >1.0 in all years). The SAE similarity signal is real and survives temporal integrity requirements.

This plan constructs a rigorous backtest to measure whether the signal produces tradeable alpha.

**Two strategies, tested sequentially:**
- **Phase 2A**: Pairs trading (mean reversion) — established benchmarks, direct comparison to Molinari Sharpe 12.18
- **Phase 2B**: Analog return prediction — closer to Hyperion product vision, bigger potential edge

**Methodology sources:**
- `docs/research/backtest-methodology-survey.md` — comprehensive framework survey
- Gatev-Goetzmann-Rouwenhorst (2006) — formation/trading mechanics
- Hoberg-Phillips (2010, 2016) — SEC filing temporal conventions
- Harvey-Liu-Zhu (2016) — statistical significance (t > 3.0)
- Bailey & López de Prado (2014) — Deflated Sharpe Ratio

**Data (all on HuggingFace, already validated in Phase 1):**
- Raw SAE features: `marco-molinari/company_reports_with_features` (~27,888 rows × 131K dims)
- Company metadata + returns: `Mateusz1017/annual_reports_tokenized_...` (cik, year, ticker, sic_code, 12 monthly logged returns)
- Pre-computed cosine sims: `v1ctor10/cos_sim_4000pca_exp` (14.9M pairs)
- Fama-French 5 factors: `mba.tuck.dartmouth.edu`

**Hardware constraint:** 16 GB RAM Mac Mini. All scripts must stream features in batches (see 2a_01 pattern).

---

## Temporal Structure

Following Hoberg-Phillips / Davis-Fama-French conventions:

- **Filing availability:** 10-K filings for fiscal year t are filed ~60 days after FYE. For Dec 31 FYE: available by ~March t+1.
- **Signal date:** July 1 of year t+1. Conservative — ensures all fiscal year t filings are available.
- **Portfolio formation:** July 1 of year t+1.
- **Holding period:** July t+1 through June t+2 (12 months).
- **Walk-forward PCA:** Fitted on all filings with fiscal year ≤ t (available by signal date).
- **Pair selection:** Within-SIC cosine similarity from walk-forward PCA features.

**Timeline example:**
- Fiscal year 2005 10-K filings → available by ~March 2006
- Signal computed: July 1, 2006 (WF-PCA on all filings through FY2005)
- Portfolio formed: July 1, 2006
- Portfolio held: July 2006 – June 2007
- Returns measured: July 2006 – June 2007

**Backtest span:** Formation years 1999–2019 → trading periods July 1999–June 2020 (21 years).
- First formation (1999): WF-PCA trained on 1996-1998 filings (3 years, sparse but usable)
- Last formation (2019): WF-PCA trained on 1996-2018 filings, trades through June 2020

---

## Script Pipeline

### ✅ 2a_01 — Walk-Forward PCA Diagnostic
**Status:** DONE. PASSED.
**Output:** `artifacts/2a_01_walkforward_pca.json`
**Result:** Spearman 0.985-0.9998, lift preservation >1.0, signal is not a PCA artifact.

### ☐ 2a_02 — Pair Universe Construction
**What:** For each formation year (1999-2019), compute walk-forward PCA, build within-SIC similarity matrices, select top pairs.
**Input:** Raw SAE features (HF), company metadata (HF)
**Output:** `artifacts/2a_02_pair_universe.parquet` — one row per pair per formation year:
  - `year`: formation year
  - `company1_idx`, `company2_idx`: HF index identifiers
  - `company1_ticker`, `company2_ticker`: tickers
  - `sic2`: shared 2-digit SIC code
  - `cosine_sim`: walk-forward PCA cosine similarity
  - `rank_in_sic`: rank within SIC group (1 = most similar)
  - `n_sic_peers`: size of SIC group

**Key decisions:**
- Use walk-forward PCA (validated by 2a_01), NOT pre-computed global cosine sims
- Stream features in batches (16 GB RAM constraint, reuse 2a_01 pattern)
- Include ALL within-SIC pairs (don't pre-filter to top-K yet — let downstream scripts choose K)
- Store cosine sims to 6 decimal places

**Verify:** Spot-check 2010 pairs against 2a_01 diagnostics. Confirm pair counts per year are reasonable.

### ☐ 2a_03 — Return Matrix Construction
**What:** Build a point-in-time monthly return matrix aligned to the trading calendar.
**Input:** Company metadata + returns (HF), Fama-French 5 factors (web)
**Output:**
  - `artifacts/2a_03_returns.parquet` — monthly returns per company, July-to-June aligned
  - `artifacts/2a_03_ff5.parquet` — Fama-French 5 factors + momentum, monthly

**Key decisions:**
- Returns are logged monthly returns from HF dataset (12 per company-year)
- Must map: company-year returns to calendar months (year Y → returns are for months in fiscal year Y)
- Download FF5 + momentum factors, align by YYYYMM date key (NOT positional — per CLAUDE.md)
- Handle delisted companies: if return series ends mid-year, fill remaining months with NaN (position closed)
- NO survivorship bias: include all companies that existed during the period, even if later delisted

**Verify:** Check return matrix coverage by year. Confirm FF5 alignment with known values (e.g., MKT-RF for specific months).

### ☐ 2a_04 — Pairs Trading Simulation
**What:** Execute Gatev-style pairs trading strategy using SAE-selected pairs.
**Input:** Pair universe (2a_02), return matrix (2a_03)
**Output:**
  - `artifacts/2a_04_strategy_returns.parquet` — monthly strategy returns
  - `artifacts/2a_04_trade_log.parquet` — individual trade records
  - `artifacts/2a_04_summary.json` — aggregate statistics

**Strategy mechanics (Gatev framework adapted for annual SAE signal):**
1. **Formation:** At each July signal date, take top-K pairs per SIC group by SAE cosine similarity
   - Test K = 10, 20, 50 (parameter sensitivity)
2. **Spread construction:** For each pair, compute normalized price spread over trading period
   - Normalize: (P_A / P_A_start) - (P_B / P_B_start)
3. **Entry signal:** Spread diverges > 2σ from formation-period mean
   - σ estimated from formation-period spread (rolling 12-month lookback)
4. **Exit signal:** Spread converges back to mean, OR 12-month holding period ends, OR stop-loss at 4σ
5. **Position sizing:** Equal dollar long-short per pair. Equal weight across all active pairs.
6. **Gatev "wait one day" rule:** Delay entry by 1 trading day after signal (eliminates bid-ask bounce)

**Transaction cost modeling:**
- Base case: 10 bps one-way (conservative for mid/large cap)
- Sensitivity: 5 bps (optimistic), 20 bps (conservative), 50 bps (stress test)
- Apply on entry AND exit
- Short-selling cost: +25 bps/year on short leg

**Monthly return computation:**
- Mark-to-market all open positions at month end
- Strategy return = sum of P&L across all positions / total capital deployed
- Track: gross return, net return (after costs), number of active pairs, turnover

**Verify:**
- Sanity check: random-pair placebo test (replace SAE pairs with random same-SIC pairs, expect ~0 return)
- Check that turnover matches expected ~5-15% monthly (annual signal → low turnover)

### ☐ 2a_05 — Factor Adjustment & Statistical Validation
**What:** Full statistical validation of strategy returns.
**Input:** Strategy returns (2a_04), FF5 + momentum (2a_03)
**Output:** `artifacts/2a_05_validation.json`

**Tests:**
1. **Time-series factor regression:**
   R_strategy - R_f = α + β₁(MKT-RF) + β₂SMB + β₃HML + β₄RMW + β₅CMA + β₆MOM + ε
   - Report α, t(α), R², all betas with standard errors
   - Newey-West standard errors, 6-lag correction
   - Target: t(α) > 3.0 (Harvey-Liu-Zhu gold standard)

2. **Deflated Sharpe Ratio (Bailey & López de Prado 2014):**
   - Correct for number of strategy variations tested
   - Must record: all K values tested, all cost assumptions, all parameter variations
   - Target: DSR > 0.95

3. **Sub-period analysis:**
   - Split into: 1999-2004, 2005-2009, 2010-2014, 2015-2019
   - Must be positive in at least 3 of 4 sub-periods
   - Also test across regimes: tech crash (2000-02), GFC (2007-09), COVID (Mar 2020)

4. **Random-pair placebo test (Gatev et al.):**
   - Replace SAE-selected pairs with random same-SIC pairs of similar market cap
   - Run same strategy
   - Expect: ~0 or slightly negative returns (confirms SAE adds value beyond SIC membership)

5. **Parameter sensitivity:**
   - K (number of pairs): 10, 20, 50
   - Entry threshold: 1.5σ, 2.0σ, 2.5σ
   - Stop-loss: 3σ, 4σ, none
   - Results should be stable across a neighborhood, not a fragile peak

6. **Delayed execution test:**
   - Re-run with 5-day delay on all trades (models slow execution)
   - Return degradation should be small given low-frequency signal

**Verify:** Cross-check Sharpe ratio against Molinari's 12.18 (theirs is pre-cost, pre-factor-adjustment, with look-ahead PCA — ours should be lower but hopefully still significant).

### ☐ 2a_06 — Results Report
**What:** Aggregate all results into a structured report.
**Input:** All artifacts from 2a_01 through 2a_05
**Output:** `artifacts/2a_06_backtest_report.json` + `artifacts/2a_06_report.md`

---

## Phase 2B: Analog Return Prediction (after 2A)

### ☐ 2b_01 — Analog Outcome Mapping
**What:** For each company-year, find top-K SAE analogs from PREVIOUS years, record what happened to those analogs AFTER the similarity was measured.
**Input:** Pair universe (2a_02), return matrix (2a_03)
**Output:** `artifacts/2b_01_analog_outcomes.parquet`

**Logic:** If Company A in 2010 is most similar to Company B in 2008, look at Company B's returns in 2009-2010 (the period AFTER the similarity was measured). Use those outcomes to predict Company A's returns in 2011.

### ☐ 2b_02 — Analog Strategy Simulation
**What:** Long companies whose analogs had positive subsequent returns, short those whose analogs had negative returns.
**Input:** Analog outcomes (2b_01), return matrix (2a_03)
**Output:** `artifacts/2b_02_analog_strategy_returns.parquet`

### ☐ 2b_03 — Analog Validation
**What:** Same validation framework as 2a_05.
**Output:** `artifacts/2b_03_analog_validation.json`

---

## Execution Notes

- **Ralph Loop:** One script per checklist item. Fresh context per iteration. AI picks top unchecked item.
- **Memory:** 16 GB RAM. All feature loading must stream in batches (see 2a_01 extract_batch pattern).
- **Seeding:** Use `np.random.default_rng(42)` consistently. Document all randomness.
- **Artifacts:** JSON for summaries, Parquet for data. All in `experiments/artifacts/`.
- **Logging:** Use `log()` pattern from existing scripts. Print progress for long operations.

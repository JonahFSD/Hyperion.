# Backtesting Methodology Survey — SAE Pairwise Similarity Signals

## Source
Deep research output, April 2026. Comprehensive survey of established frameworks for backtesting company similarity signals derived from NLP/SAE features on SEC filings.

## Key Findings

### Three Frameworks to Integrate
1. **Gatev-Goetzmann-Rouwenhorst (2006)** — Formation/trading architecture (12mo form, 6mo trade, "wait one day" rule)
2. **Hoberg-Phillips TNIC** — SEC-filing temporal conventions (Davis-Fama-French lag: fiscal year t data → July t+1 through June t+2)
3. **Harvey-Liu-Zhu (2016)** — Statistical bar (t > 3.0, Deflated Sharpe Ratio, honest multiple testing correction)

### Critical Methodological Requirements
1. **Walk-forward SAE training** — SAE + PCA must be retrained using only filings available at each signal date. Molinari et al.'s global PCA across 1996–2020 is look-ahead bias. This is the #1 thing to fix.
2. **Factor adjustment on STRATEGY RETURNS** — FF5 + momentum time-series regressions, GRS tests, Newey-West 6-lag SE. Signal-level factor adjustment (our 96-99% survival) is necessary but insufficient.
3. **Deflated Sharpe Ratio** — Bailey & López de Prado (2014). Requires recording every specification tested.

### Temporal Integrity Checklist (20 failure modes)
- Filing text only usable after EDGAR acceptance timestamp + 1 trading day
- Large accelerated filers: 10-K due 60 days after FYE, empirically clusters at 55-60 days
- December 31 FYE → filing available ~early March → signal enters portfolio July (H-P convention)
- Must use historical SIC codes, not current
- Must use historical index constituents, not current
- Survivorship: CRSP delisting returns (DLRET), -30% NYSE/AMEX or -55% Nasdaq when missing
- SAE/PCA must NOT see future filings (walk-forward only)
- Vocabulary/normalization statistics must be rolling, not full-sample

### Transaction Costs — We Have a Structural Advantage
- Annual update cycle from 10-K → estimated 5-15% monthly turnover (lowest tier)
- Comparable to value/quality factors, far below momentum (50-100%)
- Breakeven at 8% turnover + 30 bps/month gross alpha = 375 bps per trade (way above any realistic cost)
- Current realistic costs: large-cap 3-10 bps, mid-cap 10-30 bps, small-cap 30-100+ bps one-way
- Frazzini et al. (2018): actual institutional costs are ~1/10th of academic estimates

### Molinari et al. Backtest Gaps (Our Starting Point)
- Sharpe 12.18 but: zero transaction costs, no factor adjustment, global PCA (look-ahead)
- Correlation threshold 0.95 is extremely restrictive (selects near-tautological pairs)
- No FF factor adjustment on strategy returns
- ±2σ stop-losses may involve illiquid names

### Key Comparable Papers
- **Hoberg-Phillips (2010, 2016)**: TNIC text similarity, Davis-Fama-French lag conventions
- **Cohen, Malloy, Nguyen (2020) "Lazy Prices"**: 10-K/10-Q text change → 188 bps/month alpha
- **Chen, Kelly, Xiu (2026, RFS)**: LLM embeddings → return prediction, Sharpe ~3-4
- **Gatev et al. (2006)**: Canonical pairs trading, 76.4 bps/month alpha, R²=0.05-0.09
- **McLean & Pontiff (2016)**: Post-publication decay ~26% (data mining) + ~32% (crowding)
- **Do & Faff (2010, 2012)**: Pairs trading profits declined substantially post-2002
- **Rad, Low, Faff (2016)**: Cointegration methods retain 33 bps/month after costs through 2014

### Red Flags to Design Around
- Walk-forward PCA is non-negotiable (Molinari's global PCA is look-ahead)
- McLean-Pontiff: mentally discount reported performance by ≥26%
- August 2007 Quant Quake: crowding/correlation risk in mean-reversion strategies
- Regime dependence: must test across 2000-02, 2007-09, 2020 COVID, 2022 rate hikes
- Random-pair placebo test (Gatev et al.): replace real pairs with random matched securities
- SEC disclosure regulation changes alter filing language independent of economics

### Statistical Validation Stack
1. Time-series FF5+MOM regression on strategy returns (α with t > 3.0, Newey-West 6-lag)
2. GRS test across multiple test portfolios
3. Deflated Sharpe Ratio (Bailey & López de Prado 2014)
4. Hansen's SPA test (2005) for data snooping control
5. Sub-period analysis (2-4 equal periods + crisis regimes)
6. Random-pair placebo test
7. Parameter sensitivity (neighborhood stability, not fragile peak)
8. Delayed execution test (1-day lag impact)

### Implementation Architecture
- Two-stage: vectorized screening (NumPy/Pandas on Parquet/DuckDB) across all 15M pairs → full event-driven backtest on top 1K-10K pairs
- Pairs are embarrassingly parallel (Dask, Ray, multiprocessing)
- No existing framework handles 15M pairs natively; custom build required

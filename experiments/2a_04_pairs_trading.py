#!/usr/bin/env python3
"""
2a_04_pairs_trading.py — Pairs Trading Backtest

Runs the full pairs trading simulation using SAE-selected pairs and monthly returns.
Two strategies: committed pairs portfolio (academic factor-style) and conditional
entry (Gatev adaptation). Includes 100-trial random-pair placebo.

Inputs:
  experiments/artifacts/2a_02_pair_universe.parquet
  experiments/artifacts/2a_03_returns.parquet

Outputs:
  experiments/artifacts/2a_04_strategy_returns.parquet
  experiments/artifacts/2a_04_trade_log.parquet
  experiments/artifacts/2a_04_placebo.json
  experiments/artifacts/2a_04_summary.json
"""

import os
import gc
import json
import time
import numpy as np
import pandas as pd

# ═══════════════════════════════════════════════════════════
# CONFIG
# ═══════════════════════════════════════════════════════════

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ARTIFACTS_DIR = os.path.join(BASE_DIR, "artifacts")

PAIR_UNIVERSE_FILE = os.path.join(ARTIFACTS_DIR, "2a_02_pair_universe.parquet")
RETURNS_FILE = os.path.join(ARTIFACTS_DIR, "2a_03_returns.parquet")

STRATEGY_RETURNS_OUT = os.path.join(ARTIFACTS_DIR, "2a_04_strategy_returns.parquet")
TRADE_LOG_OUT = os.path.join(ARTIFACTS_DIR, "2a_04_trade_log.parquet")
PLACEBO_OUT = os.path.join(ARTIFACTS_DIR, "2a_04_placebo.json")
SUMMARY_OUT = os.path.join(ARTIFACTS_DIR, "2a_04_summary.json")

# Formation years with full (12-month) or partial trading periods
FORMATION_YEARS_FULL = list(range(1999, 2019))   # FY1999–FY2018: full 12-month trading
FORMATION_YEARS_PARTIAL = [2019]                  # FY2019: only July–Dec 2020 (6 months)
FORMATION_YEARS = FORMATION_YEARS_FULL + FORMATION_YEARS_PARTIAL

# Strategy parameters
K_VALUES = [10, 20, 50]
ENTRY_THRESHOLDS = [1.5, 2.0, 2.5]
STOP_LOSSES = [3.0, 4.0, 999.0]

# Cost levels (one-way, in decimal)
COST_LEVELS_BPS = [0, 5, 10, 20, 50]
SHORT_SELL_COST_ANNUAL = 0.0025  # 25 bps/year

# Placebo
N_PLACEBO_TRIALS = 100
PLACEBO_K = 20
PLACEBO_ENTRY = 2.0
PLACEBO_STOP = 4.0

# Minimum months required in calibration and trading periods
MIN_MONTHS_REQUIRED = 8

# Spread std floor
SPREAD_STD_FLOOR = 0.01

# Return filter: skip monthly returns beyond this (data errors / penny stocks)
MAX_MONTHLY_SIMPLE_RETURN = 2.0  # 200% per month

SEED = 42


def log(msg):
    print(f"[2a_04] {msg}", flush=True)


# ═══════════════════════════════════════════════════════════
# STEP 1: Load data
# ═══════════════════════════════════════════════════════════

t0 = time.time()
log("Step 1/7: Loading pair universe and return matrix...")

pairs_df = pd.read_parquet(PAIR_UNIVERSE_FILE)
returns_df = pd.read_parquet(RETURNS_FILE)

log(f"  Pair universe: {len(pairs_df):,} rows, formation_years {pairs_df['formation_year'].min()}–{pairs_df['formation_year'].max()}")
log(f"  Return matrix: {len(returns_df):,} rows, calendar_months {returns_df['calendar_month'].min()}–{returns_df['calendar_month'].max()}")

# ═══════════════════════════════════════════════════════════
# STEP 2: Build return lookup keyed by (CIK, YYYYMM)
# ═══════════════════════════════════════════════════════════

log("Step 2/7: Building return lookup and company_idx → CIK mapping...")

# company_idx is per-filing-row in HuggingFace, NOT a stable company ID.
# CIK is stable across years (mean 17.4 years per CIK).
# Build: idx_to_cik mapping + return_lookup keyed by (cik, calendar_month)

idx_to_cik = {}
return_lookup = {}
n_filtered = 0

for _, row in returns_df.iterrows():
    cidx = int(row['company_idx'])
    cik = int(row['cik'])
    cm = int(row['calendar_month'])
    lr = row['log_return']
    sr = row['simple_return']
    idx_to_cik[cidx] = cik
    if not np.isnan(lr):
        if abs(sr) > MAX_MONTHLY_SIMPLE_RETURN:
            n_filtered += 1
            continue
        return_lookup[(cik, cm)] = (lr, sr)

log(f"  Lookup entries: {len(return_lookup):,} (filtered {n_filtered} extreme returns > {MAX_MONTHLY_SIMPLE_RETURN:.0%})")
log(f"  Unique company_idx → CIK mappings: {len(idx_to_cik):,}")
log(f"  Unique CIKs: {len(set(idx_to_cik.values())):,}")

del returns_df
gc.collect()


# ═══════════════════════════════════════════════════════════
# HELPERS: Calendar and month sequences
# ═══════════════════════════════════════════════════════════

def get_months_sequence(start_ym, n_months):
    """Generate a list of n_months YYYYMM values starting from start_ym."""
    months = []
    y, m = divmod(start_ym, 100)
    for _ in range(n_months):
        months.append(y * 100 + m)
        m += 1
        if m > 12:
            m = 1
            y += 1
    return months


def get_calibration_months(fy):
    """July of FY through June of FY+1 (12 months before signal date)."""
    return get_months_sequence(fy * 100 + 7, 12)


def get_trading_months(fy):
    """July of FY+1 through June of FY+2 (12 months after signal date).
    For partial years (FY2019), only return available months (July–Dec of FY+1)."""
    if fy in FORMATION_YEARS_PARTIAL:
        return get_months_sequence((fy + 1) * 100 + 7, 6)
    return get_months_sequence((fy + 1) * 100 + 7, 12)


def get_returns_for_cik(cik, months):
    """Look up log and simple returns for a CIK over a list of months.
    Returns two lists: log_returns, simple_returns (with None for missing)."""
    log_rets = []
    simple_rets = []
    for m in months:
        val = return_lookup.get((cik, m))
        if val is not None:
            log_rets.append(val[0])
            simple_rets.append(val[1])
        else:
            log_rets.append(None)
            simple_rets.append(None)
    return log_rets, simple_rets


# ═══════════════════════════════════════════════════════════
# STEP 3: Build variant configurations
# ═══════════════════════════════════════════════════════════

log("Step 3/7: Building strategy variant configurations...")

variants = []

# Committed variants
for k in K_VALUES:
    variants.append({
        'name': f'committed_K{k}',
        'strategy': 'committed',
        'K': k,
        'entry_threshold': None,
        'stop_loss': None,
        'is_placebo': False,
    })

# Conditional variants
for k in K_VALUES:
    for et in ENTRY_THRESHOLDS:
        for sl in STOP_LOSSES:
            variants.append({
                'name': f'conditional_K{k}_{et}sig_{sl}stop',
                'strategy': 'conditional',
                'K': k,
                'entry_threshold': et,
                'stop_loss': sl,
                'is_placebo': False,
            })

log(f"  {len(variants)} strategy variants (excluding placebo)")


# ═══════════════════════════════════════════════════════════
# STEP 4: Core simulation engine
# ═══════════════════════════════════════════════════════════

log("Step 4/7: Running strategy simulations...")


def select_pairs(pairs_df, fy, k, rng=None):
    """Select top-K pairs per SIC2 group for a formation year.
    If rng is provided, select K random pairs per SIC2 instead."""
    fy_pairs = pairs_df[pairs_df['formation_year'] == fy]
    if rng is not None:
        selected = []
        for sic2, group in fy_pairs.groupby('sic2'):
            n_avail = len(group)
            n_select = min(k, n_avail)
            idx = rng.choice(n_avail, size=n_select, replace=False)
            selected.append(group.iloc[idx])
        if selected:
            return pd.concat(selected, ignore_index=True)
        return pd.DataFrame()
    else:
        return fy_pairs[fy_pairs['rank_in_sic'] <= k].copy()


def min_months_for_period(n_months):
    """Scale MIN_MONTHS_REQUIRED proportionally for partial trading periods."""
    return max(int(MIN_MONTHS_REQUIRED * n_months / 12), 3)


def simulate_committed(selected_pairs, fy):
    """Simulate committed pairs portfolio for one formation year."""
    cal_months = get_calibration_months(fy)
    trd_months = get_trading_months(fy)
    n_trd = len(trd_months)
    min_trd = min_months_for_period(n_trd)

    monthly_records = []
    trade_records = []
    pair_monthly = {}  # pair_key -> list of (month, return_or_None)

    for _, pair in selected_pairs.iterrows():
        c1_idx = int(pair['company1_idx'])
        c2_idx = int(pair['company2_idx'])
        c1_cik = idx_to_cik.get(c1_idx)
        c2_cik = idx_to_cik.get(c2_idx)
        if c1_cik is None or c2_cik is None:
            continue

        # Get calibration returns by CIK (spans filing years)
        c1_cal_log, _ = get_returns_for_cik(c1_cik, cal_months)
        c2_cal_log, _ = get_returns_for_cik(c2_cik, cal_months)

        cal_valid = sum(1 for a, b in zip(c1_cal_log, c2_cal_log) if a is not None and b is not None)
        if cal_valid < MIN_MONTHS_REQUIRED:
            continue

        # Get trading returns by CIK
        c1_trd_log, c1_trd_simple = get_returns_for_cik(c1_cik, trd_months)
        c2_trd_log, c2_trd_simple = get_returns_for_cik(c2_cik, trd_months)

        trd_valid = sum(1 for a, b in zip(c1_trd_simple, c2_trd_simple) if a is not None and b is not None)
        if trd_valid < min_trd:
            continue

        # Determine direction: cumulative log return over calibration
        # Only sum months where BOTH companies have returns (apples-to-apples)
        cum1 = sum(a for a, b in zip(c1_cal_log, c2_cal_log) if a is not None and b is not None)
        cum2 = sum(b for a, b in zip(c1_cal_log, c2_cal_log) if a is not None and b is not None)

        if cum1 > cum2:
            # c1 outperformed → long c2 (laggard), short c1
            long_cik, short_cik = c2_cik, c1_cik
            long_ticker = str(pair['company2_ticker'])
            short_ticker = str(pair['company1_ticker'])
        elif cum2 > cum1:
            long_cik, short_cik = c1_cik, c2_cik
            long_ticker = str(pair['company1_ticker'])
            short_ticker = str(pair['company2_ticker'])
        else:
            continue

        # Compute monthly pair returns over trading period
        pair_key = (c1_cik, c2_cik)
        pair_returns = []
        gross_cum = 0.0

        for t in range(n_trd):
            m = trd_months[t]
            long_ret = return_lookup.get((long_cik, m))
            short_ret = return_lookup.get((short_cik, m))

            if long_ret is not None and short_ret is not None:
                pr = long_ret[1] - short_ret[1]
                pair_returns.append((m, pr))
                gross_cum += pr
            else:
                pair_returns.append((m, None))

        pair_monthly[pair_key] = pair_returns

        trade_records.append({
            'company1_idx': long_cik,
            'company2_idx': short_cik,
            'company1_ticker': long_ticker,
            'company2_ticker': short_ticker,
            'sic2': str(pair['sic2']),
            'cosine_sim': float(pair['cosine_sim']),
            'formation_year': fy,
            'entry_month': trd_months[0],
            'exit_month': trd_months[-1],
            'entry_reason': 'signal_date',
            'exit_reason': 'end_of_period',
            'holding_months': n_trd,
            'gross_return': gross_cum,
            'spread_std': np.nan,
            'max_spread': np.nan,
        })

    # Aggregate monthly: equal-weighted average of all valid pair returns per month
    for t in range(n_trd):
        m = trd_months[t]
        valid_rets = []
        for pk, pr_list in pair_monthly.items():
            ret_val = pr_list[t][1]
            if ret_val is not None:
                valid_rets.append(ret_val)

        monthly_records.append({
            'date': m,
            'gross_return': np.mean(valid_rets) if valid_rets else 0.0,
            'n_pairs_active': len(valid_rets),
            'n_trades_opened': len(pair_monthly) if t == 0 else 0,
            'n_trades_closed': len(pair_monthly) if t == n_trd - 1 else 0,
            'turnover': 1.0 if t == 0 else (1.0 if t == n_trd - 1 else 0.0),
        })

    return monthly_records, trade_records


def simulate_conditional(selected_pairs, fy, entry_threshold, stop_loss):
    """Simulate conditional entry strategy for one formation year."""
    cal_months = get_calibration_months(fy)
    trd_months = get_trading_months(fy)
    n_trd = len(trd_months)
    min_trd = min_months_for_period(n_trd)

    trade_records = []
    pair_infos = []

    for _, pair in selected_pairs.iterrows():
        c1_idx = int(pair['company1_idx'])
        c2_idx = int(pair['company2_idx'])
        c1_cik = idx_to_cik.get(c1_idx)
        c2_cik = idx_to_cik.get(c2_idx)
        if c1_cik is None or c2_cik is None:
            continue

        c1_cal_log, _ = get_returns_for_cik(c1_cik, cal_months)
        c2_cal_log, _ = get_returns_for_cik(c2_cik, cal_months)

        cal_valid = sum(1 for a, b in zip(c1_cal_log, c2_cal_log) if a is not None and b is not None)
        if cal_valid < MIN_MONTHS_REQUIRED:
            continue

        c1_trd_log, c1_trd_simple = get_returns_for_cik(c1_cik, trd_months)
        c2_trd_log, c2_trd_simple = get_returns_for_cik(c2_cik, trd_months)

        trd_valid = sum(1 for a, b in zip(c1_trd_simple, c2_trd_simple) if a is not None and b is not None)
        if trd_valid < min_trd:
            continue

        # Calibration spread: cumulative log return difference at each month
        cum_spreads = []
        cum1, cum2 = 0.0, 0.0
        for a, b in zip(c1_cal_log, c2_cal_log):
            if a is not None and b is not None:
                cum1 += a
                cum2 += b
                cum_spreads.append(cum1 - cum2)

        if len(cum_spreads) < 2:
            continue

        spread_std = max(np.std(cum_spreads, ddof=1), SPREAD_STD_FLOOR)

        pair_infos.append({
            'c1_cik': c1_cik, 'c2_cik': c2_cik,
            'c1_ticker': str(pair['company1_ticker']),
            'c2_ticker': str(pair['company2_ticker']),
            'sic2': str(pair['sic2']),
            'cosine_sim': float(pair['cosine_sim']),
            'spread_std': spread_std,
            'c1_trd_log': c1_trd_log,
            'c2_trd_log': c2_trd_log,
        })

    # Simulate month by month
    positions = {i: None for i in range(len(pair_infos))}
    monthly_records = []

    for t in range(n_trd):
        m = trd_months[t]
        month_pair_returns = []
        n_opened = 0
        n_closed = 0

        for pi_idx, pi in enumerate(pair_infos):
            c1_lr = pi['c1_trd_log'][t]
            c2_lr = pi['c2_trd_log'][t]

            if c1_lr is None or c2_lr is None:
                if positions[pi_idx] is not None:
                    pos = positions[pi_idx]
                    trade_records.append({
                        'company1_idx': pos['long_cik'],
                        'company2_idx': pos['short_cik'],
                        'company1_ticker': pos['long_ticker'],
                        'company2_ticker': pos['short_ticker'],
                        'sic2': pi['sic2'],
                        'cosine_sim': pi['cosine_sim'],
                        'formation_year': fy,
                        'entry_month': pos['entry_month'],
                        'exit_month': m,
                        'entry_reason': 'divergence',
                        'exit_reason': 'missing_data',
                        'holding_months': pos['months_held'],
                        'gross_return': pos['cum_return'],
                        'spread_std': pi['spread_std'],
                        'max_spread': pos['max_spread'],
                    })
                    positions[pi_idx] = None
                    n_closed += 1
                continue

            # Cumulative spread from month 0 to t
            cum_spread = 0.0
            for tt in range(t + 1):
                lr1 = pi['c1_trd_log'][tt]
                lr2 = pi['c2_trd_log'][tt]
                if lr1 is not None and lr2 is not None:
                    cum_spread += (lr1 - lr2)

            pos = positions[pi_idx]

            if pos is not None:
                # Update position
                long_ret = return_lookup.get((pos['long_cik'], m), (0, 0))
                short_ret = return_lookup.get((pos['short_cik'], m), (0, 0))
                pair_ret = long_ret[1] - short_ret[1]
                pos['cum_return'] += pair_ret
                pos['months_held'] += 1
                pos['max_spread'] = max(pos['max_spread'], abs(cum_spread))
                month_pair_returns.append(pair_ret)

                # Check exit conditions
                should_exit = False
                exit_reason = None

                if pos['entry_spread_sign'] > 0 and cum_spread <= 0:
                    should_exit = True
                    exit_reason = 'convergence'
                elif pos['entry_spread_sign'] < 0 and cum_spread >= 0:
                    should_exit = True
                    exit_reason = 'convergence'

                if abs(cum_spread) > stop_loss * pi['spread_std']:
                    should_exit = True
                    exit_reason = 'stop_loss'

                if t == n_trd - 1:
                    should_exit = True
                    exit_reason = 'end_of_period'

                if should_exit:
                    trade_records.append({
                        'company1_idx': pos['long_cik'],
                        'company2_idx': pos['short_cik'],
                        'company1_ticker': pos['long_ticker'],
                        'company2_ticker': pos['short_ticker'],
                        'sic2': pi['sic2'],
                        'cosine_sim': pi['cosine_sim'],
                        'formation_year': fy,
                        'entry_month': pos['entry_month'],
                        'exit_month': m,
                        'entry_reason': 'divergence',
                        'exit_reason': exit_reason,
                        'holding_months': pos['months_held'],
                        'gross_return': pos['cum_return'],
                        'spread_std': pi['spread_std'],
                        'max_spread': pos['max_spread'],
                    })
                    positions[pi_idx] = None
                    n_closed += 1

            else:
                # No position — check entry (don't enter on last month)
                if t < n_trd - 1 and abs(cum_spread) > entry_threshold * pi['spread_std']:
                    if cum_spread > 0:
                        long_cik = pi['c2_cik']
                        short_cik = pi['c1_cik']
                        long_ticker = pi['c2_ticker']
                        short_ticker = pi['c1_ticker']
                    else:
                        long_cik = pi['c1_cik']
                        short_cik = pi['c2_cik']
                        long_ticker = pi['c1_ticker']
                        short_ticker = pi['c2_ticker']

                    long_ret = return_lookup.get((long_cik, m), (0, 0))
                    short_ret = return_lookup.get((short_cik, m), (0, 0))
                    pair_ret = long_ret[1] - short_ret[1]

                    positions[pi_idx] = {
                        'long_cik': long_cik,
                        'short_cik': short_cik,
                        'long_ticker': long_ticker,
                        'short_ticker': short_ticker,
                        'entry_month': m,
                        'entry_spread_sign': 1 if cum_spread > 0 else -1,
                        'cum_return': pair_ret,
                        'months_held': 1,
                        'max_spread': abs(cum_spread),
                    }
                    month_pair_returns.append(pair_ret)
                    n_opened += 1

        monthly_records.append({
            'date': m,
            'gross_return': np.mean(month_pair_returns) if month_pair_returns else 0.0,
            'n_pairs_active': len(month_pair_returns),
            'n_trades_opened': n_opened,
            'n_trades_closed': n_closed,
            'turnover': (n_opened + n_closed) / max(len(pair_infos), 1),
        })

    # Force-close any remaining positions
    for pi_idx, pos in positions.items():
        if pos is not None:
            pi = pair_infos[pi_idx]
            trade_records.append({
                'company1_idx': pos['long_cik'],
                'company2_idx': pos['short_cik'],
                'company1_ticker': pos['long_ticker'],
                'company2_ticker': pos['short_ticker'],
                'sic2': pi['sic2'],
                'cosine_sim': pi['cosine_sim'],
                'formation_year': fy,
                'entry_month': pos['entry_month'],
                'exit_month': trd_months[-1],
                'entry_reason': 'divergence',
                'exit_reason': 'end_of_period',
                'holding_months': pos['months_held'],
                'gross_return': pos['cum_return'],
                'spread_std': pi['spread_std'],
                'max_spread': pos['max_spread'],
            })

    return monthly_records, trade_records


def run_variant(variant, pairs_df, rng=None):
    """Run a full strategy variant across all formation years."""
    all_monthly = []
    all_trades = []

    for fy in FORMATION_YEARS:
        selected = select_pairs(pairs_df, fy, variant['K'], rng=rng)
        if len(selected) == 0:
            continue

        if variant['strategy'] == 'committed':
            monthly, trades = simulate_committed(selected, fy)
        else:
            monthly, trades = simulate_conditional(
                selected, fy,
                variant['entry_threshold'], variant['stop_loss']
            )

        for rec in monthly:
            rec['variant'] = variant['name']
        for rec in trades:
            rec['variant'] = variant['name']

        all_monthly.extend(monthly)
        all_trades.extend(trades)

    return all_monthly, all_trades


# Run all non-placebo variants
all_strategy_monthly = []
all_trade_logs = []

for vi, variant in enumerate(variants):
    t_start = time.time()
    monthly, trades = run_variant(variant, pairs_df)
    all_strategy_monthly.extend(monthly)
    all_trade_logs.extend(trades)
    elapsed = time.time() - t_start
    log(f"  [{vi+1}/{len(variants)}] {variant['name']}: {len(trades)} trades, "
        f"{len(monthly)} months, {elapsed:.1f}s")

log(f"  Total: {len(all_strategy_monthly):,} monthly records, {len(all_trade_logs):,} trade records")


# ═══════════════════════════════════════════════════════════
# STEP 5: Placebo (100 random trials)
# ═══════════════════════════════════════════════════════════

log("Step 5/7: Running placebo simulations (100 trials x 2 strategies)...")

rng = np.random.default_rng(SEED)

placebo_results = {}

for placebo_type in ['committed', 'conditional']:
    if placebo_type == 'committed':
        pname = f'placebo_committed_K{PLACEBO_K}'
        pvariant = {
            'name': pname,
            'strategy': 'committed',
            'K': PLACEBO_K,
            'entry_threshold': None,
            'stop_loss': None,
            'is_placebo': True,
        }
    else:
        pname = f'placebo_conditional_K{PLACEBO_K}_{PLACEBO_ENTRY}sig_{PLACEBO_STOP}stop'
        pvariant = {
            'name': pname,
            'strategy': 'conditional',
            'K': PLACEBO_K,
            'entry_threshold': PLACEBO_ENTRY,
            'stop_loss': PLACEBO_STOP,
            'is_placebo': True,
        }

    trial_sharpes = []
    trial_mean_returns = []
    all_trial_monthly_dfs = []

    for trial in range(N_PLACEBO_TRIALS):
        trial_variant = dict(pvariant)
        trial_variant['name'] = f'{pname}_trial{trial}'
        monthly, _ = run_variant(trial_variant, pairs_df, rng=rng)

        if not monthly:
            trial_sharpes.append(0.0)
            trial_mean_returns.append(0.0)
            continue

        rets = [r['gross_return'] for r in monthly]
        mean_r = np.mean(rets)
        std_r = np.std(rets, ddof=1) if len(rets) > 1 else 1e-10
        sharpe = (mean_r / std_r) * np.sqrt(12) if std_r > 1e-10 else 0.0

        trial_sharpes.append(sharpe)
        trial_mean_returns.append(mean_r)

        trial_df = pd.DataFrame(monthly)[['date', 'gross_return']]
        trial_df = trial_df.rename(columns={'gross_return': f'ret_{trial}'})
        all_trial_monthly_dfs.append(trial_df)

        if (trial + 1) % 25 == 0:
            log(f"    {pname}: {trial+1}/{N_PLACEBO_TRIALS} trials done")

    # Compute mean placebo return series
    if all_trial_monthly_dfs:
        merged = all_trial_monthly_dfs[0]
        for df in all_trial_monthly_dfs[1:]:
            merged = pd.merge(merged, df, on='date', how='outer')
        ret_cols = [c for c in merged.columns if c.startswith('ret_')]
        merged['mean_gross_return'] = merged[ret_cols].mean(axis=1)

        for _, row in merged.iterrows():
            all_strategy_monthly.append({
                'date': int(row['date']),
                'variant': pname,
                'gross_return': row['mean_gross_return'],
                'n_pairs_active': 0,
                'n_trades_opened': 0,
                'n_trades_closed': 0,
                'turnover': 0.0,
            })

    placebo_results[pname] = {
        'n_trials': N_PLACEBO_TRIALS,
        'mean_monthly_return': float(np.mean(trial_mean_returns)),
        'std_monthly_return': float(np.std(trial_mean_returns)),
        'sharpe_ratios': [float(s) for s in trial_sharpes],
        'mean_sharpe': float(np.mean(trial_sharpes)),
        'std_sharpe': float(np.std(trial_sharpes)),
        'p5_sharpe': float(np.percentile(trial_sharpes, 5)),
        'p95_sharpe': float(np.percentile(trial_sharpes, 95)),
    }
    log(f"  {pname}: mean Sharpe = {placebo_results[pname]['mean_sharpe']:.3f} "
        f"(p5={placebo_results[pname]['p5_sharpe']:.3f}, p95={placebo_results[pname]['p95_sharpe']:.3f})")


# ═══════════════════════════════════════════════════════════
# STEP 6: Save outputs
# ═══════════════════════════════════════════════════════════

log("Step 6/7: Saving output files...")

# Strategy returns
strategy_df = pd.DataFrame(all_strategy_monthly)
strategy_df = strategy_df.sort_values(['variant', 'date']).reset_index(drop=True)
strategy_df.to_parquet(STRATEGY_RETURNS_OUT, index=False)
log(f"  Strategy returns: {len(strategy_df):,} rows -> {STRATEGY_RETURNS_OUT}")

# Trade log
trade_df = pd.DataFrame(all_trade_logs)
if len(trade_df) > 0:
    trade_df = trade_df.sort_values(['variant', 'formation_year', 'entry_month']).reset_index(drop=True)
trade_df.to_parquet(TRADE_LOG_OUT, index=False)
log(f"  Trade log: {len(trade_df):,} rows -> {TRADE_LOG_OUT}")

# Placebo JSON
with open(PLACEBO_OUT, 'w') as f:
    json.dump(placebo_results, f, indent=2)
log(f"  Placebo results -> {PLACEBO_OUT}")

# Summary JSON
log("  Computing summary statistics...")

summary = {}

for vname in strategy_df['variant'].unique():
    vdf = strategy_df[strategy_df['variant'] == vname]
    rets = vdf['gross_return'].values

    mean_r = float(np.mean(rets))
    std_r = float(np.std(rets, ddof=1)) if len(rets) > 1 else 0.0
    sharpe = (mean_r / std_r) * np.sqrt(12) if std_r > 1e-10 else 0.0

    # Max drawdown on cumulative return series
    cum = np.cumsum(rets)
    running_max = np.maximum.accumulate(cum)
    drawdown = running_max - cum
    max_dd = float(np.max(drawdown)) if len(drawdown) > 0 else 0.0

    # Trade stats
    vtrades = trade_df[trade_df['variant'] == vname] if len(trade_df) > 0 else pd.DataFrame()
    n_trades = len(vtrades)
    hit_rate = float((vtrades['gross_return'] > 0).mean()) if n_trades > 0 else 0.0
    mean_holding = float(vtrades['holding_months'].mean()) if n_trades > 0 else 0.0

    # Cost-adjusted Sharpes
    cost_sharpes = {}
    for bps in COST_LEVELS_BPS:
        one_way = bps / 10000.0
        round_trip = 4 * one_way
        monthly_amortized = round_trip / 12.0
        monthly_short_cost = SHORT_SELL_COST_ANNUAL / 12.0
        total_monthly_cost = monthly_amortized + monthly_short_cost
        net_mean = mean_r - total_monthly_cost
        net_sharpe = (net_mean / std_r) * np.sqrt(12) if std_r > 1e-10 else 0.0
        cost_sharpes[f'{bps}bps'] = round(net_sharpe, 4)

    n_years = len(FORMATION_YEARS)
    mean_trades_per_year = n_trades / n_years if n_years > 0 else 0.0

    summary[vname] = {
        'mean_monthly_return': round(mean_r, 6),
        'std_monthly_return': round(std_r, 6),
        'sharpe_ratio': round(sharpe, 4),
        'total_trades': n_trades,
        'mean_trades_per_year': round(mean_trades_per_year, 1),
        'mean_holding_months': round(mean_holding, 2),
        'mean_pairs_active': round(float(vdf['n_pairs_active'].mean()), 1),
        'hit_rate': round(hit_rate, 4),
        'max_drawdown': round(max_dd, 6),
        'best_month': round(float(np.max(rets)), 6),
        'worst_month': round(float(np.min(rets)), 6),
        'cost_adjusted_sharpe': cost_sharpes,
        'n_months': len(rets),
    }

with open(SUMMARY_OUT, 'w') as f:
    json.dump(summary, f, indent=2)
log(f"  Summary -> {SUMMARY_OUT}")


# ═══════════════════════════════════════════════════════════
# STEP 7: Verification and reporting
# ═══════════════════════════════════════════════════════════

log("Step 7/7: Verification and reporting")
print("=" * 100, flush=True)

# --- Sanity Checks ---
print("\n--- SANITY CHECKS ---", flush=True)

dates = strategy_df['date'].unique()
print(f"  Date range: {min(dates)} – {max(dates)}", flush=True)
print(f"  Unique months: {len(dates)}", flush=True)

variant_names = sorted(strategy_df['variant'].unique())
print(f"  Variants: {len(variant_names)}", flush=True)

nan_count = strategy_df['gross_return'].isna().sum()
print(f"  NaN in strategy returns: {nan_count}", flush=True)

# Committed should always have active pairs
for k in K_VALUES:
    vname = f'committed_K{k}'
    vdf = strategy_df[strategy_df['variant'] == vname]
    zero_months = (vdf['n_pairs_active'] == 0).sum()
    print(f"  {vname}: months with 0 active pairs = {zero_months} "
          f"{'OK' if zero_months == 0 else 'WARNING'}", flush=True)

# K ordering check
for strategy_type in ['committed', 'conditional']:
    if strategy_type == 'conditional':
        suffix = '_2.0sig_4.0stop'
    else:
        suffix = ''
    trades_by_k = {}
    for k in K_VALUES:
        if strategy_type == 'committed':
            vname = f'committed_K{k}'
        else:
            vname = f'conditional_K{k}{suffix}'
        vtrades = trade_df[trade_df['variant'] == vname] if len(trade_df) > 0 else pd.DataFrame()
        trades_by_k[k] = len(vtrades)
    print(f"  {strategy_type} trades by K: {trades_by_k} "
          f"{'OK' if trades_by_k.get(10, 0) <= trades_by_k.get(20, 0) <= trades_by_k.get(50, 0) else 'WARNING'}", flush=True)

# --- Main Results Table ---
print("\n" + "=" * 100, flush=True)
print(f"\n{'VARIANT':<50} {'Sharpe':>8} {'Sh@10bp':>8} {'MeanRet':>10} "
      f"{'Trades':>7} {'HitRate':>8}", flush=True)
print("-" * 100, flush=True)

for vname in sorted(summary.keys()):
    if 'placebo' in vname:
        continue
    s = summary[vname]
    print(f"{vname:<50} {s['sharpe_ratio']:>8.3f} {s['cost_adjusted_sharpe']['10bps']:>8.3f} "
          f"{s['mean_monthly_return']:>10.5f} {s['total_trades']:>7} "
          f"{s['hit_rate']:>8.3f}", flush=True)

print("-" * 100, flush=True)
for vname in sorted(summary.keys()):
    if 'placebo' not in vname:
        continue
    s = summary[vname]
    print(f"{vname:<50} {s['sharpe_ratio']:>8.3f} {s['cost_adjusted_sharpe']['10bps']:>8.3f} "
          f"{s['mean_monthly_return']:>10.5f} {s['total_trades']:>7} "
          f"{s['hit_rate']:>8.3f}", flush=True)

# --- Placebo Comparison ---
print("\n--- PLACEBO COMPARISON ---", flush=True)
for pname, pres in placebo_results.items():
    real_name = pname.replace('placebo_', '')
    if real_name in summary:
        real_sharpe = summary[real_name]['sharpe_ratio']
        print(f"  {real_name}:", flush=True)
        print(f"    Real Sharpe:    {real_sharpe:.3f}", flush=True)
        print(f"    Placebo mean:   {pres['mean_sharpe']:.3f} +/- {pres['std_sharpe']:.3f}", flush=True)
        print(f"    Placebo 95th:   {pres['p95_sharpe']:.3f}", flush=True)
        beats = real_sharpe > pres['p95_sharpe']
        print(f"    Real > p95?     {'YES' if beats else 'NO'}", flush=True)

# --- Per-Formation-Year Breakdown (committed_K20) ---
print("\n--- PER-YEAR BREAKDOWN: committed_K20 ---", flush=True)
primary_name = 'committed_K20'
primary_df = strategy_df[strategy_df['variant'] == primary_name].copy()

if len(primary_df) > 0:
    def date_to_fy(d):
        y = d // 100
        m = d % 100
        if m >= 7:
            return y - 1  # July 2000 → FY1999
        else:
            return y - 2  # Jan 2001 → FY1999

    primary_df['fy'] = primary_df['date'].apply(date_to_fy)
    print(f"  {'FY':<6} {'Months':>7} {'MeanRet':>10} {'Sharpe':>8} {'Pairs':>7}", flush=True)
    print(f"  {'-'*44}", flush=True)
    for fy in sorted(primary_df['fy'].unique()):
        fydf = primary_df[primary_df['fy'] == fy]
        r = fydf['gross_return'].values
        mr = np.mean(r)
        sr_val = np.std(r, ddof=1) if len(r) > 1 else 1e-10
        sh = (mr / sr_val) * np.sqrt(12) if sr_val > 1e-10 else 0.0
        mp = fydf['n_pairs_active'].mean()
        print(f"  {fy:<6} {len(r):>7} {mr:>10.5f} {sh:>8.3f} {mp:>7.1f}", flush=True)

    # Early vs late comparison
    early = primary_df[primary_df['fy'] <= 2008]['gross_return'].values
    late = primary_df[primary_df['fy'] >= 2009]['gross_return'].values
    if len(early) > 1 and len(late) > 1:
        early_std = np.std(early, ddof=1)
        late_std = np.std(late, ddof=1)
        sh_early = (np.mean(early) / early_std) * np.sqrt(12) if early_std > 1e-10 else 0.0
        sh_late = (np.mean(late) / late_std) * np.sqrt(12) if late_std > 1e-10 else 0.0
        print(f"\n  Early (1999-2008) Sharpe: {sh_early:.3f}", flush=True)
        print(f"  Late  (2009-2019) Sharpe: {sh_late:.3f}", flush=True)
        print(f"  Late > Early? {'YES' if sh_late > sh_early else 'NO'}", flush=True)

# --- Headline Number ---
print("\n" + "=" * 100, flush=True)
if primary_name in summary:
    s = summary[primary_name]
    print(f"\n  HEADLINE: committed_K20 gross Sharpe = {s['sharpe_ratio']:.3f}", flush=True)
    print(f"  HEADLINE: committed_K20 net@10bps Sharpe = {s['cost_adjusted_sharpe']['10bps']:.3f}", flush=True)

elapsed_total = time.time() - t0
print(f"\n  Total runtime: {elapsed_total:.1f}s", flush=True)
print("=" * 100, flush=True)

log("Done.")

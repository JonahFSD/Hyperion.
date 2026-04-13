#!/usr/bin/env python3
"""
2a_03_return_matrix.py — Return Matrix Construction

Builds a point-in-time monthly return matrix aligned to the July-to-June
trading calendar, plus downloads and aligns Fama-French 5 factors + momentum.

This script is the data foundation for all downstream strategy simulations
(2a_04, 2b_02) and statistical validations (2a_05, 2b_03).

Outputs:
  experiments/artifacts/2a_03_returns.parquet
  experiments/artifacts/2a_03_factors.parquet
  experiments/artifacts/2a_03_summary.json

Cache files (avoid re-downloading):
  experiments/artifacts/ff5_monthly.csv
  experiments/artifacts/mom_monthly.csv
"""

import os
import gc
import io
import json
import zipfile
import urllib.request
import numpy as np
import pandas as pd
from datasets import load_dataset

# ═══════════════════════════════════════════════════════════
# CONFIG
# ═══════════════════════════════════════════════════════════

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ARTIFACTS_DIR = os.path.join(BASE_DIR, "artifacts")

RETURNS_OUT = os.path.join(ARTIFACTS_DIR, "2a_03_returns.parquet")
FACTORS_OUT = os.path.join(ARTIFACTS_DIR, "2a_03_factors.parquet")
SUMMARY_OUT = os.path.join(ARTIFACTS_DIR, "2a_03_summary.json")
FF5_CACHE = os.path.join(ARTIFACTS_DIR, "ff5_monthly.csv")
MOM_CACHE = os.path.join(ARTIFACTS_DIR, "mom_monthly.csv")

HF_DATASET = "Mateusz1017/annual_reports_tokenized_llama3_logged_returns_no_null_returns_and_incomplete_descriptions_24k"

FF5_URL = "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/F-F_Research_Data_5_Factors_2x3_CSV.zip"
MOM_URL = "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/F-F_Momentum_Factor_CSV.zip"


def log(msg):
    print(f"[2a_03] {msg}", flush=True)


# ═══════════════════════════════════════════════════════════
# STEP 1: Load company metadata + returns from HuggingFace
# ═══════════════════════════════════════════════════════════

log("Step 1/7: Loading company dataset from HuggingFace...")

ds = load_dataset(HF_DATASET, split="train")
log(f"  Loaded {len(ds)} rows")

# ═══════════════════════════════════════════════════════════
# STEP 2: Build return matrix
# ═══════════════════════════════════════════════════════════

log("Step 2/7: Building monthly return matrix...")

rows = []
n_nan_returns = 0
n_total_months = 0

for i in range(len(ds)):
    row = ds[i]
    company_idx = row['__index_level_0__']
    ticker = row['ticker']
    cik = row['cik']
    sic_code = row['sic_code']
    filing_year = row['year']
    log_returns = row['logged_monthly_returns_matrix']

    if log_returns is None:
        continue

    # Handle nested list format
    if isinstance(log_returns, list) and len(log_returns) > 0 and isinstance(log_returns[0], list):
        log_returns = log_returns[0]

    if len(log_returns) != 12:
        continue

    for m in range(12):
        lr = log_returns[m]
        n_total_months += 1
        calendar_month = int(filing_year) * 100 + (m + 1)  # Jan=1, ..., Dec=12

        if lr is None or (isinstance(lr, float) and np.isnan(lr)):
            n_nan_returns += 1
            rows.append({
                'company_idx': int(company_idx),
                'ticker': str(ticker),
                'cik': int(cik),
                'sic_code': str(sic_code),
                'filing_year': int(filing_year),
                'calendar_month': calendar_month,
                'log_return': np.nan,
                'simple_return': np.nan,
            })
        else:
            lr_float = float(lr)
            rows.append({
                'company_idx': int(company_idx),
                'ticker': str(ticker),
                'cik': int(cik),
                'sic_code': str(sic_code),
                'filing_year': int(filing_year),
                'calendar_month': calendar_month,
                'log_return': lr_float,
                'simple_return': np.exp(lr_float) - 1.0,
            })

    if (i + 1) % 5000 == 0:
        log(f"  Processed {i+1}/{len(ds)} companies...")

# Free HuggingFace dataset
del ds
gc.collect()

returns_df = pd.DataFrame(rows)
del rows
gc.collect()

log(f"  Return matrix: {len(returns_df)} rows")
log(f"  NaN returns: {n_nan_returns} / {n_total_months} ({100*n_nan_returns/n_total_months:.2f}%)")

# ═══════════════════════════════════════════════════════════
# STEP 3: Save returns parquet
# ═══════════════════════════════════════════════════════════

log("Step 3/7: Saving returns parquet...")
returns_df.to_parquet(RETURNS_OUT, index=False)
log(f"  Saved to {RETURNS_OUT}")


# ═══════════════════════════════════════════════════════════
# STEP 4: Download FF5 factors (or load from cache)
# ═══════════════════════════════════════════════════════════

def parse_french_zip(url, cache_path, columns, n_value_cols):
    """Download and parse a Fama-French zip file, or load from cache.

    Reuses the proven parsing pattern from 1b_factor_adjustment.py:
    look for 6-digit YYYYMM dates, stop at blank line or 4-digit annual dates.
    Values are in PERCENT — divide by 100.
    """
    if os.path.exists(cache_path):
        log(f"  Using cached file: {cache_path}")
        df = pd.read_csv(cache_path)
        df['date'] = df['date'].astype(int)
        return df

    log(f"  Downloading from: {url}")
    response = urllib.request.urlopen(url)
    zip_data = response.read()
    with zipfile.ZipFile(io.BytesIO(zip_data)) as zf:
        csv_name = [n for n in zf.namelist() if n.lower().endswith('.csv')][0]
        raw_text = zf.read(csv_name).decode('utf-8')

    # Parse: skip header lines, find the monthly data section
    lines = raw_text.split('\n')
    data_lines = []
    header_found = False
    for line in lines:
        stripped = line.strip()
        if not stripped:
            if header_found:
                break  # End of monthly section
            continue
        parts = stripped.split(',')
        first = parts[0].strip()
        if first.isdigit() and len(first) == 6:
            header_found = True
            data_lines.append(stripped)
        elif header_found and first.isdigit() and len(first) == 4:
            break  # Annual data starts

    # Build DataFrame
    rows = []
    for line in data_lines:
        parts = [p.strip() for p in line.split(',')]
        if len(parts) >= 1 + n_value_cols:
            try:
                date = int(parts[0])
                vals = [float(p) for p in parts[1:1 + n_value_cols]]
                rows.append([date] + vals)
            except ValueError:
                continue

    df = pd.DataFrame(rows, columns=columns)
    # Percent -> decimal for all value columns
    for col in columns[1:]:
        df[col] = df[col] / 100.0

    df.to_csv(cache_path, index=False)
    log(f"  Cached to: {cache_path}")
    return df


log("Step 4/7: Loading FF5 monthly factors...")
ff5 = parse_french_zip(
    url=FF5_URL,
    cache_path=FF5_CACHE,
    columns=['date', 'Mkt-RF', 'SMB', 'HML', 'RMW', 'CMA', 'RF'],
    n_value_cols=6,
)
log(f"  FF5 months: {len(ff5)}, range: {ff5['date'].min()}–{ff5['date'].max()}")

# ═══════════════════════════════════════════════════════════
# STEP 5: Download momentum factor (or load from cache)
# ═══════════════════════════════════════════════════════════

log("Step 5/7: Loading momentum factor...")
mom = parse_french_zip(
    url=MOM_URL,
    cache_path=MOM_CACHE,
    columns=['date', 'MOM'],
    n_value_cols=1,
)
log(f"  Momentum months: {len(mom)}, range: {mom['date'].min()}–{mom['date'].max()}")

# ═══════════════════════════════════════════════════════════
# STEP 6: Merge FF5 + Momentum, save factors parquet
# ═══════════════════════════════════════════════════════════

log("Step 6/7: Merging FF5 + Momentum on date (YYYYMM)...")

ff5['date'] = ff5['date'].astype(int)
mom['date'] = mom['date'].astype(int)

factors = pd.merge(ff5, mom, on='date', how='inner')
log(f"  Merged factors: {len(factors)} months, range: {factors['date'].min()}–{factors['date'].max()}")
log(f"  Columns: {list(factors.columns)}")

factors.to_parquet(FACTORS_OUT, index=False)
log(f"  Saved to {FACTORS_OUT}")

# ═══════════════════════════════════════════════════════════
# STEP 7: Verification and diagnostics
# ═══════════════════════════════════════════════════════════

log("Step 7/7: Running verification checks...")
print("=" * 80, flush=True)

# --- Return Matrix Checks ---
print("\n--- RETURN MATRIX CHECKS ---", flush=True)

# 1. Row count
n_rows = len(returns_df)
n_companies = returns_df['company_idx'].nunique()
n_nan = returns_df['log_return'].isna().sum()
print(f"  Total rows:       {n_rows:,}", flush=True)
print(f"  Unique companies: {n_companies:,}", flush=True)
print(f"  NaN returns:      {n_nan:,} ({100*n_nan/n_rows:.2f}%)", flush=True)

# 2. Year coverage
min_month = returns_df['calendar_month'].min()
max_month = returns_df['calendar_month'].max()
print(f"  Calendar month range: {min_month} – {max_month}", flush=True)

# 3. Filing year coverage
years = sorted(returns_df['filing_year'].unique())
print(f"  Filing years: {years[0]}–{years[-1]} ({len(years)} years)", flush=True)

# 4. NaN pattern by filing year
nan_by_year = returns_df.groupby('filing_year')['log_return'].apply(
    lambda x: x.isna().sum()
).reset_index(name='nan_count')
total_by_year = returns_df.groupby('filing_year').size().reset_index(name='total')
nan_summary = pd.merge(nan_by_year, total_by_year, on='filing_year')
nan_summary['pct'] = 100 * nan_summary['nan_count'] / nan_summary['total']
print("\n  NaN count by filing year:", flush=True)
for _, r in nan_summary.iterrows():
    print(f"    {int(r['filing_year'])}: {int(r['nan_count']):>5} / {int(r['total']):>5} "
          f"({r['pct']:.1f}%)", flush=True)

# 5. Return distribution by year
print("\n  Simple return distribution by year (non-NaN):", flush=True)
valid_returns = returns_df.dropna(subset=['simple_return'])
ret_stats = valid_returns.groupby('filing_year')['simple_return'].agg(['mean', 'std', 'count'])
print(f"  {'Year':<6} {'Mean':>8} {'Std':>8} {'Count':>8}", flush=True)
for yr, row in ret_stats.iterrows():
    print(f"  {int(yr):<6} {row['mean']:>8.4f} {row['std']:>8.4f} {int(row['count']):>8}", flush=True)

# 6. Spot-check AAPL
aapl = returns_df[returns_df['ticker'] == 'AAPL'].sort_values(['filing_year', 'calendar_month'])
if len(aapl) > 0:
    print(f"\n  Spot-check AAPL ({len(aapl)} rows):", flush=True)
    aapl_sample = aapl[aapl['filing_year'] == 2015]
    if len(aapl_sample) > 0:
        print(f"  AAPL filing_year=2015:", flush=True)
        for _, r in aapl_sample.iterrows():
            print(f"    {int(r['calendar_month'])}  log={r['log_return']:.6f}  "
                  f"simple={r['simple_return']:.6f}", flush=True)
    else:
        # Show whatever year is available
        sample_yr = aapl['filing_year'].iloc[len(aapl) // 2]
        aapl_sample = aapl[aapl['filing_year'] == sample_yr]
        print(f"  AAPL filing_year={int(sample_yr)}:", flush=True)
        for _, r in aapl_sample.iterrows():
            print(f"    {int(r['calendar_month'])}  log={r['log_return']:.6f}  "
                  f"simple={r['simple_return']:.6f}", flush=True)
else:
    print("\n  WARNING: AAPL not found in dataset", flush=True)

# --- Factor Matrix Checks ---
print("\n--- FACTOR MATRIX CHECKS ---", flush=True)

# 1. Date coverage
print(f"  Factor date range: {factors['date'].min()} – {factors['date'].max()}", flush=True)
print(f"  Factor columns:    {list(factors.columns)}", flush=True)
print(f"  Factor rows:       {len(factors)}", flush=True)

# 2. Spot-check Oct 2008 (GFC)
oct_2008 = factors[factors['date'] == 200810]
if len(oct_2008) > 0:
    mkt_rf = oct_2008['Mkt-RF'].iloc[0]
    print(f"\n  Oct 2008 (GFC) Mkt-RF: {mkt_rf:.6f} (expected ≈ -0.17)", flush=True)
    for col in factors.columns:
        if col != 'date':
            print(f"    {col}: {oct_2008[col].iloc[0]:.6f}", flush=True)
else:
    print("\n  WARNING: Oct 2008 not found in factors", flush=True)

# 3. No gaps in trading range 199301–202112
trading_start = 199301
trading_end = 202112
factor_dates = set(factors['date'].values)
expected_dates = []
for y in range(1993, 2022):
    for m in range(1, 13):
        ym = y * 100 + m
        if trading_start <= ym <= trading_end:
            expected_dates.append(ym)
missing_dates = [d for d in expected_dates if d not in factor_dates]
print(f"\n  Trading range {trading_start}–{trading_end}: "
      f"{len(expected_dates)} expected months", flush=True)
if missing_dates:
    print(f"  WARNING: {len(missing_dates)} missing factor months: "
          f"{missing_dates[:10]}...", flush=True)
else:
    print(f"  All {len(expected_dates)} months covered — no gaps", flush=True)

# 4. NaN check in factors for trading range
trading_factors = factors[(factors['date'] >= trading_start) & (factors['date'] <= trading_end)]
factor_nans = trading_factors.drop(columns=['date']).isna().sum()
if factor_nans.sum() > 0:
    print(f"\n  WARNING: NaN values in factors for trading range:", flush=True)
    for col, cnt in factor_nans.items():
        if cnt > 0:
            print(f"    {col}: {cnt} NaN", flush=True)
else:
    print(f"  No NaN values in any factor column for trading range", flush=True)

# --- Cross-Checks ---
print("\n--- CROSS-CHECKS ---", flush=True)

# YYYYMM alignment test: verify 12 contiguous months for a known company-year
if len(aapl) > 0:
    test_yr = 2015 if 2015 in aapl['filing_year'].values else int(aapl['filing_year'].iloc[0])
    test_rows = returns_df[
        (returns_df['ticker'] == 'AAPL') & (returns_df['filing_year'] == test_yr)
    ].sort_values('calendar_month')
    expected_months = [test_yr * 100 + m for m in range(1, 13)]
    actual_months = list(test_rows['calendar_month'].values)
    months_match = actual_months == expected_months
    print(f"  AAPL {test_yr} YYYYMM alignment: {'PASS' if months_match else 'FAIL'}", flush=True)
    print(f"    Expected: {expected_months}", flush=True)
    print(f"    Actual:   {actual_months}", flush=True)

print("\n" + "=" * 80, flush=True)

# ═══════════════════════════════════════════════════════════
# Summary JSON
# ═══════════════════════════════════════════════════════════

summary = {
    "script": "2a_03_return_matrix.py",
    "returns": {
        "total_rows": int(n_rows),
        "unique_companies": int(n_companies),
        "nan_returns": int(n_nan),
        "nan_pct": round(100 * n_nan / n_rows, 2),
        "calendar_month_range": [int(min_month), int(max_month)],
        "filing_year_range": [int(years[0]), int(years[-1])],
        "n_filing_years": len(years),
    },
    "factors": {
        "total_months": int(len(factors)),
        "date_range": [int(factors['date'].min()), int(factors['date'].max())],
        "columns": list(factors.columns),
        "trading_range_gaps": len(missing_dates),
        "trading_range_nans": int(factor_nans.sum()),
        "oct_2008_mkt_rf": float(oct_2008['Mkt-RF'].iloc[0]) if len(oct_2008) > 0 else None,
    },
    "verification": {
        "aapl_alignment_pass": bool(months_match) if len(aapl) > 0 else None,
        "return_mean_annual": {
            int(yr): round(float(row['mean']), 6) for yr, row in ret_stats.iterrows()
        },
        "return_std_annual": {
            int(yr): round(float(row['std']), 6) for yr, row in ret_stats.iterrows()
        },
    },
    "files_written": [RETURNS_OUT, FACTORS_OUT],
}

with open(SUMMARY_OUT, 'w') as f:
    json.dump(summary, f, indent=2)

log(f"Summary written to {SUMMARY_OUT}")
log("Done.")

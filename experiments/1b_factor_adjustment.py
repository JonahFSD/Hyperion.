#!/usr/bin/env python3
"""
Experiment 1B — Fama-French 5-Factor Adjustment

Does SAE's within-SIC retrieval signal survive after removing shared factor exposure?
If SAE peers just have similar betas/size/value loadings, the signal is redundant
with a free factor model.

Approach:
  1. Download FF5 monthly factors
  2. For each company, pool all monthly returns across years and regress on FF5
  3. Extract residuals (company-specific returns after removing factor exposure)
  4. Recompute T04 within-SIC precision@K using residual correlations
  5. Recompute T05 topology-vs-magnitude using residual correlations
  6. Compare raw vs factor-adjusted signal
"""

import numpy as np
import pandas as pd
import json
import gc
import os
import io
import zipfile
import urllib.request
import pyarrow.parquet as pq
from collections import defaultdict
from pathlib import Path

np.random.seed(42)

DATA_DIR = Path(__file__).resolve().parent / "phase1_artifacts"
PAIRS_FILE = DATA_DIR / "pairs.parquet"
COMPANIES_FILE = DATA_DIR / "companies.parquet"
OUTPUT_FILE = DATA_DIR / "1b_factor_adjustment_result.json"
FF5_CACHE = DATA_DIR / "ff5_factors.csv"

print("=" * 80)
print("Experiment 1B: Fama-French 5-Factor Adjustment")
print("=" * 80)

# ============================================================
# Step 0: Download FF5 monthly factors
# ============================================================
print("\n[1/7] Downloading FF5 monthly factors...")

if FF5_CACHE.exists():
    print(f"  Using cached file: {FF5_CACHE}")
    ff5 = pd.read_csv(FF5_CACHE)
else:
    url = "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/F-F_Research_Data_5_Factors_2x3_CSV.zip"
    print(f"  Fetching from: {url}")
    response = urllib.request.urlopen(url)
    zip_data = response.read()
    with zipfile.ZipFile(io.BytesIO(zip_data)) as zf:
        csv_name = [n for n in zf.namelist() if n.endswith('.CSV') or n.endswith('.csv')][0]
        raw_text = zf.read(csv_name).decode('utf-8')

    # Parse: skip header lines, find the monthly data section
    lines = raw_text.split('\n')
    data_lines = []
    header_found = False
    for line in lines:
        stripped = line.strip()
        if not stripped:
            if header_found:
                break  # End of monthly section (blank line after data = annual section follows)
            continue
        # Check if line starts with a 6-digit number (YYYYMM)
        parts = stripped.split(',')
        first = parts[0].strip()
        if first.isdigit() and len(first) == 6:
            header_found = True
            data_lines.append(stripped)
        elif header_found and first.isdigit() and len(first) == 4:
            # Annual data starts — stop
            break

    # Build DataFrame
    rows = []
    for line in data_lines:
        parts = [p.strip() for p in line.split(',')]
        if len(parts) >= 7:
            try:
                date = int(parts[0])
                vals = [float(p) for p in parts[1:7]]
                rows.append([date] + vals)
            except ValueError:
                continue

    ff5 = pd.DataFrame(rows, columns=['date', 'Mkt-RF', 'SMB', 'HML', 'RMW', 'CMA', 'RF'])
    # Divide by 100 (percentage -> decimal)
    for col in ['Mkt-RF', 'SMB', 'HML', 'RMW', 'CMA', 'RF']:
        ff5[col] = ff5[col] / 100.0

    ff5.to_csv(FF5_CACHE, index=False)
    print(f"  Cached to: {FF5_CACHE}")

ff5['date'] = ff5['date'].astype(int)
ff5_dict = {int(row['date']): row for _, row in ff5.iterrows()}

print(f"  FF5 months: {len(ff5)}")
print(f"  Date range: {ff5['date'].min()}-{ff5['date'].max()}")

ff5_date_range = f"{ff5['date'].min()}-{ff5['date'].max()}"
ff5_n_months = len(ff5)

gc.collect()

# ============================================================
# Step 1: Load companies.parquet
# ============================================================
print("\n[2/7] Loading companies.parquet...")

comps = pd.read_parquet(COMPANIES_FILE,
                        columns=['__index_level_0__', 'cik', 'year', 'sic_code',
                                 'logged_monthly_returns_matrix'])

print(f"  Rows: {len(comps)}")
print(f"  Years: {sorted(comps['year'].unique())[:5]}...{sorted(comps['year'].unique())[-3:]}")

# Build SIC mapping (fresh from parquet — NOT from /tmp/)
sic_comps = pd.read_parquet(COMPANIES_FILE, columns=['__index_level_0__', 'sic_code'])
idx_to_sic2 = dict(zip(sic_comps['__index_level_0__'],
                        sic_comps['sic_code'].astype(str).str[:2]))
del sic_comps
print(f"  SIC2 mapping: {len(idx_to_sic2)} entries")

# Verify same cik across years shares identity
print("\n  Verifying cik consistency across years:")
sample_cik = comps.groupby('cik').size().sort_values(ascending=False).index[0]
sample_rows = comps[comps['cik'] == sample_cik][['__index_level_0__', 'cik', 'year']].head(3)
print(f"    CIK {sample_cik} appears in years: {list(sample_rows['year'])}")
print(f"    __index_level_0__ values: {list(sample_rows['__index_level_0__'])}")
print(f"    (Different __index_level_0__ per year-row — CIK is the stable identity)")

# ============================================================
# Step 2: Pool across years, regress on FF5
# ============================================================
print("\n[3/7] Running FF5 regressions (pooled by company)...")

# Group rows by cik for regression
cik_groups = defaultdict(list)
for _, row in comps.iterrows():
    cik_groups[row['cik']].append(row)

n_companies_regressed = 0
n_companies_skipped = 0
obs_counts = []
r_squared_list = []

# Residuals keyed by (__index_level_0__, year) for pairs.parquet lookups
residuals_dict = {}

total_ciks = len(cik_groups)
progress_step = max(1, total_ciks // 10)

for i, (cik, rows_list) in enumerate(cik_groups.items()):
    if (i + 1) % progress_step == 0:
        print(f"  Processing company {i+1}/{total_ciks}...")

    # Stack all monthly returns across all years for this company
    monthly_data = []  # list of (yyyymm, simple_return, idx, year, month_pos)
    row_info = []  # (idx, year, n_months) for slicing residuals back

    for row in rows_list:
        idx = int(row['__index_level_0__'])
        year_int = int(row['year'])
        log_returns = row['logged_monthly_returns_matrix']

        if log_returns is None or len(log_returns) != 12:
            continue

        month_entries = []
        for m in range(12):
            lr = log_returns[m]
            if np.isnan(lr):
                month_entries.append(None)
                continue
            # Convert log return to simple return
            simple_ret = np.exp(lr) - 1.0
            yyyymm = year_int * 100 + (m + 1)  # Jan=1, ..., Dec=12
            month_entries.append((yyyymm, simple_ret))
            monthly_data.append((yyyymm, simple_ret, idx, year_int, m))

        row_info.append((idx, year_int, month_entries))

    # Filter to months with FF5 data
    valid_data = []
    for yyyymm, simple_ret, idx, year_int, month_pos in monthly_data:
        if yyyymm in ff5_dict:
            ff_row = ff5_dict[yyyymm]
            valid_data.append((yyyymm, simple_ret, idx, year_int, month_pos,
                               float(ff_row['Mkt-RF']), float(ff_row['SMB']),
                               float(ff_row['HML']), float(ff_row['RMW']),
                               float(ff_row['CMA']), float(ff_row['RF'])))

    if len(valid_data) < 24:
        n_companies_skipped += 1
        continue

    # Build regression arrays
    n = len(valid_data)
    y = np.empty(n)
    X = np.empty((n, 6))  # intercept + 5 factors

    for j, (yyyymm, ret, idx, yr, mp, mkt_rf, smb, hml, rmw, cma, rf) in enumerate(valid_data):
        y[j] = ret - rf  # excess return
        X[j, 0] = 1.0    # intercept (alpha)
        X[j, 1] = mkt_rf
        X[j, 2] = smb
        X[j, 3] = hml
        X[j, 4] = rmw
        X[j, 5] = cma

    # OLS via numpy lstsq
    coeffs, residual_ss, rank, sv = np.linalg.lstsq(X, y, rcond=None)
    fitted = X @ coeffs
    resids = y - fitted

    # R-squared
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    ss_res = np.sum(resids ** 2)
    r_sq = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0

    n_companies_regressed += 1
    obs_counts.append(n)
    r_squared_list.append(r_sq)

    # Map residuals back to (idx, year) -> [12 residuals]
    # Build a lookup from (idx, year, month_pos) -> residual
    resid_lookup = {}
    for j, (yyyymm, ret, idx, yr, mp, *_) in enumerate(valid_data):
        resid_lookup[(idx, yr, mp)] = resids[j]

    # Slice back into company-year groups of 12
    for row in rows_list:
        idx = int(row['__index_level_0__'])
        year_int = int(row['year'])
        year_resids = []
        for m in range(12):
            r = resid_lookup.get((idx, year_int, m), np.nan)
            year_resids.append(r)

        # Only store if we have enough valid residuals
        n_valid = sum(1 for r in year_resids if not np.isnan(r))
        if n_valid >= 6:
            residuals_dict[(idx, year_int)] = year_resids

print(f"\n  Regression stats:")
print(f"    Companies regressed: {n_companies_regressed}")
print(f"    Companies skipped (<24 obs): {n_companies_skipped}")
print(f"    Median obs per company: {int(np.median(obs_counts))}")
print(f"    Company-years with residuals: {len(residuals_dict)}")

r_sq_arr = np.array(r_squared_list)
r_sq_pctiles = {
    'p10': float(np.percentile(r_sq_arr, 10)),
    'p25': float(np.percentile(r_sq_arr, 25)),
    'p50': float(np.percentile(r_sq_arr, 50)),
    'p75': float(np.percentile(r_sq_arr, 75)),
    'p90': float(np.percentile(r_sq_arr, 90)),
}
print(f"    R² distribution: p10={r_sq_pctiles['p10']:.4f}, p25={r_sq_pctiles['p25']:.4f}, "
      f"p50={r_sq_pctiles['p50']:.4f}, p75={r_sq_pctiles['p75']:.4f}, p90={r_sq_pctiles['p90']:.4f}")

regression_stats = {
    'n_companies_regressed': n_companies_regressed,
    'n_companies_skipped': n_companies_skipped,
    'median_obs_per_company': int(np.median(obs_counts)),
    'median_r_squared': float(np.median(r_squared_list)),
    'r_squared_percentiles': r_sq_pctiles,
    'n_company_years_with_residuals': len(residuals_dict),
}

# Free memory from companies dataframe
del comps, cik_groups, monthly_data, valid_data
gc.collect()

# ============================================================
# Step 3: Recompute pairwise residual correlations
# (Also collect data for Steps 4 and 5 in same loop)
# ============================================================
print("\n[4/7] Computing residual correlations from pairs.parquet...")

pairs_file = pq.ParquetFile(PAIRS_FILE)

# Data collectors for Step 4 (T04: within-SIC precision@K)
# Key: (company1_idx, year) -> list of (comp2_idx, cosine_sim, residual_corr)
t04_groups = defaultdict(list)

# Data collectors for Step 5 (T05: topology-vs-magnitude)
# Per-year: list of (company1, company2, cosine_sim, residual_corr)
t05_per_year = defaultdict(list)

# Sanity check accumulators
raw_corrs_sum = 0.0
resid_corrs_sum = 0.0
n_both_valid = 0
n_pairs_total = 0
n_pairs_with_residuals = 0

for rg_idx in range(pairs_file.num_row_groups):
    rg_table = pairs_file.read_row_group(rg_idx)

    comp1_col = rg_table['Company1'].to_numpy()
    comp2_col = rg_table['Company2'].to_numpy()
    year_col = rg_table['year'].to_numpy()
    cosine_col = rg_table['cosine_similarity'].to_numpy()
    corr_col = rg_table['correlation'].to_numpy()

    n_rg = len(comp1_col)
    n_pairs_total += n_rg

    for j in range(n_rg):
        c1 = int(comp1_col[j])
        c2 = int(comp2_col[j])
        yr = int(year_col[j])
        cos_sim = float(cosine_col[j])
        raw_corr = float(corr_col[j]) if not np.isnan(corr_col[j]) else None

        key1 = (c1, yr)
        key2 = (c2, yr)

        resid1 = residuals_dict.get(key1)
        resid2 = residuals_dict.get(key2)

        if resid1 is None or resid2 is None:
            continue

        # Compute residual correlation (Pearson)
        r1 = np.array(resid1)
        r2 = np.array(resid2)
        valid_mask = ~np.isnan(r1) & ~np.isnan(r2)
        n_overlap = np.sum(valid_mask)

        if n_overlap < 6:
            continue

        r1_valid = r1[valid_mask]
        r2_valid = r2[valid_mask]

        # Pearson correlation
        r1_mean = np.mean(r1_valid)
        r2_mean = np.mean(r2_valid)
        r1_centered = r1_valid - r1_mean
        r2_centered = r2_valid - r2_mean
        denom = np.sqrt(np.sum(r1_centered ** 2) * np.sum(r2_centered ** 2))
        if denom < 1e-15:
            continue
        resid_corr = float(np.sum(r1_centered * r2_centered) / denom)

        n_pairs_with_residuals += 1

        # Sanity check accumulators
        resid_corrs_sum += resid_corr
        if raw_corr is not None:
            raw_corrs_sum += raw_corr
            n_both_valid += 1

        # Collect for T04 (within-SIC only)
        sic1 = idx_to_sic2.get(c1)
        sic2 = idx_to_sic2.get(c2)
        if sic1 is not None and sic1 == sic2:
            t04_groups[(c1, yr)].append((c2, cos_sim, resid_corr))

        # Collect for T05 (all pairs)
        t05_per_year[yr].append((c1, c2, cos_sim, resid_corr))

    del rg_table
    if (rg_idx + 1) % 3 == 0:
        print(f"  Row group {rg_idx+1}/{pairs_file.num_row_groups}: "
              f"{n_pairs_with_residuals:,} pairs with residuals so far")

print(f"\n  Total pairs in parquet: {n_pairs_total:,}")
print(f"  Pairs with valid residuals: {n_pairs_with_residuals:,}")

gc.collect()

# ============================================================
# Step 4: T04 within-SIC precision@K on residuals
# ============================================================
print("\n[5/7] Computing T04 within-SIC precision@K on residuals...")

# Filter to company-years with >= 5 same-SIC peers
t04_filtered = {k: v for k, v in t04_groups.items() if len(v) >= 5}
n_t04_total = len(t04_groups)
n_t04_filtered = len(t04_filtered)
print(f"  Company-years with >=5 same-SIC peers: {n_t04_filtered} (of {n_t04_total})")

t04_results_by_k = {1: [], 3: [], 5: [], 10: []}

for key, peers in t04_filtered.items():
    if len(peers) < 5:
        continue

    # Sort by cosine similarity (descending)
    peers_sorted = sorted(peers, key=lambda x: x[1], reverse=True)

    for K in [1, 3, 5, 10]:
        if K > len(peers_sorted):
            continue

        # SAE top-K: mean residual correlation of top-K by cosine
        top_k_resid_corrs = [rc for _, _, rc in peers_sorted[:K]]
        sae_mean = np.mean(top_k_resid_corrs)

        # Random-K: sample 100 random subsets of same-SIC peers
        random_means = []
        for _ in range(100):
            rand_idx = np.random.choice(len(peers_sorted), size=min(K, len(peers_sorted)), replace=False)
            rand_corrs = [peers_sorted[idx][2] for idx in rand_idx]
            random_means.append(np.mean(rand_corrs))

        random_mean = np.mean(random_means)
        lift = sae_mean - random_mean

        t04_results_by_k[K].append({
            'sae_mean': sae_mean,
            'random_mean': random_mean,
            'lift': lift,
        })

for K in [1, 3, 5, 10]:
    print(f"  K={K}: {len(t04_results_by_k[K])} samples")


def bootstrap_ci(values, n_resamples=1000, ci=0.95):
    """Compute bootstrap CI for mean of values."""
    if len(values) == 0:
        return None, [None, None]
    values = np.array(values)
    bootstrap_means = []
    for _ in range(n_resamples):
        resample = np.random.choice(values, size=len(values), replace=True)
        bootstrap_means.append(np.mean(resample))
    mean = np.mean(values)
    alpha = 1 - ci
    lower = np.percentile(bootstrap_means, 100 * alpha / 2)
    upper = np.percentile(bootstrap_means, 100 * (1 - alpha / 2))
    return mean, [lower, upper]


t04_residual_results = {}
for K in [1, 3, 5, 10]:
    if len(t04_results_by_k[K]) == 0:
        t04_residual_results[f'K_{K}'] = None
        continue
    lifts = [r['lift'] for r in t04_results_by_k[K]]
    sae_means = [r['sae_mean'] for r in t04_results_by_k[K]]
    random_means = [r['random_mean'] for r in t04_results_by_k[K]]

    lift_mean, lift_ci = bootstrap_ci(lifts, n_resamples=1000)
    sae_mean_val, sae_ci = bootstrap_ci(sae_means, n_resamples=1000)
    random_mean_val, random_ci = bootstrap_ci(random_means, n_resamples=1000)

    t04_residual_results[f'K_{K}'] = {
        'sae_mean_resid_corr': float(sae_mean_val),
        'random_sic_mean_resid_corr': float(random_mean_val),
        'residual_lift': float(lift_mean),
        'residual_ci_95': [float(lift_ci[0]), float(lift_ci[1])],
        'n_samples': len(lifts),
    }

    print(f"  K={K}: residual lift = {lift_mean:.6f} [{lift_ci[0]:.6f}, {lift_ci[1]:.6f}]")

del t04_groups, t04_filtered, t04_results_by_k
gc.collect()

# ============================================================
# Step 5: T05 topology-vs-magnitude on residuals
# ============================================================
print("\n[6/7] Computing T05 topology-vs-magnitude on residuals...")

nn_results = []
control_results = []

for yr in sorted(t05_per_year.keys()):
    pairs_yr = t05_per_year[yr]
    print(f"  Year {yr}: {len(pairs_yr)} pairs...", end='', flush=True)

    if len(pairs_yr) == 0:
        print(" skipped (no pairs)")
        continue

    # Build NN graph: for each company, find peer with highest cosine
    # Build bidirectional view
    company_best = {}  # company -> (peer, cosine, resid_corr)
    for c1, c2, cos_sim, resid_corr in pairs_yr:
        for comp, peer in [(c1, c2), (c2, c1)]:
            if comp not in company_best or cos_sim > company_best[comp][1]:
                company_best[comp] = (peer, cos_sim, resid_corr)

    # Deduplicate NN edges
    nn_edges = {}
    for comp, (peer, cos_sim, resid_corr) in company_best.items():
        edge = (min(comp, peer), max(comp, peer))
        if edge not in nn_edges or cos_sim > nn_edges[edge][0]:
            nn_edges[edge] = (cos_sim, resid_corr)

    print(f" {len(nn_edges)} NN edges", end='', flush=True)

    # Build lookup for control matching: list of (edge, cosine, resid_corr)
    # Use all pairs for this year
    all_pairs_sorted = sorted(pairs_yr, key=lambda x: x[2])  # sort by cosine for binary search
    # Actually, for tolerance matching, let's build a simple structure
    nn_edge_set = set(nn_edges.keys())

    matched = 0
    for edge, (nn_cos, nn_resid) in nn_edges.items():
        # Find a control pair with similar cosine but not an NN edge
        for tol in [0.01, 0.02, 0.05]:
            candidates = []
            for c1, c2, cos_sim, resid_corr in pairs_yr:
                ctrl_edge = (min(c1, c2), max(c1, c2))
                if ctrl_edge in nn_edge_set:
                    continue
                if abs(cos_sim - nn_cos) <= tol:
                    candidates.append(resid_corr)
                    if len(candidates) >= 20:
                        break  # enough candidates

            if len(candidates) > 0:
                ctrl_resid = candidates[np.random.randint(len(candidates))]
                nn_results.append({'year': yr, 'corr': float(nn_resid)})
                control_results.append({'year': yr, 'corr': float(ctrl_resid)})
                matched += 1
                break

    print(f" -> {matched} matched")

del t05_per_year
gc.collect()

# Compute T05 stats
nn_arr = np.array([r['corr'] for r in nn_results])
ctrl_arr = np.array([r['corr'] for r in control_results])
nn_mean = float(np.mean(nn_arr))
ctrl_mean = float(np.mean(ctrl_arr))
resid_diff = nn_mean - ctrl_mean

print(f"\n  T05 residual results:")
print(f"    NN mean residual corr:      {nn_mean:.6f}")
print(f"    Control mean residual corr: {ctrl_mean:.6f}")
print(f"    Difference:                 {resid_diff:.6f}")

# Bootstrap CI (10,000 resamples)
print("  Computing bootstrap CI (10,000 resamples)...")
boot_diffs = []
for i in range(10000):
    if (i + 1) % 2000 == 0:
        print(f"    {i+1}/10000")
    boot_nn = np.random.choice(nn_arr, len(nn_arr), replace=True).mean()
    boot_ctrl = np.random.choice(ctrl_arr, len(ctrl_arr), replace=True).mean()
    boot_diffs.append(boot_nn - boot_ctrl)

boot_diffs = np.array(boot_diffs)
t05_resid_ci = [float(np.percentile(boot_diffs, 2.5)), float(np.percentile(boot_diffs, 97.5))]
print(f"    95% CI: [{t05_resid_ci[0]:.6f}, {t05_resid_ci[1]:.6f}]")

gc.collect()

# ============================================================
# Step 6: Sanity checks
# ============================================================
print("\n[6.5/7] Sanity checks...")

if n_both_valid > 0:
    mean_raw = raw_corrs_sum / n_both_valid
    mean_resid = resid_corrs_sum / n_pairs_with_residuals
    print(f"  Mean raw correlation (across all pairs):     {mean_raw:.6f}")
    print(f"  Mean residual correlation (across all pairs): {mean_resid:.6f}")
    print(f"  Reduction: {mean_raw - mean_resid:.6f}")
    if mean_resid > mean_raw:
        print("  WARNING: Residual correlations are HIGHER than raw — something may be wrong!")
else:
    print("  No pairs with both raw and residual correlations for sanity check.")

print(f"  Pairs with valid residuals: {n_pairs_with_residuals:,} / {n_pairs_total:,} "
      f"({100*n_pairs_with_residuals/n_pairs_total:.1f}%)")

# ============================================================
# Step 7: Compare raw vs factor-adjusted
# ============================================================
print("\n[7/7] Comparing raw vs factor-adjusted results...")

# Load raw T04 results
with open(DATA_DIR / '1a_11_t04_result.json', 'r') as f:
    raw_t04 = json.load(f)

# Load raw T05 results
with open(DATA_DIR / '1a_11_t05_result.json', 'r') as f:
    raw_t05 = json.load(f)

# Build comparison
print("\n  T04 Within-SIC Precision@K: Raw vs Residual")
print(f"  {'K':<6} {'Raw Lift':>12} {'Resid Lift':>12} {'Survival':>10} {'Resid CI 95%':>24}")
print(f"  {'-'*64}")

t04_comparison = {}
for K in [1, 3, 5, 10]:
    k_key = f'K_{K}'
    raw_res = raw_t04['overall'].get(k_key)
    resid_res = t04_residual_results.get(k_key)

    if raw_res is None or resid_res is None:
        continue

    raw_lift = raw_res['lift']
    resid_lift = resid_res['residual_lift']
    survival = resid_lift / raw_lift if abs(raw_lift) > 1e-10 else float('nan')
    resid_ci = resid_res['residual_ci_95']

    t04_comparison[k_key] = {
        'raw_lift': float(raw_lift),
        'residual_lift': float(resid_lift),
        'survival_ratio': float(survival),
        'residual_ci_95': resid_ci,
    }

    print(f"  K={K:<4} {raw_lift:>12.6f} {resid_lift:>12.6f} {survival:>9.1%} "
          f"  [{resid_ci[0]:.6f}, {resid_ci[1]:.6f}]")

# T05 comparison
raw_t05_diff = raw_t05['overall']['difference']
t05_survival = resid_diff / raw_t05_diff if abs(raw_t05_diff) > 1e-10 else float('nan')

print(f"\n  T05 Topology vs Magnitude: Raw vs Residual")
print(f"    Raw diff:      {raw_t05_diff:.6f}")
print(f"    Residual diff: {resid_diff:.6f}")
print(f"    Survival:      {t05_survival:.1%}")
print(f"    Residual CI:   [{t05_resid_ci[0]:.6f}, {t05_resid_ci[1]:.6f}]")

t05_comparison = {
    'raw_diff': float(raw_t05_diff),
    'residual_diff': float(resid_diff),
    'survival_ratio': float(t05_survival),
    'residual_ci_95': t05_resid_ci,
}

# ============================================================
# Verdict
# ============================================================
k5_data = t04_comparison.get('K_5')
if k5_data is not None:
    ci_lo, ci_hi = k5_data['residual_ci_95']
    ci_excludes_zero = (ci_lo > 0) or (ci_hi < 0)
    survival = k5_data['survival_ratio']

    if ci_excludes_zero and survival > 0.30:
        verdict = "PASS"
        interpretation = (
            f"Factor-adjusted signal SURVIVES. K=5 residual lift = {k5_data['residual_lift']:.6f} "
            f"(CI [{ci_lo:.6f}, {ci_hi:.6f}] excludes zero), "
            f"survival ratio = {survival:.1%} (>{30}%). "
            f"SAE captures company-specific signal beyond FF5 factor exposure."
        )
    elif ci_excludes_zero and survival <= 0.30:
        verdict = "PARTIAL PASS"
        interpretation = (
            f"Factor-adjusted signal is statistically significant but weak. "
            f"K=5 residual lift = {k5_data['residual_lift']:.6f} "
            f"(CI [{ci_lo:.6f}, {ci_hi:.6f}] excludes zero), "
            f"but survival ratio = {survival:.1%} (<30%). "
            f"Most of SAE's signal is explained by shared factor exposure."
        )
    else:
        verdict = "FAIL"
        interpretation = (
            f"Factor-adjusted signal does NOT survive. K=5 residual lift = {k5_data['residual_lift']:.6f} "
            f"(CI [{ci_lo:.6f}, {ci_hi:.6f}] includes zero). "
            f"SAE's retrieval signal is largely redundant with FF5 factor loadings."
        )
else:
    verdict = "FAIL"
    interpretation = "K=5 results not available."

# ============================================================
# Write output
# ============================================================
result = {
    "test": "1B_factor_adjustment",
    "ff5_data": {
        "n_months": ff5_n_months,
        "date_range": ff5_date_range,
    },
    "regression_stats": regression_stats,
    "t04_comparison": t04_comparison,
    "t05_comparison": t05_comparison,
    "verdict": verdict,
    "interpretation": interpretation,
}

with open(OUTPUT_FILE, 'w') as f:
    json.dump(result, f, indent=2)

print(f"\n{'='*80}")
print(f"VERDICT: {verdict}")
print(f"{'='*80}")
print(f"\nInterpretation: {interpretation}")
print(f"\nResult written to: {OUTPUT_FILE}")

print(f"\n{'='*80}")
print("FULL JSON:")
print(f"{'='*80}")
print(json.dumps(result, indent=2))

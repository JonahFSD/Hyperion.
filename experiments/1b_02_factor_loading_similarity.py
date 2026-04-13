#!/usr/bin/env python3
"""
Test 1B-02: Within-Cluster Factor Loading Similarity

Red Flag 1 investigation: The 1B factor adjustment shows 98.4% survival
(median R²=3%). Is this because SAE clusters are genuinely orthogonal to
FF5 factor exposure, or because the regression failed to remove shared
factor structure?

Test: Compute factor loadings (β₁...β₅) per company via pooled FF5 OLS.
Then measure whether SAE cluster membership predicts shared factor loadings
using intraclass correlation (ICC) for each factor.

If ICC ≈ 0 for all factors → clusters don't share factor loadings → 98.4%
survival is the boring expected result → Flag 1 invalidated.

If ICC >> 0 for any factor → clusters group companies with similar factor
exposure → 98.4% survival is suspicious → deeper investigation needed.

Also computes: ANOVA F-statistics, within-cluster vs population β variance
ratios, and a multivariate test (do clusters predict the JOINT β vector?).

Data sources:
  - Companies: HuggingFace (Mateusz1017/annual_reports_tokenized...)
  - FF5 factors: Dartmouth (cached from 1B)
  - SAE clusters: company_similarity_sae/Clustering/data/Final Results/year_cluster_dfC-CD.pkl

Dependencies: numpy, pandas, scipy, datasets, pyarrow
"""

import numpy as np
import pandas as pd
import json
import pickle
import os
import io
import zipfile
import urllib.request
from collections import defaultdict
from pathlib import Path
from scipy import stats

np.random.seed(42)

# ============================================================
# Paths
# ============================================================
REPO_ROOT = Path(__file__).resolve().parent.parent
EXPERIMENTS_DIR = Path(__file__).resolve().parent
ARTIFACTS_DIR = EXPERIMENTS_DIR / "artifacts"
CLUSTER_FILE_SAE = REPO_ROOT / "company_similarity_sae" / "Clustering" / "data" / "Final Results" / "year_cluster_dfC-CD.pkl"
CLUSTER_FILE_SIC = REPO_ROOT / "company_similarity_sae" / "Clustering" / "data" / "cointegration" / "year_SIC_cluster_mapping.pkl"
FF5_CACHE = ARTIFACTS_DIR / "ff5_factors.csv"
OUTPUT_FILE = ARTIFACTS_DIR / "1b_02_factor_loading_similarity.json"

ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 80)
print("Test 1B-02: Within-Cluster Factor Loading Similarity")
print("=" * 80)

# ============================================================
# Step 1: Load FF5 factors
# ============================================================
print("\n[1/5] Loading FF5 factors...")

if FF5_CACHE.exists():
    print(f"  Using cached: {FF5_CACHE}")
    ff5 = pd.read_csv(FF5_CACHE)
else:
    url = "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/F-F_Research_Data_5_Factors_2x3_CSV.zip"
    print(f"  Downloading from: {url}")
    response = urllib.request.urlopen(url)
    zip_data = response.read()
    with zipfile.ZipFile(io.BytesIO(zip_data)) as zf:
        csv_name = [n for n in zf.namelist() if n.lower().endswith('.csv')][0]
        raw_text = zf.read(csv_name).decode('utf-8')

    lines = raw_text.split('\n')
    data_lines = []
    header_found = False
    for line in lines:
        stripped = line.strip()
        if not stripped:
            if header_found:
                break
            continue
        parts = stripped.split(',')
        first = parts[0].strip()
        if first.isdigit() and len(first) == 6:
            header_found = True
            data_lines.append(stripped)
        elif header_found and first.isdigit() and len(first) == 4:
            break

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
    for col in ['Mkt-RF', 'SMB', 'HML', 'RMW', 'CMA', 'RF']:
        ff5[col] = ff5[col] / 100.0
    ff5.to_csv(FF5_CACHE, index=False)

ff5['date'] = ff5['date'].astype(int)
ff5_dict = {int(row['date']): row for _, row in ff5.iterrows()}
print(f"  FF5 months: {len(ff5)} ({ff5['date'].min()}-{ff5['date'].max()})")

# ============================================================
# Step 2: Load companies from HuggingFace
# ============================================================
print("\n[2/5] Loading companies from HuggingFace (streaming to avoid OOM)...")

from datasets import load_dataset

ds = load_dataset(
    "Mateusz1017/annual_reports_tokenized_llama3_logged_returns_no_null_returns_and_incomplete_descriptions_24k",
    split="train"
)

# Extract only what we need without converting the entire dataset to pandas.
# We need: __index_level_0__, cik, year, logged_monthly_returns_matrix
idx_to_cik = {}
cik_monthly = defaultdict(list)  # cik -> list of (year, log_returns_12)

for i, row in enumerate(ds):
    idx = row['__index_level_0__']
    cik = row['cik']
    year = row['year']
    log_returns = row['logged_monthly_returns_matrix']

    idx_to_cik[idx] = cik

    if log_returns is not None and len(log_returns) == 12:
        cik_monthly[cik].append((int(year), log_returns))

    if (i + 1) % 5000 == 0:
        print(f"  Processed {i+1}/27888 rows...")

del ds
print(f"  Companies (unique CIK): {len(cik_monthly)}")
print(f"  idx_to_cik entries: {len(idx_to_cik)}")

# ============================================================
# Step 3: Run FF5 regressions, extract betas per company
# ============================================================
print("\n[3/5] Running FF5 regressions (pooled by CIK)...")

# cik_monthly already built in Step 2: cik -> list of (year, log_returns_12)
cik_betas = {}
r_squared_list = []
obs_counts = []

total_ciks = len(cik_monthly)
progress_step = max(1, total_ciks // 10)
n_skipped = 0

for i, (cik, year_returns) in enumerate(cik_monthly.items()):
    if (i + 1) % progress_step == 0:
        print(f"  Company {i+1}/{total_ciks}...")

    # Stack all monthly returns across all years
    monthly_data = []
    for year_int, log_returns in year_returns:
        for m in range(12):
            lr = log_returns[m]
            if lr is None or np.isnan(lr):
                continue
            simple_ret = np.exp(lr) - 1.0
            yyyymm = year_int * 100 + (m + 1)
            monthly_data.append((yyyymm, simple_ret))

    # Filter to months with FF5 data
    valid_data = []
    for yyyymm, ret in monthly_data:
        if yyyymm in ff5_dict:
            ff_row = ff5_dict[yyyymm]
            valid_data.append((ret,
                               float(ff_row['Mkt-RF']), float(ff_row['SMB']),
                               float(ff_row['HML']), float(ff_row['RMW']),
                               float(ff_row['CMA']), float(ff_row['RF'])))

    if len(valid_data) < 24:
        n_skipped += 1
        continue

    # OLS: excess_return = α + β₁·MktRF + β₂·SMB + β₃·HML + β₄·RMW + β₅·CMA + ε
    n = len(valid_data)
    y = np.empty(n)
    X = np.empty((n, 6))  # intercept + 5 factors

    for j, (ret, mkt_rf, smb, hml, rmw, cma, rf) in enumerate(valid_data):
        y[j] = ret - rf
        X[j, 0] = 1.0
        X[j, 1] = mkt_rf
        X[j, 2] = smb
        X[j, 3] = hml
        X[j, 4] = rmw
        X[j, 5] = cma

    coeffs, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
    fitted = X @ coeffs
    resids = y - fitted

    ss_tot = np.sum((y - np.mean(y)) ** 2)
    ss_res = np.sum(resids ** 2)
    r_sq = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0

    # coeffs[0] = alpha, coeffs[1:6] = [β_mkt, β_smb, β_hml, β_rmw, β_cma]
    cik_betas[cik] = coeffs[1:6].tolist()
    r_squared_list.append(r_sq)
    obs_counts.append(n)

print(f"  Companies with betas: {len(cik_betas)}")
print(f"  Companies skipped (<24 obs): {n_skipped}")
print(f"  Median R²: {np.median(r_squared_list):.4f}")
print(f"  Median obs: {int(np.median(obs_counts))}")

# ============================================================
# Step 4: Load clusters, map members to betas
# ============================================================
print("\n[4/7] Loading SAE clusters and mapping to factor betas...")

FACTOR_NAMES = ['Mkt-RF', 'SMB', 'HML', 'RMW', 'CMA']


def load_and_map_clusters(cluster_file, idx_to_cik, cik_betas, label=""):
    """Load cluster pickle and map members to beta vectors.

    Returns:
        cluster_beta_data: list of dicts with cluster_id, year, betas, n_members
        stats: dict with summary statistics
        cik_appearance_counts: Counter of how many cluster-year groups each CIK appears in
    """
    from collections import Counter

    with open(cluster_file, 'rb') as f:
        clusters_df = pickle.load(f)

    cluster_beta_data = []
    total_checked = 0
    total_singletons = 0
    total_multi = 0
    total_with_betas = 0
    cik_appearances = Counter()  # CIK -> number of cluster-year groups it appears in

    for _, row in clusters_df.iterrows():
        year = int(row['year'])
        clusters_dict = row['clusters']

        for cid, members in clusters_dict.items():
            total_checked += 1

            if len(members) < 2:
                total_singletons += 1
                continue

            total_multi += 1

            member_betas = []
            member_ciks = []
            for member_idx in members:
                cik = idx_to_cik.get(member_idx)
                if cik is None:
                    continue
                beta = cik_betas.get(cik)
                if beta is None:
                    continue
                member_betas.append(beta)
                member_ciks.append(cik)

            if len(member_betas) >= 2:
                total_with_betas += 1
                cluster_beta_data.append({
                    'cluster_id': cid,
                    'year': year,
                    'betas': np.array(member_betas),
                    'n_members': len(member_betas),
                })
                for c in member_ciks:
                    cik_appearances[c] += 1

    usable_sizes = [d['n_members'] for d in cluster_beta_data]
    stats = {
        'total_cluster_year_instances': total_checked,
        'singletons': total_singletons,
        'multi_member': total_multi,
        'multi_member_with_valid_betas': total_with_betas,
        'usable_cluster_size_distribution': {
            'min': int(min(usable_sizes)) if usable_sizes else 0,
            'max': int(max(usable_sizes)) if usable_sizes else 0,
            'median': float(np.median(usable_sizes)) if usable_sizes else 0,
            'mean': float(np.mean(usable_sizes)) if usable_sizes else 0,
        },
    }

    print(f"  [{label}] Total cluster-year instances: {total_checked}")
    print(f"  [{label}] Singletons (skipped): {total_singletons}")
    print(f"  [{label}] Multi-member clusters: {total_multi}")
    print(f"  [{label}] Multi-member with ≥2 valid betas: {total_with_betas}")
    if usable_sizes:
        print(f"  [{label}] Usable cluster size: min={min(usable_sizes)}, max={max(usable_sizes)}, "
              f"median={np.median(usable_sizes):.0f}, mean={np.mean(usable_sizes):.2f}")

    return cluster_beta_data, stats, cik_appearances


# Load SAE clusters
sae_cluster_data, sae_cluster_stats, sae_cik_appearances = load_and_map_clusters(
    CLUSTER_FILE_SAE, idx_to_cik, cik_betas, label="SAE"
)

# Repeated-measures diagnostic: how many groups does each company appear in?
appearance_counts = list(sae_cik_appearances.values())
print(f"\n  Repeated-measures diagnostic (SAE):")
print(f"    Unique CIKs across all cluster-year groups: {len(sae_cik_appearances)}")
print(f"    Mean appearances per CIK: {np.mean(appearance_counts):.1f}")
print(f"    Median appearances per CIK: {np.median(appearance_counts):.0f}")
print(f"    Max appearances: {max(appearance_counts)}")
print(f"    CIKs appearing >10 times: {sum(1 for c in appearance_counts if c > 10)}")

repeated_measures_stats = {
    'unique_ciks': len(sae_cik_appearances),
    'mean_appearances': float(np.mean(appearance_counts)),
    'median_appearances': float(np.median(appearance_counts)),
    'max_appearances': int(max(appearance_counts)),
    'ciks_above_10': int(sum(1 for c in appearance_counts if c > 10)),
}

# Load SIC clusters (sanity check — SIC SHOULD show higher ICC for market beta)
print(f"\n  Loading SIC clusters for sanity check...")
sic_cluster_data, sic_cluster_stats, sic_cik_appearances = load_and_map_clusters(
    CLUSTER_FILE_SIC, idx_to_cik, cik_betas, label="SIC"
)

# ============================================================
# Step 5: Compute ICC and ANOVA for each factor
# ============================================================
print("\n[5/7] Computing ICC and ANOVA for each factor loading...")

# Population statistics (all companies with betas)
all_betas = np.array(list(cik_betas.values()))  # shape (N, 5)
pop_means = np.mean(all_betas, axis=0)
pop_stds = np.std(all_betas, axis=0, ddof=1)

print(f"\n  Population factor loading statistics ({len(all_betas)} companies):")
for i, name in enumerate(FACTOR_NAMES):
    print(f"    {name:>6}: mean={pop_means[i]:.4f}, std={pop_stds[i]:.4f}")


def compute_icc_all_factors(cluster_beta_data, factor_names, pop_stds, label=""):
    """Compute ICC(1,1) for each factor across cluster-year groups.

    Uses harmonic mean for k_0 (correct for unbalanced designs).
    Returns dict of factor_name -> result dict.
    """
    results = {}

    for f_idx, f_name in enumerate(factor_names):
        groups = []
        all_obs = []

        for d in cluster_beta_data:
            group_vals = d['betas'][:, f_idx].tolist()
            groups.append(group_vals)
            all_obs.extend(group_vals)

        n_groups = len(groups)
        n_total = len(all_obs)
        grand_mean = np.mean(all_obs)

        # Between-group and within-group sums of squares
        ss_between = 0.0
        ss_within = 0.0
        for g in groups:
            n_g = len(g)
            g_mean = np.mean(g)
            ss_between += n_g * (g_mean - grand_mean) ** 2
            for val in g:
                ss_within += (val - g_mean) ** 2

        df_between = n_groups - 1
        df_within = n_total - n_groups

        ms_between = ss_between / df_between if df_between > 0 else 0
        ms_within = ss_within / df_within if df_within > 0 else 0

        # Harmonic mean cluster size (correct for unbalanced ICC)
        cluster_sizes = np.array([len(g) for g in groups], dtype=float)
        k_0_harmonic = float(len(cluster_sizes) / np.sum(1.0 / cluster_sizes))
        k_0_arithmetic = float(np.mean(cluster_sizes))

        # ICC(1,1) using harmonic mean
        denom = ms_between + (k_0_harmonic - 1) * ms_within
        icc = (ms_between - ms_within) / denom if denom > 0 else 0.0

        # F-statistic and p-value (one-way ANOVA)
        f_stat = ms_between / ms_within if ms_within > 0 else 0
        p_value = 1.0 - stats.f.cdf(f_stat, df_between, df_within) if df_within > 0 else 1.0

        # Within-cluster std vs population std
        weighted_within_std = np.sqrt(ms_within)
        variance_ratio = weighted_within_std / pop_stds[f_idx] if pop_stds[f_idx] > 0 else float('nan')

        results[f_name] = {
            'icc': float(icc),
            'f_statistic': float(f_stat),
            'p_value': float(p_value),
            'df_between': int(df_between),
            'df_within': int(df_within),
            'ms_between': float(ms_between),
            'ms_within': float(ms_within),
            'within_cluster_std': float(weighted_within_std),
            'population_std': float(pop_stds[f_idx]),
            'variance_ratio': float(variance_ratio),
            'n_groups': n_groups,
            'n_observations': n_total,
            'k0_harmonic': float(k_0_harmonic),
            'k0_arithmetic': float(k_0_arithmetic),
        }

        print(f"    [{label}] {f_name:>6}: ICC={icc:.4f}, F={f_stat:.2f}, p={p_value:.4e}, "
              f"within/pop std={variance_ratio:.3f}, k0_h={k_0_harmonic:.2f}")

    return results


# Compute ICC for SAE clusters
print("\n  --- SAE Clusters ---")
icc_results = compute_icc_all_factors(sae_cluster_data, FACTOR_NAMES, pop_stds, label="SAE")

# Compute ICC for SIC clusters (sanity check)
print("\n  --- SIC Clusters (sanity check) ---")
sic_icc_results = compute_icc_all_factors(sic_cluster_data, FACTOR_NAMES, pop_stds, label="SIC")

# ============================================================
# Step 6: SAE vs SIC ICC comparison
# ============================================================
print("\n[6/7] SAE vs SIC ICC comparison (sanity check)...")
print(f"\n  {'Factor':>6}  {'SAE ICC':>10}  {'SIC ICC':>10}  {'SAE < SIC?':>12}")
print(f"  {'-'*44}")
for f_name in FACTOR_NAMES:
    sae_icc = icc_results[f_name]['icc']
    sic_icc = sic_icc_results[f_name]['icc']
    comparison = "YES" if sae_icc < sic_icc else "NO"
    print(f"  {f_name:>6}  {sae_icc:>10.4f}  {sic_icc:>10.4f}  {comparison:>12}")

# Sanity check: SIC clusters SHOULD have higher ICC for at least Mkt-RF
# (market beta varies systematically by industry). If SIC ICC is also near zero,
# the computation has a bug — investigate before trusting results.
sic_mkt_icc = sic_icc_results['Mkt-RF']['icc']
if sic_mkt_icc < 0.01:
    print(f"\n  WARNING: SIC Mkt-RF ICC = {sic_mkt_icc:.4f} (near zero).")
    print(f"  Expected SIC clusters to show meaningful ICC for market beta.")
    print(f"  This may indicate a problem with the ICC computation or data mapping.")
    computation_sanity = "SUSPECT"
else:
    print(f"\n  SIC Mkt-RF ICC = {sic_mkt_icc:.4f} — SIC clusters group companies with")
    print(f"  similar market beta as expected. ICC computation appears valid.")
    computation_sanity = "VALID"

max_icc_factor = max(icc_results.keys(), key=lambda k: icc_results[k]['icc'])
max_icc_value = icc_results[max_icc_factor]['icc']

# ============================================================
# Step 7: Permutation test on ICC
# ============================================================
print(f"\n[7/7] Permutation test on {max_icc_factor} (highest SAE ICC={max_icc_value:.4f})...")

# Flatten for permutation
f_idx_max = FACTOR_NAMES.index(max_icc_factor)
flat_values = []
group_boundaries = []

pos = 0
for d in sae_cluster_data:
    vals = d['betas'][:, f_idx_max].tolist()
    flat_values.extend(vals)
    group_boundaries.append((pos, len(vals)))
    pos += len(vals)

flat_values = np.array(flat_values)
k_0_for_perm = icc_results[max_icc_factor]['k0_harmonic']


def compute_icc_from_flat(values, boundaries, k_0):
    """Compute ICC given flat values and group boundaries."""
    grand_mean = np.mean(values)
    ss_b = 0.0
    ss_w = 0.0
    for start, size in boundaries:
        g = values[start:start + size]
        g_mean = np.mean(g)
        ss_b += size * (g_mean - grand_mean) ** 2
        ss_w += np.sum((g - g_mean) ** 2)

    df_b = len(boundaries) - 1
    df_w = len(values) - len(boundaries)
    ms_b = ss_b / df_b if df_b > 0 else 0
    ms_w = ss_w / df_w if df_w > 0 else 0

    denom = ms_b + (k_0 - 1) * ms_w
    if denom > 0:
        return (ms_b - ms_w) / denom
    return 0.0


observed_icc = compute_icc_from_flat(flat_values, group_boundaries, k_0_for_perm)

N_PERMS = 1000
perm_iccs = []
for p in range(N_PERMS):
    if (p + 1) % 200 == 0:
        print(f"    Permutation {p+1}/{N_PERMS}...")
    shuffled = np.random.permutation(flat_values)
    perm_icc = compute_icc_from_flat(shuffled, group_boundaries, k_0_for_perm)
    perm_iccs.append(perm_icc)

perm_iccs = np.array(perm_iccs)
perm_p_value = float(np.mean(perm_iccs >= observed_icc))
perm_mean = float(np.mean(perm_iccs))
perm_std = float(np.std(perm_iccs))
perm_95 = float(np.percentile(perm_iccs, 95))
perm_99 = float(np.percentile(perm_iccs, 99))

print(f"    Observed ICC: {observed_icc:.4f}")
print(f"    Null distribution: mean={perm_mean:.4f}, std={perm_std:.4f}")
print(f"    Null 95th percentile: {perm_95:.4f}")
print(f"    Null 99th percentile: {perm_99:.4f}")
print(f"    Permutation p-value: {perm_p_value:.4f}")

permutation_test = {
    'factor': max_icc_factor,
    'observed_icc': float(observed_icc),
    'null_mean': perm_mean,
    'null_std': perm_std,
    'null_p95': perm_95,
    'null_p99': perm_99,
    'p_value': perm_p_value,
    'n_permutations': N_PERMS,
}

# ============================================================
# Verdict
# ============================================================
print("\n" + "=" * 80)
print("RESULTS SUMMARY")
print("=" * 80)

print(f"\n  SAE factor loading ICC by factor:")
for f_name in FACTOR_NAMES:
    r = icc_results[f_name]
    sig = "***" if r['p_value'] < 0.001 else "**" if r['p_value'] < 0.01 else "*" if r['p_value'] < 0.05 else "ns"
    print(f"    {f_name:>6}: ICC={r['icc']:.4f}  F={r['f_statistic']:.2f}  "
          f"p={r['p_value']:.4e} {sig}  within/pop std={r['variance_ratio']:.3f}")

print(f"\n  SIC factor loading ICC by factor (sanity check):")
for f_name in FACTOR_NAMES:
    r = sic_icc_results[f_name]
    sig = "***" if r['p_value'] < 0.001 else "**" if r['p_value'] < 0.01 else "*" if r['p_value'] < 0.05 else "ns"
    print(f"    {f_name:>6}: ICC={r['icc']:.4f}  F={r['f_statistic']:.2f}  "
          f"p={r['p_value']:.4e} {sig}  within/pop std={r['variance_ratio']:.3f}")

print(f"\n  Highest SAE ICC: {max_icc_factor} = {max_icc_value:.4f}")
print(f"  Permutation p-value: {perm_p_value:.4f}")
print(f"  Computation sanity (SIC check): {computation_sanity}")

# Interpretation thresholds are INFORMED by SIC baseline.
# If SIC ICC for a factor is X, and SAE ICC is << X, SAE is capturing
# something different from industry-level factor structure.
# Fixed thresholds (0.05, 0.15) are fallbacks if SIC check fails.
any_above_015 = any(r['icc'] > 0.15 for r in icc_results.values())
any_above_005 = any(r['icc'] > 0.05 for r in icc_results.values())

if computation_sanity == "SUSPECT":
    verdict = "INCONCLUSIVE"
    interpretation = (
        f"SIC sanity check failed (Mkt-RF ICC={sic_mkt_icc:.4f}). "
        f"Cannot distinguish 'SAE is orthogonal to factors' from "
        f"'ICC computation has a bug'. Investigation required."
    )
elif not any_above_005:
    verdict = "FLAG_1_INVALIDATED"
    interpretation = (
        "All SAE factor loading ICCs are below 0.05. SAE cluster membership does NOT "
        "predict shared FF5 factor exposure. The 98.4% survival ratio in 1B is the "
        "expected result: removing factors that explain 3% of variance changes "
        "correlations by approximately 3%. Flag 1 is invalidated."
    )
elif any_above_015:
    verdict = "FLAG_1_OPEN"
    interpretation = (
        f"At least one factor ({max_icc_factor}) has ICC > 0.15, indicating meaningful "
        f"clustering of factor loadings within SAE clusters. The 98.4% survival ratio "
        f"may be understating the factor overlap. Deeper investigation needed."
    )
else:
    verdict = "FLAG_1_WEAK"
    interpretation = (
        f"SAE factor loading ICCs are between 0.05 and 0.15 (max: {max_icc_factor}={max_icc_value:.4f}). "
        f"There is weak clustering of factor loadings in SAE clusters, but not enough "
        f"to explain a large deviation from 98.4% survival. Flag 1 is likely "
        f"invalidated but the weak signal warrants noting."
    )

print(f"\n  VERDICT: {verdict}")
print(f"  {interpretation}")

# ============================================================
# Write output
# ============================================================
result = {
    "test": "1B-02_factor_loading_similarity",
    "question": "Do SAE clusters share FF5 factor loadings?",
    "regression_check": {
        "n_companies_with_betas": len(cik_betas),
        "n_skipped": n_skipped,
        "median_r_squared": float(np.median(r_squared_list)),
        "median_obs_per_company": int(np.median(obs_counts)),
        "expected_from_1B": {
            "n_regressed": 1531,
            "n_skipped": 71,
            "median_r_squared": 0.0301,
            "median_obs": 240,
        },
        "factor_means": {name: float(pop_means[i]) for i, name in enumerate(FACTOR_NAMES)},
        "factor_stds": {name: float(pop_stds[i]) for i, name in enumerate(FACTOR_NAMES)},
    },
    "sae_cluster_stats": sae_cluster_stats,
    "sic_cluster_stats": sic_cluster_stats,
    "repeated_measures": repeated_measures_stats,
    "sae_icc_per_factor": icc_results,
    "sic_icc_per_factor": sic_icc_results,
    "sae_vs_sic_comparison": {
        f_name: {
            'sae_icc': icc_results[f_name]['icc'],
            'sic_icc': sic_icc_results[f_name]['icc'],
            'sae_less_than_sic': icc_results[f_name]['icc'] < sic_icc_results[f_name]['icc'],
        }
        for f_name in FACTOR_NAMES
    },
    "computation_sanity": computation_sanity,
    "permutation_test": permutation_test,
    "verdict": verdict,
    "interpretation": interpretation,
}

with open(OUTPUT_FILE, 'w') as f:
    json.dump(result, f, indent=2)

print(f"\nResult written to: {OUTPUT_FILE}")
print(f"\n{'='*80}")
print("FULL JSON:")
print(f"{'='*80}")
print(json.dumps(result, indent=2))

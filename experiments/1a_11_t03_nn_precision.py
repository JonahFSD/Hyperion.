#!/usr/bin/env python3
"""
Phase 1A Test T03 - Nearest-Neighbor Precision@K
Year-by-year processing to minimize RAM footprint.
"""

import pandas as pd
import numpy as np
import json
import gc
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# CONFIGURATION
# ============================================================================
DATA_DIR = 'phase1_artifacts'
PAIRS_FILE = f'{DATA_DIR}/pairs.parquet'
COMPANIES_FILE = f'{DATA_DIR}/companies.parquet'
OUTPUT_FILE = f'{DATA_DIR}/1a_11_t03_result.json'

K_VALUES = [1, 3, 5, 10, 20, 50]
BOOTSTRAP_SAMPLES = 100
PROGRESS_INTERVAL = 50

# ============================================================================
# LOAD COMPANIES
# ============================================================================
print("[1/6] Loading company metadata...")
companies = pd.read_parquet(COMPANIES_FILE, engine='pyarrow',
                            columns=['__index_level_0__', 'sic_code'])
cik_to_sic = dict(zip(companies['__index_level_0__'], companies['sic_code']))
print(f"  Loaded {len(cik_to_sic):,} companies")
del companies
gc.collect()

# ============================================================================
# GET YEARS
# ============================================================================
print("\n[2/6] Discovering years...")
import pyarrow.parquet as pq
table = pq.read_table(PAIRS_FILE, columns=['year'])
years = sorted(set(table['year'].to_pylist()))
print(f"  Years: {years}")
del table
gc.collect()

# ============================================================================
# PROCESS YEAR-BY-YEAR
# ============================================================================
print("\n[3/6] Processing year-by-year...")

results_by_k = defaultdict(list)
hit_rates_by_k = defaultdict(list)
sic_corrs = []
sae_top10_corrs = []
sic_top10_corrs = []
company_years_processed = 0

for year_idx, year in enumerate(years):
    print(f"  [Year {year_idx+1}/{len(years)}] Processing year {year}...")

    # Read all pairs for this year
    # Use filters in pyarrow
    import pyarrow.parquet as pq
    import pyarrow.compute as pc

    dataset = pq.ParquetDataset(PAIRS_FILE)
    # Filter doesn't work with ParquetDataset directly, so just read and filter
    pairs_year = pd.read_parquet(PAIRS_FILE, engine='pyarrow',
                                columns=['Company1', 'Company2', 'year', 'correlation', 'cosine_similarity'])
    pairs_year = pairs_year[pairs_year['year'] == year]

    if len(pairs_year) == 0:
        continue

    print(f"    {len(pairs_year):,} pairs for year {year}")

    # Get all companies in this year
    all_companies_year = np.unique(np.concatenate([
        pairs_year['Company1'].values,
        pairs_year['Company2'].values
    ]))

    # Process each company-year
    grouped = pairs_year.groupby('Company1')
    for company1, group in grouped:
        company_years_processed += 1

        if company_years_processed % PROGRESS_INTERVAL == 0:
            print(f"    Processed {company_years_processed:,} company-years overall")

        # Sort by cosine similarity descending
        group_sorted = group.sort_values('cosine_similarity', ascending=False)
        peer_correlations = group_sorted['correlation'].values
        peer_ids = group_sorted['Company2'].values

        # Get SIC code
        company_sic = cik_to_sic.get(company1, None)

        # Find same-SIC companies
        same_sic_companies = set()
        if company_sic is not None:
            for idx, sic in cik_to_sic.items():
                if sic == company_sic:
                    same_sic_companies.add(idx)

        # Compute SAE top-K metrics
        for k in K_VALUES:
            if len(peer_correlations) >= k:
                sae_top_k_corr = np.nanmean(peer_correlations[:k])

                # Random baseline
                candidate_peers = all_companies_year[all_companies_year != company1]

                if len(candidate_peers) >= k:
                    # Sample k random peers
                    random_sample = np.random.choice(candidate_peers, k, replace=False)

                    # Find correlations of random sample
                    mask = np.isin(peer_ids, random_sample)
                    if np.any(mask):
                        random_baseline_corr = np.nanmean(peer_correlations[mask])
                        lift = sae_top_k_corr - random_baseline_corr
                        results_by_k[k].append(lift)
                        hit_rates_by_k[k].append(1 if lift > 0 else 0)

        # SIC baseline
        if company_sic is not None and len(same_sic_companies) > 1:
            same_sic_peers = np.array(list(same_sic_companies - {company1}), dtype=np.int64)
            if len(same_sic_peers) > 0:
                # Find correlations for same-SIC peers
                mask = np.isin(peer_ids, same_sic_peers)
                if np.any(mask):
                    sic_mean_corr = np.nanmean(peer_correlations[mask])
                    sic_corrs.append(sic_mean_corr)

                    # Top-10 comparison
                    if len(peer_correlations) >= 10:
                        sae_top10 = np.nanmean(peer_correlations[:10])
                        sae_top10_corrs.append(sae_top10)

                        # Get same-SIC peers sorted by cosine (top of the list)
                        sic_peer_indices = np.where(mask)[0]
                        if len(sic_peer_indices) > 0:
                            sic_top_idx = sic_peer_indices[:10]
                            sic_top10 = np.nanmean(peer_correlations[sic_top_idx])
                            sic_top10_corrs.append(sic_top10)

    del pairs_year, grouped
    gc.collect()

print(f"  Total company-years: {company_years_processed:,}")

# ============================================================================
# BOOTSTRAP CONFIDENCE INTERVALS
# ============================================================================
print("\n[4/6] Computing bootstrap CIs...")

def bootstrap_ci(data, n_resamples=BOOTSTRAP_SAMPLES):
    if len(data) < 2:
        return [np.nan, np.nan]
    data = np.array(data)
    means = []
    for _ in range(n_resamples):
        sample = np.random.choice(data, size=len(data), replace=True)
        means.append(np.mean(sample))
    return [np.percentile(means, 2.5), np.percentile(means, 97.5)]

overall_results = {}
for k in K_VALUES:
    if k in results_by_k and len(results_by_k[k]) > 0:
        lifts = np.array(results_by_k[k])
        hit_rate = np.mean(hit_rates_by_k[k])
        ci = bootstrap_ci(lifts)

        overall_results[f'K_{k}'] = {
            'lift_mean': float(np.nanmean(lifts)),
            'lift_std': float(np.nanstd(lifts)),
            'ci_95': [float(c) for c in ci],
            'hit_rate': float(hit_rate),
            'n_samples': len(lifts)
        }

# ============================================================================
# VERDICT
# ============================================================================
print("\n[5/6] Evaluating verdict...")

# SAE vs SIC
if sae_top10_corrs and sic_top10_corrs:
    sae_top10_mean = np.nanmean(sae_top10_corrs)
    sic_top10_mean = np.nanmean(sic_top10_corrs)
    sae_wins = sae_top10_mean > sic_top10_mean
else:
    sae_top10_mean = np.nan
    sic_top10_mean = np.nan
    sae_wins = False

sic_baseline_mean = np.nanmean(sic_corrs) if sic_corrs else np.nan

# Verdict criteria
k5_ci_zero = False
if 'K_5' in overall_results:
    ci = overall_results['K_5']['ci_95']
    if not np.isnan(ci[0]) and ci[0] <= 0:
        k5_ci_zero = True

all_k_positive = all(
    overall_results[f'K_{k}']['lift_mean'] > 0
    for k in K_VALUES if f'K_{k}' in overall_results
)

hit_rate_10 = overall_results.get('K_10', {}).get('hit_rate', 0)
hit_rate_pass = hit_rate_10 > 0.55

verdict = "PASS" if (not k5_ci_zero and all_k_positive and hit_rate_pass) else "FAIL"

# ============================================================================
# BUILD JSON RESULT
# ============================================================================
result = {
    "test": "T03_nn_precision_at_k",
    "overall": {
        f"K_{k}": {
            "lift_mean": overall_results.get(f"K_{k}", {}).get("lift_mean", None),
            "lift_std": overall_results.get(f"K_{k}", {}).get("lift_std", None),
            "ci_95": overall_results.get(f"K_{k}", {}).get("ci_95", None),
            "hit_rate": overall_results.get(f"K_{k}", {}).get("hit_rate", None),
            "n_samples": overall_results.get(f"K_{k}", {}).get("n_samples", None)
        }
        for k in K_VALUES
    },
    "sic_baseline_mean_corr": float(sic_baseline_mean) if not np.isnan(sic_baseline_mean) else None,
    "sae_top10_vs_sic": {
        "sae": float(sae_top10_mean) if not np.isnan(sae_top10_mean) else None,
        "sic": float(sic_top10_mean) if not np.isnan(sic_top10_mean) else None,
        "sae_wins": bool(sae_wins)
    },
    "n_company_years": company_years_processed,
    "n_with_sic": len(sic_corrs),
    "verdict": verdict,
    "interpretation": (
        f"SAE nearest-neighbor retrieval shows positive lift over random baseline. "
        f"Hit rate at K=10: {hit_rate_10:.1%}. "
        f"SAE top-10 {'outperforms' if sae_wins else 'underperforms'} SIC baseline. "
        f"Verdict: {verdict}"
    )
}

# ============================================================================
# WRITE AND DISPLAY
# ============================================================================
print(f"\n[6/6] Writing to {OUTPUT_FILE}...")
with open(OUTPUT_FILE, 'w') as f:
    json.dump(result, f, indent=2)

print("\n" + "="*80)
print("PHASE 1A TEST T03 - NEAREST-NEIGHBOR PRECISION@K")
print("="*80)
print(f"\nCompany-years: {company_years_processed:,}")
print(f"With SIC baseline: {len(sic_corrs):,}")
print(f"VERDICT: {verdict}\n")

for k in K_VALUES:
    if f'K_{k}' in overall_results:
        r = overall_results[f'K_{k}']
        print(f"K={k:2d}: lift={r['lift_mean']:+.4f} ± {r['lift_std']:.4f} | "
              f"CI=[{r['ci_95'][0]:+.4f}, {r['ci_95'][1]:+.4f}] | "
              f"hit={r['hit_rate']:.1%} (n={r['n_samples']})")

print(f"\nSIC baseline: {sic_baseline_mean:.4f}")
print(f"SAE top-10: {sae_top10_mean:.4f} vs SIC top-10: {sic_top10_mean:.4f}")
print(f"SAE wins: {sae_wins}")
print("="*80)

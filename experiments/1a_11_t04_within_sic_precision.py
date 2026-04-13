#!/usr/bin/env python3
"""
Test T04 — Within-SIC Retrieval Precision@K
Minimal memory approach with pre-computed SIC mapping
"""

import numpy as np
import json
import pyarrow.parquet as pq
from collections import defaultdict, Counter
import pickle
import warnings
warnings.filterwarnings('ignore')

# Load pre-computed SIC mapping
print("Loading SIC mapping...")
with open(os.path.join(os.path.dirname(__file__), '..', 'phase1_artifacts', 'idx_to_sic2.pkl'), 'rb') as f:
    idx_to_sic2 = pickle.load(f)
print(f"SIC mapping: {len(idx_to_sic2)} entries")

# === PASS 1: Count same-SIC peers per company-year ===
print("\nPass 1: Scanning for same-SIC company-year counts...")
pairs_file = pq.ParquetFile('phase1_artifacts/pairs.parquet')

company_year_counts = Counter()
n_same_sic_total = 0

for i in range(pairs_file.num_row_groups):
    rg_table = pairs_file.read_row_group(i)

    comp1_col = rg_table['Company1'].to_numpy()
    comp2_col = rg_table['Company2'].to_numpy()
    year_col = rg_table['year'].to_numpy()

    for j in range(len(comp1_col)):
        comp1 = int(comp1_col[j])
        comp2 = int(comp2_col[j])
        year = int(year_col[j])

        sic2_1 = idx_to_sic2.get(comp1)
        sic2_2 = idx_to_sic2.get(comp2)

        if sic2_1 is not None and sic2_1 == sic2_2:
            key = (comp1, year)
            company_year_counts[key] += 1
            n_same_sic_total += 1

    if (i + 1) % 5 == 0:
        print(f"  Row group {i+1}/{pairs_file.num_row_groups}: {n_same_sic_total:,} same-SIC pairs")

# Filter to company-years with >= 5 peers
company_years_to_process = {k: v for k, v in company_year_counts.items() if v >= 5}
n_skipped = len(company_year_counts) - len(company_years_to_process)

print(f"\nTotal same-SIC pairs: {n_same_sic_total:,}")
print(f"Company-years with >=5 peers: {len(company_years_to_process):,}")
print(f"Skipped (<5 peers): {n_skipped}")

# === PASS 2: Process only relevant company-years ===
print("\nPass 2: Computing metrics...")

results_by_k = {1: [], 3: [], 5: [], 10: []}
per_year_results = defaultdict(lambda: {'lifts': [], 'hit_rates': []})
per_year_companies = Counter()

for i in range(pairs_file.num_row_groups):
    rg_table = pairs_file.read_row_group(i)

    comp1_col = rg_table['Company1'].to_numpy()
    comp2_col = rg_table['Company2'].to_numpy()
    year_col = rg_table['year'].to_numpy()
    cosine_col = rg_table['cosine_similarity'].to_numpy()
    corr_col = rg_table['correlation'].to_numpy()

    # Group by company1, year
    group_dict = defaultdict(list)
    for j in range(len(comp1_col)):
        comp1 = int(comp1_col[j])
        comp2 = int(comp2_col[j])
        year = int(year_col[j])
        cosine_sim = float(cosine_col[j])
        corr = float(corr_col[j]) if not np.isnan(corr_col[j]) else None

        sic2_1 = idx_to_sic2.get(comp1)
        sic2_2 = idx_to_sic2.get(comp2)

        if sic2_1 is not None and sic2_1 == sic2_2:
            key = (comp1, year)
            if key in company_years_to_process:
                group_dict[key].append((comp2, cosine_sim, corr))

    # Process each company-year
    for key, peers in group_dict.items():
        comp1, year = key

        if len(peers) < 5:
            continue

        per_year_companies[year] += 1

        # Sort by cosine similarity
        peers_sorted = sorted(peers, key=lambda x: x[1], reverse=True)

        for K in [1, 3, 5, 10]:
            if K > len(peers_sorted):
                continue

            # SAE top-K
            top_k_corrs = [c for _, _, c in peers_sorted[:K] if c is not None]
            if len(top_k_corrs) == 0:
                continue
            sae_mean_corr = np.mean(top_k_corrs)

            # Random-K
            random_corrs_all = []
            for _ in range(100):
                random_indices = np.random.choice(len(peers_sorted), size=min(K, len(peers_sorted)), replace=False)
                random_k_corrs = [peers_sorted[idx][2] for idx in random_indices if peers_sorted[idx][2] is not None]
                if len(random_k_corrs) > 0:
                    random_corrs_all.append(np.mean(random_k_corrs))

            if len(random_corrs_all) == 0:
                continue
            random_mean_corr = np.mean(random_corrs_all)

            lift = sae_mean_corr - random_mean_corr
            hit_rate = 1.0 if sae_mean_corr > random_mean_corr else 0.0

            results_by_k[K].append({
                'sae_mean_corr': sae_mean_corr,
                'random_mean_corr': random_mean_corr,
                'lift': lift,
                'hit_rate': hit_rate
            })

            if K == 5:
                per_year_results[year]['lifts'].append(lift)
                per_year_results[year]['hit_rates'].append(hit_rate)

    if (i + 1) % 5 == 0:
        print(f"  Row group {i+1}/{pairs_file.num_row_groups}: K=5 samples: {len(results_by_k[5])}")

print(f"\nMetrics computed:")
for K in [1, 3, 5, 10]:
    print(f"  K={K}: {len(results_by_k[K])} samples")

# Bootstrap CI
def bootstrap_ci(values, n_resamples=1000, ci=0.95):
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

print("\nAggregating results...")

overall_results = {}
for K in [1, 3, 5, 10]:
    if len(results_by_k[K]) == 0:
        overall_results[f'K_{K}'] = None
        continue

    sae_corrs = [r['sae_mean_corr'] for r in results_by_k[K]]
    random_corrs = [r['random_mean_corr'] for r in results_by_k[K]]
    lifts = [r['lift'] for r in results_by_k[K]]
    hit_rates = [r['hit_rate'] for r in results_by_k[K]]

    sae_mean, sae_ci = bootstrap_ci(sae_corrs)
    random_mean, random_ci = bootstrap_ci(random_corrs)
    lift_mean, lift_ci = bootstrap_ci(lifts)
    hit_rate_mean = np.mean(hit_rates)

    overall_results[f'K_{K}'] = {
        'sae_mean_corr': float(sae_mean),
        'sae_ci_95': [float(sae_ci[0]), float(sae_ci[1])],
        'random_sic_mean_corr': float(random_mean),
        'random_sic_ci_95': [float(random_ci[0]), float(random_ci[1])],
        'lift': float(lift_mean),
        'ci_95': [float(lift_ci[0]), float(lift_ci[1])],
        'hit_rate': float(hit_rate_mean),
        'n_samples': len(results_by_k[K])
    }

# Per-year breakdown
per_year_json = {}
years_with_lift = []
for year in sorted(per_year_results.keys()):
    if per_year_results[year]['lifts']:
        lifts = per_year_results[year]['lifts']
        hit_rates = per_year_results[year]['hit_rates']
        lift_mean, lift_ci = bootstrap_ci(lifts, n_resamples=100)
        hit_rate_mean = np.mean(hit_rates)
        n_companies = per_year_companies[year]

        per_year_json[int(year)] = {
            'lift': float(lift_mean),
            'ci_95': [float(lift_ci[0]), float(lift_ci[1])],
            'hit_rate': float(hit_rate_mean),
            'n_companies': int(n_companies),
            'n_samples': len(lifts)
        }
        if lift_mean is not None:
            years_with_lift.append(lift_mean)

# Verdict
verdict = "FAIL"
reason = []

if 'K_5' in overall_results and overall_results['K_5'] is not None:
    k5_ci = overall_results['K_5']['ci_95']
    k5_lift = overall_results['K_5']['lift']
    k5_hit_rate = overall_results['K_5']['hit_rate']

    ci_excludes_zero = (k5_ci[0] > 0 or k5_ci[1] < 0)
    if ci_excludes_zero:
        reason.append(f"K=5 lift CI [{k5_ci[0]:.4f}, {k5_ci[1]:.4f}] excludes zero")
        verdict = "PASS"

        if k5_hit_rate > 0.60 and len(years_with_lift) > 1:
            mid_idx = len(years_with_lift) // 2
            early_avg = np.mean(years_with_lift[:mid_idx]) if mid_idx > 0 else 0
            late_avg = np.mean(years_with_lift[mid_idx:])
            if late_avg > early_avg:
                verdict = "STRONG PASS"
                reason.append(f"Hit rate {k5_hit_rate:.2%} > 60% and lift improving over time")
    else:
        reason.append(f"K=5 lift CI [{k5_ci[0]:.4f}, {k5_ci[1]:.4f}] includes zero → FAIL")

# Final result
result = {
    'test': 'T04_within_sic_precision_at_k',
    'overall': overall_results,
    'per_year': per_year_json,
    'n_company_years': sum(per_year_companies.values()),
    'n_skipped_too_few_peers': n_skipped,
    'verdict': verdict,
    'interpretation': ' | '.join(reason) if reason else 'Test execution complete.',
    'timestamp': str(np.datetime64('now'))
}

output_path = 'phase1_artifacts/1a_11_t04_result.json'
with open(output_path, 'w') as f:
    json.dump(result, f, indent=2)

print(f"\n{'='*70}")
print(f"VERDICT: {verdict}")
print(f"{'='*70}")

for k_key in ['K_1', 'K_3', 'K_5', 'K_10']:
    if overall_results[k_key] is not None:
        res = overall_results[k_key]
        print(f"\n{k_key}:")
        print(f"  SAE mean corr:   {res['sae_mean_corr']:.6f} [{res['sae_ci_95'][0]:.6f}, {res['sae_ci_95'][1]:.6f}]")
        print(f"  Random SIC mean: {res['random_sic_mean_corr']:.6f}")
        print(f"  Lift:            {res['lift']:.6f} [{res['ci_95'][0]:.6f}, {res['ci_95'][1]:.6f}]")
        print(f"  Hit rate:        {res['hit_rate']:.2%}")
        print(f"  Samples:         {res['n_samples']}")

print(f"\nPer-Year (K=5):")
for year in sorted(per_year_json.keys()):
    pyr = per_year_json[year]
    print(f"  {year}: lift={pyr['lift']:.6f}, hit_rate={pyr['hit_rate']:.2%}, n={pyr['n_companies']}")

print(f"\nStats: {result['n_company_years']} company-years, {result['n_skipped_too_few_peers']} skipped")
print(f"Interpretation: {result['interpretation']}")
print(f"\nResult: {output_path}")

print(f"\n{'='*70}")
print("FULL JSON:")
print(f"{'='*70}")
print(json.dumps(result, indent=2))

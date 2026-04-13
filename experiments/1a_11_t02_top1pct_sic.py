#!/usr/bin/env python3
"""
Phase 1A Test T02: Top-1% SIC Composition Analysis

Streaming approach that processes data once to collect all statistics.
"""

import pandas as pd
import numpy as np
import json
import pyarrow.parquet as pq
import warnings
warnings.filterwarnings('ignore')

# Paths
from pathlib import Path
SCRIPT_DIR = Path(__file__).resolve().parent
PAIRS_PATH = SCRIPT_DIR / 'phase1_artifacts' / 'pairs.parquet'
COMPANIES_PATH = SCRIPT_DIR / 'phase1_artifacts' / 'companies.parquet'
OUTPUT_PATH = SCRIPT_DIR / 'phase1_artifacts' / '1a_11_t02_result.json'

print("[T02] Loading data with streaming approach...")

# Load companies (smaller)
print("Loading companies...")
companies_tbl = pq.read_table(COMPANIES_PATH, columns=['__index_level_0__', 'sic_code'])
companies_df = companies_tbl.to_pandas()
companies_df['sic_2digit'] = companies_df['sic_code'].astype(str).str[:2]
sic_map = companies_df.set_index('__index_level_0__')['sic_2digit'].to_dict()
print(f"  Created SIC-2 mapping for {len(sic_map)} companies")
del companies_df, companies_tbl

# First pass: collect all cosine similarities to compute threshold
print("\nFirst pass: collecting cosine similarities for threshold computation...")
cosine_sims = []
pairs_pf = pq.ParquetFile(PAIRS_PATH)

for i in range(pairs_pf.num_row_groups):
    print(f"  Row group {i+1}/{pairs_pf.num_row_groups}...", flush=True)
    chunk_table = pairs_pf.read_row_group(i, columns=['cosine_similarity'])
    cosine_sims.extend(chunk_table['cosine_similarity'].to_pylist())

top1pct_threshold = np.percentile(cosine_sims, 99)
print(f"\nTop-1% threshold: {top1pct_threshold:.4f}")
del cosine_sims

# Second pass: compute all statistics
print("\nSecond pass: computing statistics...")

# Accumulators for overall statistics
baseline_same_sic_sum = 0
baseline_same_sic_count = 0
baseline_corr_sum = 0
baseline_corr_count = 0

top1pct_same_sic_sum = 0
top1pct_same_sic_count = 0
top1pct_corr_sum = 0
top1pct_corr_count = 0

top1pct_same_sic_corr_sum = 0
top1pct_same_sic_corr_count = 0

top1pct_cross_sic_corr_sum = 0
top1pct_cross_sic_corr_count = 0

# Per-year accumulators
per_year_data = {}

# SIC pairing accumulators
sic_pairings = {}

for i in range(pairs_pf.num_row_groups):
    print(f"  Processing row group {i+1}/{pairs_pf.num_row_groups}...", flush=True)
    chunk_table = pairs_pf.read_row_group(i)
    chunk_df = chunk_table.to_pandas()

    # Add SIC information
    chunk_df['sic1'] = chunk_df['Company1'].map(sic_map)
    chunk_df['sic2'] = chunk_df['Company2'].map(sic_map)
    chunk_df['same_sic'] = chunk_df['sic1'] == chunk_df['sic2']

    # Filter to valid pairs
    chunk_df = chunk_df.dropna(subset=['sic1', 'sic2'])

    # Update baseline statistics
    baseline_same_sic_sum += chunk_df['same_sic'].sum()
    baseline_same_sic_count += len(chunk_df)
    baseline_corr_sum += chunk_df['correlation'].sum()
    baseline_corr_count += len(chunk_df)

    # Filter to top-1%
    top1pct_chunk = chunk_df[chunk_df['cosine_similarity'] >= top1pct_threshold]

    if len(top1pct_chunk) > 0:
        top1pct_same_sic_sum += top1pct_chunk['same_sic'].sum()
        top1pct_same_sic_count += len(top1pct_chunk)
        top1pct_corr_sum += top1pct_chunk['correlation'].sum()
        top1pct_corr_count += len(top1pct_chunk)

        # Same-SIC vs cross-SIC in top-1%
        same_sic_mask = top1pct_chunk['same_sic']
        top1pct_same_sic_corr_sum += top1pct_chunk.loc[same_sic_mask, 'correlation'].sum()
        top1pct_same_sic_corr_count += same_sic_mask.sum()

        cross_sic_mask = ~top1pct_chunk['same_sic']
        top1pct_cross_sic_corr_sum += top1pct_chunk.loc[cross_sic_mask, 'correlation'].sum()
        top1pct_cross_sic_corr_count += cross_sic_mask.sum()

        # Per-year accumulation
        for year in top1pct_chunk['year'].unique():
            year_data = top1pct_chunk[top1pct_chunk['year'] == year]
            if year not in per_year_data:
                per_year_data[year] = {
                    'baseline_same_sic_sum': 0,
                    'baseline_same_sic_count': 0,
                    'baseline_corr_sum': 0,
                    'baseline_corr_count': 0,
                    'top1pct_same_sic_sum': 0,
                    'top1pct_same_sic_count': 0,
                    'top1pct_corr_sum': 0,
                    'top1pct_corr_count': 0,
                    'top1pct_same_sic_corr_sum': 0,
                    'top1pct_same_sic_corr_count': 0,
                    'top1pct_cross_sic_corr_sum': 0,
                    'top1pct_cross_sic_corr_count': 0,
                }
            per_year_data[year]['top1pct_same_sic_sum'] += year_data['same_sic'].sum()
            per_year_data[year]['top1pct_same_sic_count'] += len(year_data)
            per_year_data[year]['top1pct_corr_sum'] += year_data['correlation'].sum()
            per_year_data[year]['top1pct_corr_count'] += len(year_data)

            y_same_sic_mask = year_data['same_sic']
            per_year_data[year]['top1pct_same_sic_corr_sum'] += year_data.loc[y_same_sic_mask, 'correlation'].sum()
            per_year_data[year]['top1pct_same_sic_corr_count'] += y_same_sic_mask.sum()

            y_cross_sic_mask = ~year_data['same_sic']
            per_year_data[year]['top1pct_cross_sic_corr_sum'] += year_data.loc[y_cross_sic_mask, 'correlation'].sum()
            per_year_data[year]['top1pct_cross_sic_corr_count'] += y_cross_sic_mask.sum()

        # SIC pairings in top-1%
        for idx, row in top1pct_chunk.iterrows():
            sic1, sic2 = row['sic1'], row['sic2']
            sic_pair = '-'.join(sorted([sic1, sic2]))
            if sic_pair not in sic_pairings:
                sic_pairings[sic_pair] = []
            sic_pairings[sic_pair].append(row['correlation'])

    # Also collect baseline per-year
    for year in chunk_df['year'].unique():
        year_baseline = chunk_df[chunk_df['year'] == year]
        if year not in per_year_data:
            per_year_data[year] = {
                'baseline_same_sic_sum': 0,
                'baseline_same_sic_count': 0,
                'baseline_corr_sum': 0,
                'baseline_corr_count': 0,
                'top1pct_same_sic_sum': 0,
                'top1pct_same_sic_count': 0,
                'top1pct_corr_sum': 0,
                'top1pct_corr_count': 0,
                'top1pct_same_sic_corr_sum': 0,
                'top1pct_same_sic_corr_count': 0,
                'top1pct_cross_sic_corr_sum': 0,
                'top1pct_cross_sic_corr_count': 0,
            }
        per_year_data[year]['baseline_same_sic_sum'] += year_baseline['same_sic'].sum()
        per_year_data[year]['baseline_same_sic_count'] += len(year_baseline)
        per_year_data[year]['baseline_corr_sum'] += year_baseline['correlation'].sum()
        per_year_data[year]['baseline_corr_count'] += len(year_baseline)

    del chunk_table, chunk_df

# Compute overall statistics
baseline_same_sic_rate = baseline_same_sic_sum / baseline_same_sic_count if baseline_same_sic_count > 0 else 0
baseline_mean_corr = baseline_corr_sum / baseline_corr_count if baseline_corr_count > 0 else 0

top1pct_same_sic_rate = top1pct_same_sic_sum / top1pct_same_sic_count if top1pct_same_sic_count > 0 else 0
top1pct_mean_corr = top1pct_corr_sum / top1pct_corr_count if top1pct_corr_count > 0 else 0

top1pct_same_sic_corr = top1pct_same_sic_corr_sum / top1pct_same_sic_corr_count if top1pct_same_sic_corr_count > 0 else np.nan
top1pct_cross_sic_corr = top1pct_cross_sic_corr_sum / top1pct_cross_sic_corr_count if top1pct_cross_sic_corr_count > 0 else np.nan

enrichment_ratio = top1pct_same_sic_rate / baseline_same_sic_rate if baseline_same_sic_rate > 0 else np.nan

print(f"\nBaseline same-SIC rate: {baseline_same_sic_rate:.4f}")
print(f"Baseline mean correlation: {baseline_mean_corr:.4f}")
print(f"Top-1% same-SIC rate: {top1pct_same_sic_rate:.4f}")
print(f"Enrichment ratio: {enrichment_ratio:.4f}x")
print(f"Top-1% mean correlation: {top1pct_mean_corr:.4f}")
print(f"Top-1% same-SIC mean correlation: {top1pct_same_sic_corr:.4f}")
print(f"Top-1% cross-SIC mean correlation: {top1pct_cross_sic_corr:.4f}")

# Compute per-year results
per_year_results = {}
for year in sorted(per_year_data.keys()):
    data = per_year_data[year]
    year_baseline_same_sic = data['baseline_same_sic_sum'] / data['baseline_same_sic_count'] if data['baseline_same_sic_count'] > 0 else 0
    year_baseline_corr = data['baseline_corr_sum'] / data['baseline_corr_count'] if data['baseline_corr_count'] > 0 else 0
    year_top1pct_same_sic = data['top1pct_same_sic_sum'] / data['top1pct_same_sic_count'] if data['top1pct_same_sic_count'] > 0 else np.nan
    year_top1pct_same_sic_corr = data['top1pct_same_sic_corr_sum'] / data['top1pct_same_sic_corr_count'] if data['top1pct_same_sic_corr_count'] > 0 else np.nan
    year_top1pct_cross_sic_corr = data['top1pct_cross_sic_corr_sum'] / data['top1pct_cross_sic_corr_count'] if data['top1pct_cross_sic_corr_count'] > 0 else np.nan
    year_enrichment = year_top1pct_same_sic / year_baseline_same_sic if year_baseline_same_sic > 0 else np.nan

    per_year_results[str(year)] = {
        'top1pct_same_sic_rate': float(year_top1pct_same_sic) if not np.isnan(year_top1pct_same_sic) else None,
        'baseline_same_sic_rate': float(year_baseline_same_sic),
        'enrichment_ratio': float(year_enrichment) if not np.isnan(year_enrichment) else None,
        'top1pct_same_sic_mean_corr': float(year_top1pct_same_sic_corr) if not np.isnan(year_top1pct_same_sic_corr) else None,
        'top1pct_cross_sic_mean_corr': float(year_top1pct_cross_sic_corr) if not np.isnan(year_top1pct_cross_sic_corr) else None,
        'baseline_mean_corr': float(year_baseline_corr),
        'n_pairs': data['baseline_same_sic_count'],
        'n_top1pct': data['top1pct_same_sic_count']
    }
    print(f"  {year}: enrichment={year_enrichment:.2f}x, same_sic_corr={year_top1pct_same_sic_corr:.4f}, cross_sic_corr={year_top1pct_cross_sic_corr:.4f}")

# Top SIC pairings
print("\nTop SIC pairings in top-1%:")
top_sic_pairings = []
for sic_pair in sorted(sic_pairings.keys(), key=lambda x: len(sic_pairings[x]), reverse=True)[:20]:
    corrs = sic_pairings[sic_pair]
    top_sic_pairings.append({
        'sic_pair': sic_pair,
        'count': len(corrs),
        'mean_corr': float(np.mean(corrs))
    })
    if len(top_sic_pairings) <= 10:
        print(f"  {sic_pair}: {len(corrs)} pairs, mean corr {np.mean(corrs):.4f}")

# Determine verdict
verdict = "FAIL"
interpretation = ""

if enrichment_ratio > 3.0:
    if not np.isnan(top1pct_same_sic_corr) and top1pct_same_sic_corr > baseline_mean_corr * 0.5:
        verdict = "PASS"
        interpretation = f"Top-1% is {enrichment_ratio:.2f}x enriched for same-SIC companies. Same-SIC pairs in top-1% have mean correlation {top1pct_same_sic_corr:.4f}, suggesting inversion is a composition effect: cosine similarity is inflated for template-similar (same-SIC) companies, and the inversion is driven by this group."
    else:
        interpretation = f"Top-1% is {enrichment_ratio:.2f}x enriched for same-SIC, but same-SIC pairs still have low correlation ({top1pct_same_sic_corr:.4f}). Template similarity doesn't explain the inversion; cosine genuinely anti-predicts at extremes."
else:
    if not np.isnan(top1pct_cross_sic_corr) and top1pct_cross_sic_corr > baseline_mean_corr * 0.5:
        interpretation = f"Top-1% is only {enrichment_ratio:.2f}x enriched for same-SIC (not >3x). Cross-SIC pairs have decent correlation ({top1pct_cross_sic_corr:.4f}). Inversion is not driven by SIC composition; cosine anti-predicts across industries."
    else:
        interpretation = f"Top-1% is only {enrichment_ratio:.2f}x enriched for same-SIC. Both same-SIC and cross-SIC pairs have low correlation. Cosine similarity genuinely anti-predicts return correlation at extremes."

# Build result
result = {
    'test': 'T02_top1pct_sic_composition',
    'overall': {
        'top1pct_same_sic_rate': float(top1pct_same_sic_rate),
        'baseline_same_sic_rate': float(baseline_same_sic_rate),
        'enrichment_ratio': float(enrichment_ratio),
        'top1pct_same_sic_mean_corr': float(top1pct_same_sic_corr) if not np.isnan(top1pct_same_sic_corr) else None,
        'top1pct_cross_sic_mean_corr': float(top1pct_cross_sic_corr) if not np.isnan(top1pct_cross_sic_corr) else None,
        'baseline_mean_corr': float(baseline_mean_corr),
        'n_pairs': baseline_same_sic_count,
        'n_top1pct': top1pct_same_sic_count
    },
    'per_year': per_year_results,
    'top_sic_pairings': top_sic_pairings,
    'verdict': verdict,
    'interpretation': interpretation
}

# Write result
with open(OUTPUT_PATH, 'w') as f:
    json.dump(result, f, indent=2)

print(f"\n{'='*80}")
print(f"VERDICT: {verdict}")
print(f"{'='*80}")
print(f"Interpretation: {interpretation}")
print(f"\nResult written to {OUTPUT_PATH}")

# Also print the full result
print(f"\n{'='*80}")
print("FULL RESULT:")
print(f"{'='*80}")
print(json.dumps(result, indent=2))

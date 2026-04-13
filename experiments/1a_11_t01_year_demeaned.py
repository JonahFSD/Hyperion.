#!/usr/bin/env python3
"""
Phase 1A Test T01: Year-Demeaned Spearman rho
Ultra-lean version: use disk-based data for bootstrap.
"""

import pandas as pd
import numpy as np
from scipy.stats import spearmanr, rankdata
import pyarrow.parquet as pq
import json
import tempfile
import os
import gc

PARQUET_PATH = 'phase1_artifacts/pairs.parquet'

def compute_year_means():
    """Pass 1: Year means"""
    print("PASS 1: Computing year-level mean correlations...")
    pf = pq.ParquetFile(PARQUET_PATH)
    year_stats = {}

    for i in range(pf.num_row_groups):
        df = pf.read_row_group(i, columns=['year', 'correlation']).to_pandas()
        print(f"  RG {i}: {len(df):,} pairs")

        for year in df['year'].unique():
            year_data = df[df['year'] == year]['correlation'].dropna().values
            if year not in year_stats:
                year_stats[year] = {'sum': 0.0, 'count': 0}
            year_stats[year]['sum'] += year_data.sum()
            year_stats[year]['count'] += len(year_data)

    year_means = {year: s['sum'] / s['count'] for year, s in year_stats.items()}
    print(f"  {len(year_means)} years: {min(year_means.keys())}-{max(year_means.keys())}\n")
    return year_means

def compute_global_spearman(year_means, tmpdir):
    """Pass 2: Global Spearman - write to disk for bootstrap"""
    print("PASS 2: Computing global Spearman rho...")
    pf = pq.ParquetFile(PARQUET_PATH)

    # Write demeaned data to CSV
    dem_file = os.path.join(tmpdir, 'demeaned.csv')
    with open(dem_file, 'w') as f:
        f.write('cs,c_dem\n')

        n_total = 0
        for i in range(pf.num_row_groups):
            df = pf.read_row_group(i, columns=['year', 'cosine_similarity', 'correlation']).to_pandas()
            print(f"  RG {i}: {len(df):,} pairs")

            # Raw
            cs_raw = df['cosine_similarity'].values
            c_raw = df['correlation'].values

            # Demeaned
            c_dem = c_raw - df['year'].map(year_means).values

            # Write non-null pairs
            for j in range(len(df)):
                if not (np.isnan(cs_raw[j]) or np.isnan(c_dem[j])):
                    f.write(f"{cs_raw[j]},{c_dem[j]}\n")
                    n_total += 1

            # Compute on-the-fly Spearman (raw)
            if i == 0:
                mask = ~(np.isnan(cs_raw) | np.isnan(c_raw))
                rho_raw_0, p_raw_0 = spearmanr(cs_raw[mask], c_raw[mask])

    print(f"  Total rows written: {n_total:,}\n")

    # Read back and compute demeaned Spearman
    print("  Computing demeaned Spearman from disk...")
    dem_df = pd.read_csv(dem_file)
    rho_dem, p_dem = spearmanr(dem_df['cs'], dem_df['c_dem'])
    print(f"    rho={rho_dem:.6f}, p={p_dem:.2e}\n")

    # Get raw rho from first RG (estimate)
    pf = pq.ParquetFile(PARQUET_PATH)
    df_first = pf.read_row_group(0, columns=['cosine_similarity', 'correlation']).to_pandas()
    cs_raw = df_first['cosine_similarity'].values
    c_raw = df_first['correlation'].values
    mask = ~(np.isnan(cs_raw) | np.isnan(c_raw))
    rho_raw, p_raw = spearmanr(cs_raw[mask], c_raw[mask])
    print(f"  Raw rho (first RG): {rho_raw:.6f}, p={p_raw:.2e}\n")

    return dem_file, rho_raw, p_raw, rho_dem, p_dem, n_total

def bootstrap_spearman(dem_file, n_boot=10000):
    """Pass 3: Bootstrap from disk"""
    print(f"PASS 3: Bootstrapping demeaned Spearman ({n_boot} resamples)...")

    # Read data once
    dem_df = pd.read_csv(dem_file)
    cs_arr = dem_df['cs'].values
    c_arr = dem_df['c_dem'].values
    n = len(cs_arr)
    print(f"  Bootstrap sample size: {n:,}\n")

    boots = []
    for i in range(n_boot):
        idx = np.random.choice(n, size=min(n, 50000), replace=True)  # Subsample for speed
        rho, _ = spearmanr(cs_arr[idx], c_arr[idx])
        boots.append(rho)

        if (i + 1) % 2000 == 0:
            print(f"  {i + 1}/{n_boot}")

    boots = np.array(boots)
    ci_lo = np.percentile(boots, 2.5)
    ci_hi = np.percentile(boots, 97.5)
    print(f"\n  CI [2.5%, 97.5%]: [{ci_lo:.6f}, {ci_hi:.6f}]\n")

    return ci_lo, ci_hi

def per_year_spearman(year_means):
    """Pass 4: Per-year Spearman"""
    print("PASS 4: Computing per-year demeaned Spearman...")
    pf = pq.ParquetFile(PARQUET_PATH)
    per_year = {}

    for i in range(pf.num_row_groups):
        df = pf.read_row_group(i, columns=['year', 'cosine_similarity', 'correlation']).to_pandas()

        for year in df['year'].unique():
            if year not in per_year:
                ydf = df[df['year'] == year]
                cs = ydf['cosine_similarity'].values
                c = ydf['correlation'].values - year_means[year]
                mask = ~(np.isnan(cs) | np.isnan(c))

                if mask.sum() >= 2:
                    rho, pval = spearmanr(cs[mask], c[mask])
                    per_year[year] = {
                        'demeaned_rho': float(rho),
                        'n_pairs': int(mask.sum()),
                        'pval': float(pval)
                    }

    print(f"  {len(per_year)} years\n")
    return per_year

def main():
    print("=" * 80)
    print("Phase 1A Test T01: Year-Demeaned Spearman rho")
    print("=" * 80 + "\n")

    with tempfile.TemporaryDirectory() as tmpdir:
        # Pass 1
        year_means = compute_year_means()

        # Pass 2
        dem_file, rho_raw, p_raw, rho_dem, p_dem, n_total = \
            compute_global_spearman(year_means, tmpdir)

        # Pass 3
        ci_lo, ci_hi = bootstrap_spearman(dem_file, n_boot=10000)
        ci_excludes_zero = (ci_lo > 0) or (ci_hi < 0)

    # Pass 4
    per_year = per_year_spearman(year_means)

    # Verdict
    print("=" * 80)
    print("VERDICT:")
    print("=" * 80)

    verdict = "FAIL"
    reason = ""

    if ci_excludes_zero:
        if rho_dem > rho_raw:
            verdict = "PASS"
            reason = f"Demeaned rho ({rho_dem:.6f}) > raw ({rho_raw:.6f}), CI excludes zero"
        elif abs(rho_dem - rho_raw) < 0.005:
            verdict = "PASS"
            reason = "Demeaned rho ≈ raw rho, CI excludes zero (no confound)"
        else:
            verdict = "PASS"
            reason = "Demeaned rho differs from raw, CI excludes zero"
    else:
        if abs(rho_dem) < 0.01:
            verdict = "FAIL"
            reason = f"Demeaned rho ≈ 0, CI includes zero (signal purely market regime)"
        else:
            verdict = "FAIL"
            reason = "Demeaned rho non-zero but CI includes zero"

    print(f"VERDICT: {verdict}")
    print(f"REASON:  {reason}\n")

    result = {
        "test": "T01_year_demeaned_spearman",
        "raw_global_rho": float(rho_raw),
        "raw_global_pval": float(p_raw),
        "demeaned_global_rho": float(rho_dem),
        "demeaned_global_pval": float(p_dem),
        "demeaned_ci_95": [float(ci_lo), float(ci_hi)],
        "ci_excludes_zero": bool(ci_excludes_zero),
        "n_bootstrap": 10000,
        "n_pairs_total": int(n_total),
        "n_years": int(len(year_means)),
        "year_range": [int(min(year_means.keys())), int(max(year_means.keys()))],
        "per_year": {str(y): {
            "demeaned_rho": float(v["demeaned_rho"]),
            "n_pairs": int(v["n_pairs"]),
            "pval": float(v["pval"])
        } for y, v in sorted(per_year.items())},
        "verdict": verdict,
        "reason": reason
    }

    outpath = 'phase1_artifacts/1a_11_t01_result.json'
    with open(outpath, 'w') as f:
        json.dump(result, f, indent=2)

    print(f"Result: {outpath}\n")
    return result

if __name__ == '__main__':
    result = main()
    print("=" * 80)
    print("FULL JSON RESULT:")
    print("=" * 80)
    print(json.dumps(result, indent=2))

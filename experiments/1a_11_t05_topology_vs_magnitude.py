#!/usr/bin/env python3
"""
Test T05: Topology vs Magnitude
Minimal memory footprint: one year at a time, no iteration.
"""

import pandas as pd
import numpy as np
import json
from pathlib import Path
import gc

np.random.seed(42)

DATA_DIR = Path(__file__).resolve().parent / "phase1_artifacts"
PAIRS_FILE = DATA_DIR / "pairs.parquet"
OUTPUT_FILE = DATA_DIR / "1a_11_t05_result.json"

print("=" * 80)
print("Test T05: Topology vs Magnitude")
print("=" * 80)

# Get all year values first (without full load)
print("\n[1/5] Reading parquet metadata...")
import pyarrow.parquet as pq
pf = pq.ParquetFile(PAIRS_FILE)
years_list = sorted(pd.read_parquet(PAIRS_FILE, columns=['year'])['year'].unique())
print(f"  Years: {years_list}")

nn_results = []
control_results = []

# Process each year separately
print("\n[2/5] Building NN graph per year...")
for year in years_list:
    print(f"  Year {year}...", end='', flush=True)
    gc.collect()

    # Load only this year
    pairs_year = pd.read_parquet(PAIRS_FILE, filters=[('year', '==', year)])
    pairs_year = pairs_year[['Company1', 'Company2', 'cosine_similarity', 'correlation']].copy()
    pairs_year = pairs_year[pairs_year['Company1'] != pairs_year['Company2']]

    # Make edge pairs (normalized)
    pairs_year['edge'] = pairs_year.apply(
        lambda r: (min(r['Company1'], r['Company2']), max(r['Company1'], r['Company2'])),
        axis=1
    )

    # Deduplicate
    pairs_year = pairs_year.drop_duplicates(subset=['edge'])

    # Build NN dict per company (vectorized)
    c1_df = pairs_year[['Company1', 'Company2', 'cosine_similarity', 'correlation']].copy()
    c1_df.columns = ['company', 'peer', 'cosine', 'corr']

    c2_df = pairs_year[['Company2', 'Company1', 'cosine_similarity', 'correlation']].copy()
    c2_df.columns = ['company', 'peer', 'cosine', 'corr']

    both = pd.concat([c1_df, c2_df], ignore_index=True)
    nn_per_co = both.loc[both.groupby('company')['cosine'].idxmax()]

    # Deduplicate edges
    nn_per_co['edge_norm'] = nn_per_co.apply(
        lambda r: (min(r['company'], r['peer']), max(r['company'], r['peer'])),
        axis=1
    )
    nn_edges = nn_per_co.drop_duplicates(subset=['edge_norm'])[['edge_norm', 'cosine', 'corr']]

    print(f" {len(nn_edges)} NN edges", end='', flush=True)

    # Match controls
    matched = 0
    for _, nn_row in nn_edges.iterrows():
        nn_cos = nn_row['cosine']
        nn_cr = nn_row['corr']
        nn_edge = nn_row['edge_norm']

        # Find control with similar cosine
        for tol in [0.01, 0.02, 0.05]:
            ctrl = pairs_year[
                (pairs_year['cosine_similarity'] >= nn_cos - tol) &
                (pairs_year['cosine_similarity'] <= nn_cos + tol) &
                (pairs_year['edge'] != nn_edge)
            ]

            if len(ctrl) > 0:
                ctrl_row = ctrl.iloc[np.random.randint(len(ctrl))]
                nn_results.append({'year': year, 'corr': float(nn_cr)})
                control_results.append({'year': year, 'corr': float(ctrl_row['correlation'])})
                matched += 1
                break

    print(f" -> {matched} matched")

    del pairs_year, c1_df, c2_df, both, nn_per_co, nn_edges
    gc.collect()

# Convert to DataFrames
print("\n[3/5] Creating result dataframes...")
nn_df = pd.DataFrame(nn_results)
ctrl_df = pd.DataFrame(control_results)

print(f"  NN pairs: {len(nn_df)}")
print(f"  Control pairs: {len(ctrl_df)}")

nn_mean = float(nn_df['corr'].mean())
ctrl_mean = float(ctrl_df['corr'].mean())
diff_mean = nn_mean - ctrl_mean

print(f"  NN mean: {nn_mean:.6f}")
print(f"  Control mean: {ctrl_mean:.6f}")

# Per-year breakdown
print(f"\n[4/5] Per-year statistics:")
per_year = {}
for year in sorted(nn_df['year'].unique()):
    nn_y = nn_df[nn_df['year'] == year]['corr']
    ctrl_y = ctrl_df[ctrl_df['year'] == year]['corr']

    per_year[int(year)] = {
        'nn_corr': float(nn_y.mean()),
        'control_corr': float(ctrl_y.mean()),
        'diff': float(nn_y.mean() - ctrl_y.mean()),
        'n': int(len(nn_y))
    }
    print(f"  {year}: NN={nn_y.mean():.6f}, Ctrl={ctrl_y.mean():.6f}, Diff={per_year[int(year)]['diff']:.6f}")

# Bootstrap CI
print(f"\n[5/5] Computing bootstrap CI (10,000 resamples)...")
nn_vals = nn_df['corr'].values
ctrl_vals = ctrl_df['corr'].values

diffs = []
for i in range(10000):
    if (i + 1) % 1000 == 0:
        print(f"  {i + 1}/10000")
    diffs.append(
        np.random.choice(nn_vals, len(nn_vals), replace=True).mean() -
        np.random.choice(ctrl_vals, len(ctrl_vals), replace=True).mean()
    )

diffs = np.array(diffs)
ci = np.percentile(diffs, [2.5, 97.5])
ci_excl = (ci[0] > 0) or (ci[1] < 0)
verdict = "PASS" if ci_excl else "FAIL"

interpretation = (
    f"NN pairs show {'significantly higher' if ci_excl else 'similar'} return correlation "
    f"vs magnitude-matched controls. "
    f"NN r={nn_mean:.6f}, Control r={ctrl_mean:.6f}, diff={diff_mean:.6f}, "
    f"95% CI=[{ci[0]:.6f}, {ci[1]:.6f}]. "
    f"Graph topology {'ADDS predictive value' if ci_excl else 'does NOT add value'} beyond cosine magnitude."
)

result = {
    "test": "T05_topology_vs_magnitude",
    "overall": {
        "nn_mean_corr": nn_mean,
        "control_mean_corr": ctrl_mean,
        "difference": diff_mean,
        "ci_95": [float(ci[0]), float(ci[1])],
        "ci_excludes_zero": bool(ci_excl),
        "n_nn_pairs": int(len(nn_df)),
        "n_matched": int(len(nn_df))
    },
    "per_year": per_year,
    "verdict": verdict,
    "interpretation": interpretation
}

with open(OUTPUT_FILE, 'w') as f:
    json.dump(result, f, indent=2)

print("\n" + "=" * 80)
print(f"VERDICT: {verdict}")
print("=" * 80)
print(f"\nResult written to: {OUTPUT_FILE}")
print("\nFull result:")
print(json.dumps(result, indent=2))

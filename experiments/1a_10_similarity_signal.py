"""
1a_10_similarity_signal.py
Phase 1A diagnostic: Does SAE cosine similarity predict return correlation?

The most direct test of whether SAE features carry useful signal,
independent of any clustering method.
"""

import json
import numpy as np
import pandas as pd
from scipy import stats

OUT = "phase1_artifacts/1a_similarity_signal.json"

# ── 1. Load data ──────────────────────────────────────────────────────────────
print("=" * 70)
print("LOADING DATA")
print("=" * 70)

pairs = pd.read_parquet("phase1_artifacts/pairs.parquet")
print(f"pairs.parquet columns: {list(pairs.columns)}")
print(f"pairs shape: {pairs.shape}")
print(f"years: {sorted(pairs['year'].unique())}")
print()

companies = pd.read_parquet(
    "phase1_artifacts/companies.parquet",
    columns=["cik", "year", "sic_code"],
)
print(f"companies.parquet columns (subset): {list(companies.columns)}")
print(f"companies shape: {companies.shape}")
print()

# ── 2. Identify columns ──────────────────────────────────────────────────────
cos_col = "cosine_similarity"
corr_col = "correlation"
print(f"Using cosine similarity column: {cos_col}")
print(f"Using return correlation column: {corr_col}")
print(f"  cos_sim range: [{pairs[cos_col].min():.4f}, {pairs[cos_col].max():.4f}]")
print(f"  correlation range: [{pairs[corr_col].min():.4f}, {pairs[corr_col].max():.4f}]")
print()

# Drop rows with NaN in key columns
n_before = len(pairs)
pairs = pairs.dropna(subset=[cos_col, corr_col])
n_after = len(pairs)
if n_before != n_after:
    print(f"Dropped {n_before - n_after} rows with NaN in key columns")
    print()

results = {}

# ── 3. OVERALL SIGNAL TEST ────────────────────────────────────────────────────
print("=" * 70)
print("OVERALL SIGNAL TEST")
print("=" * 70)

rho, pval = stats.spearmanr(pairs[cos_col], pairs[corr_col])
print(f"Spearman rho:  {rho:.6f}")
print(f"p-value:       {pval:.2e}")
print(f"N pairs:       {len(pairs):,}")
print()

results["overall"] = {
    "spearman_rho": round(float(rho), 6),
    "p_value": float(pval),
    "n_pairs": int(len(pairs)),
}

# ── 4. PER-YEAR SIGNAL TEST ──────────────────────────────────────────────────
print("=" * 70)
print("PER-YEAR SIGNAL TEST")
print("=" * 70)

# Build SIC lookup: __index_level_0__ -> sic_2digit
# Company1/Company2 in pairs are __index_level_0__ from companies, NOT CIKs
companies_full = pd.read_parquet(
    "phase1_artifacts/companies.parquet",
    columns=["__index_level_0__", "sic_code"],
)
companies_sic = companies_full.dropna(subset=["sic_code"])
# Take last occurrence per __index_level_0__ (most recent year)
companies_sic = companies_sic.drop_duplicates("__index_level_0__", keep="last")
sic_map = dict(
    zip(
        companies_sic["__index_level_0__"],
        companies_sic["sic_code"].astype(str).str[:2],
    )
)
print(f"SIC map covers {len(sic_map)} unique company IDs")

# Add SIC 2-digit to pairs
pairs["sic2_1"] = pairs["Company1"].map(sic_map)
pairs["sic2_2"] = pairs["Company2"].map(sic_map)
pairs["same_sic2"] = pairs["sic2_1"] == pairs["sic2_2"]
has_sic = pairs["sic2_1"].notna() & pairs["sic2_2"].notna()
print(f"Pairs with SIC for both companies: {has_sic.sum():,} / {len(pairs):,}")
print()

yearly_results = {}
years = sorted(pairs["year"].unique())

print(f"{'Year':>6} | {'All rho':>10} {'N':>10} | {'Same SIC':>10} {'N':>8} | {'Diff SIC':>10} {'N':>10}")
print("-" * 85)

for yr in years:
    mask_yr = pairs["year"] == yr
    subset = pairs[mask_yr]

    rho_all, p_all = stats.spearmanr(subset[cos_col], subset[corr_col])

    # Same SIC
    same = subset[subset["same_sic2"] & has_sic[mask_yr]]
    if len(same) > 10:
        rho_same, p_same = stats.spearmanr(same[cos_col], same[corr_col])
    else:
        rho_same, p_same = float("nan"), float("nan")

    # Different SIC
    diff = subset[~subset["same_sic2"] & has_sic[mask_yr]]
    if len(diff) > 10:
        rho_diff, p_diff = stats.spearmanr(diff[cos_col], diff[corr_col])
    else:
        rho_diff, p_diff = float("nan"), float("nan")

    print(f"{yr:>6} | {rho_all:>10.4f} {len(subset):>10,} | {rho_same:>10.4f} {len(same):>8,} | {rho_diff:>10.4f} {len(diff):>10,}")

    yearly_results[int(yr)] = {
        "all_rho": round(float(rho_all), 6),
        "all_p": float(p_all),
        "all_n": int(len(subset)),
        "same_sic_rho": round(float(rho_same), 6) if not np.isnan(rho_same) else None,
        "same_sic_n": int(len(same)),
        "diff_sic_rho": round(float(rho_diff), 6) if not np.isnan(rho_diff) else None,
        "diff_sic_n": int(len(diff)),
    }

results["per_year"] = yearly_results
print()

# ── 5. BINNED ANALYSIS ───────────────────────────────────────────────────────
print("=" * 70)
print("BINNED ANALYSIS (20 ventiles overall)")
print("=" * 70)

# Overall ventiles
pairs["ventile"] = pd.qcut(pairs[cos_col], 20, labels=False, duplicates="drop")
binned = pairs.groupby("ventile").agg(
    mean_cos=(cos_col, "mean"),
    mean_corr=(corr_col, "mean"),
    count=(corr_col, "count"),
).reset_index()

print(f"{'Bin':>4} | {'Mean CosSim':>12} | {'Mean RetCorr':>12} | {'Count':>10}")
print("-" * 50)
prev_corr = -999
monotonic = True
for _, row in binned.iterrows():
    print(f"{int(row['ventile']):>4} | {row['mean_cos']:>12.6f} | {row['mean_corr']:>12.6f} | {int(row['count']):>10,}")
    if row["mean_corr"] < prev_corr:
        monotonic = False
    prev_corr = row["mean_corr"]

print(f"\nMonotonically increasing: {monotonic}")

# Check approximate monotonicity (allow 1-2 violations)
corr_values = binned["mean_corr"].values
n_violations = sum(1 for i in range(1, len(corr_values)) if corr_values[i] < corr_values[i - 1])
approx_monotonic = n_violations <= 2
print(f"Monotonicity violations: {n_violations} / {len(corr_values) - 1}")
print(f"Approximately monotonic (<=2 violations): {approx_monotonic}")
print()

results["binned_overall"] = {
    "bins": [
        {
            "ventile": int(row["ventile"]),
            "mean_cos_sim": round(float(row["mean_cos"]), 6),
            "mean_ret_corr": round(float(row["mean_corr"]), 6),
            "count": int(row["count"]),
        }
        for _, row in binned.iterrows()
    ],
    "strictly_monotonic": bool(monotonic),
    "monotonicity_violations": int(n_violations),
    "approx_monotonic": bool(approx_monotonic),
}

# Per-year deciles
print("=" * 70)
print("BINNED ANALYSIS (10 deciles per year)")
print("=" * 70)

yearly_binned = {}
for yr in years:
    subset = pairs[pairs["year"] == yr].copy()
    subset["decile"] = pd.qcut(subset[cos_col], 10, labels=False, duplicates="drop")
    yb = subset.groupby("decile").agg(
        mean_cos=(cos_col, "mean"),
        mean_corr=(corr_col, "mean"),
        count=(corr_col, "count"),
    ).reset_index()

    corr_vals = yb["mean_corr"].values
    yr_violations = sum(1 for i in range(1, len(corr_vals)) if corr_vals[i] < corr_vals[i - 1])

    yearly_binned[int(yr)] = {
        "bins": [
            {
                "decile": int(row["decile"]),
                "mean_cos_sim": round(float(row["mean_cos"]), 6),
                "mean_ret_corr": round(float(row["mean_corr"]), 6),
                "count": int(row["count"]),
            }
            for _, row in yb.iterrows()
        ],
        "monotonicity_violations": int(yr_violations),
    }

# Print summary table for per-year binned
print(f"{'Year':>6} | {'D0 corr':>8} | {'D9 corr':>8} | {'Spread':>8} | {'Violations':>10}")
print("-" * 55)
for yr in years:
    yb = yearly_binned[int(yr)]
    d0 = yb["bins"][0]["mean_ret_corr"]
    d9 = yb["bins"][-1]["mean_ret_corr"]
    spread = d9 - d0
    print(f"{yr:>6} | {d0:>8.4f} | {d9:>8.4f} | {spread:>8.4f} | {yb['monotonicity_violations']:>10}")

results["binned_per_year"] = yearly_binned
print()

# ── 6. COMPARISON TO RANDOM ──────────────────────────────────────────────────
print("=" * 70)
print("COMPARISON TO RANDOM (top-k% vs baseline)")
print("=" * 70)

baseline_corr = pairs[corr_col].mean()
n_total = len(pairs)

top_pcts = [1, 5, 10]
comparison = {"baseline_mean_corr": round(float(baseline_corr), 6), "n_total": int(n_total)}

print(f"Baseline (all pairs) mean return correlation: {baseline_corr:.6f}")
print(f"N total pairs: {n_total:,}")
print()

# Sort by cosine similarity descending for top-k
cos_sorted = pairs[cos_col].values
corr_sorted_by_cos = pairs[corr_col].values[np.argsort(-cos_sorted)]
cos_sorted_vals = np.sort(cos_sorted)[::-1]

for pct in top_pcts:
    k = max(1, int(n_total * pct / 100))
    top_corr = corr_sorted_by_cos[:k].mean()
    top_cos_threshold = cos_sorted_vals[k - 1]
    lift = top_corr - baseline_corr

    print(f"Top {pct:>2}%: N={k:>10,}  mean_cos>={top_cos_threshold:.4f}  mean_ret_corr={top_corr:.6f}  lift={lift:+.6f}")
    comparison[f"top_{pct}pct"] = {
        "n": int(k),
        "cos_threshold": round(float(top_cos_threshold), 4),
        "mean_ret_corr": round(float(top_corr), 6),
        "lift_vs_baseline": round(float(lift), 6),
    }

# Also bottom percentiles for contrast
print()
for pct in top_pcts:
    k = max(1, int(n_total * pct / 100))
    bottom_corr = corr_sorted_by_cos[-k:].mean()
    bottom_cos_threshold = cos_sorted_vals[-(k)]
    lift = bottom_corr - baseline_corr

    print(f"Bot {pct:>2}%: N={k:>10,}  mean_cos<={bottom_cos_threshold:.4f}  mean_ret_corr={bottom_corr:.6f}  lift={lift:+.6f}")
    comparison[f"bottom_{pct}pct"] = {
        "n": int(k),
        "cos_threshold": round(float(bottom_cos_threshold), 4),
        "mean_ret_corr": round(float(bottom_corr), 6),
        "lift_vs_baseline": round(float(lift), 6),
    }

results["comparison_to_random"] = comparison
print()

# ── 7. Save results ──────────────────────────────────────────────────────────
with open(OUT, "w") as f:
    json.dump(results, f, indent=2)
print(f"Results saved to {OUT}")
print()

# ── 8. SUMMARY ────────────────────────────────────────────────────────────────
print("=" * 70)
print("SUMMARY")
print("=" * 70)

print(f"\n1. OVERALL SIGNAL:")
print(f"   Spearman rho = {results['overall']['spearman_rho']:.6f}  (p = {results['overall']['p_value']:.2e})")
print(f"   N = {results['overall']['n_pairs']:,} pairs")

print(f"\n2. TOP-K% vs BASELINE:")
print(f"   Baseline mean ret corr:  {comparison['baseline_mean_corr']:.6f}")
for pct in top_pcts:
    t = comparison[f"top_{pct}pct"]
    print(f"   Top {pct:>2}% mean ret corr:  {t['mean_ret_corr']:.6f}  (lift: {t['lift_vs_baseline']:+.6f})")

print(f"\n3. BINNED MONOTONICITY:")
print(f"   Strictly monotonic: {results['binned_overall']['strictly_monotonic']}")
print(f"   Violations: {results['binned_overall']['monotonicity_violations']}/19")
print(f"   Approx monotonic (<=2 violations): {results['binned_overall']['approx_monotonic']}")

all_rhos = [v["all_rho"] for v in results["per_year"].values()]
same_rhos = [v["same_sic_rho"] for v in results["per_year"].values() if v["same_sic_rho"] is not None]
diff_rhos = [v["diff_sic_rho"] for v in results["per_year"].values() if v["diff_sic_rho"] is not None]

print(f"\n4. PER-YEAR SPEARMAN RHOS:")
print(f"   All pairs:     min={min(all_rhos):.4f}  max={max(all_rhos):.4f}  mean={np.mean(all_rhos):.4f}")
if same_rhos:
    print(f"   Same SIC 2dig:  min={min(same_rhos):.4f}  max={max(same_rhos):.4f}  mean={np.mean(same_rhos):.4f}")
if diff_rhos:
    print(f"   Diff SIC 2dig:  min={min(diff_rhos):.4f}  max={max(diff_rhos):.4f}  mean={np.mean(diff_rhos):.4f}")

# Interpretation
print(f"\n5. INTERPRETATION:")
if abs(rho) < 0.02:
    print("   ⚠ Overall Spearman rho is near zero — SAE cosine similarity has")
    print("     negligible linear rank relationship with return correlation.")
elif rho > 0.02:
    print(f"   ✓ Positive Spearman rho ({rho:.4f}) — SAE cosine similarity")
    print("     has some predictive signal for return correlation.")
else:
    print(f"   ✗ Negative Spearman rho ({rho:.4f}) — SAE cosine similarity")
    print("     is inversely related to return correlation.")

if same_rhos and diff_rhos:
    mean_same = np.mean(same_rhos)
    mean_diff = np.mean(diff_rhos)
    if mean_diff > 0.01:
        print(f"   ✓ Signal persists within different-SIC pairs (mean rho={mean_diff:.4f})")
        print("     → SAE captures similarity BEYOND industry classification.")
    else:
        print(f"   ⚠ Signal within different-SIC pairs is weak (mean rho={mean_diff:.4f})")
        print("     → SAE signal may be mostly industry membership.")

print()

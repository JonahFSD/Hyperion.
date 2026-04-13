"""
1a_09_cluster_size_control.py — Cluster Size Confound Diagnostic

The SAE advantage might be partly mechanical: SAE clusters shrink over time
(2.16 → 1.48 companies/cluster) while SIC grows (2.71 → 4.34) and SBERT
grows (14.1 → 29.1). Smaller clusters mechanically produce higher MC because
they average fewer, tighter pairs. This script controls for that.

Three analyses:
  1. Pair-count weighted MC: weight each cluster's MC by its number of pairs
     instead of equal-weighting. If SAE still wins, the advantage isn't just
     from having tiny clusters.
  2. Size-stratified MC: bin clusters by size, compare methods within each bin.
     Shows whether SAE wins at every scale or only in the small-cluster regime.
  3. Temporal trend with pair-weighted MC: does the doubling survive?

Inputs:
  phase1_artifacts/pairs.parquet
  phase1_artifacts/clusters/*.pkl

Outputs:
  phase1_artifacts/1a_cluster_size_control.json
  phase1_artifacts/1a_report_09.md
"""

import json
import math
import os
import pickle
import sys

import numpy as np
import pandas as pd

ARTIFACTS = "phase1_artifacts"
CLUSTERS_DIR = os.path.join(ARTIFACTS, "clusters")

# Only the three methods that matter for delta tests
METHODS = ["sae_cd", "sic", "sbert"]

# Size bins: [2, 3-5, 6-15, 16-50, 51+]
SIZE_BINS = [
    (2, 2, "2"),
    (3, 5, "3-5"),
    (6, 15, "6-15"),
    (16, 50, "16-50"),
    (51, 99999, "51+"),
]


def load_pairs():
    path = os.path.join(ARTIFACTS, "pairs.parquet")
    print(f"  Loading {path}...")
    df = pd.read_parquet(path)
    df["year"] = df["year"].astype(int)
    return df


def load_clusters(method):
    path = os.path.join(CLUSTERS_DIR, f"{method}.pkl")
    with open(path, "rb") as f:
        df = pickle.load(f)
    df["year"] = df["year"].astype(int)
    return df


def compute_mc_with_sizes(pairs_df, cluster_df):
    """
    Compute MC three ways:
      1. Equal-weighted (original, for verification)
      2. Pair-count weighted (main diagnostic)
      3. Size-stratified (per size bin)

    Returns dict with all results per year.
    """
    pairs_by_year = {yr: grp for yr, grp in pairs_df.groupby("year")}

    yearly = {}

    for _, row in cluster_df.iterrows():
        year = int(row["year"])
        clusters = row["clusters"]
        year_pairs = pairs_by_year.get(year)

        if year_pairs is None:
            continue

        # Collect per-cluster stats
        cluster_stats = []  # (cluster_size, n_pairs, mean_corr)

        for cluster_id, members in clusters.items():
            if len(members) <= 1:
                continue

            members_set = set(members)
            mask = (
                year_pairs["Company1"].isin(members_set)
                & year_pairs["Company2"].isin(members_set)
            )
            cluster_pairs = year_pairs.loc[mask, "correlation"]

            if len(cluster_pairs) > 0:
                cluster_stats.append((
                    len(members),
                    len(cluster_pairs),
                    float(cluster_pairs.mean()),
                ))

        if not cluster_stats:
            continue

        sizes = np.array([s[0] for s in cluster_stats])
        n_pairs = np.array([s[1] for s in cluster_stats])
        means = np.array([s[2] for s in cluster_stats])

        # 1. Equal-weighted MC (original method)
        equal_mc = float(means.mean())

        # 2. Pair-count weighted MC
        total_pairs = n_pairs.sum()
        if total_pairs > 0:
            pair_weighted_mc = float((means * n_pairs).sum() / total_pairs)
        else:
            pair_weighted_mc = float("nan")

        # 3. Size-stratified MC
        size_strat = {}
        for lo, hi, label in SIZE_BINS:
            mask = (sizes >= lo) & (sizes <= hi)
            if mask.sum() > 0:
                bin_means = means[mask]
                bin_n_pairs = n_pairs[mask]
                size_strat[label] = {
                    "n_clusters": int(mask.sum()),
                    "n_pairs": int(bin_n_pairs.sum()),
                    "equal_mc": float(bin_means.mean()),
                    "pair_weighted_mc": float(
                        (bin_means * bin_n_pairs).sum() / bin_n_pairs.sum()
                    ) if bin_n_pairs.sum() > 0 else None,
                    "avg_cluster_size": float(sizes[mask].mean()),
                }

        yearly[year] = {
            "equal_mc": equal_mc,
            "pair_weighted_mc": pair_weighted_mc,
            "n_clusters_with_pairs": len(cluster_stats),
            "total_pairs": int(total_pairs),
            "avg_cluster_size": float(sizes.mean()),
            "median_cluster_size": float(np.median(sizes)),
            "size_stratified": size_strat,
            "cluster_size_distribution": {
                "pct_size_2": float((sizes == 2).sum() / len(sizes)),
                "pct_size_3_5": float(((sizes >= 3) & (sizes <= 5)).sum() / len(sizes)),
                "pct_size_6_plus": float((sizes >= 6).sum() / len(sizes)),
            },
        }

    return yearly


def ols_slope(x, y):
    """Simple OLS slope + intercept."""
    x = np.array(x, dtype=float)
    y = np.array(y, dtype=float)
    n = len(x)
    x_bar = x.mean()
    y_bar = y.mean()
    ss_xy = ((x - x_bar) * (y - y_bar)).sum()
    ss_xx = ((x - x_bar) ** 2).sum()
    if ss_xx == 0:
        return 0.0, y_bar
    slope = ss_xy / ss_xx
    intercept = y_bar - slope * x_bar
    return float(slope), float(intercept)


def bootstrap_slope_ci(x, y, n_boot=10000, seed=42, alpha=0.05):
    """Bootstrap CI on OLS slope."""
    rng = np.random.RandomState(seed)
    x = np.array(x, dtype=float)
    y = np.array(y, dtype=float)
    n = len(x)
    slopes = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.choice(n, size=n, replace=True)
        slopes[b] = ols_slope(x[idx], y[idx])[0]
    lo = float(np.percentile(slopes, 100 * alpha / 2))
    hi = float(np.percentile(slopes, 100 * (1 - alpha / 2)))
    return lo, hi


def main():
    print("=" * 70)
    print("1a_09: Cluster Size Confound Diagnostic")
    print("=" * 70)

    pairs_df = load_pairs()

    results = {}
    for method in METHODS:
        print(f"\nProcessing {method}...")
        cluster_df = load_clusters(method)
        yearly = compute_mc_with_sizes(pairs_df, cluster_df)
        results[method] = yearly

        # Quick summary
        years_sorted = sorted(yearly.keys())
        if years_sorted:
            first, last = years_sorted[0], years_sorted[-1]
            print(f"  {first}: equal={yearly[first]['equal_mc']:.4f}, "
                  f"pair_wt={yearly[first]['pair_weighted_mc']:.4f}, "
                  f"avg_size={yearly[first]['avg_cluster_size']:.2f}")
            print(f"  {last}: equal={yearly[last]['equal_mc']:.4f}, "
                  f"pair_wt={yearly[last]['pair_weighted_mc']:.4f}, "
                  f"avg_size={yearly[last]['avg_cluster_size']:.2f}")

    # ── Analysis 1: Overall pair-weighted MC ──
    print("\n" + "=" * 70)
    print("ANALYSIS 1: Pair-Count Weighted MC (overall)")
    print("=" * 70)

    # Overlapping years for all three methods
    common_years = sorted(
        set(results["sae_cd"].keys())
        & set(results["sic"].keys())
        & set(results["sbert"].keys())
    )

    overall = {}
    for method in METHODS:
        eq_vals = [results[method][y]["equal_mc"] for y in common_years]
        pw_vals = [results[method][y]["pair_weighted_mc"] for y in common_years]
        overall[method] = {
            "equal_mc": float(np.mean(eq_vals)),
            "pair_weighted_mc": float(np.mean(pw_vals)),
        }
        print(f"  {method:>8}: equal={overall[method]['equal_mc']:.4f}, "
              f"pair_weighted={overall[method]['pair_weighted_mc']:.4f}")

    # Deltas
    print("\n  --- Deltas (SAE minus baseline) ---")
    for baseline in ["sic", "sbert"]:
        eq_delta = overall["sae_cd"]["equal_mc"] - overall[baseline]["equal_mc"]
        pw_delta = overall["sae_cd"]["pair_weighted_mc"] - overall[baseline]["pair_weighted_mc"]
        pct_change = (pw_delta - eq_delta) / eq_delta * 100 if eq_delta != 0 else float("nan")
        print(f"  SAE-{baseline.upper():>5}: equal_delta={eq_delta:+.4f}, "
              f"pair_wt_delta={pw_delta:+.4f}, "
              f"change={pct_change:+.1f}%")

    # ── Analysis 2: Year-by-year pair-weighted deltas + trend ──
    print("\n" + "=" * 70)
    print("ANALYSIS 2: Temporal Trend (pair-weighted)")
    print("=" * 70)

    trend_results = {}
    for baseline in ["sic", "sbert"]:
        label = f"sae_minus_{baseline}"

        eq_deltas = [
            results["sae_cd"][y]["equal_mc"] - results[baseline][y]["equal_mc"]
            for y in common_years
        ]
        pw_deltas = [
            results["sae_cd"][y]["pair_weighted_mc"] - results[baseline][y]["pair_weighted_mc"]
            for y in common_years
        ]

        eq_slope, eq_int = ols_slope(common_years, eq_deltas)
        pw_slope, pw_int = ols_slope(common_years, pw_deltas)

        eq_ci = bootstrap_slope_ci(common_years, eq_deltas)
        pw_ci = bootstrap_slope_ci(common_years, pw_deltas)

        trend_results[label] = {
            "equal_slope": eq_slope,
            "equal_ci": list(eq_ci),
            "equal_ci_excludes_zero": not (eq_ci[0] <= 0 <= eq_ci[1]),
            "pair_weighted_slope": pw_slope,
            "pair_weighted_ci": list(pw_ci),
            "pw_ci_excludes_zero": not (pw_ci[0] <= 0 <= pw_ci[1]),
        }

        print(f"\n  {label}:")
        print(f"    Equal-wt slope:    {eq_slope:+.6f}  CI=[{eq_ci[0]:.6f}, {eq_ci[1]:.6f}]  "
              f"{'EXCLUDES' if trend_results[label]['equal_ci_excludes_zero'] else 'INCLUDES'} zero")
        print(f"    Pair-wt slope:     {pw_slope:+.6f}  CI=[{pw_ci[0]:.6f}, {pw_ci[1]:.6f}]  "
              f"{'EXCLUDES' if trend_results[label]['pw_ci_excludes_zero'] else 'INCLUDES'} zero")

    # ── Analysis 3: Size-stratified comparison ──
    print("\n" + "=" * 70)
    print("ANALYSIS 3: Size-Stratified MC")
    print("=" * 70)

    # Aggregate size-stratified MC across years
    strat_agg = {}
    for method in METHODS:
        strat_agg[method] = {}
        for lo, hi, label in SIZE_BINS:
            bin_mcs = []
            bin_n_pairs = []
            bin_n_clusters = []
            for y in common_years:
                yd = results[method][y].get("size_stratified", {})
                if label in yd:
                    bin_mcs.append(yd[label]["equal_mc"])
                    bin_n_pairs.append(yd[label]["n_pairs"])
                    bin_n_clusters.append(yd[label]["n_clusters"])
            if bin_mcs:
                strat_agg[method][label] = {
                    "mean_mc": float(np.mean(bin_mcs)),
                    "n_year_bins": len(bin_mcs),
                    "total_pairs": sum(bin_n_pairs),
                    "total_clusters": sum(bin_n_clusters),
                }

    print(f"\n  {'Size Bin':>8} | {'SAE MC':>8} {'(n_cl)':>8} | "
          f"{'SIC MC':>8} {'(n_cl)':>8} | {'SBERT MC':>8} {'(n_cl)':>8} | "
          f"{'SAE-SIC':>8} {'SAE-SBERT':>10}")
    print("  " + "-" * 95)

    for lo, hi, label in SIZE_BINS:
        parts = []
        vals = {}
        for method in METHODS:
            if label in strat_agg[method]:
                d = strat_agg[method][label]
                vals[method] = d["mean_mc"]
                parts.append(f"{d['mean_mc']:>8.4f} {d['total_clusters']:>7d} ")
            else:
                vals[method] = None
                parts.append(f"{'--':>8} {'--':>8} ")

        sae_sic = (vals["sae_cd"] - vals["sic"]) if vals.get("sae_cd") and vals.get("sic") else None
        sae_sbert = (vals["sae_cd"] - vals["sbert"]) if vals.get("sae_cd") and vals.get("sbert") else None

        delta_str1 = f"{sae_sic:+8.4f}" if sae_sic is not None else f"{'--':>8}"
        delta_str2 = f"{sae_sbert:+10.4f}" if sae_sbert is not None else f"{'--':>10}"

        print(f"  {label:>8} | {'|'.join(parts)}| {delta_str1} {delta_str2}")

    # ── Analysis 4: Cluster size distribution over time ──
    print("\n" + "=" * 70)
    print("ANALYSIS 4: Cluster Size Distribution Over Time")
    print("=" * 70)

    print(f"\n  {'Year':>6} | SAE: {'%sz2':>5} {'%3-5':>5} {'%6+':>5} {'med':>4} | "
          f"SIC: {'%sz2':>5} {'%3-5':>5} {'%6+':>5} {'med':>4} | "
          f"SBERT: {'%sz2':>5} {'%3-5':>5} {'%6+':>5} {'med':>4}")
    print("  " + "-" * 100)

    for y in common_years[::5]:  # Every 5 years for readability
        parts = []
        for method in METHODS:
            d = results[method][y]["cluster_size_distribution"]
            med = results[method][y]["median_cluster_size"]
            parts.append(f"{d['pct_size_2']*100:>5.1f} {d['pct_size_3_5']*100:>5.1f} "
                        f"{d['pct_size_6_plus']*100:>5.1f} {med:>4.0f}")
        print(f"  {y:>6} | {' | '.join(parts)}")
    # Also print 2020
    y = 2020
    if y in common_years:
        parts = []
        for method in METHODS:
            d = results[method][y]["cluster_size_distribution"]
            med = results[method][y]["median_cluster_size"]
            parts.append(f"{d['pct_size_2']*100:>5.1f} {d['pct_size_3_5']*100:>5.1f} "
                        f"{d['pct_size_6_plus']*100:>5.1f} {med:>4.0f}")
        print(f"  {y:>6} | {' | '.join(parts)}")

    # ── Verdict ──
    print("\n" + "=" * 70)
    print("VERDICT")
    print("=" * 70)

    # Key question: does SAE pair-weighted MC still beat SIC and SBERT?
    pw_sae = overall["sae_cd"]["pair_weighted_mc"]
    pw_sic = overall["sic"]["pair_weighted_mc"]
    pw_sbert = overall["sbert"]["pair_weighted_mc"]

    sae_wins_sic = pw_sae > pw_sic
    sae_wins_sbert = pw_sae > pw_sbert
    pw_delta_sic = pw_sae - pw_sic
    pw_delta_sbert = pw_sae - pw_sbert
    eq_delta_sic = overall["sae_cd"]["equal_mc"] - overall["sic"]["equal_mc"]
    eq_delta_sbert = overall["sae_cd"]["equal_mc"] - overall["sbert"]["equal_mc"]

    # How much of the advantage survives?
    survival_sic = pw_delta_sic / eq_delta_sic * 100 if eq_delta_sic != 0 else float("nan")
    survival_sbert = pw_delta_sbert / eq_delta_sbert * 100 if eq_delta_sbert != 0 else float("nan")

    print(f"\n  SAE pair-weighted MC > SIC:   {'YES' if sae_wins_sic else 'NO'} "
          f"(delta={pw_delta_sic:+.4f}, {survival_sic:.0f}% of equal-wt advantage survives)")
    print(f"  SAE pair-weighted MC > SBERT: {'YES' if sae_wins_sbert else 'NO'} "
          f"(delta={pw_delta_sbert:+.4f}, {survival_sbert:.0f}% of equal-wt advantage survives)")

    # Temporal trend survival
    for baseline in ["sic", "sbert"]:
        label = f"sae_minus_{baseline}"
        tr = trend_results[label]
        eq_sig = tr["equal_ci_excludes_zero"]
        pw_sig = tr["pw_ci_excludes_zero"]
        if eq_sig and pw_sig:
            print(f"\n  Temporal trend ({label}): SURVIVES pair-weighting")
        elif eq_sig and not pw_sig:
            print(f"\n  Temporal trend ({label}): KILLED by pair-weighting — "
                  f"cluster size confound explains the trend")
        else:
            print(f"\n  Temporal trend ({label}): was not significant either way")

    # ── Save results ──
    output = {
        "overall": overall,
        "trend_results": trend_results,
        "size_stratified_aggregate": strat_agg,
        "yearly_detail": {
            method: {
                str(y): {
                    "equal_mc": d["equal_mc"],
                    "pair_weighted_mc": d["pair_weighted_mc"],
                    "n_clusters_with_pairs": d["n_clusters_with_pairs"],
                    "total_pairs": d["total_pairs"],
                    "avg_cluster_size": d["avg_cluster_size"],
                    "median_cluster_size": d["median_cluster_size"],
                    "cluster_size_distribution": d["cluster_size_distribution"],
                }
                for y, d in sorted(results[method].items())
            }
            for method in METHODS
        },
        "common_years": common_years,
        "verdict": {
            "sae_pw_beats_sic": sae_wins_sic,
            "sae_pw_beats_sbert": sae_wins_sbert,
            "pw_delta_sic": round(pw_delta_sic, 6),
            "pw_delta_sbert": round(pw_delta_sbert, 6),
            "pct_advantage_surviving_sic": round(survival_sic, 1),
            "pct_advantage_surviving_sbert": round(survival_sbert, 1),
            "temporal_trend_sic_survives": trend_results["sae_minus_sic"]["pw_ci_excludes_zero"],
            "temporal_trend_sbert_survives": trend_results["sae_minus_sbert"]["pw_ci_excludes_zero"],
        },
    }

    output_path = os.path.join(ARTIFACTS, "1a_cluster_size_control.json")
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved: {output_path}")

    # ── Write report section ──
    report_lines = []
    report_lines.append("## Step 9: Cluster Size Confound Diagnostic")
    report_lines.append("")
    report_lines.append("### Motivation")
    report_lines.append("")
    report_lines.append("SAE clusters shrink over time (2.16 → 1.48 companies/cluster) while")
    report_lines.append("SIC grows (2.71 → 4.34) and SBERT grows (14.1 → 29.1). Smaller clusters")
    report_lines.append("mechanically produce higher MC. This diagnostic controls for cluster size")
    report_lines.append("by weighting each cluster's MC by its pair count instead of equal-weighting.")
    report_lines.append("")
    report_lines.append("### Results")
    report_lines.append("")
    report_lines.append("**Overall MC (common years only)**")
    report_lines.append("")
    report_lines.append(f"| Method | Equal-Wt MC | Pair-Wt MC | Change |")
    report_lines.append(f"|--------|------------|------------|--------|")
    for method in METHODS:
        eq = overall[method]["equal_mc"]
        pw = overall[method]["pair_weighted_mc"]
        chg = pw - eq
        report_lines.append(f"| {method} | {eq:.4f} | {pw:.4f} | {chg:+.4f} |")
    report_lines.append("")

    report_lines.append("**Deltas (pair-weighted)**")
    report_lines.append("")
    report_lines.append(f"| Comparison | Equal-Wt Delta | Pair-Wt Delta | % Surviving |")
    report_lines.append(f"|------------|---------------|---------------|-------------|")
    report_lines.append(f"| SAE - SIC | {eq_delta_sic:+.4f} | {pw_delta_sic:+.4f} | {survival_sic:.0f}% |")
    report_lines.append(f"| SAE - SBERT | {eq_delta_sbert:+.4f} | {pw_delta_sbert:+.4f} | {survival_sbert:.0f}% |")
    report_lines.append("")

    report_lines.append("**Temporal Trend**")
    report_lines.append("")
    for baseline in ["sic", "sbert"]:
        label = f"sae_minus_{baseline}"
        tr = trend_results[label]
        report_lines.append(f"- {label}: equal-wt slope={tr['equal_slope']:+.6f} "
                          f"(CI [{tr['equal_ci'][0]:.6f}, {tr['equal_ci'][1]:.6f}]), "
                          f"pair-wt slope={tr['pair_weighted_slope']:+.6f} "
                          f"(CI [{tr['pair_weighted_ci'][0]:.6f}, {tr['pair_weighted_ci'][1]:.6f}])")
    report_lines.append("")

    report_path = os.path.join(ARTIFACTS, "1a_report_09.md")
    with open(report_path, "w") as f:
        f.write("\n".join(report_lines))
    print(f"Saved: {report_path}")

    print("\nAll checks completed.")
    sys.exit(0)


if __name__ == "__main__":
    main()

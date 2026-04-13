"""
1a_02_replicate.py — MC Replication (mean + median)

Computes mean AND median within-cluster correlation for all 7 clustering
methods, replicating ACL paper Table 1 / GCD_Clustering_SAEs.py lines 60-109
for the mean computation. Median is added as a robustness check.

Inputs:
  phase1_artifacts/pairs.parquet
  phase1_artifacts/clusters/*.pkl

Outputs:
  phase1_artifacts/1a_replication.json
  phase1_artifacts/1a_mc_by_year.json
  phase1_artifacts/1a_report_02.md
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

# Published values from ACL paper (GCD line 367)
PUBLISHED = {
    "sae_cd": 0.359,
    "sic": 0.231,
}

# Tolerance for match: within 1e-3 (per PLAN.md)
TOLERANCE = 1e-3

METHODS = ["sae_cd", "bert", "sbert", "palm", "rolling_cd", "sic", "industry"]


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


def compute_mc(pairs_df, cluster_df):
    """
    Compute mean and median MC, matching ACL code structure:
      - For each year, for each cluster with >1 member:
        - Filter pairs where both companies are in the cluster
        - Compute mean correlation for that cluster (for mean MC)
        - Compute median correlation for that cluster (for median MC)
      - Average cluster-level stats across clusters (equal-weighted) per year
      - Average across years

    Returns:
        mean_mc (float): Overall mean MC across years
        median_mc (float): Overall median MC across years
        yearly_mean (dict): {year: mean_mc} per year
        yearly_median (dict): {year: median_mc} per year
    """
    pairs_by_year = {yr: grp for yr, grp in pairs_df.groupby("year")}

    yearly_mean = {}
    yearly_median = {}

    for _, row in cluster_df.iterrows():
        year = int(row["year"])
        clusters = row["clusters"]
        year_pairs = pairs_by_year.get(year)

        if year_pairs is None:
            yearly_mean[year] = float("nan")
            yearly_median[year] = float("nan")
            continue

        cluster_means = []
        cluster_medians = []

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
                cluster_means.append(cluster_pairs.mean())
                cluster_medians.append(cluster_pairs.median())

        if cluster_means:
            yearly_mean[year] = sum(cluster_means) / len(cluster_means)
            yearly_median[year] = sum(cluster_medians) / len(cluster_medians)
        else:
            yearly_mean[year] = float("nan")
            yearly_median[year] = float("nan")

    # Overall MC = mean of yearly values, skipping NaN
    valid_means = [v for v in yearly_mean.values() if not math.isnan(v)]
    valid_medians = [v for v in yearly_median.values() if not math.isnan(v)]

    mean_mc = sum(valid_means) / len(valid_means) if valid_means else float("nan")
    median_mc = sum(valid_medians) / len(valid_medians) if valid_medians else float("nan")

    return mean_mc, median_mc, yearly_mean, yearly_median


def write_report(results, population_mean, population_median):
    """Write phase1_artifacts/1a_report_02.md."""
    lines = []
    lines.append("## Step 2: MC Replication (Mean + Median)")
    lines.append("")

    # --- Observations ---
    lines.append("### Observations")
    lines.append("")

    # Replication comparison table
    lines.append("**MC Replication vs. Published Values**")
    lines.append("")
    lines.append(f"| {'Method':<14} | {'Mean MC':>8} | {'Published':>10} | {'Delta':>8} | {'Match?':>7} |")
    lines.append(f"|{'-'*16}|{'-'*10}|{'-'*12}|{'-'*10}|{'-'*9}|")
    lines.append(f"| {'population':<14} | {population_mean:>8.4f} | {'--':>10} | {'--':>8} | {'--':>7} |")

    for method in METHODS:
        r = results[method]
        pub = PUBLISHED.get(method)
        if pub is not None:
            delta = r["mean_mc"] - pub
            match = abs(delta) <= TOLERANCE
            match_str = "PASS" if match else "FAIL"
            lines.append(
                f"| {method:<14} | {r['mean_mc']:>8.4f} | {pub:>10.3f} | {delta:>+8.4f} | {match_str:>7} |"
            )
        else:
            lines.append(
                f"| {method:<14} | {r['mean_mc']:>8.4f} | {'--':>10} | {'--':>8} | {'--':>7} |"
            )
    lines.append("")

    # Mean vs Median table
    lines.append("**Mean vs. Median MC by Method**")
    lines.append("")
    lines.append(f"| {'Method':<14} | {'Mean MC':>8} | {'Median MC':>10} | {'Ratio':>7} |")
    lines.append(f"|{'-'*16}|{'-'*10}|{'-'*12}|{'-'*9}|")
    lines.append(
        f"| {'population':<14} | {population_mean:>8.4f} | {population_median:>10.4f} "
        f"| {population_median / population_mean if population_mean != 0 else float('nan'):>7.3f} |"
    )
    for method in METHODS:
        r = results[method]
        ratio = r["median_mc"] / r["mean_mc"] if r["mean_mc"] != 0 else float("nan")
        lines.append(
            f"| {method:<14} | {r['mean_mc']:>8.4f} | {r['median_mc']:>10.4f} | {ratio:>7.3f} |"
        )
    lines.append("")

    # --- Interpretation ---
    lines.append("### Interpretation")
    lines.append("")
    lines.append("_To be filled after running the script with actual data._")
    lines.append("")
    lines.append("Key questions for interpretation:")
    lines.append("")
    lines.append("- Do our mean MC values match the published values within tolerance (1e-3)?")
    lines.append("- Are mean and median MC telling the same story across methods?")
    lines.append("  If mean >> median for a method, the MC is driven by outlier pairs,")
    lines.append("  not broad cluster quality.")
    lines.append("- Which methods show the largest mean-median divergence? Does SAE's")
    lines.append("  advantage hold under median as well as mean?")
    lines.append("")

    report_path = os.path.join(ARTIFACTS, "1a_report_02.md")
    with open(report_path, "w") as f:
        f.write("\n".join(lines))
    print(f"Saved: {report_path}")


def main():
    print("=== 1a_02_replicate.py: MC Replication (mean + median) ===\n")

    # Load data
    pairs_df = load_pairs()
    population_mean = float(pairs_df["correlation"].mean())
    population_median = float(pairs_df["correlation"].median())
    print(f"  Population baseline — mean: {population_mean:.6f}, median: {population_median:.6f}")

    results = {}
    mc_by_year = {}

    for method in METHODS:
        print(f"\nComputing MC for {method}...")
        cluster_df = load_clusters(method)
        mean_mc, median_mc, yearly_mean, yearly_median = compute_mc(pairs_df, cluster_df)
        results[method] = {"mean_mc": mean_mc, "median_mc": median_mc}
        mc_by_year[method] = {
            "mean": {str(k): v for k, v in sorted(yearly_mean.items())},
            "median": {str(k): v for k, v in sorted(yearly_median.items())},
        }
        print(f"  {method}: mean MC = {mean_mc:.6f}, median MC = {median_mc:.6f}")

    # Print comparison table
    print("\n" + "=" * 72)
    print(f"{'Method':<14} {'Mean MC':>8} {'Median MC':>10} {'Published':>10} {'Match?':>8}")
    print("-" * 72)
    print(f"{'population':<14} {population_mean:>8.4f} {population_median:>10.4f} {'--':>10} {'--':>8}")
    for method in METHODS:
        r = results[method]
        pub = PUBLISHED.get(method)
        if pub is not None:
            match = abs(r["mean_mc"] - pub) <= TOLERANCE
            match_str = "PASS" if match else "FAIL"
            print(f"{method:<14} {r['mean_mc']:>8.4f} {r['median_mc']:>10.4f} {pub:>10.3f} {match_str:>8}")
        else:
            print(f"{method:<14} {r['mean_mc']:>8.4f} {r['median_mc']:>10.4f} {'--':>10} {'--':>8}")
    print("=" * 72)

    # Check pass/fail: published match within 1e-3
    passed = True
    for method, pub in PUBLISHED.items():
        our_mc = results[method]["mean_mc"]
        delta = abs(our_mc - pub)
        if delta > TOLERANCE:
            print(f"\nFAIL: {method} mean MC {our_mc:.6f} differs from published {pub:.3f} by {delta:.6f} (tolerance: {TOLERANCE})")
            passed = False
        else:
            print(f"\nPASS: {method} mean MC {our_mc:.6f} matches published {pub:.3f} (delta: {delta:.6f})")

    # Save 1a_replication.json — schema from PLAN.md
    replication = {
        "population_baseline_mc": round(population_mean, 6),
        "population_baseline_median": round(population_median, 6),
        "methods": {},
    }
    for method in METHODS:
        replication["methods"][method] = {
            "mean_mc": round(results[method]["mean_mc"], 6),
            "median_mc": round(results[method]["median_mc"], 6),
        }

    replication_path = os.path.join(ARTIFACTS, "1a_replication.json")
    with open(replication_path, "w") as f:
        json.dump(replication, f, indent=2)
    print(f"\nSaved: {replication_path}")

    # Save 1a_mc_by_year.json — with both mean and median per method per year
    mc_by_year_clean = {}
    for method, yearly in mc_by_year.items():
        mc_by_year_clean[method] = {}
        for stat_type in ("mean", "median"):
            mc_by_year_clean[method][stat_type] = {
                k: (round(v, 6) if not math.isnan(v) else None)
                for k, v in yearly[stat_type].items()
            }

    mc_by_year_path = os.path.join(ARTIFACTS, "1a_mc_by_year.json")
    with open(mc_by_year_path, "w") as f:
        json.dump(mc_by_year_clean, f, indent=2)
    print(f"Saved: {mc_by_year_path}")

    # Write report section
    write_report(results, population_mean, population_median)

    if passed:
        print("\nAll replication checks PASSED.")
        sys.exit(0)
    else:
        print("\nReplication checks FAILED.")
        sys.exit(1)


if __name__ == "__main__":
    main()

"""
Phase 1A Task 2: Compute Mean Correlation (MC) for all clustering methods.
Replicates ACL paper Table 1 / GCD_Clustering_SAEs.py lines 60-109.
"""

import json
import math
import os
import pickle
import sys

import pandas as pd
import numpy as np

ARTIFACTS = "phase1_artifacts"
CLUSTERS_DIR = os.path.join(ARTIFACTS, "clusters")

# Published values from ACL paper (GCD line 367)
PUBLISHED = {
    "sae_cd": 0.359,
    "sic": 0.231,
}

# Tolerance for pass/fail
TOLERANCE = 0.10  # 10%

METHODS = ["sae_cd", "bert", "sbert", "palm", "rolling_cd", "sic", "industry"]


def load_pairs():
    df = pd.read_parquet(os.path.join(ARTIFACTS, "pairs.parquet"))
    df["year"] = df["year"].astype(int)
    return df


def load_clusters(method):
    path = os.path.join(CLUSTERS_DIR, f"{method}.pkl")
    with open(path, "rb") as f:
        df = pickle.load(f)
    df["year"] = df["year"].astype(int)
    return df


def compute_mc(pairs_df, cluster_df):
    """Compute MC exactly matching ACL code: equal-weighted across clusters, then years."""
    # Pre-group pairs by year for speed
    pairs_by_year = {yr: grp for yr, grp in pairs_df.groupby("year")}

    yearly_mc = {}
    for _, row in cluster_df.iterrows():
        year = int(row["year"])
        clusters = row["clusters"]
        year_pairs = pairs_by_year.get(year)
        if year_pairs is None:
            yearly_mc[year] = float("nan")
            continue

        cluster_stats = []
        for cluster_id, members in clusters.items():
            if len(members) <= 1:
                continue
            members_set = set(members)
            mask = year_pairs["Company1"].isin(members_set) & year_pairs["Company2"].isin(members_set)
            cluster_pairs = year_pairs.loc[mask, "correlation"]
            if len(cluster_pairs) > 0:
                cluster_stats.append(cluster_pairs.mean())

        if cluster_stats:
            yearly_mc[year] = sum(cluster_stats) / len(cluster_stats)
        else:
            yearly_mc[year] = float("nan")

    # Method MC = mean of yearly MCs, skipping NaN
    valid = [v for v in yearly_mc.values() if not math.isnan(v)]
    method_mc = sum(valid) / len(valid) if valid else float("nan")
    return method_mc, yearly_mc


def main():
    print("Loading pairs data...")
    pairs_df = load_pairs()
    population_baseline = pairs_df["correlation"].mean()
    print(f"Population baseline MC: {population_baseline:.6f}")

    # Load verification baseline for comparison
    with open(os.path.join(ARTIFACTS, "data_verification.json")) as f:
        dv = json.load(f)
    verified_baseline = dv["population_baseline_mc"]
    print(f"Verified baseline (from data_verification.json): {verified_baseline:.6f}")

    results = {}
    mc_by_year = {}

    for method in METHODS:
        print(f"\nComputing MC for {method}...")
        cluster_df = load_clusters(method)
        method_mc, yearly = compute_mc(pairs_df, cluster_df)
        results[method] = method_mc
        mc_by_year[method] = {str(k): v for k, v in sorted(yearly.items())}
        print(f"  {method}: MC = {method_mc:.6f}")

    # Print comparison table
    print("\n" + "=" * 65)
    print(f"{'Method':<14} {'Our MC':>8} {'Published':>10} {'Match?':>8}")
    print("-" * 65)
    print(f"{'population':<14} {population_baseline:>8.4f} {'0.161':>10} {'':>8}")
    for method in METHODS:
        mc = results[method]
        pub = PUBLISHED.get(method)
        if pub is not None:
            lo = pub * (1 - TOLERANCE)
            hi = pub * (1 + TOLERANCE)
            match = lo <= mc <= hi
            match_str = "PASS" if match else "FAIL"
            print(f"{method:<14} {mc:>8.4f} {pub:>10.3f} {match_str:>8}")
        else:
            print(f"{method:<14} {mc:>8.4f} {'--':>10} {'--':>8}")
    print("=" * 65)

    # Check pass/fail criteria
    sae_mc = results["sae_cd"]
    sic_mc = results["sic"]
    sae_ok = 0.32 <= sae_mc <= 0.40
    sic_ok = 0.20 <= sic_mc <= 0.26

    if not sae_ok:
        print(f"\nFAIL: SAE MC {sae_mc:.4f} outside [0.32, 0.40]")
    if not sic_ok:
        print(f"\nFAIL: SIC MC {sic_mc:.4f} outside [0.20, 0.26]")

    # Save results
    replication = {
        "population_baseline_mc": round(population_baseline, 6),
        "verified_baseline_mc": verified_baseline,
    }
    for method in METHODS:
        replication[f"{method}_mc"] = round(results[method], 6)

    with open(os.path.join(ARTIFACTS, "1a_replication.json"), "w") as f:
        json.dump(replication, f, indent=2)
    print(f"\nSaved: {ARTIFACTS}/1a_replication.json")

    # Save per-year MC
    # Convert NaN to None for JSON serialization
    mc_by_year_clean = {}
    for method, yearly in mc_by_year.items():
        mc_by_year_clean[method] = {
            k: (round(v, 6) if not math.isnan(v) else None) for k, v in yearly.items()
        }
    with open(os.path.join(ARTIFACTS, "1a_mc_by_year.json"), "w") as f:
        json.dump(mc_by_year_clean, f, indent=2)
    print(f"Saved: {ARTIFACTS}/1a_mc_by_year.json")

    if sae_ok and sic_ok:
        print("\nAll checks PASSED.")
        sys.exit(0)
    else:
        print("\nChecks FAILED.")
        sys.exit(1)


if __name__ == "__main__":
    main()

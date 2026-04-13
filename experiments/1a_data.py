"""
1a_data.py — Phase 1A data loading and verification.

Loads all source data for Hyperion Phase 1A validation,
saves to phase1_artifacts/, and verifies against expected values.
"""

import os
import sys
import json
import time
import pandas as pd
from datasets import load_dataset

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ARTIFACTS_DIR = os.path.join(BASE_DIR, "phase1_artifacts")
CLUSTERS_DIR = os.path.join(ARTIFACTS_DIR, "clusters")
ACL_DIR = os.path.join(BASE_DIR, "company_similarity_sae", "Clustering")

CLUSTER_SOURCES = {
    "sae_cd": os.path.join(ACL_DIR, "data", "Final Results", "year_cluster_dfC-CD.pkl"),
    "bert": os.path.join(ACL_DIR, "data", "Final Results", "year_cluster_dfBERT.pkl"),
    "sbert": os.path.join(ACL_DIR, "data", "Final Results", "year_cluster_dfSBERT.pkl"),
    "palm": os.path.join(ACL_DIR, "data", "Final Results", "year_cluster_dfPaLM-gecko.pkl"),
    "rolling_cd": os.path.join(ACL_DIR, "data", "Final Results", "year_cluster_dfrollingCD.pkl"),
    "sic": os.path.join(ACL_DIR, "data", "cointegration", "year_SIC_cluster_mapping.pkl"),
    "industry": os.path.join(ACL_DIR, "data", "cointegration", "year_Industry_cluster_mapping.pkl"),
}


def log(msg):
    print(f"[1a_data] {msg}")


def load_pairs():
    """Load pairs dataset, drop NaN correlations (matching GCD_Clustering_SAEs.py line 262)."""
    log("Loading pairs dataset from HuggingFace...")
    t0 = time.time()
    ds = load_dataset("v1ctor10/cos_sim_4000pca_exp", split="train")
    pairs_df = ds.to_pandas()
    log(f"  Raw rows: {len(pairs_df):,} ({time.time() - t0:.1f}s)")

    before = len(pairs_df)
    pairs_df = pairs_df.dropna(subset=["correlation"]).reset_index(drop=True)
    pairs_df["year"] = pairs_df["year"].astype(int)
    log(f"  After dropna(correlation): {len(pairs_df):,} (dropped {before - len(pairs_df):,})")
    return pairs_df


def load_companies():
    """Load company metadata dataset."""
    log("Loading company metadata from HuggingFace...")
    t0 = time.time()
    ds = load_dataset(
        "Mateusz1017/annual_reports_tokenized_llama3_logged_returns_no_null_returns_and_incomplete_descriptions_24k",
        split="train",
    )
    df = ds.to_pandas()
    log(f"  Rows: {len(df):,}, Columns: {list(df.columns)} ({time.time() - t0:.1f}s)")
    return df


def load_clusters():
    """Load all cluster label pickles."""
    clusters = {}
    for name, path in CLUSTER_SOURCES.items():
        log(f"Loading cluster pickle: {name} ← {os.path.basename(path)}")
        if not os.path.exists(path):
            log(f"  ERROR: file not found: {path}")
            sys.exit(1)
        clusters[name] = pd.read_pickle(path)
        df = clusters[name]
        log(f"  Rows: {len(df)}, Columns: {list(df.columns)}")
    return clusters


def save_artifacts(pairs_df, companies_df, clusters):
    """Save all data to phase1_artifacts/."""
    os.makedirs(CLUSTERS_DIR, exist_ok=True)

    log(f"Saving pairs.parquet ({len(pairs_df):,} rows)...")
    pairs_df.to_parquet(os.path.join(ARTIFACTS_DIR, "pairs.parquet"), index=False)

    log(f"Saving companies.parquet ({len(companies_df):,} rows)...")
    companies_df.to_parquet(os.path.join(ARTIFACTS_DIR, "companies.parquet"), index=False)

    for name, df in clusters.items():
        path = os.path.join(CLUSTERS_DIR, f"{name}.pkl")
        log(f"Saving {path}")
        df.to_pickle(path)


def verify(pairs_df, companies_df, clusters):
    """Run all verification checks. Returns dict of results. Exits 1 on failure."""
    checks = {}
    failures = []

    # --- Pairs checks ---
    pairs_path = os.path.join(ARTIFACTS_DIR, "pairs.parquet")
    checks["pairs_file_exists"] = os.path.exists(pairs_path)

    n_pairs = len(pairs_df)
    checks["pairs_row_count"] = n_pairs
    checks["pairs_rows_in_range"] = 14_000_000 <= n_pairs <= 16_000_000

    required_cols = {"Company1", "Company2", "year", "cosine_similarity", "correlation"}
    actual_cols = set(pairs_df.columns)
    checks["pairs_has_required_columns"] = required_cols.issubset(actual_cols)
    checks["pairs_columns"] = sorted(pairs_df.columns.tolist())

    pop_baseline = float(pairs_df["correlation"].mean())
    checks["population_baseline_mc"] = round(pop_baseline, 6)
    checks["baseline_in_range"] = 0.14 <= pop_baseline <= 0.18

    checks["pairs_year_range"] = [int(pairs_df["year"].min()), int(pairs_df["year"].max())]
    checks["pairs_nan_correlations"] = int(pairs_df["correlation"].isna().sum())

    # --- Companies checks ---
    companies_path = os.path.join(ARTIFACTS_DIR, "companies.parquet")
    checks["companies_file_exists"] = os.path.exists(companies_path)

    n_companies = len(companies_df)
    checks["companies_row_count"] = n_companies
    checks["companies_rows_in_range"] = 20_000 <= n_companies <= 30_000
    checks["companies_columns"] = sorted(companies_df.columns.tolist())

    # --- Cluster checks ---
    for name, df in clusters.items():
        prefix = f"cluster_{name}"
        has_year = "year" in df.columns
        has_clusters = "clusters" in df.columns
        checks[f"{prefix}_has_year_and_clusters"] = has_year and has_clusters
        checks[f"{prefix}_row_count"] = len(df)
        if has_year:
            years = sorted(df["year"].unique().tolist())
            checks[f"{prefix}_year_range"] = [int(years[0]), int(years[-1])]

    # SAE C-CD year range check
    sae_df = clusters["sae_cd"]
    sae_years = sorted(sae_df["year"].unique().tolist())
    checks["sae_cd_covers_1996_2020"] = int(sae_years[0]) <= 1996 and int(sae_years[-1]) >= 2020

    # --- Collect failures ---
    for key, val in checks.items():
        if isinstance(val, bool) and not val:
            failures.append(key)

    checks["all_passed"] = len(failures) == 0
    checks["failures"] = failures

    # Save verification JSON
    json_path = os.path.join(ARTIFACTS_DIR, "data_verification.json")
    with open(json_path, "w") as f:
        json.dump(checks, f, indent=2, default=str)
    log(f"Saved {json_path}")

    return checks, failures


def print_summary(pairs_df, companies_df, clusters, checks):
    """Print a summary table."""
    print("\n" + "=" * 70)
    print("PHASE 1A DATA SUMMARY")
    print("=" * 70)

    print(f"\n{'Dataset':<25} {'Rows':>12} {'Year Range':>15}")
    print("-" * 55)
    print(f"{'pairs':<25} {len(pairs_df):>12,} {checks['pairs_year_range'][0]}-{checks['pairs_year_range'][1]:>4}")
    print(f"{'companies':<25} {len(companies_df):>12,}")

    print(f"\n{'Cluster Pickle':<25} {'Rows':>8} {'Year Range':>15}")
    print("-" * 50)
    for name in CLUSTER_SOURCES:
        prefix = f"cluster_{name}"
        yr = checks.get(f"{prefix}_year_range", ["?", "?"])
        print(f"  {name:<23} {checks[f'{prefix}_row_count']:>8} {yr[0]}-{yr[1]:>4}")

    pop_mc = checks["population_baseline_mc"]
    print(f"\nPopulation baseline MC (mean correlation): {pop_mc:.6f}")
    print(f"  ACL paper reports: 0.161")
    print(f"  Acceptable range: [0.14, 0.18]")
    print(f"  Status: {'PASS' if checks['baseline_in_range'] else 'FAIL'}")

    print(f"\nVerification: {'ALL CHECKS PASSED' if checks['all_passed'] else 'FAILURES: ' + ', '.join(checks['failures'])}")
    print("=" * 70)


def main():
    t_start = time.time()

    pairs_df = load_pairs()
    companies_df = load_companies()
    clusters = load_clusters()

    save_artifacts(pairs_df, companies_df, clusters)

    checks, failures = verify(pairs_df, companies_df, clusters)
    print_summary(pairs_df, companies_df, clusters, checks)

    log(f"Total time: {time.time() - t_start:.1f}s")

    if failures:
        log(f"FAILED checks: {failures}")
        sys.exit(1)
    else:
        log("All checks passed.")


if __name__ == "__main__":
    main()

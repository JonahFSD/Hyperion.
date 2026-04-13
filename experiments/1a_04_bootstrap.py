"""
1a_04_bootstrap.py — Bootstrap CIs + Influence Diagnostics

Phase 1A Step 4/7: Resamples tickers (the independent unit) to compute
BCa confidence intervals for SAE, SIC, and SBERT mean correlation (MC),
tests SAE superiority via bootstrap delta distributions, and identifies
fragile tickers via leave-one-out jackknife influence diagnostics.

Inputs:
  phase1_artifacts/pairs.parquet
  phase1_artifacts/companies.parquet
  phase1_artifacts/clusters/{sae_cd,sic,sbert}.pkl
  phase1_artifacts/1a_replication.json

Outputs:
  phase1_artifacts/1a_bootstrap.json
  phase1_artifacts/1a_report_04.md
"""

import json
import math
import os
import pickle
import sys
import time

import numpy as np
import pandas as pd
from scipy.stats import norm, skew

ARTIFACTS = "phase1_artifacts"
CLUSTERS_DIR = os.path.join(ARTIFACTS, "clusters")

N_BOOTSTRAP = 10000
SEED = 42
CI_LEVEL = 0.95


# ============================================================
# Data loading
# ============================================================

def load_pairs():
    df = pd.read_parquet(
        os.path.join(ARTIFACTS, "pairs.parquet"),
        columns=["Company1", "Company2", "year", "correlation"],
    )
    df["year"] = df["year"].astype(int)
    return df


def load_companies():
    df = pd.read_parquet(os.path.join(ARTIFACTS, "companies.parquet"))
    return df


def load_clusters(method):
    path = os.path.join(CLUSTERS_DIR, f"{method}.pkl")
    with open(path, "rb") as f:
        df = pickle.load(f)
    df["year"] = df["year"].astype(int)
    return df


def load_replication():
    """Load replicated MC values from 1a_02 output.

    Handles both the current flat-key format and the PLAN-specified
    nested format (with 'methods' dict), so this script works regardless
    of whether 1a_02 has been updated yet.
    """
    path = os.path.join(ARTIFACTS, "1a_replication.json")
    with open(path) as f:
        data = json.load(f)

    if "methods" in data:
        return {
            "sae": data["methods"]["sae_cd"]["mean_mc"],
            "sic": data["methods"]["sic"]["mean_mc"],
            "sbert": data["methods"]["sbert"]["mean_mc"],
            "population_baseline": data["population_baseline_mc"],
        }
    else:
        return {
            "sae": data["sae_cd_mc"],
            "sic": data["sic_mc"],
            "sbert": data["sbert_mc"],
            "population_baseline": data["population_baseline_mc"],
        }


# ============================================================
# Company-to-ticker mapping
# ============================================================

def build_ticker_mapping(companies_df):
    """Build company ID -> ticker mapping.

    The ticker column in companies.parquet stores arrays (one company can
    have multiple tickers across years). We take the first element as the
    primary ticker, following the ACL convention.

    Returns: (comp_to_ticker, ticker_to_idx, unique_tickers)
    """
    primary_ticker = companies_df["ticker"].apply(
        lambda x: x[0] if hasattr(x, "__len__") and len(x) > 0 else x
    )
    valid = primary_ticker.notna()
    comp_ids = companies_df.loc[valid, "__index_level_0__"]
    tickers = primary_ticker[valid]
    comp_to_ticker = dict(zip(comp_ids, tickers))

    unique_tickers = sorted(set(comp_to_ticker.values()))
    ticker_to_idx = {t: i for i, t in enumerate(unique_tickers)}
    return comp_to_ticker, ticker_to_idx, unique_tickers


# ============================================================
# Precompute within-cluster pairs
# ============================================================

def precompute_within_cluster_pairs(pairs_df, cluster_df, comp_to_ticker, ticker_to_idx):
    """Build flat arrays of (ticker1_idx, ticker2_idx, correlation) grouped
    by (year, cluster).

    Each group corresponds to one (year, cluster_id) combination with >1
    member. group_starts[i] marks where group i begins in the flat arrays.
    This structure enables O(1) weighted MC computation via np.add.reduceat.
    """
    pairs_by_year = {yr: grp for yr, grp in pairs_df.groupby("year")}

    all_t1, all_t2, all_corr = [], [], []
    group_years = []
    group_idx = 0

    for _, row in cluster_df.iterrows():
        year = int(row["year"])
        clusters = row["clusters"]
        yp = pairs_by_year.get(year)
        if yp is None:
            continue

        # Build company -> cluster label mapping for this year
        c2cl = {}
        for cid, members in clusters.items():
            for m in members:
                c2cl[m] = cid

        # Find within-cluster pairs
        c1_cl = yp["Company1"].map(c2cl)
        c2_cl = yp["Company2"].map(c2cl)
        same = (c1_cl == c2_cl) & c1_cl.notna() & c2_cl.notna()
        within = yp[same]
        if len(within) == 0:
            continue

        # Map company IDs to ticker indices
        t1 = within["Company1"].map(comp_to_ticker)
        t2 = within["Company2"].map(comp_to_ticker)
        valid = t1.notna() & t2.notna()
        if valid.sum() == 0:
            continue

        t1_idx = t1[valid].map(ticker_to_idx)
        t2_idx = t2[valid].map(ticker_to_idx)
        tv = t1_idx.notna() & t2_idx.notna()
        if tv.sum() == 0:
            continue

        t1_arr = t1_idx[tv].values.astype(int)
        t2_arr = t2_idx[tv].values.astype(int)
        corr_arr = within.loc[tv.values if hasattr(tv, "values") else tv, "correlation"].values
        cl_arr = c1_cl[same][valid][tv].values

        # Split by cluster label — each cluster is a separate group
        unique_cls = np.unique(cl_arr)
        for cl in unique_cls:
            cl_key = cl
            if cl_key not in clusters:
                # Try int conversion for numeric keys stored as float
                try:
                    cl_key = int(cl)
                except (ValueError, TypeError):
                    pass
            members = clusters.get(cl_key, [])
            if len(members) <= 1:
                continue
            mask = cl_arr == cl
            if mask.sum() == 0:
                continue
            all_t1.append(t1_arr[mask])
            all_t2.append(t2_arr[mask])
            all_corr.append(corr_arr[mask])
            group_years.append(year)
            group_idx += 1

    if group_idx == 0:
        return None

    flat_t1 = np.concatenate(all_t1)
    flat_t2 = np.concatenate(all_t2)
    flat_corr = np.concatenate(all_corr)

    sizes = np.array([len(a) for a in all_t1])
    group_starts = np.zeros(len(sizes), dtype=int)
    group_starts[1:] = np.cumsum(sizes[:-1])

    return {
        "t1": flat_t1,
        "t2": flat_t2,
        "corr": flat_corr,
        "group_starts": group_starts,
        "group_years": np.array(group_years),
        "unique_years": np.array(sorted(set(group_years))),
        "n_groups": group_idx,
        "n_pairs": len(flat_t1),
    }


# ============================================================
# Weighted MC computation
# ============================================================

def compute_weighted_mc(pdata, multiplicities):
    """Compute weighted MC from ticker multiplicities.

    For each (year, cluster) group:
      weighted_mean = sum(w_ij * corr_ij) / sum(w_ij)
      where w_ij = multiplicities[ticker_i] * multiplicities[ticker_j]
    Then average across clusters within each year (equal-weighted),
    then average across years. Matches ACL's calculate_avg_correlation.
    """
    w = multiplicities[pdata["t1"]] * multiplicities[pdata["t2"]]
    wc = w * pdata["corr"]

    gw = np.add.reduceat(w, pdata["group_starts"])
    gwc = np.add.reduceat(wc, pdata["group_starts"])

    n_groups = pdata["n_groups"]
    cmc = np.full(n_groups, np.nan)
    valid_g = gw > 0
    if valid_g.any():
        cmc[valid_g] = gwc[valid_g] / gw[valid_g]

    group_years = pdata["group_years"]
    year_mcs = []
    for yr in pdata["unique_years"]:
        yr_mask = group_years == yr
        yr_cmc = cmc[yr_mask]
        yr_valid = yr_cmc[~np.isnan(yr_cmc)]
        if len(yr_valid) > 0:
            year_mcs.append(yr_valid.mean())

    return float(np.mean(year_mcs)) if year_mcs else float("nan")


# ============================================================
# BCa confidence interval
# ============================================================

def bca_ci(boot_dist, original, jack_stats, alpha=0.05):
    """Compute BCa (bias-corrected and accelerated) confidence interval.

    Args:
        boot_dist: array of bootstrap statistics (length B)
        original: observed statistic
        jack_stats: array of leave-one-out jackknife statistics (length n)
        alpha: significance level (default 0.05 for 95% CI)

    Returns:
        (ci_lower, ci_upper, z0, a_hat)
    """
    B = len(boot_dist)

    # Bias correction z0 = Phi^{-1}(proportion of bootstrap < observed)
    prop = np.clip(np.sum(boot_dist < original) / B, 1e-10, 1 - 1e-10)
    z0 = float(norm.ppf(prop))

    # Acceleration from jackknife: a = sum(d^3) / (6 * (sum(d^2))^{3/2})
    theta_dot = np.mean(jack_stats)
    d = theta_dot - jack_stats
    denom = np.sum(d ** 2)
    a_hat = float(np.sum(d ** 3) / (6.0 * denom ** 1.5)) if denom > 0 else 0.0

    # Adjusted percentiles
    z_lo = norm.ppf(alpha / 2)
    z_hi = norm.ppf(1 - alpha / 2)

    def adj(z_a):
        num = z0 + z_a
        den = 1.0 - a_hat * num
        if abs(den) < 1e-10:
            return 0.5
        return float(norm.cdf(z0 + num / den))

    a1 = np.clip(adj(z_lo), 0.5 / B, 1.0 - 0.5 / B)
    a2 = np.clip(adj(z_hi), 0.5 / B, 1.0 - 0.5 / B)

    ci_lo = float(np.percentile(boot_dist, 100 * a1))
    ci_hi = float(np.percentile(boot_dist, 100 * a2))
    return ci_lo, ci_hi, z0, a_hat


# ============================================================
# Report generation
# ============================================================

def write_report(output):
    """Write phase1_artifacts/1a_report_04.md from computed results."""
    cfg = output["config"]
    methods = output["methods"]
    deltas = output["deltas"]
    influence = output["influence"]

    lines = []
    lines.append("## 4. Bootstrap CIs + Influence Diagnostics")
    lines.append("")
    lines.append("### Observations")
    lines.append("")

    # Config
    lines.append(
        f"**Configuration:** {cfg['n_bootstrap']:,} bootstrap iterations, "
        f"{cfg['n_tickers']} tickers resampled, seed={cfg['seed']}, "
        f"{cfg['ci_level'] * 100:.0f}% BCa confidence intervals."
    )
    lines.append("")

    # Method CIs table
    lines.append("**Method CIs:**")
    lines.append("")
    lines.append("| Method | MC | 95% CI | z0 | a | Boot Mean | Boot Std |")
    lines.append("|--------|---:|-------:|---:|--:|----------:|---------:|")
    for label in ["sae", "sic", "sbert"]:
        m = methods[label]
        lines.append(
            f"| {label.upper()} | {m['mc']:.6f} "
            f"| [{m['ci_lower']:.6f}, {m['ci_upper']:.6f}] "
            f"| {m['z0']:.4f} | {m['a']:.6f} "
            f"| {m['bootstrap_mean']:.6f} | {m['bootstrap_std']:.6f} |"
        )
    lines.append("")

    # Delta tests table
    lines.append("**Delta Tests:**")
    lines.append("")
    lines.append("| Comparison | Delta | 95% CI | p-value | t-stat | z0 | a |")
    lines.append("|------------|------:|-------:|--------:|-------:|---:|--:|")
    delta_labels = {
        "sae_minus_sic": "SAE - SIC",
        "sae_minus_sbert": "SAE - SBERT",
        "sae_minus_baseline": "SAE - baseline",
    }
    for name in ["sae_minus_sic", "sae_minus_sbert", "sae_minus_baseline"]:
        d = deltas[name]
        lines.append(
            f"| {delta_labels[name]} | {d['delta']:.6f} "
            f"| [{d['ci_lower']:.6f}, {d['ci_upper']:.6f}] "
            f"| {d['p_value']:.6f} | {d['t_stat']:.2f} "
            f"| {d['z0']:.4f} | {d['a']:.6f} |"
        )
    lines.append("")

    # Influence diagnostics
    lines.append("**Influence Diagnostics (SAE, leave-one-ticker-out):**")
    lines.append("")
    lines.append(
        f"- Conclusion-flipping tickers (removing causes SAE MC < SIC MC): "
        f"**{influence['n_tickers_that_flip_conclusion']}**"
    )
    if influence["flipping_tickers"]:
        lines.append(f"  - Tickers: {', '.join(influence['flipping_tickers'])}")
    lines.append(
        f"- Influence distribution: mean={influence['influence_mean']:.6f}, "
        f"std={influence['influence_std']:.6f}, "
        f"skewness={influence['influence_skewness']:.4f}"
    )
    lines.append(
        f"- Max absolute influence: {influence['influence_max_abs']:.6f} "
        f"({influence['influence_max_ticker']})"
    )
    lines.append("")
    lines.append("Top 20 tickers by absolute influence:")
    lines.append("")
    lines.append("| Rank | Ticker | Influence |")
    lines.append("|-----:|--------|----------:|")
    for i, entry in enumerate(influence["top_20_by_abs_influence"]):
        lines.append(f"| {i + 1} | {entry['ticker']} | {entry['influence']:.6f} |")
    lines.append("")

    # Interpretation
    lines.append("### Interpretation")
    lines.append("")
    lines.append("*(To be written after reviewing the numbers above.)*")
    lines.append("")
    lines.append("Key questions:")
    lines.append(
        "- Is the result broad-based or concentrated? "
        "(Check influence distribution skewness and max.)"
    )
    lines.append(
        "- What does z0 tell us about bias in the bootstrap? "
        "(z0 near 0 = symmetric, large |z0| = skewed.)"
    )
    lines.append(
        "- Are there fragilities? "
        "(Any conclusion-flipping tickers indicate the result depends "
        "on specific companies.)"
    )
    lines.append("- Do t-statistics exceed HLZ gold standard of 3.0?")
    lines.append("")

    report_path = os.path.join(ARTIFACTS, "1a_report_04.md")
    with open(report_path, "w") as f:
        f.write("\n".join(lines))
    print(f"Saved: {report_path}")


# ============================================================
# Main
# ============================================================

def main():
    print("=" * 70)
    print("Phase 1A Step 4/7: Bootstrap CIs + Influence Diagnostics")
    print("=" * 70)
    t_start = time.time()

    # ----------------------------------------------------------
    # 1. Load data
    # ----------------------------------------------------------
    print("\n--- Loading data ---")
    pairs_df = load_pairs()
    companies_df = load_companies()
    replication = load_replication()
    print(f"  Pairs: {len(pairs_df):,}")
    print(f"  Companies: {len(companies_df):,}")
    print(
        f"  Replicated MCs: SAE={replication['sae']:.6f}, "
        f"SIC={replication['sic']:.6f}, SBERT={replication['sbert']:.6f}"
    )
    print(f"  Population baseline: {replication['population_baseline']:.6f}")

    # ----------------------------------------------------------
    # 2. Build ticker mapping
    # ----------------------------------------------------------
    print("\n--- Building ticker mapping ---")
    comp_to_ticker, ticker_to_idx, unique_tickers = build_ticker_mapping(
        companies_df
    )
    n_tickers = len(unique_tickers)
    print(f"  Unique tickers: {n_tickers}")

    # ----------------------------------------------------------
    # 3. Precompute within-cluster pairs for SAE, SIC, SBERT
    # ----------------------------------------------------------
    methods_map = {"sae": "sae_cd", "sic": "sic", "sbert": "sbert"}
    pdata = {}
    orig_mc = {}

    for label, method in methods_map.items():
        print(f"\n--- Precomputing within-cluster pairs: {label} ---")
        cluster_df = load_clusters(method)
        pd_ = precompute_within_cluster_pairs(
            pairs_df, cluster_df, comp_to_ticker, ticker_to_idx
        )
        if pd_ is None:
            print(f"  FATAL: No within-cluster pairs for {label}")
            sys.exit(1)
        pdata[label] = pd_

        # Compute MC with uniform weights (verification baseline)
        ones = np.ones(n_tickers, dtype=np.float64)
        orig_mc[label] = compute_weighted_mc(pd_, ones)
        print(
            f"  {label}: MC={orig_mc[label]:.6f}, "
            f"pairs={pd_['n_pairs']:,}, groups={pd_['n_groups']}"
        )

    # ----------------------------------------------------------
    # 4. Verification: uniform-weight MC must match 1a_replication.json
    # ----------------------------------------------------------
    print("\n--- Verification: uniform weights vs replicated MC ---")
    verification_ok = True
    for label in ["sae", "sic", "sbert"]:
        diff = abs(orig_mc[label] - replication[label])
        status = "OK" if diff < 1e-6 else "MISMATCH"
        if diff >= 1e-6:
            verification_ok = False
        print(
            f"  {label}: weighted={orig_mc[label]:.6f}, "
            f"replicated={replication[label]:.6f}, "
            f"diff={diff:.2e} [{status}]"
        )

    if not verification_ok:
        print("\n  FATAL: Weighted MC does not match replicated MC within 1e-6.")
        print("  The pair data structures are inconsistent with 1a_02. Halting.")
        sys.exit(1)
    print("  Verification PASSED.")

    # ----------------------------------------------------------
    # 5. Jackknife: leave-one-ticker-out
    #    Needed for BCa acceleration parameter AND influence diagnostics
    # ----------------------------------------------------------
    print(f"\n--- Jackknife ({n_tickers} tickers) ---")
    t_jack = time.time()
    jack = {label: np.zeros(n_tickers) for label in methods_map}

    for i in range(n_tickers):
        if (i + 1) % 500 == 0:
            elapsed = time.time() - t_jack
            rate = (i + 1) / elapsed
            remaining = (n_tickers - i - 1) / rate
            print(
                f"  {i + 1}/{n_tickers} "
                f"({elapsed:.0f}s, ~{remaining:.0f}s remaining)"
            )
        mult = np.ones(n_tickers, dtype=np.float64)
        mult[i] = 0.0
        for label in methods_map:
            jack[label][i] = compute_weighted_mc(pdata[label], mult)

    jack_elapsed = time.time() - t_jack
    print(f"  Done in {jack_elapsed:.1f}s")

    # ----------------------------------------------------------
    # 6. Bootstrap: 10,000 iterations, resample tickers with replacement
    # ----------------------------------------------------------
    print(f"\n--- Bootstrap ({N_BOOTSTRAP} iterations, seed={SEED}) ---")
    t_boot = time.time()
    rng = np.random.default_rng(SEED)
    boot = {label: np.zeros(N_BOOTSTRAP) for label in methods_map}

    for b in range(N_BOOTSTRAP):
        if (b + 1) % 2000 == 0:
            elapsed = time.time() - t_boot
            rate = (b + 1) / elapsed
            remaining = (N_BOOTSTRAP - b - 1) / rate
            print(
                f"  {b + 1}/{N_BOOTSTRAP} "
                f"({elapsed:.0f}s, ~{remaining:.0f}s remaining)"
            )
        draw = rng.choice(n_tickers, n_tickers, replace=True)
        mult = np.bincount(draw, minlength=n_tickers).astype(np.float64)
        for label in methods_map:
            boot[label][b] = compute_weighted_mc(pdata[label], mult)

    boot_elapsed = time.time() - t_boot
    print(f"  Done in {boot_elapsed:.1f}s")

    # ----------------------------------------------------------
    # 7. BCa CIs for each method
    # ----------------------------------------------------------
    print("\n--- BCa Confidence Intervals ---")
    method_results = {}
    for label in methods_map:
        ci_lo, ci_hi, z0, a = bca_ci(
            boot[label], orig_mc[label], jack[label]
        )
        method_results[label] = {
            "mc": round(orig_mc[label], 6),
            "ci_lower": round(ci_lo, 6),
            "ci_upper": round(ci_hi, 6),
            "z0": round(z0, 6),
            "a": round(a, 6),
            "bootstrap_mean": round(float(np.mean(boot[label])), 6),
            "bootstrap_std": round(float(np.std(boot[label])), 6),
        }
        print(
            f"  {label}: MC={orig_mc[label]:.6f}, "
            f"95% CI=[{ci_lo:.6f}, {ci_hi:.6f}], "
            f"z0={z0:.4f}, a={a:.6f}"
        )

    # ----------------------------------------------------------
    # 8. Delta tests: SAE-SIC, SAE-SBERT, SAE-baseline
    # ----------------------------------------------------------
    print("\n--- Delta Tests ---")
    population_baseline = replication["population_baseline"]

    delta_configs = [
        ("sae_minus_sic", "sae", "sic"),
        ("sae_minus_sbert", "sae", "sbert"),
        ("sae_minus_baseline", "sae", None),
    ]
    deltas = {}
    for name, ma, mb in delta_configs:
        if mb is not None:
            orig_d = orig_mc[ma] - orig_mc[mb]
            boot_d = boot[ma] - boot[mb]
            jack_d = jack[ma] - jack[mb]
        else:
            # SAE vs population baseline (a constant)
            orig_d = orig_mc[ma] - population_baseline
            boot_d = boot[ma] - population_baseline
            jack_d = jack[ma] - population_baseline

        ci_lo, ci_hi, z0, a = bca_ci(boot_d, orig_d, jack_d)
        p_val = float(np.mean(boot_d <= 0))
        boot_std = float(np.std(boot_d))
        t_stat = float(orig_d / boot_std) if boot_std > 0 else float("inf")

        deltas[name] = {
            "delta": round(float(orig_d), 6),
            "ci_lower": round(ci_lo, 6),
            "ci_upper": round(ci_hi, 6),
            "z0": round(z0, 6),
            "a": round(a, 6),
            "p_value": round(p_val, 6),
            "t_stat": round(t_stat, 6),
        }
        print(
            f"  {name}: delta={orig_d:.6f}, "
            f"CI=[{ci_lo:.6f}, {ci_hi:.6f}], "
            f"p={p_val:.6f}, t={t_stat:.2f}"
        )

    # ----------------------------------------------------------
    # 9. Jackknife influence diagnostics (SAE)
    #    Reuses the jackknife values already computed for BCa.
    # ----------------------------------------------------------
    print("\n--- Influence Diagnostics ---")
    sae_full_mc = orig_mc["sae"]
    sic_full_mc = orig_mc["sic"]

    # influence_i = MC_full - MC_without_i
    # Positive = ticker contributes positively (removing it decreases MC)
    influence = sae_full_mc - jack["sae"]

    # Conclusion-flipping: removing ticker i causes SAE MC to drop below SIC MC
    flipping_mask = jack["sae"] < sic_full_mc
    n_flipping = int(flipping_mask.sum())
    flipping_tickers = sorted(
        [unique_tickers[i] for i in np.where(flipping_mask)[0]]
    )

    # Summary statistics on influence distribution
    inf_mean = float(np.mean(influence))
    inf_std = float(np.std(influence))
    inf_max_abs_idx = int(np.argmax(np.abs(influence)))
    inf_max_abs = float(np.abs(influence[inf_max_abs_idx]))
    inf_max_ticker = unique_tickers[inf_max_abs_idx]
    inf_skewness = float(skew(influence))

    # Top 20 by absolute influence
    sorted_idx = np.argsort(np.abs(influence))[::-1]
    top_20 = [
        {
            "ticker": unique_tickers[i],
            "influence": round(float(influence[i]), 6),
        }
        for i in sorted_idx[:20]
    ]

    print(f"  Conclusion-flipping tickers: {n_flipping}")
    if n_flipping > 0:
        preview = ", ".join(flipping_tickers[:10])
        suffix = "..." if n_flipping > 10 else ""
        print(f"    Tickers: {preview}{suffix}")
    print(
        f"  Influence: mean={inf_mean:.6f}, std={inf_std:.6f}, "
        f"max_abs={inf_max_abs:.6f} ({inf_max_ticker})"
    )
    print(f"  Skewness: {inf_skewness:.4f}")

    influence_result = {
        "n_tickers_that_flip_conclusion": n_flipping,
        "flipping_tickers": flipping_tickers,
        "influence_mean": round(inf_mean, 6),
        "influence_std": round(inf_std, 6),
        "influence_max_abs": round(inf_max_abs, 6),
        "influence_max_ticker": inf_max_ticker,
        "influence_skewness": round(inf_skewness, 6),
        "top_20_by_abs_influence": top_20,
    }

    # ----------------------------------------------------------
    # 10. Assemble output JSON
    # ----------------------------------------------------------
    output = {
        "config": {
            "n_bootstrap": N_BOOTSTRAP,
            "n_tickers": n_tickers,
            "seed": SEED,
            "ci_level": CI_LEVEL,
            "population_baseline": round(population_baseline, 6),
        },
        "methods": method_results,
        "deltas": deltas,
        "influence": influence_result,
    }

    output_path = os.path.join(ARTIFACTS, "1a_bootstrap.json")
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved: {output_path}")

    # ----------------------------------------------------------
    # 11. Write report section
    # ----------------------------------------------------------
    write_report(output)

    # ----------------------------------------------------------
    # 12. Verification checks
    # ----------------------------------------------------------
    print("\n" + "=" * 70)
    print("VERIFICATION CHECKS")
    print("=" * 70)

    ok = True

    def check(name, passed):
        nonlocal ok
        ok &= passed
        print(f"  [{'PASS' if passed else 'FAIL'}] {name}")

    check(
        "Uniform-weight MC matches replicated MC within 1e-6",
        verification_ok,
    )

    sae_ci = method_results["sae"]
    baseline_excluded = (
        sae_ci["ci_lower"] > population_baseline
        or sae_ci["ci_upper"] < population_baseline
    )
    check(
        f"SAE CI [{sae_ci['ci_lower']:.4f}, {sae_ci['ci_upper']:.4f}] "
        f"excludes population baseline {population_baseline:.4f}",
        baseline_excluded,
    )

    check(
        f"SAE-SIC delta CI lower > 0 "
        f"({deltas['sae_minus_sic']['ci_lower']:.6f})",
        deltas["sae_minus_sic"]["ci_lower"] > 0,
    )

    check(
        f"SAE-SBERT delta CI lower > 0 "
        f"({deltas['sae_minus_sbert']['ci_lower']:.6f})",
        deltas["sae_minus_sbert"]["ci_lower"] > 0,
    )

    check(
        "All z0 values finite",
        all(math.isfinite(method_results[m]["z0"]) for m in method_results),
    )

    check(
        "All acceleration (a) values finite",
        all(math.isfinite(method_results[m]["a"]) for m in method_results),
    )

    check("Influence diagnostics complete", True)

    total = time.time() - t_start
    print(f"\nTotal elapsed: {total:.1f}s")

    if ok:
        print("\nAll checks PASSED.")
        sys.exit(0)
    else:
        print("\nSome checks FAILED.")
        sys.exit(1)


if __name__ == "__main__":
    main()

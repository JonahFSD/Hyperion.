"""
Phase 1A Task 4: Statistical rigor for MC replication.

Theta sensitivity sweep, bootstrap CIs with BCa correction,
random baseline, and train/test split evaluation.
"""

import json
import math
import os
import pickle
import sys
import time

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import minimum_spanning_tree, connected_components
from scipy.stats import norm
from sklearn.preprocessing import StandardScaler

ARTIFACTS = "phase1_artifacts"
CLUSTERS_DIR = os.path.join(ARTIFACTS, "clusters")
POPULATION_BASELINE = 0.1609


# ============================================================
# Data loading
# ============================================================

def load_pairs():
    df = pd.read_parquet(
        os.path.join(ARTIFACTS, "pairs.parquet"),
        columns=["Company1", "Company2", "year", "correlation", "cosine_similarity"],
    )
    df["year"] = df["year"].astype(int)
    return df


def load_companies():
    df = pd.read_parquet(
        os.path.join(ARTIFACTS, "companies.parquet"),
        columns=["__index_level_0__", "ticker"],
    )
    return df


def load_clusters(method):
    path = os.path.join(CLUSTERS_DIR, f"{method}.pkl")
    with open(path, "rb") as f:
        df = pickle.load(f)
    df["year"] = df["year"].astype(int)
    return df


# ============================================================
# Scaling and MST construction
# ============================================================

def scale_distances(pairs_df, train_years):
    """Fit StandardScaler on cosine_distance for train_years, transform all."""
    pairs_df = pairs_df.copy()
    pairs_df["cosine_distance"] = 1.0 - pairs_df["cosine_similarity"]
    scaler = StandardScaler()
    train_mask = pairs_df["year"].isin(set(train_years))
    scaler.fit(pairs_df.loc[train_mask, ["cosine_distance"]])
    pairs_df["cosine_distance_scaled"] = scaler.transform(
        pairs_df[["cosine_distance"]]
    )
    return pairs_df, scaler


def build_msts(pairs_df, years):
    """Build MST for each year using scipy.sparse.csgraph."""
    msts = {}
    for year in years:
        year_df = pairs_df[pairs_df["year"] == year]
        companies = sorted(set(year_df["Company1"]) | set(year_df["Company2"]))
        comp_to_idx = {c: i for i, c in enumerate(companies)}
        n = len(companies)

        rows = year_df["Company1"].map(comp_to_idx).values
        cols = year_df["Company2"].map(comp_to_idx).values
        vals = year_df["cosine_distance_scaled"].values

        # Ensure upper triangle (row < col) for scipy
        swap = rows > cols
        rows_ut = np.where(swap, cols, rows)
        cols_ut = np.where(swap, rows, cols)

        mat = csr_matrix((vals, (rows_ut, cols_ut)), shape=(n, n))
        mst = minimum_spanning_tree(mat)
        mst_coo = mst.tocoo()

        msts[year] = {
            "rows": mst_coo.row,
            "cols": mst_coo.col,
            "weights": mst_coo.data,
            "companies": companies,
            "comp_to_idx": comp_to_idx,
            "n": n,
        }
    return msts


def threshold_mst(mst_info, theta):
    """Threshold MST at theta, return connected component labels."""
    rows, cols, weights = mst_info["rows"], mst_info["cols"], mst_info["weights"]
    n = mst_info["n"]

    keep = weights <= theta
    if not keep.any():
        return np.arange(n)

    kr, kc = rows[keep], cols[keep]
    ones = np.ones(kr.shape[0])
    sym = csr_matrix(
        (np.concatenate([ones, ones]),
         (np.concatenate([kr, kc]), np.concatenate([kc, kr]))),
        shape=(n, n),
    )
    _, labels = connected_components(sym, directed=False)
    return labels


def compute_mc_from_labels(labels, c1_local, c2_local, corr):
    """Compute MC given cluster labels and pre-indexed pair data."""
    c1_lab = labels[c1_local]
    c2_lab = labels[c2_local]
    same = c1_lab == c2_lab
    if not same.any():
        return float("nan")

    wl = c1_lab[same]
    wc = corr[same]
    n_labels = labels.max() + 1
    sums = np.bincount(wl, weights=wc, minlength=n_labels)
    counts = np.bincount(wl, minlength=n_labels)
    valid = counts > 0
    if not valid.any():
        return float("nan")
    return float((sums[valid] / counts[valid]).mean())


# ============================================================
# Part A: Theta sweep
# ============================================================

def theta_sweep(pairs_df, msts, years):
    print("\n=== Part A: Theta Sweep ===")
    t0 = time.time()

    all_weights = np.concatenate([msts[y]["weights"] for y in years])
    lo, hi = np.percentile(all_weights, [5, 95])
    thresholds = np.linspace(lo, hi, 60)
    print(f"  Threshold range: [{lo:.4f}, {hi:.4f}]")

    # Pre-index pairs per year for MC computation
    year_pair_data = {}
    for year in years:
        ydf = pairs_df[pairs_df["year"] == year]
        c2i = msts[year]["comp_to_idx"]
        year_pair_data[year] = (
            ydf["Company1"].map(c2i).values.astype(int),
            ydf["Company2"].map(c2i).values.astype(int),
            ydf["correlation"].values,
        )

    mc_curve = []
    for theta in thresholds:
        yearly_mcs = []
        for year in years:
            labels = threshold_mst(msts[year], theta)
            c1, c2, co = year_pair_data[year]
            mc = compute_mc_from_labels(labels, c1, c2, co)
            if not math.isnan(mc):
                yearly_mcs.append(mc)
        mc_curve.append(np.mean(yearly_mcs) if yearly_mcs else float("nan"))

    mc_curve = np.array(mc_curve)
    best_idx = int(np.nanargmax(mc_curve))
    optimal_theta = float(thresholds[best_idx])
    optimal_mc = float(mc_curve[best_idx])

    # CV within +/-10% of optimal theta
    abs_theta = abs(optimal_theta) if optimal_theta != 0 else 1.0
    band = 0.10 * abs_theta
    near = (thresholds >= optimal_theta - band) & (thresholds <= optimal_theta + band)
    if near.sum() >= 2:
        near_mcs = mc_curve[near]
        near_mcs = near_mcs[~np.isnan(near_mcs)]
        cv = float(np.std(near_mcs) / np.mean(near_mcs)) if len(near_mcs) >= 2 else float("nan")
    else:
        cv = float("nan")

    if math.isnan(cv):
        cv_class = "insufficient_data"
    elif cv <= 0.10:
        cv_class = "stable"
    elif cv <= 0.20:
        cv_class = "moderate"
    else:
        cv_class = "brittle"

    elapsed = time.time() - t0
    print(f"  Optimal theta: {optimal_theta:.4f} (ACL: -2.700)")
    print(f"  Optimal MC: {optimal_mc:.6f}")
    print(f"  CV near optimal (+/-10%): {cv:.6f} ({cv_class})")
    print(f"  Elapsed: {elapsed:.1f}s")

    return {
        "thresholds": [round(float(t), 6) for t in thresholds],
        "mc_values": [round(float(m), 6) if not math.isnan(m) else None for m in mc_curve],
        "optimal_theta": round(optimal_theta, 6),
        "optimal_mc": round(optimal_mc, 6),
        "acl_theta": -2.7,
        "cv_near_optimal": round(cv, 6) if not math.isnan(cv) else None,
        "cv_classification": cv_class,
        "n_thresholds": 60,
    }


# ============================================================
# Part B: Bootstrap CIs
# ============================================================

def precompute_within_cluster_pairs(pairs_df, cluster_df, comp_to_ticker, ticker_to_idx):
    """Build flat arrays of (ticker1_idx, ticker2_idx, correlation) grouped by (year, cluster)."""
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

        # Build company -> cluster label mapping
        c2cl = {}
        for cid, members in clusters.items():
            for m in members:
                c2cl[m] = cid

        # Map pair endpoints to cluster labels
        c1_cl = yp["Company1"].map(c2cl)
        c2_cl = yp["Company2"].map(c2cl)
        same = (c1_cl == c2_cl) & c1_cl.notna() & c2_cl.notna()
        within = yp[same]
        if len(within) == 0:
            continue

        # Map to ticker indices
        t1 = within["Company1"].map(comp_to_ticker)
        t2 = within["Company2"].map(comp_to_ticker)
        valid = t1.notna() & t2.notna()
        if valid.sum() == 0:
            continue

        t1_mapped = t1[valid].map(ticker_to_idx)
        t2_mapped = t2[valid].map(ticker_to_idx)
        tv = t1_mapped.notna() & t2_mapped.notna()
        if tv.sum() == 0:
            continue

        t1_arr = t1_mapped[tv].values.astype(int)
        t2_arr = t2_mapped[tv].values.astype(int)
        corr_arr = within.loc[tv.values if hasattr(tv, 'values') else tv, "correlation"].values
        cl_arr = c1_cl[same][valid][tv].values

        # Split by cluster label — each cluster is a separate group
        unique_cls = np.unique(cl_arr)
        for cl in unique_cls:
            members = clusters.get(cl, clusters.get(int(cl), []))
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

    # Compute group start indices
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


def compute_weighted_mc(pdata, multiplicities):
    """Compute weighted MC from ticker multiplicities."""
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


def bca_ci(boot_dist, original, jack_stats, alpha=0.05):
    """BCa confidence interval."""
    B = len(boot_dist)

    # Bias correction
    prop = np.clip(np.sum(boot_dist < original) / B, 1e-10, 1 - 1e-10)
    z0 = float(norm.ppf(prop))

    # Acceleration from jackknife
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


def bootstrap_cis(pairs_df, comp_to_ticker, ticker_to_idx, n_tickers, n_bootstrap=10000):
    print("\n=== Part B: Bootstrap CIs ===")
    t0 = time.time()
    rng = np.random.default_rng(42)

    methods = {"sae": "sae_cd", "sic": "sic", "sbert": "sbert"}
    pdata = {}
    orig_mc = {}

    for label, method in methods.items():
        print(f"  Pre-computing within-cluster pairs for {label}...")
        cluster_df = load_clusters(method)
        pd_ = precompute_within_cluster_pairs(
            pairs_df, cluster_df, comp_to_ticker, ticker_to_idx
        )
        if pd_ is None:
            print(f"    WARNING: no within-cluster pairs for {label}")
            sys.exit(1)
        pdata[label] = pd_
        ones = np.ones(n_tickers, dtype=np.float64)
        orig_mc[label] = compute_weighted_mc(pd_, ones)
        print(f"    {label}: MC={orig_mc[label]:.6f}, {pd_['n_pairs']:,} pairs, {pd_['n_groups']} groups")

    # Jackknife: leave one ticker out
    print(f"  Jackknife ({n_tickers} tickers)...")
    t_jack = time.time()
    jack = {label: np.zeros(n_tickers) for label in methods}
    for i in range(n_tickers):
        mult = np.ones(n_tickers, dtype=np.float64)
        mult[i] = 0.0
        for label in methods:
            jack[label][i] = compute_weighted_mc(pdata[label], mult)
    print(f"    Done in {time.time() - t_jack:.1f}s")

    # Bootstrap
    print(f"  Bootstrap ({n_bootstrap} iterations)...")
    t_boot = time.time()
    boot = {label: np.zeros(n_bootstrap) for label in methods}
    for b in range(n_bootstrap):
        if (b + 1) % 2000 == 0:
            el = time.time() - t_boot
            rate = (b + 1) / el
            print(f"    {b+1}/{n_bootstrap} ({el:.0f}s, ~{(n_bootstrap-b-1)/rate:.0f}s left)")
        draw = rng.choice(n_tickers, n_tickers, replace=True)
        mult = np.bincount(draw, minlength=n_tickers).astype(np.float64)
        for label in methods:
            boot[label][b] = compute_weighted_mc(pdata[label], mult)
    print(f"    Done in {time.time() - t_boot:.1f}s")

    # BCa CIs for each method
    method_results = {}
    for label in methods:
        ci_lo, ci_hi, z0, a = bca_ci(boot[label], orig_mc[label], jack[label])
        method_results[label] = {
            "mc": round(orig_mc[label], 6),
            "ci_lower": round(ci_lo, 6),
            "ci_upper": round(ci_hi, 6),
            "z0": round(z0, 6),
            "a": round(a, 6),
            "bootstrap_mean": round(float(np.mean(boot[label])), 6),
            "bootstrap_std": round(float(np.std(boot[label])), 6),
        }
        print(f"  {label}: MC={orig_mc[label]:.6f}, 95% CI=[{ci_lo:.6f}, {ci_hi:.6f}], z0={z0:.4f}, a={a:.6f}")

    # Deltas
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
            orig_d = orig_mc[ma] - POPULATION_BASELINE
            boot_d = boot[ma] - POPULATION_BASELINE
            jack_d = jack[ma] - POPULATION_BASELINE

        ci_lo, ci_hi, z0, a = bca_ci(boot_d, orig_d, jack_d)
        p_val = float(np.mean(boot_d <= 0))

        deltas[name] = {
            "original_delta": round(float(orig_d), 6),
            "ci_lower": round(ci_lo, 6),
            "ci_upper": round(ci_hi, 6),
            "z0": round(z0, 6),
            "a": round(a, 6),
            "p_value": round(p_val, 6),
        }
        print(f"  {name}: delta={orig_d:.6f}, 95% CI=[{ci_lo:.6f}, {ci_hi:.6f}], p={p_val:.6f}")

    print(f"  Total Part B: {time.time() - t0:.1f}s")
    return {"n_iterations": n_bootstrap, "methods": method_results, "deltas": deltas}


# ============================================================
# Part C: Random baseline
# ============================================================

def random_baseline(pairs_df, n_iterations=1000):
    print("\n=== Part C: Random Baseline ===")
    t0 = time.time()
    rng = np.random.default_rng(123)

    sae_clusters = load_clusters("sae_cd")
    years = sorted(sae_clusters["year"].unique())

    # Pre-index per year
    year_data = {}
    pairs_by_year = {yr: grp for yr, grp in pairs_df.groupby("year")}

    for _, row in sae_clusters.iterrows():
        year = int(row["year"])
        clusters = row["clusters"]

        # Collect all companies and cluster sizes
        all_comps = []
        cluster_sizes = []
        for cid in sorted(clusters.keys()):
            members = clusters[cid]
            all_comps.extend(members)
            cluster_sizes.append(len(members))

        all_comps = np.array(all_comps)
        cluster_sizes = np.array(cluster_sizes)
        n_comp = len(all_comps)

        # Label template: first s0 get label 0, next s1 get label 1, etc.
        labels_template = np.concatenate(
            [np.full(s, i, dtype=int) for i, s in enumerate(cluster_sizes)]
        )

        # Get pairs for this year, restricted to companies in clusters
        yp = pairs_by_year.get(year)
        if yp is None:
            continue
        comp_set = set(all_comps)
        mask = yp["Company1"].isin(comp_set) & yp["Company2"].isin(comp_set)
        yp = yp[mask]
        if len(yp) == 0:
            continue

        comp_to_local = {c: i for i, c in enumerate(all_comps)}
        c1_local = yp["Company1"].map(comp_to_local)
        c2_local = yp["Company2"].map(comp_to_local)
        valid = c1_local.notna() & c2_local.notna()
        c1_local = c1_local[valid].values.astype(int)
        c2_local = c2_local[valid].values.astype(int)
        corr = yp.loc[valid.values, "correlation"].values

        year_data[year] = {
            "n_companies": n_comp,
            "cluster_sizes": cluster_sizes,
            "labels_template": labels_template,
            "c1_local": c1_local,
            "c2_local": c2_local,
            "corr": corr,
            "n_clusters": len(cluster_sizes),
        }

    random_mcs = np.zeros(n_iterations)
    for it in range(n_iterations):
        if (it + 1) % 200 == 0:
            print(f"    Iteration {it + 1}/{n_iterations}")

        yearly_mcs = []
        for year in years:
            yd = year_data.get(year)
            if yd is None:
                continue
            perm = rng.permutation(yd["n_companies"])
            labels = np.empty(yd["n_companies"], dtype=int)
            labels[perm] = yd["labels_template"]

            c1_lab = labels[yd["c1_local"]]
            c2_lab = labels[yd["c2_local"]]
            same = c1_lab == c2_lab
            if not same.any():
                continue

            wl = c1_lab[same]
            wc = yd["corr"][same]
            n_cl = yd["n_clusters"]
            sums = np.bincount(wl, weights=wc, minlength=n_cl)
            counts = np.bincount(wl, minlength=n_cl)
            valid = counts > 0
            if valid.any():
                yearly_mcs.append((sums[valid] / counts[valid]).mean())

        random_mcs[it] = np.mean(yearly_mcs) if yearly_mcs else float("nan")

    mean_mc = float(np.nanmean(random_mcs))
    std_mc = float(np.nanstd(random_mcs))
    elapsed = time.time() - t0

    print(f"  Random baseline: {mean_mc:.6f} +/- {std_mc:.6f}")
    print(f"  Range check (0.13-0.19): {'PASS' if 0.13 <= mean_mc <= 0.19 else 'FAIL'}")
    print(f"  Elapsed: {elapsed:.1f}s")

    return {
        "n_iterations": n_iterations,
        "mean": round(mean_mc, 6),
        "std": round(std_mc, 6),
        "min": round(float(np.nanmin(random_mcs)), 6),
        "max": round(float(np.nanmax(random_mcs)), 6),
    }


# ============================================================
# Part D: Train/test split
# ============================================================

def train_test_evaluation(pairs_df_raw):
    print("\n=== Part D: Train/Test Split ===")
    t0 = time.time()

    all_years = sorted(pairs_df_raw["year"].unique())
    train_years = [y for y in all_years if y <= 2010]
    test_years = [y for y in all_years if y > 2010]
    print(f"  Train: {min(train_years)}-{max(train_years)} ({len(train_years)} years)")
    print(f"  Test: {min(test_years)}-{max(test_years)} ({len(test_years)} years)")

    # Separate scaler on train years only
    pairs_df, _ = scale_distances(pairs_df_raw, train_years)

    # Build MSTs for all years
    print("  Building MSTs (train/test scaler)...")
    msts = build_msts(pairs_df, all_years)

    # Pre-index pairs
    year_pair_data = {}
    for year in all_years:
        ydf = pairs_df[pairs_df["year"] == year]
        c2i = msts[year]["comp_to_idx"]
        year_pair_data[year] = (
            ydf["Company1"].map(c2i).values.astype(int),
            ydf["Company2"].map(c2i).values.astype(int),
            ydf["correlation"].values,
        )

    # Threshold range from all MSTs
    all_w = np.concatenate([msts[y]["weights"] for y in all_years])
    lo, hi = np.percentile(all_w, [5, 95])
    thresholds = np.linspace(lo, hi, 60)

    # Sweep theta on train years only
    train_curve = []
    for theta in thresholds:
        ymcs = []
        for year in train_years:
            labels = threshold_mst(msts[year], theta)
            c1, c2, co = year_pair_data[year]
            mc = compute_mc_from_labels(labels, c1, c2, co)
            if not math.isnan(mc):
                ymcs.append(mc)
        train_curve.append(np.mean(ymcs) if ymcs else float("nan"))

    train_curve = np.array(train_curve)
    best_idx = int(np.nanargmax(train_curve))
    train_theta = float(thresholds[best_idx])
    train_mc = float(train_curve[best_idx])

    # Apply training-optimal theta to test years
    test_ymcs = []
    for year in test_years:
        labels = threshold_mst(msts[year], train_theta)
        c1, c2, co = year_pair_data[year]
        mc = compute_mc_from_labels(labels, c1, c2, co)
        if not math.isnan(mc):
            test_ymcs.append(mc)
    test_mc = float(np.mean(test_ymcs)) if test_ymcs else float("nan")
    gap = train_mc - test_mc

    elapsed = time.time() - t0
    print(f"  Train optimal theta: {train_theta:.4f}")
    print(f"  Train MC: {train_mc:.6f}")
    print(f"  Held-out MC: {test_mc:.6f}")
    print(f"  Gap: {gap:.6f}")
    print(f"  Elapsed: {elapsed:.1f}s")

    return {
        "train_years": f"{min(train_years)}-{max(train_years)}",
        "test_years": f"{min(test_years)}-{max(test_years)}",
        "train_optimal_theta": round(train_theta, 6),
        "train_mc": round(train_mc, 6),
        "test_mc": round(test_mc, 6),
        "gap": round(gap, 6),
    }


# ============================================================
# Main
# ============================================================

def main():
    print("Phase 1A Statistics")
    print("=" * 80)

    t_start = time.time()
    print("\nLoading data...")
    pairs_df_raw = load_pairs()
    companies_df = load_companies()
    print(f"  Pairs: {len(pairs_df_raw):,} rows")
    print(f"  Companies: {len(companies_df):,} rows")

    # Company -> ticker mapping (ticker column stores arrays; take first element)
    primary_ticker = companies_df["ticker"].apply(
        lambda x: x[0] if hasattr(x, "__len__") and len(x) > 0 else x
    )
    comp_to_ticker = dict(
        zip(companies_df["__index_level_0__"], primary_ticker)
    )
    unique_tickers = sorted(set(comp_to_ticker.values()))
    ticker_to_idx = {t: i for i, t in enumerate(unique_tickers)}
    n_tickers = len(unique_tickers)
    print(f"  Unique tickers: {n_tickers}")

    # Scale distances (ACL: fit on first 75% of years)
    all_years = sorted(pairs_df_raw["year"].unique())
    split_end = int(0.75 * len(all_years))
    train_years_acl = list(all_years[:split_end])
    print(f"  ACL scaler years: {min(train_years_acl)}-{max(train_years_acl)} ({len(train_years_acl)} years)")

    pairs_df, _ = scale_distances(pairs_df_raw, train_years_acl)

    # Build MSTs
    print("  Building MSTs...")
    msts = build_msts(pairs_df, list(all_years))
    print(f"  MSTs built for {len(msts)} years")

    results = {}

    # Part A
    results["theta_sweep"] = theta_sweep(pairs_df, msts, list(all_years))

    # Free scaled data before bootstrap (bootstrap only needs correlations)
    del pairs_df, msts

    # Part B
    results["bootstrap"] = bootstrap_cis(
        pairs_df_raw, comp_to_ticker, ticker_to_idx, n_tickers, n_bootstrap=10000
    )

    # Part C
    results["random_baseline"] = random_baseline(pairs_df_raw, n_iterations=1000)

    # Part D
    results["train_test"] = train_test_evaluation(pairs_df_raw)

    # ============================================================
    # Summary
    # ============================================================
    print("\n" + "=" * 80)
    print("RESULTS SUMMARY")
    print("=" * 80)

    ts = results["theta_sweep"]
    print(f"\n--- Theta Sweep ---")
    print(f"  60 thresholds: [{ts['thresholds'][0]:.4f}, {ts['thresholds'][-1]:.4f}]")
    print(f"  Optimal theta: {ts['optimal_theta']:.4f} (ACL: {ts['acl_theta']})")
    print(f"  Optimal MC: {ts['optimal_mc']:.6f}")
    cv_str = f"{ts['cv_near_optimal']:.6f}" if ts['cv_near_optimal'] is not None else "N/A"
    print(f"  CV near optimal: {cv_str} ({ts['cv_classification']})")

    bs = results["bootstrap"]
    print(f"\n--- Bootstrap CIs (n={bs['n_iterations']}) ---")
    for label, data in bs["methods"].items():
        print(f"  {label}: MC={data['mc']:.6f}, 95% CI=[{data['ci_lower']:.6f}, {data['ci_upper']:.6f}]")
        print(f"         z0={data['z0']:.4f}, a={data['a']:.6f}")

    sae_ci = bs["methods"]["sae"]
    baseline_in_ci = sae_ci["ci_lower"] <= POPULATION_BASELINE <= sae_ci["ci_upper"]
    status = "FAIL: baseline in CI" if baseline_in_ci else "PASS: baseline excluded"
    print(f"  SAE CI vs baseline ({POPULATION_BASELINE}): {status}")

    print(f"\n  Deltas:")
    for dname, dd in bs["deltas"].items():
        print(f"    {dname}: delta={dd['original_delta']:.6f}, "
              f"CI=[{dd['ci_lower']:.6f}, {dd['ci_upper']:.6f}], p={dd['p_value']:.6f}")

    rb = results["random_baseline"]
    print(f"\n--- Random Baseline (n={rb['n_iterations']}) ---")
    print(f"  Mean: {rb['mean']:.6f} +/- {rb['std']:.6f}")
    in_range = 0.13 <= rb["mean"] <= 0.19
    print(f"  Range check (0.13-0.19): {'PASS' if in_range else 'FAIL'}")

    tt = results["train_test"]
    print(f"\n--- Train/Test Split ---")
    print(f"  Train ({tt['train_years']}): theta={tt['train_optimal_theta']:.4f}, MC={tt['train_mc']:.6f}")
    print(f"  Test ({tt['test_years']}): MC={tt['test_mc']:.6f}")
    print(f"  Gap: {tt['gap']:.6f}")

    # Save
    output_path = os.path.join(ARTIFACTS, "1a_statistics.json")
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved: {output_path}")

    # ============================================================
    # Verification
    # ============================================================
    print("\n" + "=" * 80)
    print("VERIFICATION CHECKS")
    print("=" * 80)

    ok = True

    def check(name, passed):
        nonlocal ok
        ok &= passed
        print(f"  [{'PASS' if passed else 'FAIL'}] {name}")

    check("Theta curve has 60 points", len(ts["thresholds"]) == 60)
    check(f"Optimal theta reported ({ts['optimal_theta']:.4f})", ts["optimal_theta"] is not None)
    check(f"CV classified: {ts['cv_classification']}",
          ts["cv_classification"] in ("stable", "moderate", "brittle"))
    check(f"Bootstrap ran {bs['n_iterations']} iterations", bs["n_iterations"] == 10000)
    check(f"SAE CI [{sae_ci['ci_lower']:.4f}, {sae_ci['ci_upper']:.4f}] excludes baseline {POPULATION_BASELINE}",
          not baseline_in_ci)
    check("BCa z0, a reported for all methods",
          all("z0" in bs["methods"][m] and "a" in bs["methods"][m] for m in bs["methods"]))
    check("Delta CIs for SAE-SIC and SAE-SBERT",
          "sae_minus_sic" in bs["deltas"] and "sae_minus_sbert" in bs["deltas"])
    check(f"Random baseline mean {rb['mean']:.6f} in [0.13, 0.19]",
          0.13 <= rb["mean"] <= 0.19)
    check(f"Train MC={tt['train_mc']:.6f}, Test MC={tt['test_mc']:.6f} both reported",
          not math.isnan(tt["train_mc"]) and not math.isnan(tt["test_mc"]))

    total = time.time() - t_start
    print(f"\nTotal: {total:.1f}s")

    if ok:
        print("\nAll verification checks PASSED.")
        sys.exit(0)
    else:
        print("\nSome verification checks FAILED.")
        sys.exit(1)


if __name__ == "__main__":
    main()

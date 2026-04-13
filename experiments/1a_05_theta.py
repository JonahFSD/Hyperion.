"""
1a_05_theta.py — Theta Sensitivity (Diagnostic Only)

Re-derives clusters from cosine_similarity in pairs.parquet to verify
the shape of the theta curve. This is NOT a formal test — it confirms
the ACL's theta choice (-2.7) is not fragile.

Note: MC values here will NOT match 1a_02 because clusters are re-derived
from scratch, not loaded from ACL pre-computed labels.
"""

import json
import math
import os
import sys
import time

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import minimum_spanning_tree, connected_components

ARTIFACTS = "phase1_artifacts"


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


# ============================================================
# Scaling
# ============================================================

def scale_distances(pairs_df):
    """Compute cosine distance and fit StandardScaler on ALL years.

    This differs from the ACL code, which fits on the first 75% of years.
    We fit on all years because this is a diagnostic (not a replication)
    and we want the fullest possible picture of the theta landscape.
    """
    pairs_df = pairs_df.copy()
    pairs_df["cosine_distance"] = 1.0 - pairs_df["cosine_similarity"]

    # Manual StandardScaler: (x - mean) / std
    cd_mean = pairs_df["cosine_distance"].mean()
    cd_std = pairs_df["cosine_distance"].std(ddof=0)  # population std like sklearn
    pairs_df["cosine_distance_scaled"] = (pairs_df["cosine_distance"] - cd_mean) / cd_std
    return pairs_df


# ============================================================
# MST construction
# ============================================================

def build_msts(pairs_df, years):
    """Build MST for each year using scipy.sparse.csgraph.minimum_spanning_tree."""
    msts = {}
    for year in years:
        year_df = pairs_df[pairs_df["year"] == year]
        companies = sorted(set(year_df["Company1"]) | set(year_df["Company2"]))
        comp_to_idx = {c: i for i, c in enumerate(companies)}
        n = len(companies)

        rows = year_df["Company1"].map(comp_to_idx).values
        cols = year_df["Company2"].map(comp_to_idx).values
        vals = year_df["cosine_distance_scaled"].values

        # scipy minimum_spanning_tree needs upper triangle
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


# ============================================================
# Clustering and MC computation
# ============================================================

def threshold_mst(mst_info, theta):
    """Threshold MST at theta: keep edges with weight <= theta.
    Returns connected component labels."""
    rows, cols, weights = mst_info["rows"], mst_info["cols"], mst_info["weights"]
    n = mst_info["n"]

    keep = weights <= theta
    if not keep.any():
        # No edges kept — every node is its own cluster
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
    """Compute MC given cluster labels and pre-indexed pair data.

    Equal-weighted across clusters within a year (matching ACL method).
    """
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

    # Only include clusters with >1 member (i.e., at least one pair)
    valid = counts > 0
    if not valid.any():
        return float("nan")
    return float((sums[valid] / counts[valid]).mean())


def compute_mc_all_years(msts, year_pair_data, years, theta):
    """Compute MC at a given theta across all years."""
    yearly_mcs = []
    for year in years:
        labels = threshold_mst(msts[year], theta)
        c1, c2, co = year_pair_data[year]
        mc = compute_mc_from_labels(labels, c1, c2, co)
        if not math.isnan(mc):
            yearly_mcs.append(mc)
    return float(np.mean(yearly_mcs)) if yearly_mcs else float("nan")


# ============================================================
# Main
# ============================================================

def main():
    print("1a_05_theta.py — Theta Sensitivity (Diagnostic Only)")
    print("=" * 70)
    t_start = time.time()

    # --- Load and scale ---
    print("\nLoading pairs data...")
    pairs_df = load_pairs()
    print(f"  {len(pairs_df):,} rows")

    print("Scaling cosine distances (StandardScaler fit on ALL years)...")
    pairs_df = scale_distances(pairs_df)

    years = sorted(pairs_df["year"].unique())
    print(f"  Years: {years[0]}-{years[-1]} ({len(years)} years)")

    # --- Build MSTs ---
    print("Building MSTs per year...")
    t_mst = time.time()
    msts = build_msts(pairs_df, years)
    print(f"  {len(msts)} MSTs built in {time.time() - t_mst:.1f}s")

    # --- Pre-index pairs per year ---
    year_pair_data = {}
    for year in years:
        ydf = pairs_df[pairs_df["year"] == year]
        c2i = msts[year]["comp_to_idx"]
        year_pair_data[year] = (
            ydf["Company1"].map(c2i).values.astype(int),
            ydf["Company2"].map(c2i).values.astype(int),
            ydf["correlation"].values,
        )

    # --- Determine threshold range ---
    all_weights = np.concatenate([msts[y]["weights"] for y in years])
    lo, hi = np.percentile(all_weights, [5, 95])
    n_thresholds = 100
    thresholds = np.linspace(lo, hi, n_thresholds)
    print(f"\nSweeping {n_thresholds} thresholds: [{lo:.4f}, {hi:.4f}]")

    # --- Sweep ---
    t_sweep = time.time()
    mc_values = []
    for i, theta in enumerate(thresholds):
        mc = compute_mc_all_years(msts, year_pair_data, years, theta)
        mc_values.append(mc)
        if (i + 1) % 25 == 0:
            print(f"  {i + 1}/{n_thresholds} thresholds evaluated...")

    mc_values = np.array(mc_values)
    print(f"  Sweep complete in {time.time() - t_sweep:.1f}s")

    # --- Find optimal ---
    valid_mask = ~np.isnan(mc_values)
    if not valid_mask.any():
        print("\nFAIL: All MC values are NaN.")
        sys.exit(1)

    best_idx = int(np.nanargmax(mc_values))
    optimal_theta = float(thresholds[best_idx])
    optimal_mc = float(mc_values[best_idx])

    # --- Compute MC at ACL theta = -2.7 ---
    acl_theta = -2.7
    mc_at_acl = compute_mc_all_years(msts, year_pair_data, years, acl_theta)

    # Ratio: how close is ACL theta to optimal?
    if not math.isnan(mc_at_acl) and not math.isnan(optimal_mc) and optimal_mc > 0:
        ratio = mc_at_acl / optimal_mc
    else:
        ratio = float("nan")

    # --- Print results ---
    print(f"\nResults:")
    print(f"  Optimal theta:        {optimal_theta:.6f}")
    print(f"  Optimal MC:           {optimal_mc:.6f}")
    print(f"  ACL theta:            {acl_theta}")
    print(f"  MC at ACL theta:      {mc_at_acl:.6f}")
    print(f"  Ratio (ACL/optimal):  {ratio:.6f}")

    # --- Success criteria ---
    ok = True

    # Curve has a clear peak (not monotonic)
    mc_valid = mc_values[valid_mask]
    is_monotonic = np.all(np.diff(mc_valid) >= 0) or np.all(np.diff(mc_valid) <= 0)
    if is_monotonic:
        print("\n  WARNING: MC curve appears monotonic (no clear peak)")
        ok = False
    else:
        print("  CHECK: MC curve has a clear peak (not monotonic) — OK")

    # Optimal theta within 1.0 of ACL's -2.7
    theta_dist = abs(optimal_theta - acl_theta)
    if theta_dist <= 1.0:
        print(f"  CHECK: Optimal theta within 1.0 of ACL theta (distance={theta_dist:.4f}) — OK")
    else:
        print(f"  WARNING: Optimal theta is {theta_dist:.4f} from ACL theta (>1.0)")
        ok = False

    # MC at ACL theta is reported (not NaN)
    if math.isnan(mc_at_acl):
        print("  WARNING: MC at ACL theta is NaN")
        ok = False
    else:
        print(f"  CHECK: MC at ACL theta reported — OK")

    # --- Save JSON ---
    output = {
        "scaler": "StandardScaler fit on all years (1996-2020)",
        "n_thresholds": n_thresholds,
        "thresholds": [round(float(t), 6) for t in thresholds],
        "mc_values": [
            round(float(m), 6) if not math.isnan(m) else None
            for m in mc_values
        ],
        "optimal_theta": round(optimal_theta, 6),
        "optimal_mc": round(optimal_mc, 6),
        "acl_theta": acl_theta,
        "mc_at_acl_theta": round(mc_at_acl, 6) if not math.isnan(mc_at_acl) else None,
        "ratio_acl_to_optimal": round(ratio, 6) if not math.isnan(ratio) else None,
        "note": "Diagnostic only. Clusters re-derived from cosine similarity matrix, not ACL pre-computed labels.",
    }

    output_path = os.path.join(ARTIFACTS, "1a_theta.json")
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved: {output_path}")

    # --- Write report section ---
    write_report(output)

    elapsed = time.time() - t_start
    print(f"\nTotal time: {elapsed:.1f}s")

    if ok:
        print("All checks passed.")
        sys.exit(0)
    else:
        print("Some checks raised warnings (diagnostic — not a hard failure).")
        sys.exit(0)  # Diagnostic only — warnings do not cause failure


def write_report(output):
    """Write the report section for 1a_05."""
    acl_mc = output["mc_at_acl_theta"]
    acl_mc_str = f"{acl_mc:.6f}" if acl_mc is not None else "N/A"
    ratio = output["ratio_acl_to_optimal"]
    ratio_str = f"{ratio:.6f}" if ratio is not None else "N/A"

    report = f"""## 5. Theta Sensitivity (Diagnostic)

### Observations

- **Scaler:** {output['scaler']}
- **Threshold range:** {output['n_thresholds']} thresholds from {output['thresholds'][0]:.4f} to {output['thresholds'][-1]:.4f} (5th–95th percentile of MST edge weights)
- **Optimal theta:** {output['optimal_theta']:.6f}
- **MC at optimal theta:** {output['optimal_mc']:.6f}
- **ACL theta:** {output['acl_theta']}
- **MC at ACL theta:** {acl_mc_str}
- **Ratio (ACL / optimal):** {ratio_str}

Note: These MC values do not match Section 2 (MC Replication) because clusters are re-derived here from the cosine similarity matrix, not loaded from ACL pre-computed labels.

### Interpretation

*To be completed after running the script.*

Key questions:
- Is the theta curve flat or sharp near the peak? A flat peak means MC is insensitive to the exact threshold choice; a sharp peak means the result is fragile.
- How does the ratio of MC at ACL theta to optimal MC compare? A ratio > 0.95 means the ACL's choice of -2.7 barely matters — any nearby theta gives similar performance.
- Does the optimal theta found here agree with the ACL's -2.7? Agreement is expected but not guaranteed, because our scaler differs (fit on all years vs. first 75%).
"""

    report_path = os.path.join(ARTIFACTS, "1a_report_05.md")
    with open(report_path, "w") as f:
        f.write(report)
    print(f"Saved: {report_path}")


if __name__ == "__main__":
    main()

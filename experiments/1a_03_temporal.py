"""
Phase 1A Task 3: Temporal Delta Analysis.
Computes year-by-year deltas (SAE minus SIC, SAE minus SBERT),
OLS regression of delta vs year, and bootstrap CI on the slope.
"""

import json
import os
import sys

import numpy as np

ARTIFACTS = "phase1_artifacts"
INPUT_FILE = os.path.join(ARTIFACTS, "1a_mc_by_year.json")
OUTPUT_FILE = os.path.join(ARTIFACTS, "1a_temporal.json")
REPORT_FILE = os.path.join(ARTIFACTS, "1a_report_03.md")

N_BOOTSTRAP = 10000
SEED = 42
CI_LEVEL = 0.95


def load_mc_by_year():
    with open(INPUT_FILE) as f:
        return json.load(f)


def get_year_values(mc_by_year, method):
    """Return {int_year: float_mc} excluding null entries.

    Handles both old format {year: float} and new format {year: {"mean": float, "median": float}}.
    Uses "mean" sub-key when present.
    """
    method_data = mc_by_year[method]
    # Detect new nested format: if first non-null value is a dict, use "mean" sub-key
    sample_val = next((v for v in method_data.values() if v is not None), None)
    if isinstance(sample_val, dict):
        # New format: {"mean": {...}, "median": {...}}
        mean_data = method_data.get("mean", {})
        return {
            int(y): v for y, v in mean_data.items() if v is not None
        }
    else:
        # Old format: {year: float}
        return {
            int(y): v for y, v in method_data.items() if v is not None
        }


def compute_deltas(sae_vals, baseline_vals):
    """Compute delta = SAE - baseline for overlapping years. Returns sorted arrays."""
    overlap_years = sorted(set(sae_vals) & set(baseline_vals))
    years = np.array(overlap_years)
    deltas = np.array([sae_vals[y] - baseline_vals[y] for y in overlap_years])
    return years, deltas


def ols_slope_intercept(x, y):
    """Ordinary least squares: y = intercept + slope * x."""
    n = len(x)
    x_mean = x.mean()
    y_mean = y.mean()
    ss_xy = np.sum((x - x_mean) * (y - y_mean))
    ss_xx = np.sum((x - x_mean) ** 2)
    slope = ss_xy / ss_xx
    intercept = y_mean - slope * x_mean
    return slope, intercept


def bootstrap_slope_ci(years, deltas, n_bootstrap, seed, ci_level):
    """Bootstrap CI on OLS slope by resampling (year, delta) pairs."""
    rng = np.random.RandomState(seed)
    n = len(years)
    slopes = np.empty(n_bootstrap)
    for i in range(n_bootstrap):
        idx = rng.randint(0, n, size=n)
        slopes[i] = ols_slope_intercept(years[idx], deltas[idx])[0]

    alpha = 1.0 - ci_level
    lower = np.percentile(slopes, 100 * alpha / 2)
    upper = np.percentile(slopes, 100 * (1 - alpha / 2))
    return lower, upper


def analyze_comparison(sae_vals, baseline_vals, label):
    """Full temporal delta analysis for one SAE-vs-baseline comparison."""
    years, deltas = compute_deltas(sae_vals, baseline_vals)
    n = len(years)
    if n == 0:
        print(f"  ERROR: No overlapping years for {label}")
        return None

    # Summary statistics
    mean_delta = float(deltas.mean())
    min_idx = int(deltas.argmin())
    max_idx = int(deltas.argmax())

    negative_mask = deltas <= 0
    negative_years = [int(y) for y in years[negative_mask]]

    # OLS
    slope, intercept = ols_slope_intercept(years.astype(float), deltas)

    # Bootstrap CI on slope
    ci_lower, ci_upper = bootstrap_slope_ci(
        years.astype(float), deltas, N_BOOTSTRAP, SEED, CI_LEVEL
    )
    ci_includes_zero = bool(ci_lower <= 0 <= ci_upper)

    print(f"  {label}: {n} years, mean delta = {mean_delta:.6f}")
    print(f"    slope = {slope:.6f}, CI = [{ci_lower:.6f}, {ci_upper:.6f}]")
    print(f"    CI includes zero: {ci_includes_zero}")
    if negative_years:
        print(f"    Negative delta years: {negative_years}")

    return {
        "deltas_by_year": {str(int(y)): round(float(d), 6) for y, d in zip(years, deltas)},
        "mean_delta": round(mean_delta, 6),
        "min_delta": round(float(deltas[min_idx]), 6),
        "min_delta_year": int(years[min_idx]),
        "max_delta": round(float(deltas[max_idx]), 6),
        "max_delta_year": int(years[max_idx]),
        "n_years_negative": int(negative_mask.sum()),
        "negative_years": negative_years,
        "ols_slope": round(float(slope), 6),
        "ols_intercept": round(float(intercept), 6),
        "slope_ci_lower": round(float(ci_lower), 6),
        "slope_ci_upper": round(float(ci_upper), 6),
        "slope_ci_includes_zero": ci_includes_zero,
    }


def write_report(results):
    """Write the report section separating Observations from Interpretation."""
    sic = results["sae_minus_sic"]
    sbert = results["sae_minus_sbert"]

    lines = []
    lines.append("## 3. Temporal Delta Analysis\n")

    # --- Observations ---
    lines.append("### Observations\n")

    lines.append("**Year-by-year deltas (SAE MC minus baseline MC):**\n")
    lines.append("| Year | SAE − SIC | SAE − SBERT |")
    lines.append("|------|-----------|-------------|")
    all_years = sorted(set(sic["deltas_by_year"]) | set(sbert["deltas_by_year"]))
    for y in all_years:
        d_sic = sic["deltas_by_year"].get(y, "—")
        d_sbert = sbert["deltas_by_year"].get(y, "—")
        sic_str = f"{d_sic:+.4f}" if isinstance(d_sic, (int, float)) else d_sic
        sbert_str = f"{d_sbert:+.4f}" if isinstance(d_sbert, (int, float)) else d_sbert
        lines.append(f"| {y} | {sic_str} | {sbert_str} |")

    lines.append("")
    lines.append("**Summary statistics:**\n")
    lines.append("| Metric | SAE − SIC | SAE − SBERT |")
    lines.append("|--------|-----------|-------------|")
    lines.append(f"| Mean delta | {sic['mean_delta']:+.4f} | {sbert['mean_delta']:+.4f} |")
    lines.append(f"| Min delta | {sic['min_delta']:+.4f} ({sic['min_delta_year']}) | {sbert['min_delta']:+.4f} ({sbert['min_delta_year']}) |")
    lines.append(f"| Max delta | {sic['max_delta']:+.4f} ({sic['max_delta_year']}) | {sbert['max_delta']:+.4f} ({sbert['max_delta_year']}) |")
    lines.append(f"| Years with delta ≤ 0 | {sic['n_years_negative']} | {sbert['n_years_negative']} |")

    if sic["negative_years"]:
        lines.append(f"\nSAE − SIC negative years: {sic['negative_years']}")
    if sbert["negative_years"]:
        lines.append(f"\nSAE − SBERT negative years: {sbert['negative_years']}")

    lines.append("")
    lines.append("**OLS trend (delta = intercept + slope × year):**\n")
    lines.append("| Metric | SAE − SIC | SAE − SBERT |")
    lines.append("|--------|-----------|-------------|")
    lines.append(f"| Slope (MC/year) | {sic['ols_slope']:+.6f} | {sbert['ols_slope']:+.6f} |")
    lines.append(f"| Intercept | {sic['ols_intercept']:+.4f} | {sbert['ols_intercept']:+.4f} |")
    lines.append(f"| 95% CI lower | {sic['slope_ci_lower']:+.6f} | {sbert['slope_ci_lower']:+.6f} |")
    lines.append(f"| 95% CI upper | {sic['slope_ci_upper']:+.6f} | {sbert['slope_ci_upper']:+.6f} |")
    lines.append(f"| CI includes zero | {sic['slope_ci_includes_zero']} | {sbert['slope_ci_includes_zero']} |")

    # --- Interpretation ---
    lines.append("")
    lines.append("### Interpretation\n")

    # SAE vs SIC stability
    if sic["n_years_negative"] == 0:
        lines.append("SAE outperforms SIC in every year of the sample. The advantage is consistent, not driven by a subset of years.")
    else:
        lines.append(f"SAE underperforms SIC in {sic['n_years_negative']} out of {len(sic['deltas_by_year'])} years ({sic['negative_years']}). These periods warrant investigation.")

    # SAE vs SBERT stability
    if sbert["n_years_negative"] == 0:
        lines.append(f"\nSAE outperforms SBERT in every year of the sample.")
    else:
        lines.append(f"\nSAE underperforms SBERT in {sbert['n_years_negative']} out of {len(sbert['deltas_by_year'])} years ({sbert['negative_years']}). These are the fairer comparison (both use MST + theta clustering) so negative years here are more concerning than SAE vs SIC.")

    # Trend interpretation
    lines.append("")
    for label, res in [("SAE − SIC", sic), ("SAE − SBERT", sbert)]:
        if res["slope_ci_includes_zero"]:
            lines.append(f"**{label} trend:** Slope CI [{res['slope_ci_lower']:+.6f}, {res['slope_ci_upper']:+.6f}] includes zero. The SAE advantage is stable over time — no statistically detectable drift.")
        else:
            direction = "growing" if res["ols_slope"] > 0 else "shrinking"
            lines.append(f"**{label} trend:** Slope CI [{res['slope_ci_lower']:+.6f}, {res['slope_ci_upper']:+.6f}] excludes zero. The SAE advantage is {direction} at {abs(res['ols_slope']):.4f} MC units/year. This temporal trend should be investigated in 1B — it could reflect genuine improvement or confounding with time-varying macro factors.")

    lines.append("")

    with open(REPORT_FILE, "w") as f:
        f.write("\n".join(lines))
    print(f"Saved: {REPORT_FILE}")


def main():
    print("=== 1a_03_temporal: Temporal Delta Analysis ===\n")

    # Load input
    print("Loading MC by year data...")
    mc_by_year = load_mc_by_year()

    sae_vals = get_year_values(mc_by_year, "sae_cd")
    sic_vals = get_year_values(mc_by_year, "sic")
    sbert_vals = get_year_values(mc_by_year, "sbert")

    print(f"  SAE years: {sorted(sae_vals)[:3]}...{sorted(sae_vals)[-1]} ({len(sae_vals)} total)")
    print(f"  SIC years: {sorted(sic_vals)[:3]}...{sorted(sic_vals)[-1]} ({len(sic_vals)} total)")
    print(f"  SBERT years: {sorted(sbert_vals)[:3]}...{sorted(sbert_vals)[-1]} ({len(sbert_vals)} total)")

    # Compute temporal analysis for each comparison
    print("\nAnalyzing SAE minus SIC...")
    sae_minus_sic = analyze_comparison(sae_vals, sic_vals, "SAE − SIC")

    print("\nAnalyzing SAE minus SBERT...")
    sae_minus_sbert = analyze_comparison(sae_vals, sbert_vals, "SAE − SBERT")

    if sae_minus_sic is None or sae_minus_sbert is None:
        print("\nFAIL: Could not compute deltas for one or both comparisons.")
        sys.exit(1)

    results = {
        "sae_minus_sic": sae_minus_sic,
        "sae_minus_sbert": sae_minus_sbert,
    }

    # Validation checks
    ok = True

    # Check: OLS slope and CI are finite
    for label, res in results.items():
        for field in ["ols_slope", "ols_intercept", "slope_ci_lower", "slope_ci_upper"]:
            if not np.isfinite(res[field]):
                print(f"FAIL: {label}.{field} is not finite: {res[field]}")
                ok = False

    # Check: mean delta > 0 for both
    for label, res in results.items():
        if res["mean_delta"] <= 0:
            print(f"FAIL: {label} mean delta = {res['mean_delta']:.6f} <= 0")
            ok = False

    # Save JSON output
    with open(OUTPUT_FILE, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved: {OUTPUT_FILE}")

    # Write report section
    write_report(results)

    if ok:
        print("\nAll checks PASSED.")
        sys.exit(0)
    else:
        print("\nChecks FAILED.")
        sys.exit(1)


if __name__ == "__main__":
    main()

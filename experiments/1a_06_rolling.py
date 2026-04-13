"""
Phase 1A Script 6/7: Rolling 5-year temporal holdout.

Reads per-year MC values and computes rolling windows to test whether
the SAE advantage holds across different market regimes.

Windows: [1996-2000], [1997-2001], ..., [2016-2020] = 21 windows.
For each: mean SAE/SIC/SBERT MC and deltas.
Reports win rates and worst-case windows. No arbitrary thresholds.
"""

import json
import os
import sys

ARTIFACTS = "phase1_artifacts"
WINDOW_SIZE = 5
START_YEAR = 1996
END_YEAR = 2020


def load_mc_by_year():
    path = os.path.join(ARTIFACTS, "1a_mc_by_year.json")
    with open(path) as f:
        return json.load(f)


def get_yearly_values(method_data, years):
    """Extract MC values for a list of years, skipping nulls."""
    vals = []
    for y in years:
        v = method_data.get(str(y))
        if v is not None:
            vals.append(v)
    return vals


def unwrap_method_data(method_data):
    """Handle both old format {year: float} and new format {"mean": {...}, "median": {...}}.
    Returns {year_str: float} using mean values."""
    sample_val = next((v for v in method_data.values() if v is not None), None)
    if isinstance(sample_val, dict):
        return method_data.get("mean", {})
    return method_data


def compute_rolling_windows(mc_by_year):
    sae_data = unwrap_method_data(mc_by_year["sae_cd"])
    sic_data = unwrap_method_data(mc_by_year["sic"])
    sbert_data = unwrap_method_data(mc_by_year["sbert"])

    # Generate all 5-year windows from 1996-2020
    windows = []
    first_start = START_YEAR
    last_start = END_YEAR - WINDOW_SIZE + 1  # 2016

    for start in range(first_start, last_start + 1):
        end = start + WINDOW_SIZE - 1
        years = list(range(start, end + 1))

        sae_vals = get_yearly_values(sae_data, years)
        sic_vals = get_yearly_values(sic_data, years)
        sbert_vals = get_yearly_values(sbert_data, years)

        if not sae_vals:
            print(f"  WARNING: No SAE data for window {start}-{end}, skipping")
            continue

        sae_mc = sum(sae_vals) / len(sae_vals)
        sic_mc = sum(sic_vals) / len(sic_vals) if sic_vals else None
        sbert_mc = sum(sbert_vals) / len(sbert_vals) if sbert_vals else None

        delta_sic = round(sae_mc - sic_mc, 6) if sic_mc is not None else None
        delta_sbert = round(sae_mc - sbert_mc, 6) if sbert_mc is not None else None

        windows.append({
            "start_year": start,
            "end_year": end,
            "sae_mc": round(sae_mc, 6),
            "sic_mc": round(sic_mc, 6) if sic_mc is not None else None,
            "sbert_mc": round(sbert_mc, 6) if sbert_mc is not None else None,
            "delta_sic": delta_sic,
            "delta_sbert": delta_sbert,
        })

    return windows


def summarize_deltas(windows, delta_key):
    """Compute summary stats for a delta across all windows."""
    valid = [(w, w[delta_key]) for w in windows if w[delta_key] is not None]
    if not valid:
        return None

    deltas = [d for _, d in valid]
    n_positive = sum(1 for d in deltas if d > 0)

    min_delta = min(deltas)
    max_delta = max(deltas)
    min_w = next(w for w, d in valid if d == min_delta)
    max_w = next(w for w, d in valid if d == max_delta)

    return {
        "n_positive": n_positive,
        "n_total": len(valid),
        "mean_delta": round(sum(deltas) / len(deltas), 6),
        "min_delta": round(min_delta, 6),
        "min_delta_window": f"{min_w['start_year']}-{min_w['end_year']}",
        "max_delta": round(max_delta, 6),
        "max_delta_window": f"{max_w['start_year']}-{max_w['end_year']}",
    }


def write_report(results):
    """Write the report section for script 06."""
    windows = results["windows"]
    sae_sic = results["sae_vs_sic"]
    sae_sbert = results["sae_vs_sbert"]

    lines = []
    lines.append("## Rolling Temporal Holdout")
    lines.append("")
    lines.append(f"Window size: {results['window_size_years']} years "
                 f"({results['window_size_rationale']})")
    lines.append(f"Number of windows: {results['n_windows']}")
    lines.append("")

    # Observations
    lines.append("### Observations")
    lines.append("")

    # Full window table
    lines.append("| Window | SAE MC | SIC MC | SBERT MC | Delta(SIC) | Delta(SBERT) |")
    lines.append("|--------|--------|--------|----------|------------|--------------|")
    for w in windows:
        sic_str = f"{w['sic_mc']:.4f}" if w['sic_mc'] is not None else "N/A"
        sbert_str = f"{w['sbert_mc']:.4f}" if w['sbert_mc'] is not None else "N/A"
        d_sic_str = f"{w['delta_sic']:+.4f}" if w['delta_sic'] is not None else "N/A"
        d_sbert_str = f"{w['delta_sbert']:+.4f}" if w['delta_sbert'] is not None else "N/A"
        lines.append(f"| {w['start_year']}-{w['end_year']} "
                     f"| {w['sae_mc']:.4f} "
                     f"| {sic_str} "
                     f"| {sbert_str} "
                     f"| {d_sic_str} "
                     f"| {d_sbert_str} |")
    lines.append("")

    # Win rates
    lines.append("**Win rates:**")
    lines.append("")
    if sae_sic:
        lines.append(f"- SAE > SIC: {sae_sic['n_positive']}/{sae_sic['n_total']} windows")
        lines.append(f"  - Mean delta: {sae_sic['mean_delta']:+.4f}")
        lines.append(f"  - Worst window: {sae_sic['min_delta_window']} "
                     f"(delta = {sae_sic['min_delta']:+.4f})")
        lines.append(f"  - Best window: {sae_sic['max_delta_window']} "
                     f"(delta = {sae_sic['max_delta']:+.4f})")
    if sae_sbert:
        lines.append(f"- SAE > SBERT: {sae_sbert['n_positive']}/{sae_sbert['n_total']} windows")
        lines.append(f"  - Mean delta: {sae_sbert['mean_delta']:+.4f}")
        lines.append(f"  - Worst window: {sae_sbert['min_delta_window']} "
                     f"(delta = {sae_sbert['min_delta']:+.4f})")
        lines.append(f"  - Best window: {sae_sbert['max_delta_window']} "
                     f"(delta = {sae_sbert['max_delta']:+.4f})")
    lines.append("")

    # Negative-delta windows
    neg_sic = [w for w in windows if w["delta_sic"] is not None and w["delta_sic"] <= 0]
    neg_sbert = [w for w in windows if w["delta_sbert"] is not None and w["delta_sbert"] <= 0]
    if neg_sic or neg_sbert:
        lines.append("**Negative-delta windows:**")
        lines.append("")
        if neg_sic:
            for w in neg_sic:
                lines.append(f"- vs SIC: {w['start_year']}-{w['end_year']} "
                             f"(delta = {w['delta_sic']:+.4f})")
        if neg_sbert:
            for w in neg_sbert:
                lines.append(f"- vs SBERT: {w['start_year']}-{w['end_year']} "
                             f"(delta = {w['delta_sbert']:+.4f})")
        lines.append("")
    else:
        lines.append("No negative-delta windows for either baseline.")
        lines.append("")

    # Interpretation
    lines.append("### Interpretation")
    lines.append("")
    lines.append("*Does the SAE advantage hold across market regimes? "
                 "Which periods are weakest?*")
    lines.append("")
    lines.append("_(Populated after data is available.)_")
    lines.append("")

    report_path = os.path.join(ARTIFACTS, "1a_report_06.md")
    with open(report_path, "w") as f:
        f.write("\n".join(lines))
    print(f"Saved: {report_path}")


def main():
    print("Phase 1A Step 6/7: Rolling Temporal Holdout")
    print("=" * 60)

    print("\nLoading per-year MC data...")
    mc_by_year = load_mc_by_year()

    # Verify required methods exist
    for method in ["sae_cd", "sic", "sbert"]:
        if method not in mc_by_year:
            print(f"FAIL: Method '{method}' not found in 1a_mc_by_year.json")
            sys.exit(1)

    print("Computing rolling 5-year windows...")
    windows = compute_rolling_windows(mc_by_year)
    n_windows = len(windows)
    print(f"  Computed {n_windows} windows")

    if n_windows == 0:
        print("FAIL: No windows computed")
        sys.exit(1)

    sae_vs_sic = summarize_deltas(windows, "delta_sic")
    sae_vs_sbert = summarize_deltas(windows, "delta_sbert")

    # Print summary
    print(f"\n--- SAE vs SIC ---")
    if sae_vs_sic:
        print(f"  Win rate: {sae_vs_sic['n_positive']}/{sae_vs_sic['n_total']}")
        print(f"  Mean delta: {sae_vs_sic['mean_delta']:+.6f}")
        print(f"  Worst: {sae_vs_sic['min_delta_window']} (delta = {sae_vs_sic['min_delta']:+.6f})")

    print(f"\n--- SAE vs SBERT ---")
    if sae_vs_sbert:
        print(f"  Win rate: {sae_vs_sbert['n_positive']}/{sae_vs_sbert['n_total']}")
        print(f"  Mean delta: {sae_vs_sbert['mean_delta']:+.6f}")
        print(f"  Worst: {sae_vs_sbert['min_delta_window']} (delta = {sae_vs_sbert['min_delta']:+.6f})")

    # Assemble output
    results = {
        "window_size_years": WINDOW_SIZE,
        "window_size_rationale": "One business cycle (NBER average expansion+contraction)",
        "n_windows": n_windows,
        "windows": windows,
        "sae_vs_sic": sae_vs_sic,
        "sae_vs_sbert": sae_vs_sbert,
    }

    # Save JSON
    output_path = os.path.join(ARTIFACTS, "1a_rolling.json")
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved: {output_path}")

    # Write report section
    write_report(results)

    # Verification checks
    print("\n" + "=" * 60)
    print("VERIFICATION")
    ok = True

    def check(name, passed):
        nonlocal ok
        ok &= passed
        print(f"  [{'PASS' if passed else 'FAIL'}] {name}")

    check(f"All 21 windows computed (got {n_windows})", n_windows == 21)
    check("SAE vs SIC summary computed", sae_vs_sic is not None)
    check("SAE vs SBERT summary computed", sae_vs_sbert is not None)
    if sae_vs_sic:
        check(f"Mean delta SAE-SIC > 0 ({sae_vs_sic['mean_delta']:+.6f})",
              sae_vs_sic["mean_delta"] > 0)
    if sae_vs_sbert:
        check(f"Mean delta SAE-SBERT > 0 ({sae_vs_sbert['mean_delta']:+.6f})",
              sae_vs_sbert["mean_delta"] > 0)

    if ok:
        print("\nAll checks PASSED.")
        sys.exit(0)
    else:
        print("\nSome checks FAILED.")
        sys.exit(1)


if __name__ == "__main__":
    main()

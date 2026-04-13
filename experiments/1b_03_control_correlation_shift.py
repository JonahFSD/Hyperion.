#!/usr/bin/env python3
"""
Test 1B-03: Control Correlation Shift Decomposition

Red Flag 2 investigation: T05 topology survival ratio is 129.6% — the signal
gets STRONGER after FF5 factor removal. Why?

This test decomposes the 129.6% into its components:
  - Δ_NN = residual_NN_mean - raw_NN_mean
  - Δ_control = residual_control_mean - raw_control_mean

Three possible outcomes:
  (a) Δ_NN > 0 (NN pairs gain correlation after factor removal)
      → Factor exposure was masking structural signal. Real amplification.
  (b) |Δ_control| >> |Δ_NN| (controls drop more than NN pairs)
      → Mechanical gap widening. Factor removal disproportionately hits
        non-structural pairs. Valid result but different interpretation.
  (c) Δ_NN ≈ Δ_control → need pair-level distributional analysis.

Also reports: tolerance band distribution of control matches, per-year
decomposition, and pair-level shift distributions.

Recomputes residuals using same methodology as 1B (pooled FF5 OLS).
Recomputes T05 on both raw and residual correlations to get all four means.

Data sources:
  - Companies: HuggingFace (Mateusz1017/annual_reports_tokenized...)
  - Pairs: HuggingFace (v1ctor10/cos_sim_4000pca_exp)
  - FF5 factors: cached from 1B (experiments/artifacts/ff5_factors.csv)
  - Raw T05 result: experiments/artifacts/1a_11_t05_result.json (for validation)
  - 1B T05 result: experiments/artifacts/1b_factor_adjustment_result.json (for validation)
"""

import numpy as np
import pandas as pd
import json
import gc
import io
import zipfile
import urllib.request
from collections import defaultdict
from pathlib import Path

np.random.seed(42)

EXPERIMENTS_DIR = Path(__file__).resolve().parent
ARTIFACTS_DIR = EXPERIMENTS_DIR / "artifacts"
FF5_CACHE = ARTIFACTS_DIR / "ff5_factors.csv"
OUTPUT_FILE = ARTIFACTS_DIR / "1b_03_control_correlation_shift.json"

ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 80)
print("Test 1B-03: Control Correlation Shift Decomposition")
print("=" * 80)

# ============================================================
# Step 1: Load FF5 factors
# ============================================================
print("\n[1/6] Loading FF5 factors...")

if FF5_CACHE.exists():
    print(f"  Using cached: {FF5_CACHE}")
    ff5 = pd.read_csv(FF5_CACHE)
else:
    url = "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/F-F_Research_Data_5_Factors_2x3_CSV.zip"
    print(f"  Downloading from: {url}")
    response = urllib.request.urlopen(url)
    zip_data = response.read()
    with zipfile.ZipFile(io.BytesIO(zip_data)) as zf:
        csv_name = [n for n in zf.namelist() if n.lower().endswith('.csv')][0]
        raw_text = zf.read(csv_name).decode('utf-8')

    lines = raw_text.split('\n')
    data_lines = []
    header_found = False
    for line in lines:
        stripped = line.strip()
        if not stripped:
            if header_found:
                break
            continue
        parts = stripped.split(',')
        first = parts[0].strip()
        if first.isdigit() and len(first) == 6:
            header_found = True
            data_lines.append(stripped)
        elif header_found and first.isdigit() and len(first) == 4:
            break

    rows = []
    for line in data_lines:
        parts = [p.strip() for p in line.split(',')]
        if len(parts) >= 7:
            try:
                date = int(parts[0])
                vals = [float(p) for p in parts[1:7]]
                rows.append([date] + vals)
            except ValueError:
                continue

    ff5 = pd.DataFrame(rows, columns=['date', 'Mkt-RF', 'SMB', 'HML', 'RMW', 'CMA', 'RF'])
    for col in ['Mkt-RF', 'SMB', 'HML', 'RMW', 'CMA', 'RF']:
        ff5[col] = ff5[col] / 100.0
    ff5.to_csv(FF5_CACHE, index=False)

ff5['date'] = ff5['date'].astype(int)
ff5_dict = {int(row['date']): row for _, row in ff5.iterrows()}
print(f"  FF5 months: {len(ff5)} ({ff5['date'].min()}-{ff5['date'].max()})")

# ============================================================
# Step 2: Load companies, run FF5 regressions, extract residuals
# ============================================================
print("\n[2/6] Loading companies and running FF5 regressions...")

from datasets import load_dataset

ds = load_dataset(
    "Mateusz1017/annual_reports_tokenized_llama3_logged_returns_no_null_returns_and_incomplete_descriptions_24k",
    split="train"
)

# Stream into compact structures
idx_to_cik = {}
idx_year_map = {}
cik_monthly = defaultdict(list)

for i, row in enumerate(ds):
    idx = row['__index_level_0__']
    cik = row['cik']
    year = int(row['year'])
    log_returns = row['logged_monthly_returns_matrix']
    idx_to_cik[idx] = cik
    idx_year_map[idx] = year
    if log_returns is not None and len(log_returns) == 12:
        cik_monthly[cik].append((year, log_returns))
    if (i + 1) % 5000 == 0:
        print(f"  Loaded {i+1}/27888 rows...")

del ds
print(f"  Unique CIKs: {len(cik_monthly)}")

# Build reverse map: (cik, year) -> [idx, ...] for fast residual assignment
cik_year_to_idxs = defaultdict(list)
for idx, cik in idx_to_cik.items():
    yr = idx_year_map.get(idx)
    if yr is not None:
        cik_year_to_idxs[(cik, yr)].append(idx)

# Regress each company, store residuals keyed by (idx, year)
residuals_dict = {}
n_regressed = 0
n_skipped = 0

total_ciks = len(cik_monthly)
progress_step = max(1, total_ciks // 10)

for i, (cik, year_returns) in enumerate(cik_monthly.items()):
    if (i + 1) % progress_step == 0:
        print(f"  Regressing company {i+1}/{total_ciks}...")

    monthly_data = []
    for year_int, log_returns in year_returns:
        for m in range(12):
            lr = log_returns[m]
            if lr is None or np.isnan(lr):
                continue
            simple_ret = np.exp(lr) - 1.0
            yyyymm = year_int * 100 + (m + 1)
            monthly_data.append((yyyymm, simple_ret, year_int, m))

    valid_data = []
    for yyyymm, ret, yr, mp in monthly_data:
        if yyyymm in ff5_dict:
            ff_row = ff5_dict[yyyymm]
            valid_data.append((ret, yr, mp,
                               float(ff_row['Mkt-RF']), float(ff_row['SMB']),
                               float(ff_row['HML']), float(ff_row['RMW']),
                               float(ff_row['CMA']), float(ff_row['RF'])))

    if len(valid_data) < 24:
        n_skipped += 1
        continue

    n = len(valid_data)
    y = np.empty(n)
    X = np.empty((n, 6))

    for j, (ret, yr, mp, mkt_rf, smb, hml, rmw, cma, rf) in enumerate(valid_data):
        y[j] = ret - rf
        X[j, 0] = 1.0
        X[j, 1] = mkt_rf
        X[j, 2] = smb
        X[j, 3] = hml
        X[j, 4] = rmw
        X[j, 5] = cma

    coeffs, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
    resids = y - X @ coeffs
    n_regressed += 1

    resid_lookup = {}
    for j, (ret, yr, mp, *_) in enumerate(valid_data):
        resid_lookup[(yr, mp)] = resids[j]

    # Map residuals back to all idx values for this CIK
    for year_int, log_returns in year_returns:
        year_resids = []
        for m in range(12):
            r = resid_lookup.get((year_int, m), np.nan)
            year_resids.append(r)
        n_valid = sum(1 for r in year_resids if not np.isnan(r))
        if n_valid >= 6:
            for idx in cik_year_to_idxs.get((cik, year_int), []):
                    residuals_dict[(idx, year_int)] = year_resids

del cik_monthly
gc.collect()

print(f"  Regressed: {n_regressed}, Skipped: {n_skipped}")
print(f"  Company-years with residuals: {len(residuals_dict)}")

# Validate against 1B's known regression counts
EXPECTED_REGRESSED = 1531
EXPECTED_SKIPPED = 71
if n_regressed != EXPECTED_REGRESSED or n_skipped != EXPECTED_SKIPPED:
    print(f"  *** WARNING: Regression counts differ from 1B! ***")
    print(f"      Expected: {EXPECTED_REGRESSED} regressed, {EXPECTED_SKIPPED} skipped")
    print(f"      Got:      {n_regressed} regressed, {n_skipped} skipped")
    print(f"      Delta:    {n_regressed - EXPECTED_REGRESSED} regressed, {n_skipped - EXPECTED_SKIPPED} skipped")
else:
    print(f"  Regression counts match 1B exactly: {n_regressed}/{n_skipped} ✓")

# ============================================================
# Step 3: Load pairs, compute T05 with BOTH raw and residual correlations
# ============================================================
print("\n[3/6] Scanning pairs for T05 (raw + residual)...")

pairs_ds = load_dataset("v1ctor10/cos_sim_4000pca_exp", split="train")
print(f"  Pairs dataset: {len(pairs_ds)} rows")

# Collect per-year data for T05
# For each year: list of (c1, c2, cosine, raw_corr, resid_corr_or_None)
t05_per_year = defaultdict(list)
n_total = 0
n_with_residuals = 0

for i, row in enumerate(pairs_ds):
    c1 = int(row['Company1'])
    c2 = int(row['Company2'])
    yr = int(row['year'])
    cos_sim = float(row['cosine_similarity'])
    raw_corr = row['correlation']

    if raw_corr is None or np.isnan(raw_corr):
        continue

    raw_corr = float(raw_corr)
    n_total += 1

    # Compute residual correlation if available
    resid1 = residuals_dict.get((c1, yr))
    resid2 = residuals_dict.get((c2, yr))
    resid_corr = None

    if resid1 is not None and resid2 is not None:
        r1 = np.array(resid1)
        r2 = np.array(resid2)
        valid_mask = ~np.isnan(r1) & ~np.isnan(r2)
        n_overlap = np.sum(valid_mask)
        if n_overlap >= 6:
            r1v = r1[valid_mask]
            r2v = r2[valid_mask]
            r1c = r1v - np.mean(r1v)
            r2c = r2v - np.mean(r2v)
            denom = np.sqrt(np.sum(r1c ** 2) * np.sum(r2c ** 2))
            if denom > 1e-15:
                resid_corr = float(np.sum(r1c * r2c) / denom)
                n_with_residuals += 1

    t05_per_year[yr].append((c1, c2, cos_sim, raw_corr, resid_corr))

    if (i + 1) % 3000000 == 0:
        print(f"  Processed {i+1}/{len(pairs_ds)} pairs, {n_with_residuals} with residuals...")

del pairs_ds
gc.collect()

print(f"  Total valid pairs: {n_total}")
print(f"  Pairs with residual correlations: {n_with_residuals}")

# ============================================================
# Step 4: Build T05 NN graph and control matching
# ============================================================
print("\n[4/6] Building T05 NN graphs and matching controls...")

# For each year: build NN graph, match controls, record all four correlations
# per matched pair: (raw_nn, raw_ctrl, resid_nn, resid_ctrl, tolerance_used)
matched_pairs = []  # list of dicts

for yr in sorted(t05_per_year.keys()):
    pairs_yr = t05_per_year[yr]
    print(f"  Year {yr}: {len(pairs_yr)} pairs...", end='', flush=True)

    if not pairs_yr:
        print(" skipped")
        continue

    # Only use pairs where BOTH raw and residual correlations exist
    pairs_both = [(c1, c2, cos, raw, res) for c1, c2, cos, raw, res in pairs_yr if res is not None]
    print(f" ({len(pairs_both)} with both)...", end='', flush=True)

    if len(pairs_both) < 10:
        print(" skipped (too few)")
        continue

    # Build NN graph: for each company, find peer with highest cosine
    company_best = {}
    for c1, c2, cos_sim, raw_corr, resid_corr in pairs_both:
        for comp, peer, rc, rrc in [(c1, c2, raw_corr, resid_corr), (c2, c1, raw_corr, resid_corr)]:
            if comp not in company_best or cos_sim > company_best[comp][1]:
                company_best[comp] = (peer, cos_sim, rc, rrc)

    # Deduplicate NN edges
    nn_edges = {}
    for comp, (peer, cos_sim, raw_corr, resid_corr) in company_best.items():
        edge = (min(comp, peer), max(comp, peer))
        if edge not in nn_edges or cos_sim > nn_edges[edge][0]:
            nn_edges[edge] = (cos_sim, raw_corr, resid_corr)

    nn_edge_set = set(nn_edges.keys())

    # Match each NN edge to a control
    matched_yr = 0
    for edge, (nn_cos, nn_raw, nn_resid) in nn_edges.items():
        for tol in [0.01, 0.02, 0.05]:
            candidates = []
            for c1, c2, cos_sim, raw_corr, resid_corr in pairs_both:
                ctrl_edge = (min(c1, c2), max(c1, c2))
                if ctrl_edge in nn_edge_set:
                    continue
                if abs(cos_sim - nn_cos) <= tol:
                    candidates.append((raw_corr, resid_corr))
                    if len(candidates) >= 20:
                        break

            if candidates:
                pick = candidates[np.random.randint(len(candidates))]
                matched_pairs.append({
                    'year': yr,
                    'nn_raw': nn_raw,
                    'nn_resid': nn_resid,
                    'ctrl_raw': pick[0],
                    'ctrl_resid': pick[1],
                    'tolerance': tol,
                })
                matched_yr += 1
                break

    print(f" {len(nn_edges)} NN, {matched_yr} matched")

del t05_per_year
gc.collect()

print(f"\n  Total matched pairs: {len(matched_pairs)}")

# ============================================================
# Step 5: Decompose the shift
# ============================================================
print("\n[5/6] Decomposing correlation shift...")

nn_raw_arr = np.array([p['nn_raw'] for p in matched_pairs])
nn_resid_arr = np.array([p['nn_resid'] for p in matched_pairs])
ctrl_raw_arr = np.array([p['ctrl_raw'] for p in matched_pairs])
ctrl_resid_arr = np.array([p['ctrl_resid'] for p in matched_pairs])
tolerances = [p['tolerance'] for p in matched_pairs]

# Aggregate means
nn_raw_mean = float(np.mean(nn_raw_arr))
nn_resid_mean = float(np.mean(nn_resid_arr))
ctrl_raw_mean = float(np.mean(ctrl_raw_arr))
ctrl_resid_mean = float(np.mean(ctrl_resid_arr))

delta_nn = nn_resid_mean - nn_raw_mean
delta_ctrl = ctrl_resid_mean - ctrl_raw_mean

raw_gap = nn_raw_mean - ctrl_raw_mean
resid_gap = nn_resid_mean - ctrl_resid_mean

print(f"\n  Four component means:")
print(f"    NN raw:      {nn_raw_mean:.6f}")
print(f"    NN residual: {nn_resid_mean:.6f}")
print(f"    Ctrl raw:      {ctrl_raw_mean:.6f}")
print(f"    Ctrl residual: {ctrl_resid_mean:.6f}")
print(f"\n  Shifts after factor removal:")
print(f"    Δ_NN   = {delta_nn:+.6f}")
print(f"    Δ_ctrl = {delta_ctrl:+.6f}")
print(f"    Δ_NN - Δ_ctrl = {delta_nn - delta_ctrl:+.6f}")
print(f"\n  Gaps:")
print(f"    Raw gap:      {raw_gap:.6f}")
print(f"    Residual gap: {resid_gap:.6f}")
print(f"    Survival ratio: {resid_gap / raw_gap:.3f}" if abs(raw_gap) > 1e-10 else "    (undefined)")

# Tolerance distribution
tol_counts = {0.01: 0, 0.02: 0, 0.05: 0}
for t in tolerances:
    tol_counts[t] += 1
print(f"\n  Control matching tolerance distribution:")
for t in [0.01, 0.02, 0.05]:
    pct = 100 * tol_counts[t] / len(tolerances) if tolerances else 0
    print(f"    ±{t}: {tol_counts[t]} ({pct:.1f}%)")

# Per-pair shift distributions
pair_delta_nn = nn_resid_arr - nn_raw_arr
pair_delta_ctrl = ctrl_resid_arr - ctrl_raw_arr
pair_gap_change = pair_delta_nn - pair_delta_ctrl

print(f"\n  Per-pair Δ_NN distribution:")
print(f"    mean={np.mean(pair_delta_nn):.6f}, median={np.median(pair_delta_nn):.6f}")
print(f"    std={np.std(pair_delta_nn):.6f}")
print(f"    % positive: {100*np.mean(pair_delta_nn > 0):.1f}%")

print(f"\n  Per-pair Δ_ctrl distribution:")
print(f"    mean={np.mean(pair_delta_ctrl):.6f}, median={np.median(pair_delta_ctrl):.6f}")
print(f"    std={np.std(pair_delta_ctrl):.6f}")
print(f"    % positive: {100*np.mean(pair_delta_ctrl > 0):.1f}%")

print(f"\n  Per-pair gap change (Δ_NN - Δ_ctrl):")
print(f"    mean={np.mean(pair_gap_change):.6f}, median={np.median(pair_gap_change):.6f}")
print(f"    % where NN shifted more favorably: {100*np.mean(pair_gap_change > 0):.1f}%")

# Bootstrap CI on the aggregate shifts
print("\n  Bootstrap CIs (10,000 resamples)...")
n_boot = 10000
boot_delta_nn = []
boot_delta_ctrl = []
boot_gap_change = []

for b in range(n_boot):
    if (b + 1) % 2000 == 0:
        print(f"    {b+1}/{n_boot}...")
    idx = np.random.choice(len(matched_pairs), len(matched_pairs), replace=True)
    b_nn_raw = np.mean(nn_raw_arr[idx])
    b_nn_res = np.mean(nn_resid_arr[idx])
    b_ctrl_raw = np.mean(ctrl_raw_arr[idx])
    b_ctrl_res = np.mean(ctrl_resid_arr[idx])
    boot_delta_nn.append(b_nn_res - b_nn_raw)
    boot_delta_ctrl.append(b_ctrl_res - b_ctrl_raw)
    boot_gap_change.append((b_nn_res - b_nn_raw) - (b_ctrl_res - b_ctrl_raw))

delta_nn_ci = [float(np.percentile(boot_delta_nn, 2.5)), float(np.percentile(boot_delta_nn, 97.5))]
delta_ctrl_ci = [float(np.percentile(boot_delta_ctrl, 2.5)), float(np.percentile(boot_delta_ctrl, 97.5))]
gap_change_ci = [float(np.percentile(boot_gap_change, 2.5)), float(np.percentile(boot_gap_change, 97.5))]

print(f"\n  Δ_NN:   {delta_nn:+.6f}  CI [{delta_nn_ci[0]:+.6f}, {delta_nn_ci[1]:+.6f}]")
print(f"  Δ_ctrl: {delta_ctrl:+.6f}  CI [{delta_ctrl_ci[0]:+.6f}, {delta_ctrl_ci[1]:+.6f}]")
print(f"  Gap Δ:  {delta_nn - delta_ctrl:+.6f}  CI [{gap_change_ci[0]:+.6f}, {gap_change_ci[1]:+.6f}]")

# Per-year decomposition
print("\n  Per-year decomposition:")
print(f"  {'Year':>6} {'Δ_NN':>10} {'Δ_ctrl':>10} {'Gap Δ':>10} {'N':>6}")
print(f"  {'-'*46}")

years_in_data = sorted(set(p['year'] for p in matched_pairs))
per_year_decomp = {}

for yr in years_in_data:
    yr_pairs = [p for p in matched_pairs if p['year'] == yr]
    yr_nn_raw = np.mean([p['nn_raw'] for p in yr_pairs])
    yr_nn_res = np.mean([p['nn_resid'] for p in yr_pairs])
    yr_ctrl_raw = np.mean([p['ctrl_raw'] for p in yr_pairs])
    yr_ctrl_res = np.mean([p['ctrl_resid'] for p in yr_pairs])
    yr_d_nn = yr_nn_res - yr_nn_raw
    yr_d_ctrl = yr_ctrl_res - yr_ctrl_raw

    per_year_decomp[str(yr)] = {
        'nn_raw': float(yr_nn_raw),
        'nn_resid': float(yr_nn_res),
        'ctrl_raw': float(yr_ctrl_raw),
        'ctrl_resid': float(yr_ctrl_res),
        'delta_nn': float(yr_d_nn),
        'delta_ctrl': float(yr_d_ctrl),
        'gap_change': float(yr_d_nn - yr_d_ctrl),
        'n': len(yr_pairs),
    }

    print(f"  {yr:>6} {yr_d_nn:>+10.4f} {yr_d_ctrl:>+10.4f} {yr_d_nn - yr_d_ctrl:>+10.4f} {len(yr_pairs):>6}")

# ============================================================
# Step 6: Validate against existing results
# ============================================================
print("\n[6/6] Validation against existing results...")

# Load raw T05 for comparison
with open(ARTIFACTS_DIR / '1a_11_t05_result.json', 'r') as f:
    raw_t05 = json.load(f)

# Load 1B T05 for comparison
with open(ARTIFACTS_DIR / '1b_factor_adjustment_result.json', 'r') as f:
    oneb_result = json.load(f)

expected_raw_nn = raw_t05['overall']['nn_mean_corr']
expected_raw_ctrl = raw_t05['overall']['control_mean_corr']
expected_raw_diff = raw_t05['overall']['difference']
expected_resid_diff = oneb_result['t05_comparison']['residual_diff']

print(f"  Raw NN mean:      ours={nn_raw_mean:.6f}  expected={expected_raw_nn:.6f}  delta={nn_raw_mean - expected_raw_nn:.6f}")
print(f"  Raw ctrl mean:    ours={ctrl_raw_mean:.6f}  expected={expected_raw_ctrl:.6f}  delta={ctrl_raw_mean - expected_raw_ctrl:.6f}")
print(f"  Raw gap:          ours={raw_gap:.6f}  expected={expected_raw_diff:.6f}  delta={raw_gap - expected_raw_diff:.6f}")
print(f"  Residual gap:     ours={resid_gap:.6f}  expected={expected_resid_diff:.6f}  delta={resid_gap - expected_resid_diff:.6f}")

# Note: exact match is NOT expected for two reasons:
# (1) This script's NN graph is built from the intersection of pairs with BOTH
#     raw and residual correlations. The original T05 (1a_11) used ALL pairs with
#     valid raw correlation. The ~71 companies skipped in regression change some
#     companies' nearest neighbors, which is the primary source of divergence.
# (2) Control matching is stochastic (random selection among candidates).
# What matters is the decomposition pattern, not exact numerical match.

# ============================================================
# Verdict
# ============================================================
print("\n" + "=" * 80)
print("VERDICT")
print("=" * 80)

# Classify the mechanism
if delta_nn > 0 and gap_change_ci[0] > 0:
    # NN pairs gained correlation AND gap widened significantly
    verdict = "FLAG_2_REAL_AMPLIFICATION"
    interpretation = (
        f"Factor removal genuinely amplifies structural signal. "
        f"NN pairs GAINED correlation (Δ_NN={delta_nn:+.4f}, CI [{delta_nn_ci[0]:+.4f}, {delta_nn_ci[1]:+.4f}]). "
        f"Controls shifted by Δ_ctrl={delta_ctrl:+.4f}. "
        f"The 129.6% survival ratio reflects real signal amplification: "
        f"factor exposure was masking structural similarity between NN pairs. "
        f"Combined with 1B-02 (SAE clusters have divergent factor loadings), "
        f"this is the expected outcome. Flag 2 is a finding, not a red flag."
    )
elif delta_ctrl < delta_nn and gap_change_ci[0] > 0:
    # Controls dropped more, gap widened mechanically
    if delta_nn >= 0:
        verdict = "FLAG_2_MECHANICAL_PLUS_REAL"
        interpretation = (
            f"Mixed mechanism. NN pairs shifted by Δ_NN={delta_nn:+.4f}, "
            f"controls shifted by Δ_ctrl={delta_ctrl:+.4f}. "
            f"Gap widened because controls lost more correlation than NN pairs. "
            f"Factor removal disproportionately hits non-structural pairs "
            f"while preserving structural ones. Flag 2 is explained."
        )
    else:
        verdict = "FLAG_2_MECHANICAL"
        interpretation = (
            f"Mechanical gap widening. Both NN and controls lost correlation, "
            f"but controls lost more (Δ_NN={delta_nn:+.4f} vs Δ_ctrl={delta_ctrl:+.4f}). "
            f"The 129.6% survival is not signal amplification — it's differential "
            f"correlation loss. Still a valid result (factor removal improves "
            f"discrimination) but the interpretation differs."
        )
else:
    verdict = "FLAG_2_INCONCLUSIVE"
    interpretation = (
        f"Shifts are ambiguous. Δ_NN={delta_nn:+.4f}, Δ_ctrl={delta_ctrl:+.4f}. "
        f"Gap change CI [{gap_change_ci[0]:+.4f}, {gap_change_ci[1]:+.4f}] "
        f"{'excludes' if gap_change_ci[0] > 0 or gap_change_ci[1] < 0 else 'includes'} zero. "
        f"Cannot cleanly attribute the 129.6% survival to a single mechanism."
    )

print(f"\n  {verdict}")
print(f"  {interpretation}")

# ============================================================
# Write output
# ============================================================
result = {
    "test": "1B-03_control_correlation_shift",
    "question": "Why does T05 topology signal get stronger after factor removal (129.6%)?",
    "component_means": {
        "nn_raw": nn_raw_mean,
        "nn_residual": nn_resid_mean,
        "control_raw": ctrl_raw_mean,
        "control_residual": ctrl_resid_mean,
    },
    "shifts": {
        "delta_nn": delta_nn,
        "delta_nn_ci_95": delta_nn_ci,
        "delta_control": delta_ctrl,
        "delta_control_ci_95": delta_ctrl_ci,
        "gap_change": delta_nn - delta_ctrl,
        "gap_change_ci_95": gap_change_ci,
    },
    "gaps": {
        "raw_gap": raw_gap,
        "residual_gap": resid_gap,
        "survival_ratio": resid_gap / raw_gap if abs(raw_gap) > 1e-10 else None,
    },
    "tolerance_distribution": {
        str(t): {"count": tol_counts[t], "pct": 100 * tol_counts[t] / len(tolerances)}
        for t in [0.01, 0.02, 0.05]
    },
    "pair_level_distributions": {
        "delta_nn": {
            "mean": float(np.mean(pair_delta_nn)),
            "median": float(np.median(pair_delta_nn)),
            "std": float(np.std(pair_delta_nn)),
            "pct_positive": float(100 * np.mean(pair_delta_nn > 0)),
        },
        "delta_control": {
            "mean": float(np.mean(pair_delta_ctrl)),
            "median": float(np.median(pair_delta_ctrl)),
            "std": float(np.std(pair_delta_ctrl)),
            "pct_positive": float(100 * np.mean(pair_delta_ctrl > 0)),
        },
        "gap_change": {
            "mean": float(np.mean(pair_gap_change)),
            "median": float(np.median(pair_gap_change)),
            "std": float(np.std(pair_gap_change)),
            "pct_nn_shifted_more_favorably": float(100 * np.mean(pair_gap_change > 0)),
        },
    },
    "per_year": per_year_decomp,
    "validation": {
        "expected_raw_nn": expected_raw_nn,
        "expected_raw_ctrl": expected_raw_ctrl,
        "expected_raw_diff": expected_raw_diff,
        "expected_resid_diff": expected_resid_diff,
        "our_raw_diff": raw_gap,
        "our_resid_diff": resid_gap,
        "note": "Exact match not expected — control matching is stochastic. Pattern should match.",
    },
    "n_matched_pairs": len(matched_pairs),
    "verdict": verdict,
    "interpretation": interpretation,
}

with open(OUTPUT_FILE, 'w') as f:
    json.dump(result, f, indent=2)

print(f"\nResult written to: {OUTPUT_FILE}")
print(f"\n{'='*80}")
print("FULL JSON:")
print(f"{'='*80}")
print(json.dumps(result, indent=2))

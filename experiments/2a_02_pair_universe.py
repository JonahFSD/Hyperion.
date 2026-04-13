#!/usr/bin/env python3
"""
2a_02_pair_universe.py — Pair Universe Construction

For each formation year (1999–2020), fit walk-forward PCA on all filings
through that year, then compute within-SIC2 pairwise cosine similarities.

Outputs a flat parquet of all unique within-SIC ordered pairs per year,
scored by walk-forward PCA cosine similarity. This is the foundation for
all downstream backtest scripts (2a_03, 2a_04, 2a_05).

Temporal contract: walk-forward PCA is fitted ONLY on filings with year
<= formation_year. Never sees future data.

Outputs:
  experiments/artifacts/2a_02_pair_universe.parquet
  experiments/artifacts/2a_02_summary.json

Schema:
  formation_year   int    Filing year (1999–2020)
  signal_year      int    formation_year + 1 (portfolio forms July signal_year)
  company1_idx     int    __index_level_0__ from HuggingFace (company1_idx < company2_idx)
  company2_idx     int    __index_level_0__ from HuggingFace
  company1_ticker  str    Ticker symbol
  company2_ticker  str    Ticker symbol
  sic2             str    Shared 2-digit SIC code (zero-padded: "01" not "1")
  cosine_sim       float  Walk-forward PCA cosine similarity (6 decimal places)
  rank_in_sic      int    Rank within SIC2 group this year (1 = most similar pair)
  n_sic_peers      int    Number of companies in this SIC2 group this year
  wf_pca_components int   PCA components used (may be <4000 for early years)
  wf_pca_variance  float  Variance explained by walk-forward PCA
"""

import os
import gc
import json
import time
import warnings
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity
from datasets import load_dataset

warnings.filterwarnings('ignore')

# ═══════════════════════════════════════════════════════════
# CONFIG
# ═══════════════════════════════════════════════════════════

N_COMPONENTS = 4000         # Target PCA components
N_FEATURES_KEEP = 8000      # Top features by variance to keep before PCA
FORMATION_YEARS = list(range(1999, 2021))  # 1999–2020 inclusive (22 years)
MIN_PEERS = 2               # Minimum companies in SIC2 group to form pairs
SEED = 42
BATCH = 500                 # Rows per streaming batch

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ARTIFACTS_DIR = os.path.join(BASE_DIR, "artifacts")


def log(msg):
    print(f"[2a_02] {msg}", flush=True)


def mem_mb():
    """Current process RSS in MB (macOS)."""
    try:
        import resource
        return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024 * 1024)
    except Exception:
        return 0


def extract_batch(batch_features):
    """Convert a batch of feature rows to float32 numpy, handling nested lists."""
    rows = []
    for row in batch_features:
        if isinstance(row, list) and len(row) > 0 and isinstance(row[0], list):
            row = row[0]
        arr = np.asarray(row, dtype=np.float32).flatten()
        rows.append(arr)
    lengths = [len(r) for r in rows]
    mode_len = max(set(lengths), key=lengths.count)
    out = np.zeros((len(rows), mode_len), dtype=np.float32)
    for i, r in enumerate(rows):
        if len(r) == mode_len:
            out[i] = r
    return out


def sic2_from_code(x):
    """Zero-pad SIC to 4 digits, return first 2. Returns None for invalid input."""
    try:
        return str(int(float(x))).zfill(4)[:2]
    except (ValueError, TypeError):
        return None


def main():
    rng = np.random.default_rng(SEED)  # noqa: F841 — reserved for downstream randomness
    os.makedirs(ARTIFACTS_DIR, exist_ok=True)
    t_start = time.time()

    # ═══════════════════════════════════════════════════════════
    # STEP 1+2: LOAD & FILTER SAE FEATURES (streaming, 2 passes)
    #
    # Raw matrix: ~28K × 131K × 4 bytes ≈ 14.7 GB — doesn't fit in RAM.
    # Pass 1: Stream to compute per-feature variance → select top 8K features.
    # Pass 2: Stream again, load only those 8K columns → ~0.9 GB resident.
    # ═══════════════════════════════════════════════════════════

    log("=" * 60)
    log("STEP 1+2: LOAD & FILTER SAE FEATURES (streaming)")
    log(f"  Memory: {mem_mb():.0f} MB")
    log("=" * 60)

    log("Loading features dataset from HuggingFace...")
    t0 = time.time()
    features_ds = load_dataset('marco-molinari/company_reports_with_features', split='train')
    log(f"  Loaded in {time.time()-t0:.1f}s | rows: {len(features_ds):,} | cols: {features_ds.column_names}")

    n_rows = len(features_ds)

    # Extract join key: __index_level_0__
    if '__index_level_0__' in features_ds.column_names:
        feat_hf_indices = list(features_ds['__index_level_0__'])
        log("  Found __index_level_0__ in features dataset")
    else:
        feat_hf_indices = list(range(n_rows))
        log("  WARNING: No __index_level_0__ — using row position as index")

    # --- Pass 1: Compute per-feature variance (streaming) ---
    log(f"  Pass 1: Computing per-feature variance (streaming, {(n_rows + BATCH - 1) // BATCH} batches)...")
    t0 = time.time()
    feat_sum = None
    feat_sq_sum = None
    n_raw = None
    n_batches = (n_rows + BATCH - 1) // BATCH

    for start in range(0, n_rows, BATCH):
        end = min(start + BATCH, n_rows)
        batch_arr = extract_batch(features_ds[start:end]['features'])
        if feat_sum is None:
            n_raw = batch_arr.shape[1]
            feat_sum = np.zeros(n_raw, dtype=np.float64)
            feat_sq_sum = np.zeros(n_raw, dtype=np.float64)
        feat_sum += batch_arr.sum(axis=0).astype(np.float64)
        feat_sq_sum += (batch_arr.astype(np.float64) ** 2).sum(axis=0)
        batch_num = start // BATCH
        if batch_num % 10 == 0:
            log(f"    Batch {batch_num + 1}/{n_batches} ({mem_mb():.0f} MB)")

    feat_mean = feat_sum / n_rows
    feat_var = feat_sq_sum / n_rows - feat_mean ** 2

    n_keep = min(N_FEATURES_KEEP, n_raw)
    top_indices = np.argsort(feat_var)[::-1][:n_keep]
    top_indices.sort()  # Sort for stable column order
    combined_mask = np.zeros(n_raw, dtype=bool)
    combined_mask[top_indices] = True
    n_filtered = int(combined_mask.sum())

    del feat_sum, feat_sq_sum, feat_mean, feat_var, top_indices
    gc.collect()

    log(f"  Pass 1 done in {time.time()-t0:.1f}s | raw: {n_raw:,} | selected: {n_filtered:,}")
    log(f"  Filtered matrix will be: {n_rows} × {n_filtered} = {n_rows * n_filtered * 4 / 1e9:.2f} GB")

    # --- Pass 2: Build filtered feature matrix ---
    log(f"  Pass 2: Building filtered feature matrix ({n_batches} batches)...")
    t0 = time.time()
    features = np.empty((n_rows, n_filtered), dtype=np.float32)

    for start in range(0, n_rows, BATCH):
        end = min(start + BATCH, n_rows)
        batch_arr = extract_batch(features_ds[start:end]['features'])
        features[start:end] = batch_arr[:, combined_mask]
        batch_num = start // BATCH
        if batch_num % 10 == 0:
            log(f"    Batch {batch_num + 1}/{n_batches} ({mem_mb():.0f} MB)")

    del features_ds, combined_mask
    gc.collect()

    log(f"  Pass 2 done in {time.time()-t0:.1f}s | shape: {features.shape} | Memory: {mem_mb():.0f} MB")

    # ═══════════════════════════════════════════════════════════
    # STEP 3: LOAD COMPANY METADATA
    # ═══════════════════════════════════════════════════════════

    log("")
    log("=" * 60)
    log("STEP 3: LOAD COMPANY METADATA")
    log("=" * 60)

    log("Loading company dataset from HuggingFace...")
    t0 = time.time()
    company_ds = load_dataset(
        'Mateusz1017/annual_reports_tokenized_llama3_logged_returns_no_null_returns_and_incomplete_descriptions_24k',
        split='train',
    )
    log(f"  Loaded in {time.time()-t0:.1f}s | rows: {len(company_ds):,}")

    meta_df = pd.DataFrame({
        'hf_index': (
            company_ds['__index_level_0__']
            if '__index_level_0__' in company_ds.column_names
            else range(len(company_ds))
        ),
        'year': [int(y) for y in company_ds['year']],
        'sic_code': company_ds['sic_code'],
        'ticker': company_ds['ticker'],
        'cik': company_ds['cik'],
    })
    del company_ds
    gc.collect()

    # Drop rows with missing SIC
    meta_df = meta_df.dropna(subset=['sic_code']).copy()

    # SIC2: zero-pad to 4 digits, take first 2.
    # e.g. SIC 111 (Agriculture) → "0111" → "01", not "11" (Tobacco).
    meta_df['sic2'] = meta_df['sic_code'].apply(sic2_from_code)
    meta_df = meta_df.dropna(subset=['sic2']).copy()

    # Map hf_index → row position in features array
    hf_to_feat_row = {int(idx): i for i, idx in enumerate(feat_hf_indices)}
    meta_df['feat_row'] = meta_df['hf_index'].map(hf_to_feat_row)
    meta_df = meta_df.dropna(subset=['feat_row']).copy()
    meta_df['feat_row'] = meta_df['feat_row'].astype(int)
    meta_df['hf_index'] = meta_df['hf_index'].astype(int)

    log(f"  Companies with features + SIC: {len(meta_df):,}")
    log(f"  Year range: {meta_df['year'].min()}-{meta_df['year'].max()}")
    log(f"  Unique SIC2 codes: {meta_df['sic2'].nunique()}")

    # --- SIC2 zero-padding verification ---
    unique_sic2 = sorted(meta_df['sic2'].unique())
    single_digit_check = [s for s in unique_sic2 if len(s) == 1]
    if single_digit_check:
        log(f"  WARNING: Found single-digit SIC2 codes: {single_digit_check}")
    else:
        log(f"  SIC2 zero-padding OK — no single-digit codes found")
    log(f"  SIC2 sample (first 20): {unique_sic2[:20]}")

    # --- No-leakage verification ---
    n_train_1999 = len(meta_df[meta_df['year'] <= 1999])
    n_test_1999 = len(meta_df[meta_df['year'] == 1999])
    log(f"  Leakage check: year<=1999 training rows = {n_train_1999:,}")
    log(f"  Leakage check: year==1999 test rows = {n_test_1999:,}")
    log(f"  Memory: {mem_mb():.0f} MB")

    # ═══════════════════════════════════════════════════════════
    # STEP 4: WALK-FORWARD PAIR CONSTRUCTION (1999–2020)
    #
    # For each formation year FY:
    #   1. Fit PCA on all filings with year <= FY (walk-forward)
    #   2. Transform FY filings into WF PCA space
    #   3. Group FY companies by SIC2
    #   4. Compute pairwise cosine similarity within each group
    #   5. Store all pairs where company1_idx < company2_idx
    # ═══════════════════════════════════════════════════════════

    log("")
    log("=" * 60)
    log(f"STEP 4: WALK-FORWARD PAIR CONSTRUCTION ({FORMATION_YEARS[0]}–{FORMATION_YEARS[-1]})")
    log("=" * 60)

    all_year_dfs = []
    per_year_summary = {}

    for fy in FORMATION_YEARS:
        log(f"\n--- Formation year: {fy} (signal_year: {fy + 1}) ---")
        t_fy = time.time()

        # Training set: all filings with year <= fy (temporal integrity)
        wf_mask = meta_df['year'] <= fy
        wf_meta = meta_df[wf_mask]
        test_meta = meta_df[meta_df['year'] == fy].copy()

        n_train = len(wf_meta)
        n_test = len(test_meta)
        log(f"  Training (year <= {fy}): {n_train:,} | Test (year == {fy}): {n_test:,}")

        if n_test < 2:
            log(f"  SKIP: too few test filings ({n_test})")
            per_year_summary[fy] = {'status': 'skipped', 'reason': 'too few test filings'}
            continue

        # Fit walk-forward PCA on training rows
        wf_n_comp = min(N_COMPONENTS, n_train - 1, features.shape[1])
        log(f"  Fitting WF PCA: {wf_n_comp} components on {n_train} samples...")
        t0 = time.time()

        wf_feat_rows = wf_meta['feat_row'].values
        wf_pca = PCA(n_components=wf_n_comp, random_state=SEED)
        wf_pca.fit(features[wf_feat_rows])
        wf_var = float(wf_pca.explained_variance_ratio_.sum())
        log(f"  WF PCA: variance={wf_var:.6f}, time={time.time()-t0:.1f}s")

        # Transform test filings into WF PCA space
        test_feat_rows = test_meta['feat_row'].values
        wf_test_pca = wf_pca.transform(features[test_feat_rows])  # (n_test, wf_n_comp)

        del wf_pca
        gc.collect()

        # Map feat_row → local row index in wf_test_pca
        local_idx_map = {int(fr): i for i, fr in enumerate(test_feat_rows)}

        # Build pairs within each SIC2 group
        sic2_groups = test_meta.groupby('sic2', sort=True)
        n_sic_groups = 0
        n_pairs_year = 0
        year_rows = []

        for sic2, group in sic2_groups:
            n_group = len(group)
            if n_group < MIN_PEERS:
                continue
            n_sic_groups += 1

            group_feat_rows = group['feat_row'].values
            group_hf_indices = group['hf_index'].values
            group_tickers = group['ticker'].values

            # Local indices into wf_test_pca
            local_idxs = np.array([local_idx_map[int(fr)] for fr in group_feat_rows])

            # Pairwise cosine similarity matrix: (n_group, n_group)
            cos_matrix = cosine_similarity(wf_test_pca[local_idxs])

            # Extract upper triangle (i < j by group position, not by hf_index)
            rows_i, rows_j = np.triu_indices(n_group, k=1)
            hf_i_raw = group_hf_indices[rows_i].astype(int)
            hf_j_raw = group_hf_indices[rows_j].astype(int)
            cos_arr = cos_matrix[rows_i, rows_j]
            ticker_i_raw = group_tickers[rows_i]
            ticker_j_raw = group_tickers[rows_j]

            # Ensure company1_idx < company2_idx (unique canonical ordering)
            swap_mask = hf_i_raw > hf_j_raw
            company1_idx = np.where(swap_mask, hf_j_raw, hf_i_raw)
            company2_idx = np.where(swap_mask, hf_i_raw, hf_j_raw)
            company1_ticker = np.where(swap_mask, ticker_j_raw, ticker_i_raw)
            company2_ticker = np.where(swap_mask, ticker_i_raw, ticker_j_raw)

            # Rank pairs by cosine_sim descending within this SIC2 group
            # rank 1 = most similar pair
            rank_arr = (-cos_arr).argsort().argsort() + 1

            n_pairs_group = len(cos_arr)
            group_df = pd.DataFrame({
                'formation_year': np.full(n_pairs_group, fy, dtype=np.int32),
                'signal_year': np.full(n_pairs_group, fy + 1, dtype=np.int32),
                'company1_idx': company1_idx,
                'company2_idx': company2_idx,
                'company1_ticker': company1_ticker,
                'company2_ticker': company2_ticker,
                'sic2': sic2,
                'cosine_sim': np.round(cos_arr, 6),
                'rank_in_sic': rank_arr.astype(np.int32),
                'n_sic_peers': np.full(n_pairs_group, n_group, dtype=np.int32),
                'wf_pca_components': np.full(n_pairs_group, wf_n_comp, dtype=np.int32),
                'wf_pca_variance': np.full(n_pairs_group, wf_var, dtype=np.float64),
            })
            year_rows.append(group_df)
            n_pairs_year += n_pairs_group

        if year_rows:
            year_df = pd.concat(year_rows, ignore_index=True)
            all_year_dfs.append(year_df)

            cos_vals = year_df['cosine_sim']
            log(f"  Year {fy}: {n_sic_groups} SIC2 groups, {n_test:,} companies, {n_pairs_year:,} pairs")
            log(f"  Cosine sim: mean={cos_vals.mean():.4f}, median={cos_vals.median():.4f}, "
                f"std={cos_vals.std():.4f}, min={cos_vals.min():.4f}, max={cos_vals.max():.4f}")

            per_year_summary[fy] = {
                'n_train': int(n_train),
                'n_test': int(n_test),
                'n_sic2_groups': int(n_sic_groups),
                'n_pairs': int(n_pairs_year),
                'wf_pca_components': int(wf_n_comp),
                'wf_pca_variance': float(wf_var),
                'cosine_sim_mean': float(cos_vals.mean()),
                'cosine_sim_median': float(cos_vals.median()),
                'cosine_sim_std': float(cos_vals.std()),
                'cosine_sim_min': float(cos_vals.min()),
                'cosine_sim_max': float(cos_vals.max()),
                'time_seconds': float(time.time() - t_fy),
            }
        else:
            log(f"  Year {fy}: no valid SIC2 groups (all < {MIN_PEERS} peers)")
            per_year_summary[fy] = {
                'status': 'no_groups',
                'n_train': int(n_train),
                'n_test': int(n_test),
            }

        log(f"  Memory: {mem_mb():.0f} MB | Year elapsed: {time.time()-t_fy:.1f}s")

    # ═══════════════════════════════════════════════════════════
    # STEP 5: SAVE OUTPUTS
    # ═══════════════════════════════════════════════════════════

    log("")
    log("=" * 60)
    log("STEP 5: SAVING OUTPUTS")
    log("=" * 60)

    final_df = pd.concat(all_year_dfs, ignore_index=True)
    log(f"  Total rows: {len(final_df):,}")
    log(f"  Columns: {list(final_df.columns)}")
    log(f"  Formation years present: {sorted(final_df['formation_year'].unique())}")

    parquet_path = os.path.join(ARTIFACTS_DIR, '2a_02_pair_universe.parquet')
    final_df.to_parquet(parquet_path, index=False)
    file_size_mb = os.path.getsize(parquet_path) / 1e6
    log(f"  Saved: {parquet_path}")
    log(f"  Parquet size: {file_size_mb:.1f} MB")

    summary_data = {
        'test': '2a_02_pair_universe',
        'description': 'Walk-forward SAE pair universe: all within-SIC2 pairs per formation year',
        'config': {
            'n_components': N_COMPONENTS,
            'n_features_keep': N_FEATURES_KEEP,
            'n_features_filtered': int(n_filtered),
            'n_features_raw': int(n_raw),
            'min_peers': MIN_PEERS,
            'formation_years': FORMATION_YEARS,
            'seed': SEED,
            'n_companies_total': int(len(meta_df)),
        },
        'per_year': {str(k): v for k, v in per_year_summary.items()},
        'totals': {
            'n_rows': int(len(final_df)),
            'n_formation_years': len(FORMATION_YEARS),
            'parquet_size_mb': float(file_size_mb),
            'total_time_seconds': float(time.time() - t_start),
        },
    }
    summary_path = os.path.join(ARTIFACTS_DIR, '2a_02_summary.json')
    with open(summary_path, 'w') as f:
        json.dump(summary_data, f, indent=2)
    log(f"  Saved: {summary_path}")

    # ═══════════════════════════════════════════════════════════
    # STEP 6: VERIFICATION
    # ═══════════════════════════════════════════════════════════

    log("")
    log("=" * 60)
    log("STEP 6: VERIFICATION")
    log("=" * 60)

    # Per-year stats table
    log(f"\n{'Year':>6}  {'N_pairs':>10}  {'N_groups':>9}  {'N_test':>7}  {'cos_mean':>9}  {'cos_median':>11}")
    log("-" * 65)
    total_pairs = 0
    for fy in FORMATION_YEARS:
        s = per_year_summary.get(fy, {})
        if s.get('status') in ('skipped', 'no_groups'):
            log(f"{fy:>6}  {'SKIP':>10}  {'':>9}  {'':>7}  {'':>9}  {'':>11}")
            continue
        if 'n_pairs' not in s:
            continue
        total_pairs += s['n_pairs']
        log(f"{fy:>6}  {s['n_pairs']:>10,}  {s['n_sic2_groups']:>9}  "
            f"{s['n_test']:>7,}  {s['cosine_sim_mean']:>9.4f}  {s['cosine_sim_median']:>11.4f}")
    log("-" * 65)
    log(f"{'TOTAL':>6}  {total_pairs:>10,}")

    # Cross-check vs 2a_01 (which used min_peers=5; our min_peers=2 should be >= those numbers)
    log("")
    log("Cross-check vs 2a_01 (min_peers=5 → our min_peers=2 should be >= these):")
    check_years = {2002: (39, 768), 2005: (43, 823), 2010: (49, 1151), 2015: (53, 1208), 2018: (51, 1290)}
    all_ok = True
    for fy, (exp_groups, exp_companies) in sorted(check_years.items()):
        s = per_year_summary.get(fy, {})
        if s.get('status') in ('skipped', 'no_groups') or 'n_sic2_groups' not in s:
            log(f"  {fy}: MISSING")
            all_ok = False
            continue
        ok_groups = s['n_sic2_groups'] >= exp_groups
        ok_comp = s['n_test'] >= exp_companies
        status = "OK" if (ok_groups and ok_comp) else "FAIL"
        if status == "FAIL":
            all_ok = False
        log(f"  {fy}: groups {s['n_sic2_groups']:>3} (expect >= {exp_groups}) "
            f"companies {s['n_test']:>5,} (expect >= {exp_companies})  [{status}]")
    log(f"  Cross-check overall: {'PASS' if all_ok else 'FAIL'}")

    # SIC2 zero-padding verification
    log("")
    log("SIC2 zero-padding verification:")
    unique_sic2_final = sorted(final_df['sic2'].unique())
    bad_sic2 = [s for s in unique_sic2_final if len(s) != 2]
    if bad_sic2:
        log(f"  WARNING: non-2-digit SIC2 codes found: {bad_sic2}")
    else:
        log(f"  All SIC2 codes are 2-digit strings. OK")
    log(f"  Unique SIC2 codes ({len(unique_sic2_final)}): {unique_sic2_final}")

    # No-leakage check for formation_year=1999
    log("")
    log("No-leakage check (formation_year=1999):")
    log(f"  Expected n_train = companies with year<=1999 = {n_train_1999:,}")
    s1999 = per_year_summary.get(1999, {})
    if 'n_train' in s1999:
        match = s1999['n_train'] == n_train_1999
        log(f"  Recorded n_train = {s1999['n_train']:,}  [{'OK' if match else 'MISMATCH'}]")
    else:
        log(f"  1999 data missing from summary")

    # Parquet file size check
    log("")
    log(f"Parquet file size: {file_size_mb:.1f} MB")
    log(f"Total rows: {len(final_df):,}")
    log(f"Total elapsed: {time.time()-t_start:.1f}s")
    log("")
    log("DONE.")


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""
2a_01_walkforward_pca_diagnostic.py — Walk-Forward PCA Stability Test

THE QUESTION: Does walk-forward PCA (fitted only on filings available at
signal date) materially change the SAE similarity signal compared to the
global PCA used by Molinari et al.?

This is the highest-risk unknown before building the full backtest.
Molinari's global PCA across 1996-2020 is textbook look-ahead bias — the
dimensionality reduction sees future filing language. Every quant who reads
the paper will spot this. If walk-forward PCA degrades the within-SIC
re-ranking lift (T04's +26%), the signal may not survive temporal integrity
requirements. If it holds, we're clear to build.

WHAT IT DOES:
1. Loads raw 131K SAE features from HuggingFace
2. Applies feature filtering (matching run_phase1.py)
3. For each test year (2002, 2005, 2010, 2015, 2018):
   a. Fits PCA on filings <= test_year only (walk-forward)
   b. Fits PCA on ALL filings (global, Molinari baseline)
   c. Computes within-SIC cosine similarities in both PCA spaces
   d. Compares: rank correlation, precision lift, signal preservation

KEY METRICS:
- Spearman rank correlation between WF and global cosine sims
- Within-SIC precision@K lift for walk-forward vs random
- Lift preservation ratio: WF lift / global lift (>0.8 = safe, <0.5 = problem)

Output: experiments/artifacts/2a_01_walkforward_pca.json

RUNTIME: ~15-30 min depending on hardware. Peak RAM ~4-8 GB.
"""

import os
import sys
import gc
import json
import time
import warnings
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity
from scipy.stats import spearmanr
from datasets import load_dataset
from collections import defaultdict

warnings.filterwarnings('ignore')

# ═══════════════════════════════════════════════════════════
# CONFIG
# ═══════════════════════════════════════════════════════════

N_COMPONENTS = 4000
N_FEATURES_KEEP = 8000  # Top features by variance to keep (must be > N_COMPONENTS)
TEST_YEARS = [2002, 2005, 2010, 2015, 2018]
K_VALUES = [1, 3, 5, 10]
N_RANDOM_TRIALS = 100
MIN_PEERS = 5          # Minimum same-SIC peers for a company to be included
SEED = 42

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ARTIFACTS_DIR = os.path.join(BASE_DIR, "artifacts")


def log(msg):
    print(f"[2a_01] {msg}", flush=True)


def mem_mb():
    """Current process RSS in MB."""
    try:
        import resource
        return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024 * 1024)  # macOS returns bytes
    except Exception:
        return 0


def bootstrap_ci(values, n_resamples=2000, ci=0.95):
    """Bootstrap confidence interval for mean."""
    if len(values) == 0:
        return None, [None, None]
    values = np.array(values, dtype=np.float64)
    rng = np.random.default_rng(SEED)
    boot_means = np.array([
        rng.choice(values, size=len(values), replace=True).mean()
        for _ in range(n_resamples)
    ])
    alpha = 1 - ci
    return float(np.mean(values)), [
        float(np.percentile(boot_means, 100 * alpha / 2)),
        float(np.percentile(boot_means, 100 * (1 - alpha / 2))),
    ]


def main():
    np.random.seed(SEED)
    os.makedirs(ARTIFACTS_DIR, exist_ok=True)
    t_start = time.time()

    # ═══════════════════════════════════════════════════════════
    # STEP 1 + 2: LOAD & FILTER SAE FEATURES (streaming)
    #
    # The raw feature matrix is 27,888 × 131,072 × 4 bytes = ~14.6 GB.
    # That doesn't fit in 16 GB RAM. So we stream in batches:
    #   Pass 1: Compute activation rates (for feature filter mask)
    #   Pass 2: Load only the filtered columns (~5-20K features)
    # Peak memory: batch_size × 131K × 4 + n_rows × n_filtered × 4
    # With 500-row batches and ~10K filtered features: ~1.4 GB peak.
    # ═══════════════════════════════════════════════════════════

    log("=" * 60)
    log("STEP 1+2: LOAD & FILTER SAE FEATURES (streaming)")
    log(f"  (Memory: {mem_mb():.0f} MB)")
    log("=" * 60)

    log("Loading features dataset from HuggingFace...")
    log("  (marco-molinari/company_reports_with_features — ~27K rows × 131K dims)")
    t0 = time.time()
    features_ds = load_dataset('marco-molinari/company_reports_with_features', split='train')
    log(f"  Dataset loaded in {time.time()-t0:.1f}s")
    log(f"  Columns: {features_ds.column_names}")
    log(f"  Rows: {len(features_ds):,}")

    n_rows = len(features_ds)
    BATCH = 500  # rows per batch — keeps peak memory low

    # Extract __index_level_0__ for joining with pairs/company data
    if '__index_level_0__' in features_ds.column_names:
        feat_hf_indices = features_ds['__index_level_0__']
        log("  Found __index_level_0__ in features dataset")
    else:
        feat_hf_indices = list(range(n_rows))
        log("  WARNING: No __index_level_0__ — using row position as index")

    def extract_batch(batch_features):
        """Convert a batch of feature rows to float32 numpy, handling nested lists."""
        rows = []
        for row in batch_features:
            # Unwrap one level of nesting if present
            if isinstance(row, list) and len(row) > 0 and isinstance(row[0], list):
                row = row[0]
            arr = np.asarray(row, dtype=np.float32).flatten()
            rows.append(arr)

        lengths = [len(r) for r in rows]
        mode_len = max(set(lengths), key=lengths.count)

        # Build output, filling any malformed rows with zeros to preserve indexing
        out = np.zeros((len(rows), mode_len), dtype=np.float32)
        for i, r in enumerate(rows):
            if len(r) == mode_len:
                out[i] = r
        return out

    # --- Pass 1: Compute per-feature variance (streaming) ---
    # NOTE: The ACL paper does NOT filter features — it PCA's all 131K directly.
    # But 28K×131K×4B = 14.7 GB won't fit in RAM. Instead we select the top
    # N_FEATURES_KEEP features by variance (most informative for PCA) and load
    # only those in Pass 2. Variance is computed via streaming E[X] and E[X²].
    # The filter is computed on the FULL corpus (all years). For this diagnostic
    # we hold filtering constant to isolate the PCA variable only.
    log("  Pass 1: Computing per-feature variance (streaming)...")
    t0 = time.time()
    feat_sum = None
    feat_sq_sum = None
    n_raw = None

    for start in range(0, n_rows, BATCH):
        end = min(start + BATCH, n_rows)
        batch_arr = extract_batch(features_ds[start:end]['features'])
        if feat_sum is None:
            n_raw = batch_arr.shape[1]
            feat_sum = np.zeros(n_raw, dtype=np.float64)
            feat_sq_sum = np.zeros(n_raw, dtype=np.float64)
        feat_sum += batch_arr.sum(axis=0).astype(np.float64)
        feat_sq_sum += (batch_arr.astype(np.float64) ** 2).sum(axis=0)
        if (start // BATCH) % 10 == 0:
            log(f"    Batch {start // BATCH + 1}/{(n_rows + BATCH - 1) // BATCH} ({mem_mb():.0f} MB)")

    feat_mean = feat_sum / n_rows
    feat_var = feat_sq_sum / n_rows - feat_mean ** 2
    # Select top N_FEATURES_KEEP by variance
    n_keep = min(N_FEATURES_KEEP, n_raw)
    top_indices = np.argsort(feat_var)[::-1][:n_keep]
    top_indices.sort()  # Sort for consistent column ordering
    combined_mask = np.zeros(n_raw, dtype=bool)
    combined_mask[top_indices] = True
    n_filtered = int(combined_mask.sum())
    del feat_sum, feat_sq_sum, feat_mean, feat_var, top_indices
    gc.collect()

    log(f"  Pass 1 done in {time.time()-t0:.1f}s")
    log(f"  Raw features: {n_raw:,}")
    log(f"  Selected top {n_filtered:,} features by variance")
    log(f"  Filtered matrix will be: {n_rows} × {n_filtered} = {n_rows * n_filtered * 4 / 1e9:.2f} GB")

    # --- Pass 2: Build filtered feature matrix ---
    log("  Pass 2: Building filtered feature matrix...")
    t0 = time.time()
    features = np.empty((n_rows, n_filtered), dtype=np.float32)

    for start in range(0, n_rows, BATCH):
        end = min(start + BATCH, n_rows)
        batch_arr = extract_batch(features_ds[start:end]['features'])
        features[start:end] = batch_arr[:, combined_mask]
        if (start // BATCH) % 10 == 0:
            log(f"    Batch {start // BATCH + 1}/{(n_rows + BATCH - 1) // BATCH} ({mem_mb():.0f} MB)")

    # Free the HF dataset object
    del features_ds
    gc.collect()

    log(f"  Pass 2 done in {time.time()-t0:.1f}s")
    log(f"  Filtered matrix: {features.shape}")
    log(f"  Memory: {mem_mb():.0f} MB")

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
    log(f"  Loaded in {time.time()-t0:.1f}s, rows: {len(company_ds):,}")

    # Build metadata DataFrame
    meta_df = pd.DataFrame({
        'hf_index': company_ds['__index_level_0__'] if '__index_level_0__' in company_ds.column_names else range(len(company_ds)),
        'year': [int(y) for y in company_ds['year']],
        'sic_code': company_ds['sic_code'],
        'ticker': company_ds['ticker'],
        'cik': company_ds['cik'],
    })
    del company_ds
    gc.collect()

    # Clean
    meta_df = meta_df.dropna(subset=['sic_code']).copy()
    meta_df['sic2'] = meta_df['sic_code'].astype(str).str[:2]

    # Map hf_index -> feature row position
    hf_to_feat_row = {idx: i for i, idx in enumerate(feat_hf_indices)}
    meta_df['feat_row'] = meta_df['hf_index'].map(hf_to_feat_row)
    meta_df = meta_df.dropna(subset=['feat_row']).copy()
    meta_df['feat_row'] = meta_df['feat_row'].astype(int)
    meta_df['hf_index'] = meta_df['hf_index'].astype(int)

    log(f"  Companies with features + SIC: {len(meta_df):,}")
    log(f"  Year range: {meta_df['year'].min()}-{meta_df['year'].max()}")
    log(f"  Unique SIC2 codes: {meta_df['sic2'].nunique()}")

    # ═══════════════════════════════════════════════════════════
    # STEP 4: LOAD PAIRS (for correlation ground truth)
    # ═══════════════════════════════════════════════════════════

    log("")
    log("=" * 60)
    log("STEP 4: LOAD PAIRS DATA (correlation ground truth)")
    log("=" * 60)

    log("Loading pre-computed pairs from HuggingFace...")
    t0 = time.time()
    pairs_ds = load_dataset("v1ctor10/cos_sim_4000pca_exp", split="train")
    pairs_df = pairs_ds.to_pandas()
    del pairs_ds
    gc.collect()

    pairs_df = pairs_df.dropna(subset=["correlation"]).reset_index(drop=True)
    pairs_df["year"] = pairs_df["year"].astype(int)
    pairs_df["Company1"] = pairs_df["Company1"].astype(int)
    pairs_df["Company2"] = pairs_df["Company2"].astype(int)
    log(f"  Pairs: {len(pairs_df):,} ({time.time()-t0:.1f}s)")
    log(f"  Memory: {mem_mb():.0f} MB")

    # ═══════════════════════════════════════════════════════════
    # STEP 5: FIT GLOBAL PCA (Molinari baseline)
    # ═══════════════════════════════════════════════════════════

    log("")
    log("=" * 60)
    log("STEP 5: FIT GLOBAL PCA (Molinari baseline)")
    log("=" * 60)

    n_comp = min(N_COMPONENTS, features.shape[0] - 1, features.shape[1])
    log(f"Fitting global PCA: {n_comp} components on {features.shape[0]} × {features.shape[1]}...")
    t0 = time.time()

    global_pca = PCA(n_components=n_comp)
    global_pca_features = global_pca.fit_transform(features)
    global_var = float(global_pca.explained_variance_ratio_.sum())

    log(f"  Variance preserved: {global_var:.4f} ({global_var*100:.2f}%)")
    log(f"  Output shape: {global_pca_features.shape}")
    log(f"  Time: {time.time()-t0:.1f}s")

    # ═══════════════════════════════════════════════════════════
    # STEP 6: WALK-FORWARD PCA DIAGNOSTIC (per test year)
    # ═══════════════════════════════════════════════════════════

    results = {
        'test': '2a_01_walkforward_pca_diagnostic',
        'description': 'Walk-forward PCA stability test: does temporal PCA preserve SAE re-ranking lift?',
        'config': {
            'n_components': n_comp,
            'n_features_keep': N_FEATURES_KEEP,
            'n_features_filtered': int(features.shape[1]),
            'n_features_raw': int(n_raw),
            'feature_selection': 'top_by_variance',
            'global_pca_variance': global_var,
            'test_years': TEST_YEARS,
            'k_values': K_VALUES,
            'n_random_trials': N_RANDOM_TRIALS,
            'min_peers': MIN_PEERS,
            'seed': SEED,
            'n_companies_total': len(meta_df),
        },
        'years': {},
        'summary': {},
    }

    rng = np.random.default_rng(SEED)

    for test_year in TEST_YEARS:
        log(f"\n{'='*60}")
        log(f"TEST YEAR: {test_year}")
        log(f"{'='*60}")
        t_year = time.time()

        # --- Define training and test sets ---
        wf_meta = meta_df[meta_df['year'] <= test_year]
        test_meta = meta_df[meta_df['year'] == test_year].copy()

        wf_feat_rows = wf_meta['feat_row'].values
        test_feat_rows = test_meta['feat_row'].values

        n_train = len(wf_feat_rows)
        n_test = len(test_feat_rows)
        log(f"  Training (years <= {test_year}): {n_train:,} filings")
        log(f"  Test (year == {test_year}): {n_test:,} filings")

        if n_test < 20:
            log(f"  SKIP: too few test filings")
            results['years'][str(test_year)] = {'status': 'skipped', 'reason': 'too few test filings'}
            continue

        # --- Fit walk-forward PCA ---
        wf_n_comp = min(N_COMPONENTS, n_train - 1, features.shape[1])
        log(f"  Fitting walk-forward PCA ({wf_n_comp} components on {n_train} samples)...")
        t0 = time.time()

        wf_pca = PCA(n_components=wf_n_comp)
        wf_pca.fit(features[wf_feat_rows])
        wf_var = float(wf_pca.explained_variance_ratio_.sum())
        log(f"  WF PCA variance: {wf_var:.4f} ({wf_var*100:.2f}%) in {time.time()-t0:.1f}s")

        # --- Transform test filings in both PCA spaces ---
        wf_test_pca = wf_pca.transform(features[test_feat_rows])
        global_test_pca = global_pca_features[test_feat_rows]

        del wf_pca
        gc.collect()

        # --- Build correlation lookup for this year ---
        log(f"  Building correlation lookup for year {test_year}...")
        year_pairs = pairs_df[pairs_df['year'] == test_year]
        # Vectorized dict construction (iterrows is ~100x slower)
        c1_arr = year_pairs['Company1'].values
        c2_arr = year_pairs['Company2'].values
        corr_arr = year_pairs['correlation'].values
        cos_arr = year_pairs['cosine_similarity'].values
        corr_lookup = {(int(c1_arr[i]), int(c2_arr[i])): float(corr_arr[i]) for i in range(len(c1_arr))}
        global_cos_lookup = {(int(c1_arr[i]), int(c2_arr[i])): float(cos_arr[i]) for i in range(len(c1_arr))}
        log(f"  Correlation pairs for {test_year}: {len(corr_lookup):,}")

        # --- Build local index mappings ---
        # Map: feat_row -> position in test arrays (0, 1, 2, ...)
        feat_row_to_local = {fr: i for i, fr in enumerate(test_feat_rows)}

        # --- Compute within-SIC metrics ---
        log(f"  Computing within-SIC precision metrics...")
        t0 = time.time()

        sic2_groups = test_meta.groupby('sic2')

        all_wf_cosines = []
        all_global_cosines = []
        # Sanity check: aligned pairs where we have BOTH recomputed and pre-computed global cosines
        sanity_recomputed = []
        sanity_precomputed = []

        precision_wf = {k: [] for k in K_VALUES}
        precision_global = {k: [] for k in K_VALUES}

        n_groups_processed = 0
        n_companies_processed = 0

        for sic2, group in sic2_groups:
            if len(group) < MIN_PEERS:
                continue

            group_feat_rows = group['feat_row'].values
            group_hf_indices = group['hf_index'].values
            n_group = len(group)

            # Get local indices into test PCA arrays
            local_idxs = np.array([feat_row_to_local[fr] for fr in group_feat_rows])

            # Compute cosine similarity matrices
            wf_cos_matrix = cosine_similarity(wf_test_pca[local_idxs])
            global_cos_matrix = cosine_similarity(global_test_pca[local_idxs])

            n_groups_processed += 1

            # For each company in the group: compute precision@K
            for i in range(n_group):
                hf_i = int(group_hf_indices[i])

                # Collect peers with known correlations
                peers = []
                for j in range(n_group):
                    if i == j:
                        continue
                    hf_j = int(group_hf_indices[j])

                    # Look up return correlation
                    corr = corr_lookup.get((hf_i, hf_j))
                    if corr is None:
                        corr = corr_lookup.get((hf_j, hf_i))
                    if corr is None:
                        continue

                    # Look up pre-computed global cosine sim (sanity check)
                    precomp_cos = global_cos_lookup.get((hf_i, hf_j))
                    if precomp_cos is None:
                        precomp_cos = global_cos_lookup.get((hf_j, hf_i))

                    peers.append({
                        'wf_cos': float(wf_cos_matrix[i, j]),
                        'global_cos': float(global_cos_matrix[i, j]),
                        'precomp_cos': precomp_cos,
                        'corr': corr,
                    })

                if len(peers) < MIN_PEERS:
                    continue

                n_companies_processed += 1

                # Collect cosine sim pairs for rank correlation
                for p in peers:
                    all_wf_cosines.append(p['wf_cos'])
                    all_global_cosines.append(p['global_cos'])
                    # Sanity check: only collect ALIGNED pairs where precomp exists
                    if p['precomp_cos'] is not None:
                        sanity_recomputed.append(p['global_cos'])
                        sanity_precomputed.append(p['precomp_cos'])

                # Sort peers by each cosine sim measure
                peers_by_wf = sorted(peers, key=lambda x: x['wf_cos'], reverse=True)
                peers_by_global = sorted(peers, key=lambda x: x['global_cos'], reverse=True)

                for K in K_VALUES:
                    if K > len(peers):
                        continue

                    # Walk-forward top-K mean correlation
                    wf_topk = np.mean([p['corr'] for p in peers_by_wf[:K]])

                    # Global top-K mean correlation
                    global_topk = np.mean([p['corr'] for p in peers_by_global[:K]])

                    # Random baseline (same for both — same peer pool)
                    random_means = []
                    for _ in range(N_RANDOM_TRIALS):
                        rand_idx = rng.choice(len(peers), size=K, replace=False)
                        random_means.append(np.mean([peers[idx]['corr'] for idx in rand_idx]))
                    random_baseline = np.mean(random_means)

                    precision_wf[K].append({
                        'topk_corr': float(wf_topk),
                        'random_corr': float(random_baseline),
                        'lift': float(wf_topk - random_baseline),
                    })
                    precision_global[K].append({
                        'topk_corr': float(global_topk),
                        'random_corr': float(random_baseline),
                        'lift': float(global_topk - random_baseline),
                    })

        log(f"  Processed {n_groups_processed} SIC2 groups, {n_companies_processed} company-years")
        log(f"  Cosine sim pairs: {len(all_wf_cosines):,}")
        log(f"  Time: {time.time()-t0:.1f}s")

        # --- Compute aggregate metrics ---

        # 1. Rank correlation between WF and global cosine sims
        if len(all_wf_cosines) > 10:
            spearman_rho, spearman_p = spearmanr(all_wf_cosines, all_global_cosines)
        else:
            spearman_rho, spearman_p = None, None

        # 2. Sanity check: correlation between recomputed global and pre-computed
        #    Arrays are collected in lockstep — only where precomp_cos is not None
        if len(sanity_recomputed) > 10:
            sanity_rho, sanity_p = spearmanr(sanity_recomputed, sanity_precomputed)
        else:
            sanity_rho, sanity_p = None, None

        # 3. Precision lift comparison
        year_result = {
            'n_train': n_train,
            'n_test': n_test,
            'n_sic2_groups': n_groups_processed,
            'n_companies': n_companies_processed,
            'n_cosine_pairs': len(all_wf_cosines),
            'wf_pca_variance': wf_var,
            'wf_pca_components': wf_n_comp,
            'spearman_wf_vs_global': {
                'rho': float(spearman_rho) if spearman_rho is not None else None,
                'p_value': float(spearman_p) if spearman_p is not None else None,
            },
            'sanity_recomputed_vs_precomputed': {
                'rho': float(sanity_rho) if sanity_rho is not None else None,
                'p_value': float(sanity_p) if sanity_p is not None else None,
                'n_pairs': len(sanity_precomputed),
            },
            'precision_at_k': {},
        }

        for K in K_VALUES:
            wf_lifts = [r['lift'] for r in precision_wf[K]]
            global_lifts = [r['lift'] for r in precision_global[K]]

            if len(wf_lifts) == 0:
                year_result['precision_at_k'][f'K_{K}'] = None
                continue

            wf_lift_mean, wf_lift_ci = bootstrap_ci(wf_lifts)
            global_lift_mean, global_lift_ci = bootstrap_ci(global_lifts)

            # Lift preservation ratio
            if global_lift_mean and abs(global_lift_mean) > 1e-6:
                preservation = wf_lift_mean / global_lift_mean
            else:
                preservation = None

            # Hit rates
            wf_hit_rate = np.mean([1.0 if r['lift'] > 0 else 0.0 for r in precision_wf[K]])
            global_hit_rate = np.mean([1.0 if r['lift'] > 0 else 0.0 for r in precision_global[K]])

            # Mean top-K correlations
            wf_topk_mean = np.mean([r['topk_corr'] for r in precision_wf[K]])
            global_topk_mean = np.mean([r['topk_corr'] for r in precision_global[K]])
            random_mean = np.mean([r['random_corr'] for r in precision_wf[K]])

            year_result['precision_at_k'][f'K_{K}'] = {
                'n_samples': len(wf_lifts),
                'walkforward': {
                    'topk_mean_corr': float(wf_topk_mean),
                    'lift': float(wf_lift_mean),
                    'lift_ci_95': [float(wf_lift_ci[0]), float(wf_lift_ci[1])],
                    'hit_rate': float(wf_hit_rate),
                },
                'global': {
                    'topk_mean_corr': float(global_topk_mean),
                    'lift': float(global_lift_mean),
                    'lift_ci_95': [float(global_lift_ci[0]), float(global_lift_ci[1])],
                    'hit_rate': float(global_hit_rate),
                },
                'random_baseline_corr': float(random_mean),
                'lift_preservation_ratio': float(preservation) if preservation is not None else None,
            }

            pres_str = f"{preservation:.3f}" if preservation is not None else "N/A"
            log(f"  K={K}: WF lift={wf_lift_mean:.4f}, Global lift={global_lift_mean:.4f}, "
                f"Preservation={pres_str}, "
                f"WF hit={wf_hit_rate:.1%}, n={len(wf_lifts)}")

        log(f"  Spearman(WF vs Global cosine): {spearman_rho:.4f}" if spearman_rho else "  Spearman: N/A")
        if sanity_rho:
            log(f"  Sanity check (recomputed vs pre-computed global): {sanity_rho:.4f}")

        results['years'][str(test_year)] = year_result
        log(f"  Year {test_year} complete in {time.time()-t_year:.1f}s")

        # Cleanup
        del wf_test_pca, global_test_pca, year_pairs
        gc.collect()

    # ═══════════════════════════════════════════════════════════
    # STEP 7: SUMMARY AND VERDICT
    # ═══════════════════════════════════════════════════════════

    log(f"\n{'='*60}")
    log("SUMMARY")
    log(f"{'='*60}")

    # Aggregate across years
    all_spearman = []
    all_preservation_k1 = []
    all_preservation_k5 = []

    for yr_str, yr_data in results['years'].items():
        if isinstance(yr_data, dict) and yr_data.get('status') == 'skipped':
            continue

        rho = yr_data['spearman_wf_vs_global']['rho']
        if rho is not None:
            all_spearman.append(rho)

        k1 = yr_data['precision_at_k'].get('K_1')
        k5 = yr_data['precision_at_k'].get('K_5')
        if k1 and k1['lift_preservation_ratio'] is not None:
            all_preservation_k1.append(k1['lift_preservation_ratio'])
        if k5 and k5['lift_preservation_ratio'] is not None:
            all_preservation_k5.append(k5['lift_preservation_ratio'])

    # Verdict logic
    verdict = "INCONCLUSIVE"
    interpretation = []

    if all_spearman:
        mean_rho = np.mean(all_spearman)
        min_rho = np.min(all_spearman)
        interpretation.append(f"Spearman(WF vs Global): mean={mean_rho:.4f}, min={min_rho:.4f}")

        if min_rho >= 0.95:
            interpretation.append("Cosine sim rankings near-identical (rho >= 0.95 in all years)")
        elif min_rho >= 0.85:
            interpretation.append("Cosine sim rankings highly correlated (rho >= 0.85)")
        else:
            interpretation.append(f"WARNING: Cosine sim rankings diverge (min rho={min_rho:.4f})")

    if all_preservation_k1:
        mean_pres_k1 = np.mean(all_preservation_k1)
        min_pres_k1 = np.min(all_preservation_k1)
        interpretation.append(f"K=1 lift preservation: mean={mean_pres_k1:.3f}, min={min_pres_k1:.3f}")

    if all_preservation_k5:
        mean_pres_k5 = np.mean(all_preservation_k5)
        min_pres_k5 = np.min(all_preservation_k5)
        interpretation.append(f"K=5 lift preservation: mean={mean_pres_k5:.3f}, min={min_pres_k5:.3f}")

    # Decision criteria
    if all_spearman and all_preservation_k5:
        mean_rho = np.mean(all_spearman)
        mean_pres = np.mean(all_preservation_k5)
        min_pres = np.min(all_preservation_k5)

        if mean_rho >= 0.90 and min_pres >= 0.70:
            verdict = "PASS"
            interpretation.append("Walk-forward PCA preserves signal. Safe to proceed with backtest.")
        elif mean_rho >= 0.80 and min_pres >= 0.50:
            verdict = "QUALIFIED PASS"
            interpretation.append("Some degradation but signal survives. Proceed with caution.")
        else:
            verdict = "FAIL"
            interpretation.append("Walk-forward PCA materially degrades signal. Investigate before proceeding.")

    results['summary'] = {
        'verdict': verdict,
        'interpretation': interpretation,
        'mean_spearman_rho': float(np.mean(all_spearman)) if all_spearman else None,
        'min_spearman_rho': float(np.min(all_spearman)) if all_spearman else None,
        'mean_k1_preservation': float(np.mean(all_preservation_k1)) if all_preservation_k1 else None,
        'mean_k5_preservation': float(np.mean(all_preservation_k5)) if all_preservation_k5 else None,
        'min_k5_preservation': float(np.min(all_preservation_k5)) if all_preservation_k5 else None,
        'total_time_seconds': time.time() - t_start,
    }

    # Print summary
    log("")
    log(f"VERDICT: {verdict}")
    log("")
    for line in interpretation:
        log(f"  {line}")

    log("")
    log("Per-year detail:")
    print(f"  {'Year':<6} {'Spearman':>10} {'WF Var':>8} {'K=1 Pres':>10} {'K=5 Pres':>10} {'K=1 WF Lift':>12} {'K=5 WF Lift':>12}")
    print(f"  {'-'*6} {'-'*10} {'-'*8} {'-'*10} {'-'*10} {'-'*12} {'-'*12}")

    for yr_str in sorted(results['years'].keys()):
        yr = results['years'][yr_str]
        if isinstance(yr, dict) and yr.get('status') == 'skipped':
            print(f"  {yr_str:<6} {'SKIPPED':>10}")
            continue

        rho = yr['spearman_wf_vs_global']['rho']
        wf_var_yr = yr['wf_pca_variance']
        k1 = yr['precision_at_k'].get('K_1')
        k5 = yr['precision_at_k'].get('K_5')

        k1_pres = k1['lift_preservation_ratio'] if k1 and k1.get('lift_preservation_ratio') is not None else None
        k5_pres = k5['lift_preservation_ratio'] if k5 and k5.get('lift_preservation_ratio') is not None else None
        k1_wf_lift = k1['walkforward']['lift'] if k1 else None
        k5_wf_lift = k5['walkforward']['lift'] if k5 else None

        cols = [f"  {yr_str:<6}"]
        cols.append(f"{rho:>10.4f}" if rho is not None else f"{'N/A':>10}")
        cols.append(f"{wf_var_yr:>8.4f}")
        cols.append(f"{k1_pres:>10.3f}" if k1_pres is not None else f"{'N/A':>10}")
        cols.append(f"{k5_pres:>10.3f}" if k5_pres is not None else f"{'N/A':>10}")
        cols.append(f"{k1_wf_lift:>12.4f}" if k1_wf_lift is not None else f"{'N/A':>12}")
        cols.append(f"{k5_wf_lift:>12.4f}" if k5_wf_lift is not None else f"{'N/A':>12}")
        print(" ".join(cols))

    # Save results
    output_path = os.path.join(ARTIFACTS_DIR, "2a_01_walkforward_pca.json")
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2, default=str)

    log(f"\nResults saved: {output_path}")
    log(f"Total time: {time.time() - t_start:.1f}s")
    log(f"Peak memory: {mem_mb():.0f} MB")


if __name__ == "__main__":
    main()

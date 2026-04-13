#!/usr/bin/env python3
"""
2b_analog_prediction.py — Analog Return Prediction

Tests whether SAE-matched historical analogs predict future company returns.
For each target company in each formation year, finds the most structurally
similar companies from PRIOR years using walk-forward PCA + cosine similarity
within SIC2 industries, and uses their realized market-adjusted outcomes to
predict the target's future return.

Inputs:
  HuggingFace: marco-molinari/company_reports_with_features
  HuggingFace: Mateusz1017/annual_reports_tokenized_llama3_...
  experiments/artifacts/2a_03_returns.parquet
  experiments/artifacts/2a_03_factors.parquet

Outputs:
  experiments/artifacts/2b_strategy_returns.parquet
  experiments/artifacts/2b_analog_details.parquet
  experiments/artifacts/2b_placebo.json
  experiments/artifacts/2b_summary.json
"""

import os
import gc
import json
import time
import warnings
import numpy as np
import pandas as pd
from datasets import load_dataset
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity
from scipy.stats import spearmanr
from statsmodels.regression.linear_model import OLS
from statsmodels.tools import add_constant

warnings.filterwarnings('ignore', category=FutureWarning)

# ═══════════════════════════════════════════════════════════
# CONFIG
# ═══════════════════════════════════════════════════════════

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ARTIFACTS_DIR = os.path.join(BASE_DIR, "artifacts")
os.makedirs(ARTIFACTS_DIR, exist_ok=True)

RETURNS_FILE = os.path.join(ARTIFACTS_DIR, "2a_03_returns.parquet")
FACTORS_FILE = os.path.join(ARTIFACTS_DIR, "2a_03_factors.parquet")

STRATEGY_RETURNS_OUT = os.path.join(ARTIFACTS_DIR, "2b_strategy_returns.parquet")
ANALOG_DETAILS_OUT = os.path.join(ARTIFACTS_DIR, "2b_analog_details.parquet")
PLACEBO_OUT = os.path.join(ARTIFACTS_DIR, "2b_placebo.json")
SUMMARY_OUT = os.path.join(ARTIFACTS_DIR, "2b_summary.json")

BATCH = 500
N_FEATURES_KEEP = 8000
N_COMPONENTS = 4000
TRANSFORM_CHUNK = 5000

FORMATION_YEARS_FULL = list(range(1999, 2019))
FORMATION_YEARS_PARTIAL = [2019]
FORMATION_YEARS = FORMATION_YEARS_FULL + FORMATION_YEARS_PARTIAL

K_VALUES = [1, 3, 5, 10]
PRIMARY_K = 5
MIN_ANALOGS = 3
MIN_TARGETS = 2
MIN_OUTCOME_MONTHS = 8
MIN_QUINTILE_SIZE = 10

N_PLACEBO_TRIALS = 100
MAX_MONTHLY_SIMPLE_RETURN = 2.0
SEED = 42


def log(msg):
    print(f"[2b] {msg}", flush=True)


def mem_mb():
    try:
        import resource
        return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024 * 1024)
    except Exception:
        return 0


def get_months_sequence(start_ym, n_months):
    months = []
    y, m = divmod(start_ym, 100)
    for _ in range(n_months):
        months.append(y * 100 + m)
        m += 1
        if m > 12:
            m = 1
            y += 1
    return months


def get_outcome_months(filing_year):
    """Analog outcome period: Jul FY+1 through Jun FY+2."""
    return get_months_sequence((filing_year + 1) * 100 + 7, 12)


def get_trading_months(fy):
    """Target trading period: Jul FY+1 through Jun FY+2 (6 months for FY2019)."""
    if fy in FORMATION_YEARS_PARTIAL:
        return get_months_sequence((fy + 1) * 100 + 7, 6)
    return get_months_sequence((fy + 1) * 100 + 7, 12)


def clean_ticker(t):
    t = str(t)
    if t.startswith('[') and t.endswith(']'):
        t = t[1:-1].strip("'\"")
    return t


def min_months_for_period(n_months):
    """Scale MIN_OUTCOME_MONTHS proportionally for partial trading periods."""
    return max(int(MIN_OUTCOME_MONTHS * n_months / 12), 3)


def extract_batch(batch_features):
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


t_start = time.time()

# ═══════════════════════════════════════════════════════════
# STEP 1: Load SAE features (two-pass streaming)
# ═══════════════════════════════════════════════════════════

log("=" * 60)
log("STEP 1: Load SAE features (two-pass streaming)")
log("=" * 60)

log("Loading features dataset from HuggingFace...")
t0 = time.time()
features_ds = load_dataset('marco-molinari/company_reports_with_features', split='train')
n_rows = len(features_ds)
feat_hf_indices = features_ds['__index_level_0__']
log(f"  Loaded in {time.time()-t0:.1f}s | rows: {n_rows:,} | Memory: {mem_mb():.0f} MB")

# Pass 1: Per-feature variance
n_batches = (n_rows + BATCH - 1) // BATCH
log(f"Pass 1: Computing per-feature variance ({n_batches} batches)...")
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
    batch_num = start // BATCH
    if batch_num % 10 == 0:
        log(f"    Batch {batch_num + 1}/{n_batches} ({mem_mb():.0f} MB)")

feat_mean = feat_sum / n_rows
feat_var = feat_sq_sum / n_rows - feat_mean ** 2
n_keep = min(N_FEATURES_KEEP, n_raw)
top_indices = np.argsort(feat_var)[::-1][:n_keep]
top_indices.sort()
combined_mask = np.zeros(n_raw, dtype=bool)
combined_mask[top_indices] = True
n_filtered = int(combined_mask.sum())
del feat_sum, feat_sq_sum, feat_mean, feat_var, top_indices
gc.collect()
log(f"  Pass 1 done in {time.time()-t0:.1f}s | raw: {n_raw:,} | selected: {n_filtered:,}")

# Pass 2: Build filtered feature matrix
log(f"Pass 2: Building filtered feature matrix ({n_batches} batches)...")
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
# STEP 2: Load company metadata
# ═══════════════════════════════════════════════════════════

log("")
log("=" * 60)
log("STEP 2: Load company metadata")
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


def sic2_from_code(x):
    try:
        return str(int(float(x))).zfill(4)[:2]
    except (ValueError, TypeError):
        return None


meta_df = meta_df.dropna(subset=['sic_code']).copy()
meta_df['cik'] = pd.to_numeric(meta_df['cik'], errors='coerce')
meta_df = meta_df.dropna(subset=['cik']).copy()
meta_df['sic2'] = meta_df['sic_code'].apply(sic2_from_code)
meta_df = meta_df.dropna(subset=['sic2']).copy()

hf_to_feat_row = {int(idx): i for i, idx in enumerate(feat_hf_indices)}
meta_df['feat_row'] = meta_df['hf_index'].map(hf_to_feat_row)
meta_df = meta_df.dropna(subset=['feat_row']).copy()
meta_df['feat_row'] = meta_df['feat_row'].astype(int)
meta_df['hf_index'] = meta_df['hf_index'].astype(int)
meta_df['cik'] = meta_df['cik'].astype(int)
meta_df['ticker'] = meta_df['ticker'].apply(clean_ticker)

log(f"  Companies with features + SIC: {len(meta_df):,}")
log(f"  Year range: {meta_df['year'].min()}-{meta_df['year'].max()}")
log(f"  Unique SIC2: {meta_df['sic2'].nunique()}, Unique CIKs: {meta_df['cik'].nunique()}")

# Build hf_index -> cik mapping from metadata
hf_to_cik = dict(zip(meta_df['hf_index'].values, meta_df['cik'].values))

# ═══════════════════════════════════════════════════════════
# STEP 3: Load returns and factors, build lookups
# ═══════════════════════════════════════════════════════════

log("")
log("=" * 60)
log("STEP 3: Load returns and factors, build lookups")
log("=" * 60)

returns_df = pd.read_parquet(RETURNS_FILE)
factors_df = pd.read_parquet(FACTORS_FILE)
log(f"  Returns: {len(returns_df):,} rows | Factors: {len(factors_df):,} rows")

# Return lookup: (cik, YYYYMM) -> simple_return
n_extreme = 0
return_lookup = {}
for row in returns_df.itertuples(index=False):
    sr = row.simple_return
    if np.isnan(sr):
        continue
    if abs(sr) > MAX_MONTHLY_SIMPLE_RETURN:
        n_extreme += 1
        continue
    return_lookup[(int(row.cik), int(row.calendar_month))] = float(sr)

log(f"  Return lookup: {len(return_lookup):,} entries (filtered {n_extreme} extreme)")

# Factor lookup: YYYYMM -> market_return and full factor dict
factor_lookup = {}
factor_full = {}
for row in factors_df.itertuples(index=False):
    d = int(row.date)
    mkt_rf = float(getattr(row, 'Mkt_RF', 0) if hasattr(row, 'Mkt_RF') else row[1])
    # itertuples renames Mkt-RF to _1 — use positional access
factor_lookup = {}
factor_full = {}
for _, row in factors_df.iterrows():
    d = int(row['date'])
    factor_lookup[d] = row['Mkt-RF'] + row['RF']
    factor_full[d] = {
        'Mkt-RF': float(row['Mkt-RF']), 'SMB': float(row['SMB']),
        'HML': float(row['HML']), 'RMW': float(row['RMW']),
        'CMA': float(row['CMA']), 'RF': float(row['RF']),
        'MOM': float(row['MOM']),
    }

log(f"  Factor lookup: {len(factor_lookup):,} months")
del returns_df, factors_df
gc.collect()

# ═══════════════════════════════════════════════════════════
# STEP 4: Pre-compute analog outcomes (market-adjusted)
# ═══════════════════════════════════════════════════════════

log("")
log("=" * 60)
log("STEP 4: Pre-compute analog outcomes (market-adjusted)")
log("=" * 60)

t0 = time.time()
analog_outcomes = {}
outcome_valid_months = {}

meta_records = meta_df[['hf_index', 'year', 'cik']].to_dict('records')
for rec in meta_records:
    cidx = int(rec['hf_index'])
    fy = int(rec['year'])
    cik = int(rec['cik'])

    outcome_months = get_outcome_months(fy)
    cum = 1.0
    n_valid = 0
    for yyyymm in outcome_months:
        r = return_lookup.get((cik, yyyymm))
        mkt = factor_lookup.get(yyyymm)
        if r is not None and mkt is not None:
            cum *= (1.0 + r - mkt)
            n_valid += 1

    if n_valid >= MIN_OUTCOME_MONTHS:
        analog_outcomes[cidx] = cum - 1.0
        outcome_valid_months[cidx] = n_valid

log(f"  Outcomes computed: {len(analog_outcomes):,} / {len(meta_df):,} ({len(analog_outcomes)/len(meta_df)*100:.1f}%)")
log(f"  Mean valid months: {np.mean(list(outcome_valid_months.values())):.1f}")
log(f"  Mean market-adj return: {np.mean(list(analog_outcomes.values())):.4f}")
log(f"  Time: {time.time()-t0:.1f}s")

# ═══════════════════════════════════════════════════════════
# STEP 5: Cross-year analog matching (walk-forward PCA)
# ═══════════════════════════════════════════════════════════

log("")
log("=" * 60)
log("STEP 5: Cross-year analog matching (walk-forward PCA)")
log("=" * 60)

all_predictions = []
primary_analog_matches = []
analog_pools_for_placebo = {}

for fy_idx, fy in enumerate(FORMATION_YEARS):
    log(f"\n--- Formation year {fy} ({fy_idx+1}/{len(FORMATION_YEARS)}) ---")
    t_fy = time.time()

    wf_meta = meta_df[meta_df['year'] <= fy]
    target_meta = meta_df[meta_df['year'] == fy]
    analog_meta = meta_df[(meta_df['year'] < fy) & (meta_df['hf_index'].isin(analog_outcomes))]

    n_train = len(wf_meta)
    n_targets = len(target_meta)
    n_analogs = len(analog_meta)
    log(f"  Train: {n_train:,} | Targets: {n_targets:,} | Analogs w/outcomes: {n_analogs:,}")

    if n_targets < MIN_TARGETS or n_analogs < MIN_ANALOGS:
        log(f"  SKIP: insufficient data")
        continue

    # Fit walk-forward PCA
    wf_n_comp = min(N_COMPONENTS, n_train - 1, features.shape[1])
    wf_feat_rows = wf_meta['feat_row'].values

    log(f"  Fitting PCA: {wf_n_comp} components on {n_train} samples...")
    t0 = time.time()
    wf_pca = PCA(n_components=wf_n_comp, svd_solver='randomized', random_state=SEED)
    wf_pca.fit(features[wf_feat_rows])
    wf_var = float(wf_pca.explained_variance_ratio_.sum())
    log(f"  PCA fit: var={wf_var:.4f}, {time.time()-t0:.1f}s")

    # Transform in chunks (float32 output to save memory)
    t0 = time.time()
    wf_pca_all = np.empty((n_train, wf_n_comp), dtype=np.float32)
    for cs in range(0, n_train, TRANSFORM_CHUNK):
        ce = min(cs + TRANSFORM_CHUNK, n_train)
        wf_pca_all[cs:ce] = wf_pca.transform(
            features[wf_feat_rows[cs:ce]]
        ).astype(np.float32)

    del wf_pca
    gc.collect()
    log(f"  PCA transform: {wf_pca_all.shape}, {time.time()-t0:.1f}s | Mem: {mem_mb():.0f} MB")

    # Map feat_row -> position in wf_pca_all
    feat_row_to_local = {int(fr): i for i, fr in enumerate(wf_feat_rows)}

    # Process each SIC2 group
    target_sic2_groups = target_meta.groupby('sic2')
    n_groups = 0
    n_scored = 0

    for sic2, tgt_group in target_sic2_groups:
        sic2_analogs = analog_meta[analog_meta['sic2'] == sic2]
        if len(sic2_analogs) < MIN_ANALOGS or len(tgt_group) < MIN_TARGETS:
            continue

        # PCA embeddings
        tgt_local = np.array([feat_row_to_local[int(fr)]
                              for fr in tgt_group['feat_row'].values])
        ana_local = np.array([feat_row_to_local[int(fr)]
                              for fr in sic2_analogs['feat_row'].values])
        tgt_pca = wf_pca_all[tgt_local]
        ana_pca = wf_pca_all[ana_local]

        # Cross-year cosine similarity: (n_targets, n_analogs)
        cos_mat = cosine_similarity(tgt_pca, ana_pca)

        # Analog data arrays
        ana_cidxs = sic2_analogs['hf_index'].values.astype(int)
        ana_outcome_vals = np.array([analog_outcomes[int(c)] for c in ana_cidxs])
        ana_tickers = sic2_analogs['ticker'].values
        ana_years = sic2_analogs['year'].values

        # Target data arrays
        tgt_cidxs = tgt_group['hf_index'].values.astype(int)
        tgt_ciks = tgt_group['cik'].values.astype(int)
        tgt_tickers = tgt_group['ticker'].values

        # Save analog pool for placebo
        analog_pools_for_placebo[(fy, sic2)] = {
            'target_cidxs': tgt_cidxs.tolist(),
            'target_ciks': tgt_ciks.tolist(),
            'analog_outcomes': ana_outcome_vals.tolist(),
        }

        # Compute predictions for each K and weighting
        for K in K_VALUES:
            actual_k = min(K, len(ana_cidxs))
            for weighting in ['simweight', 'equalweight']:
                variant = f"analog_K{K}_{weighting}"
                group_preds = []

                for ti in range(len(tgt_group)):
                    sims = cos_mat[ti]
                    top_idx = np.argsort(sims)[::-1][:actual_k]
                    top_sims = sims[top_idx]
                    top_outcomes = ana_outcome_vals[top_idx]

                    if weighting == 'simweight':
                        w = top_sims.copy()
                        ws = w.sum()
                        if ws > 1e-10:
                            w /= ws
                        else:
                            w = np.ones(len(w)) / len(w)
                        pred = float(np.dot(w, top_outcomes))
                    else:
                        pred = float(np.mean(top_outcomes))

                    group_preds.append({
                        'variant': variant,
                        'formation_year': fy,
                        'company_idx': int(tgt_cidxs[ti]),
                        'cik': int(tgt_ciks[ti]),
                        'ticker': str(tgt_tickers[ti]),
                        'sic2': sic2,
                        'predicted_return': pred,
                        'top_sim': float(top_sims[0]) if len(top_sims) > 0 else 0.0,
                        'mean_sim': float(np.mean(top_sims)) if len(top_sims) > 0 else 0.0,
                    })

                    # Store analog details for primary variant
                    if K == PRIMARY_K and weighting == 'simweight':
                        for ki in range(actual_k):
                            ai = top_idx[ki]
                            primary_analog_matches.append({
                                'formation_year': fy,
                                'target_idx': int(tgt_cidxs[ti]),
                                'target_ticker': str(tgt_tickers[ti]),
                                'sic2': sic2,
                                'analog_idx': int(ana_cidxs[ai]),
                                'analog_ticker': str(ana_tickers[ai]),
                                'analog_year': int(ana_years[ai]),
                                'cosine_sim': float(sims[ai]),
                                'analog_outcome': float(ana_outcome_vals[ai]),
                                'predicted_return': pred,
                            })

                # Z-score within SIC2 group
                if len(group_preds) >= 2:
                    preds_arr = np.array([p['predicted_return']
                                          for p in group_preds])
                    mu = preds_arr.mean()
                    sigma = preds_arr.std(ddof=1)
                    if sigma > 1e-10:
                        z_arr = (preds_arr - mu) / sigma
                    else:
                        z_arr = np.zeros(len(preds_arr))
                    for i, p in enumerate(group_preds):
                        p['z_scored_prediction'] = float(z_arr[i])
                    all_predictions.extend(group_preds)

        n_groups += 1
        n_scored += len(tgt_group)

    del wf_pca_all, feat_row_to_local
    gc.collect()
    log(f"  Groups: {n_groups}, Targets scored: {n_scored}, "
        f"Time: {time.time()-t_fy:.1f}s")

log(f"\nTotal predictions: {len(all_predictions):,}")
predictions_df = pd.DataFrame(all_predictions)
del all_predictions
gc.collect()

# Free features matrix — no longer needed
del features
gc.collect()
log(f"  Freed features matrix | Memory: {mem_mb():.0f} MB")

# ═══════════════════════════════════════════════════════════
# STEP 6: Portfolio construction & monthly strategy returns
# ═══════════════════════════════════════════════════════════

log("")
log("=" * 60)
log("STEP 6: Portfolio construction & monthly strategy returns")
log("=" * 60)

all_strategy_rows = []
primary_quintile_map = {}   # (fy, company_idx) -> quintile

variants_list = [f"analog_K{K}_{w}"
                 for K in K_VALUES for w in ['simweight', 'equalweight']]
primary_name = f"analog_K{PRIMARY_K}_simweight"

for variant in variants_list:
    vdf = predictions_df[predictions_df['variant'] == variant]

    for fy in FORMATION_YEARS:
        fy_df = vdf[vdf['formation_year'] == fy].copy()
        if len(fy_df) < MIN_QUINTILE_SIZE:
            continue

        # Filter targets with valid trading-period returns
        trading_months = get_trading_months(fy)
        min_months_req = min_months_for_period(len(trading_months))
        valid_flags = []
        for row in fy_df.itertuples(index=False):
            cik = int(row.cik)
            n_valid = sum(1 for m in trading_months
                          if (cik, m) in return_lookup)
            valid_flags.append(n_valid >= min_months_req)
        fy_df = fy_df[valid_flags]

        if len(fy_df) < MIN_QUINTILE_SIZE:
            continue

        # Assign quintiles by z_scored_prediction (descending)
        fy_df = fy_df.sort_values('z_scored_prediction',
                                   ascending=False).reset_index(drop=True)
        n = len(fy_df)
        qs = n // 5

        q_map = {}  # company_idx -> (quintile, cik)
        for i in range(n):
            cidx = int(fy_df.iloc[i]['company_idx'])
            cik = int(fy_df.iloc[i]['cik'])
            if i < qs:
                q = 1
            elif i < 2 * qs:
                q = 2
            elif i < 3 * qs:
                q = 3
            elif i < 4 * qs:
                q = 4
            else:
                q = 5
            q_map[cidx] = (q, cik)
            if variant == primary_name:
                primary_quintile_map[(fy, cidx)] = q

        q1_ciks = [cik for cidx, (q, cik) in q_map.items() if q == 1]
        q5_ciks = [cik for cidx, (q, cik) in q_map.items() if q == 5]

        for m in trading_months:
            q1_r = [return_lookup[(c, m)]
                    for c in q1_ciks if (c, m) in return_lookup]
            q5_r = [return_lookup[(c, m)]
                    for c in q5_ciks if (c, m) in return_lookup]
            strat_ret = ((np.mean(q1_r) if q1_r else 0.0) -
                         (np.mean(q5_r) if q5_r else 0.0))
            all_strategy_rows.append({
                'date': m,
                'variant': variant,
                'gross_return': strat_ret,
                'n_long': len(q1_r),
                'n_short': len(q5_r),
                'n_total_scored': n,
            })

strategy_df = pd.DataFrame(all_strategy_rows)
log(f"Strategy return rows: {len(strategy_df):,}")
for v in variants_list:
    vr = strategy_df[strategy_df['variant'] == v]
    if len(vr) > 0:
        log(f"  {v}: {len(vr)} months, mean={vr['gross_return'].mean():.6f}")

# ═══════════════════════════════════════════════════════════
# Build analog details parquet
# ═══════════════════════════════════════════════════════════

log("\nBuilding analog details...")

# Pre-build z-score lookup
z_score_lookup = {}
primary_preds = predictions_df[predictions_df['variant'] == primary_name]
for row in primary_preds.itertuples(index=False):
    z_score_lookup[(int(row.formation_year), int(row.company_idx))] = \
        float(row.z_scored_prediction)

# Pre-build realized return cache
realized_cache = {}
for match in primary_analog_matches:
    key = (match['formation_year'], match['target_idx'])
    if key not in realized_cache:
        cik = hf_to_cik.get(match['target_idx'])
        if cik is not None:
            trd = get_trading_months(match['formation_year'])
            cum = 1.0
            for m in trd:
                r = return_lookup.get((int(cik), m))
                if r is not None:
                    cum *= (1.0 + r)
            realized_cache[key] = cum - 1.0
        else:
            realized_cache[key] = np.nan

for match in primary_analog_matches:
    fy = match['formation_year']
    tidx = match['target_idx']
    match['quintile'] = primary_quintile_map.get((fy, tidx), 0)
    match['z_scored_prediction'] = z_score_lookup.get((fy, tidx), np.nan)
    match['realized_return'] = realized_cache.get((fy, tidx), np.nan)

analog_details_df = pd.DataFrame(primary_analog_matches)
log(f"  Analog details: {len(analog_details_df):,} rows")

# ═══════════════════════════════════════════════════════════
# STEP 7: Strategy statistics
# ═══════════════════════════════════════════════════════════

log("")
log("=" * 60)
log("STEP 7: Strategy statistics")
log("=" * 60)

variant_stats = {}
for v in variants_list:
    vdf = strategy_df[strategy_df['variant'] == v].sort_values('date')
    rets = vdf['gross_return'].values
    if len(rets) == 0:
        continue
    mean_r = float(np.mean(rets))
    std_r = float(np.std(rets, ddof=1)) if len(rets) > 1 else 1e-10
    sharpe = (mean_r / std_r) * np.sqrt(12) if std_r > 1e-10 else 0.0
    cum = np.cumsum(rets)
    max_dd = float(np.max(np.maximum.accumulate(cum) - cum)) if len(cum) > 0 else 0.0

    variant_stats[v] = {
        'mean_monthly_return': round(mean_r, 6),
        'std_monthly_return': round(std_r, 6),
        'sharpe_ratio': round(sharpe, 4),
        'n_months': len(rets),
        'max_drawdown': round(max_dd, 6),
        'mean_n_long': round(float(vdf['n_long'].mean()), 1),
        'mean_n_short': round(float(vdf['n_short'].mean()), 1),
        'mean_total_scored': round(float(vdf['n_total_scored'].mean()), 1),
    }
    log(f"  {v}: Sharpe={sharpe:.4f}, mean={mean_r:.6f}, months={len(rets)}")

# ═══════════════════════════════════════════════════════════
# STEP 8: Prediction-realization correlation
# ═══════════════════════════════════════════════════════════

log("")
log("=" * 60)
log("STEP 8: Prediction-realization correlation")
log("=" * 60)

pv = predictions_df[predictions_df['variant'] == primary_name].copy()

# Compute realized return per target
realized_returns = {}
for row in pv.itertuples(index=False):
    cidx = int(row.company_idx)
    if cidx in realized_returns:
        continue
    cik = int(row.cik)
    fy = int(row.formation_year)
    trd = get_trading_months(fy)
    cum = 1.0
    n_valid = 0
    for m in trd:
        r = return_lookup.get((cik, m))
        if r is not None:
            cum *= (1.0 + r)
            n_valid += 1
    if n_valid >= min_months_for_period(len(trd)):
        realized_returns[cidx] = cum - 1.0

pv['realized_return'] = pv['company_idx'].map(realized_returns)
pv_valid = pv.dropna(subset=['realized_return'])

year_corrs = []
for fy in FORMATION_YEARS:
    fd = pv_valid[pv_valid['formation_year'] == fy]
    if len(fd) < 10:
        continue
    rho, p = spearmanr(fd['z_scored_prediction'].values,
                       fd['realized_return'].values)
    year_corrs.append({
        'year': fy,
        'spearman_rho': round(float(rho), 6),
        'p_value': round(float(p), 6),
        'n': len(fd),
    })
    log(f"  FY{fy}: rho={rho:.4f}, p={p:.4f}, n={len(fd)}")

mean_rho = np.mean([c['spearman_rho'] for c in year_corrs]) if year_corrs else 0.0
log(f"\n  Mean Spearman rho: {mean_rho:.4f}")

# ═══════════════════════════════════════════════════════════
# STEP 9: Quintile return analysis (primary variant)
# ═══════════════════════════════════════════════════════════

log("")
log("=" * 60)
log("STEP 9: Quintile return analysis (primary variant)")
log("=" * 60)

quintile_cum_returns = {q: [] for q in range(1, 6)}
quintile_sizes_per_year = {}

for fy in FORMATION_YEARS:
    fd = pv_valid[pv_valid['formation_year'] == fy].copy()
    if len(fd) < MIN_QUINTILE_SIZE:
        continue
    fd = fd.sort_values('z_scored_prediction',
                        ascending=False).reset_index(drop=True)
    n = len(fd)
    qs_size = n // 5
    year_sizes = {q: 0 for q in range(1, 6)}

    for i in range(n):
        if i < qs_size:
            q = 1
        elif i < 2 * qs_size:
            q = 2
        elif i < 3 * qs_size:
            q = 3
        elif i < 4 * qs_size:
            q = 4
        else:
            q = 5
        quintile_cum_returns[q].append(fd.iloc[i]['realized_return'])
        year_sizes[q] += 1

    quintile_sizes_per_year[fy] = year_sizes

log(f"\n  {'Quintile':>10}  {'Mean':>10}  {'Median':>10}  {'N':>6}")
log(f"  {'-'*40}")
quintile_means = {}
for q in range(1, 6):
    rets = quintile_cum_returns[q]
    if rets:
        quintile_means[q] = float(np.mean(rets))
        log(f"  Q{q:>9}  {np.mean(rets):>10.4f}  {np.median(rets):>10.4f}"
            f"  {len(rets):>6}")
    else:
        quintile_means[q] = 0.0
        log(f"  Q{q:>9}  {'N/A':>10}  {'N/A':>10}  {0:>6}")

q1q5 = quintile_means.get(1, 0) - quintile_means.get(5, 0)
monotone = all(quintile_means.get(q, 0) >= quintile_means.get(q + 1, 0)
               for q in range(1, 5))
log(f"\n  Q1-Q5 spread: {q1q5:.4f}")
log(f"  Monotone: {'YES' if monotone else 'NO'}")

# Print quintile sizes per year
log(f"\n  Quintile sizes per formation year:")
log(f"  {'FY':>6}  {'Q1':>5}  {'Q2':>5}  {'Q3':>5}  {'Q4':>5}  {'Q5':>5}")
for fy in sorted(quintile_sizes_per_year.keys()):
    ys = quintile_sizes_per_year[fy]
    log(f"  {fy:>6}  {ys[1]:>5}  {ys[2]:>5}  {ys[3]:>5}  {ys[4]:>5}  {ys[5]:>5}")

# ═══════════════════════════════════════════════════════════
# STEP 10: Placebo test (100 trials, random analog selection)
# ═══════════════════════════════════════════════════════════

log("")
log("=" * 60)
log("STEP 10: Placebo test (100 trials, random analog selection)")
log("=" * 60)

t0 = time.time()
rng = np.random.default_rng(SEED)

# Pre-group pools by FY for efficiency
pools_by_fy = {}
for (pool_fy, pool_sic2), pool_data in analog_pools_for_placebo.items():
    if pool_fy not in pools_by_fy:
        pools_by_fy[pool_fy] = []
    pools_by_fy[pool_fy].append((pool_sic2, pool_data))

placebo_sharpes = []

for trial in range(N_PLACEBO_TRIALS):
    trial_monthly = []

    for fy in FORMATION_YEARS:
        trading_months = get_trading_months(fy)
        min_months_req = min_months_for_period(len(trading_months))
        all_fy_targets = []  # (cidx, cik, z_pred)

        for sic2, pool_data in pools_by_fy.get(fy, []):
            target_cidxs = pool_data['target_cidxs']
            target_ciks = pool_data['target_ciks']
            ana_outcomes_pool = np.array(pool_data['analog_outcomes'])

            if (len(ana_outcomes_pool) < MIN_ANALOGS or
                    len(target_cidxs) < MIN_TARGETS):
                continue

            # Random analog selection for each target
            actual_k = min(PRIMARY_K, len(ana_outcomes_pool))
            preds = []
            for _ in target_cidxs:
                sampled = rng.choice(len(ana_outcomes_pool),
                                     size=actual_k, replace=False)
                preds.append(float(np.mean(ana_outcomes_pool[sampled])))

            # Z-score within SIC2
            preds_arr = np.array(preds)
            if len(preds_arr) >= 2:
                sigma = preds_arr.std(ddof=1)
                if sigma > 1e-10:
                    z = (preds_arr - preds_arr.mean()) / sigma
                else:
                    z = np.zeros(len(preds_arr))
            else:
                z = np.zeros(len(preds_arr))

            for i, cidx in enumerate(target_cidxs):
                cik = target_ciks[i]
                n_valid = sum(1 for m in trading_months
                              if (cik, m) in return_lookup)
                if n_valid >= min_months_req:
                    all_fy_targets.append((cidx, cik, float(z[i])))

        if len(all_fy_targets) < MIN_QUINTILE_SIZE:
            continue

        # Form quintiles (same logic as real strategy)
        all_fy_targets.sort(key=lambda x: x[2], reverse=True)
        n = len(all_fy_targets)
        qs = n // 5

        q1_ciks = [t[1] for i, t in enumerate(all_fy_targets) if i < qs]
        q5_ciks = [t[1] for i, t in enumerate(all_fy_targets)
                   if i >= 4 * qs]

        for m in trading_months:
            q1_r = [return_lookup[(c, m)]
                    for c in q1_ciks if (c, m) in return_lookup]
            q5_r = [return_lookup[(c, m)]
                    for c in q5_ciks if (c, m) in return_lookup]
            ret = ((np.mean(q1_r) if q1_r else 0.0) -
                   (np.mean(q5_r) if q5_r else 0.0))
            trial_monthly.append(ret)

    if trial_monthly:
        arr = np.array(trial_monthly)
        m_val = float(np.mean(arr))
        s_val = float(np.std(arr, ddof=1))
        placebo_sharpes.append(
            (m_val / s_val) * np.sqrt(12) if s_val > 1e-10 else 0.0)
    else:
        placebo_sharpes.append(0.0)

    if (trial + 1) % 25 == 0:
        log(f"  Trial {trial+1}/{N_PLACEBO_TRIALS} ({time.time()-t0:.1f}s)")

real_sharpe = variant_stats.get(primary_name, {}).get('sharpe_ratio', 0.0)
placebo_arr = np.array(placebo_sharpes)
placebo_mean = float(np.mean(placebo_arr))
placebo_std = float(np.std(placebo_arr))
placebo_p95 = float(np.percentile(placebo_arr, 95))
pctile_rank = float(np.sum(placebo_arr < real_sharpe) /
                     len(placebo_arr) * 100)

log(f"\n  Real Sharpe:       {real_sharpe:.4f}")
log(f"  Placebo mean:      {placebo_mean:.4f} +/- {placebo_std:.4f}")
log(f"  Placebo 95th pctl: {placebo_p95:.4f}")
log(f"  Real percentile:   {pctile_rank:.1f}%")
log(f"  PASS: {'YES' if real_sharpe > placebo_p95 else 'NO'}")

placebo_results = {
    'n_trials': N_PLACEBO_TRIALS,
    'primary_variant': primary_name,
    'real_sharpe': round(real_sharpe, 6),
    'placebo_sharpes': [round(s, 6) for s in placebo_sharpes],
    'placebo_mean_sharpe': round(placebo_mean, 6),
    'placebo_std_sharpe': round(placebo_std, 6),
    'placebo_p5': round(float(np.percentile(placebo_arr, 5)), 6),
    'placebo_p95': round(placebo_p95, 6),
    'percentile_rank': round(pctile_rank, 2),
    'passes_95th': bool(real_sharpe > placebo_p95),
}

# ═══════════════════════════════════════════════════════════
# STEP 11: Factor regression (FF5 + MOM, Newey-West)
# ═══════════════════════════════════════════════════════════

log("")
log("=" * 60)
log("STEP 11: Factor regression (FF5 + MOM, Newey-West)")
log("=" * 60)

regression_results = None

primary_strat = strategy_df[strategy_df['variant'] == primary_name] \
    .sort_values('date')
strat_dates = primary_strat['date'].values
strat_rets = primary_strat['gross_return'].values

factor_rows = []
rf_vals = []
valid_idx = []
for i, d in enumerate(strat_dates):
    fd = factor_full.get(int(d))
    if fd is not None:
        factor_rows.append([fd['Mkt-RF'], fd['SMB'], fd['HML'],
                            fd['RMW'], fd['CMA'], fd['MOM']])
        rf_vals.append(fd['RF'])
        valid_idx.append(i)

if len(valid_idx) >= 12:
    factor_mat = np.array(factor_rows)
    rf_arr = np.array(rf_vals)
    y = strat_rets[valid_idx] - rf_arr

    X = add_constant(factor_mat)
    model = OLS(y, X).fit(cov_type='HAC', cov_kwds={'maxlags': 6})

    fnames = ['const', 'Mkt-RF', 'SMB', 'HML', 'RMW', 'CMA', 'MOM']
    log(f"\n  {'Factor':>10}  {'Coef':>10}  {'t-stat':>8}  {'p-value':>8}")
    log(f"  {'-'*40}")
    for i, fn in enumerate(fnames):
        log(f"  {fn:>10}  {model.params[i]:>10.6f}  "
            f"{model.tvalues[i]:>8.3f}  {model.pvalues[i]:>8.4f}")

    alpha = float(model.params[0])
    t_alpha = float(model.tvalues[0])
    r2 = float(model.rsquared)

    log(f"\n  Alpha (annualized): {alpha * 12:.4f}")
    log(f"  t(alpha): {t_alpha:.3f}")
    log(f"  R-squared: {r2:.4f}")

    regression_results = {
        'alpha_monthly': round(alpha, 8),
        'alpha_annualized': round(alpha * 12, 6),
        't_alpha': round(t_alpha, 4),
        'r_squared': round(r2, 6),
        'n_obs': len(valid_idx),
        'betas': {fn: round(float(model.params[i]), 6)
                  for i, fn in enumerate(fnames)},
        't_stats': {fn: round(float(model.tvalues[i]), 4)
                    for i, fn in enumerate(fnames)},
    }
else:
    log("  ERROR: Not enough factor data for regression")

# ═══════════════════════════════════════════════════════════
# STEP 12: Sub-period analysis
# ═══════════════════════════════════════════════════════════

log("")
log("=" * 60)
log("STEP 12: Sub-period analysis")
log("=" * 60)


def compute_subperiod_sharpe(dates_arr, rets_arr, start_ym, end_ym):
    mask = (dates_arr >= start_ym) & (dates_arr <= end_ym)
    r = rets_arr[mask]
    if len(r) < 3:
        return None, len(r)
    m_val = np.mean(r)
    s_val = np.std(r, ddof=1)
    return (round((m_val / s_val) * np.sqrt(12), 4)
            if s_val > 1e-10 else 0.0), len(r)


primary_dates = primary_strat['date'].values
primary_rets = primary_strat['gross_return'].values

subperiods = [
    ('Early (Jul 2000 - Jun 2010)', 200007, 201006),
    ('Late (Jul 2010 - Dec 2020)', 201007, 202012),
    ('Tech crash (Jul 2001 - Jun 2003)', 200107, 200306),
    ('GFC (Jul 2008 - Jun 2010)', 200807, 201006),
    ('COVID (Jan 2020 - Dec 2020)', 202001, 202012),
]

subperiod_results = {}
log(f"\n  {'Period':>35}  {'Sharpe':>8}  {'Months':>8}")
log(f"  {'-'*55}")
for name, start, end in subperiods:
    sharpe, n = compute_subperiod_sharpe(primary_dates, primary_rets,
                                          start, end)
    subperiod_results[name] = {'sharpe': sharpe, 'n_months': n}
    log(f"  {name:>35}  {str(sharpe):>8}  {n:>8}")

# ═══════════════════════════════════════════════════════════
# STEP 13: Analog similarity distribution
# ═══════════════════════════════════════════════════════════

log("")
log("=" * 60)
log("STEP 13: Analog similarity distribution")
log("=" * 60)

sim_stats = None
if len(analog_details_df) > 0:
    sims = analog_details_df['cosine_sim'].values
    sim_stats = {
        'mean': round(float(np.mean(sims)), 4),
        'median': round(float(np.median(sims)), 4),
    }
    log(f"  Mean:   {np.mean(sims):.4f}")
    log(f"  Median: {np.median(sims):.4f}")
    log(f"  Range:  [{np.min(sims):.4f}, {np.max(sims):.4f}]")
    log(f"  > 0.3:  {(sims > 0.3).sum()} / {len(sims)} "
        f"({(sims > 0.3).mean()*100:.1f}%)")
    log(f"  > 0.5:  {(sims > 0.5).sum()} / {len(sims)} "
        f"({(sims > 0.5).mean()*100:.1f}%)")

    log(f"\n  Top 10 highest-similarity matches:")
    top10 = analog_details_df.nlargest(10, 'cosine_sim')
    for _, r in top10.iterrows():
        log(f"    {r['target_ticker']} FY{r['formation_year']} <- "
            f"{r['analog_ticker']} FY{r['analog_year']}: "
            f"sim={r['cosine_sim']:.4f}, outcome={r['analog_outcome']:+.4f}")

# ═══════════════════════════════════════════════════════════
# STEP 14: Save all outputs
# ═══════════════════════════════════════════════════════════

log("")
log("=" * 60)
log("STEP 14: Save outputs")
log("=" * 60)

strategy_df.to_parquet(STRATEGY_RETURNS_OUT, index=False)
log(f"  {STRATEGY_RETURNS_OUT}: {len(strategy_df):,} rows")

analog_details_df.to_parquet(ANALOG_DETAILS_OUT, index=False)
log(f"  {ANALOG_DETAILS_OUT}: {len(analog_details_df):,} rows")

with open(PLACEBO_OUT, 'w') as f:
    json.dump(placebo_results, f, indent=2)
log(f"  {PLACEBO_OUT}")

summary = {
    'script': '2b_analog_prediction',
    'config': {
        'formation_years': FORMATION_YEARS,
        'k_values': K_VALUES,
        'primary_variant': primary_name,
        'min_analogs': MIN_ANALOGS,
        'min_outcome_months': MIN_OUTCOME_MONTHS,
        'n_features_keep': N_FEATURES_KEEP,
        'n_components': N_COMPONENTS,
        'n_placebo_trials': N_PLACEBO_TRIALS,
        'seed': SEED,
    },
    'variant_stats': variant_stats,
    'prediction_realization': {
        'mean_spearman_rho': round(mean_rho, 6),
        'per_year': year_corrs,
    },
    'quintile_analysis': {
        'quintile_means': {str(q): round(v, 6)
                           for q, v in quintile_means.items()},
        'q1_q5_spread': round(q1q5, 6),
        'monotone': monotone,
    },
    'factor_regression': regression_results,
    'subperiod_analysis': subperiod_results,
    'placebo': {k: v for k, v in placebo_results.items()
                if k != 'placebo_sharpes'},
    'analog_similarity': sim_stats,
    'total_runtime_seconds': round(time.time() - t_start, 1),
}

with open(SUMMARY_OUT, 'w') as f:
    json.dump(summary, f, indent=2, default=str)
log(f"  {SUMMARY_OUT}")

# ═══════════════════════════════════════════════════════════
# FINAL SUMMARY
# ═══════════════════════════════════════════════════════════

log("")
log("=" * 60)
log("FINAL SUMMARY")
log("=" * 60)

log(f"\n{'Variant':>30}  {'Sharpe':>8}  {'Mean Ret':>10}  {'Months':>8}")
log(f"{'-'*60}")
for v in variants_list:
    vs = variant_stats.get(v, {})
    log(f"{v:>30}  {vs.get('sharpe_ratio', 0):>8.4f}  "
        f"{vs.get('mean_monthly_return', 0):>10.6f}  "
        f"{vs.get('n_months', 0):>8}")

if regression_results:
    log(f"\nFactor Regression (primary variant):")
    log(f"  Alpha (ann): {regression_results['alpha_annualized']:.4f}")
    log(f"  t(alpha):    {regression_results['t_alpha']:.3f}")
    log(f"  R-squared:   {regression_results['r_squared']:.4f}")

log(f"\nPrediction-Realization: mean rho = {mean_rho:.4f}")
log(f"Quintile Spread (Q1-Q5): {q1q5:.4f}")
log(f"\nPlacebo: real={real_sharpe:.4f} vs p95={placebo_p95:.4f} "
    f"-> {'PASS' if placebo_results['passes_95th'] else 'FAIL'}")

log(f"\nSub-periods:")
for name, res in subperiod_results.items():
    log(f"  {name}: Sharpe={res['sharpe']}")

log(f"\nTotal runtime: {time.time()-t_start:.1f}s "
    f"({(time.time()-t_start)/60:.1f} min)")
log("Done.")

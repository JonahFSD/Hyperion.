#!/usr/bin/env python3
"""
Hyperion Phase 1 Validation Suite
Experiments 1A, 1B, 1C — Replication on ACL paper data

Run with: python3 run_phase1.py
Output: phase1_results.json + console summary
"""

import json
import os
import pickle
import time
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from scipy.spatial.distance import cosine
from sklearn.decomposition import PCA

warnings.filterwarnings('ignore')

RESULTS = {}
START_TIME = time.time()

def log(msg):
    elapsed = time.time() - START_TIME
    print(f"[{elapsed:7.1f}s] {msg}")

# ═══════════════════════════════════════════════════════════
# STEP 1: DOWNLOAD AND LOAD DATA
# ═══════════════════════════════════════════════════════════

log("=" * 60)
log("PHASE 1 VALIDATION SUITE — STARTING")
log("=" * 60)

# --- Load SAE features from HuggingFace ---
log("Downloading SAE features from HuggingFace (this may take a while)...")
from datasets import load_dataset

try:
    features_ds = load_dataset(
        'marco-molinari/company_reports_with_features',
        split='train'
    )
    log(f"SAE features loaded: {len(features_ds)} rows")
except Exception as e:
    log(f"ERROR loading SAE features: {e}")
    log("Retrying with streaming download...")
    features_ds = load_dataset(
        'marco-molinari/company_reports_with_features',
        split='train',
        download_mode='force_redownload'
    )
    log(f"SAE features loaded: {len(features_ds)} rows")

# --- Load company data ---
log("Downloading company data from HuggingFace...")
try:
    company_ds = load_dataset(
        'Mateusz1017/annual_reports_tokenized_llama3_logged_returns_no_null_returns_and_incomplete_descriptions_24k',
        split='train'
    )
    log(f"Company data loaded: {len(company_ds)} rows")
except Exception as e:
    log(f"ERROR loading company data: {e}")
    log("Retrying...")
    company_ds = load_dataset(
        'Mateusz1017/annual_reports_tokenized_llama3_logged_returns_no_null_returns_and_incomplete_descriptions_24k',
        split='train',
        download_mode='force_redownload'
    )
    log(f"Company data loaded: {len(company_ds)} rows")

# --- Extract into arrays ---
log("Extracting feature vectors...")
feature_vectors = np.array(features_ds['features'])
log(f"Feature matrix shape: {feature_vectors.shape}")
RESULTS['n_documents'] = feature_vectors.shape[0]
RESULTS['n_raw_features'] = feature_vectors.shape[1] if len(feature_vectors.shape) > 1 else 'nested'

# Handle nested list structure
if len(feature_vectors.shape) == 1:
    log("Features are nested lists, converting...")
    feature_vectors = np.array([np.array(f) for f in features_ds['features']])
    log(f"Feature matrix shape after conversion: {feature_vectors.shape}")
    RESULTS['n_raw_features'] = feature_vectors.shape[1]

# --- Extract company metadata ---
log("Extracting company metadata...")
company_df = pd.DataFrame({
    'cik': company_ds['cik'],
    'year': company_ds['year'],
    'company_name': company_ds['company_name'],
    'sic_code': company_ds['sic_code'],
    'ticker': company_ds['ticker'],
})

# Extract return matrices
log("Extracting return matrices...")
returns_matrices = company_ds['logged_monthly_returns_matrix']

log(f"Companies: {len(company_df)}")
log(f"Unique tickers: {company_df['ticker'].nunique()}")
log(f"Year range: {company_df['year'].min()} - {company_df['year'].max()}")

RESULTS['n_companies_unique'] = int(company_df['ticker'].nunique())
RESULTS['year_range'] = [int(company_df['year'].min()), int(company_df['year'].max())]

# ═══════════════════════════════════════════════════════════
# STEP 2: FEATURE FILTERING
# ═══════════════════════════════════════════════════════════

log("")
log("=" * 60)
log("STEP 2: FEATURE FILTERING")
log("=" * 60)

# Remove dead features (activate on < 0.1% of documents)
activation_rate = (feature_vectors > 0).mean(axis=0)
alive_mask = activation_rate >= 0.001
log(f"Dead features removed (<0.1% activation): {(~alive_mask).sum()}")

# Remove overly dense features (activation density > 0.01)
density_mask = activation_rate <= 0.01
log(f"Overly dense features removed (>1% density): {(alive_mask & ~density_mask).sum()}")

combined_mask = alive_mask & density_mask
filtered_features = feature_vectors[:, combined_mask]
log(f"Features after basic filtering: {filtered_features.shape[1]}")
RESULTS['n_filtered_features'] = int(filtered_features.shape[1])

# ═══════════════════════════════════════════════════════════
# STEP 3: PCA COMPRESSION
# ═══════════════════════════════════════════════════════════

log("")
log("=" * 60)
log("STEP 3: PCA COMPRESSION")
log("=" * 60)

n_components = min(4000, filtered_features.shape[0] - 1, filtered_features.shape[1])
log(f"Fitting PCA with {n_components} components on {filtered_features.shape[0]} samples x {filtered_features.shape[1]} features...")

pca = PCA(n_components=n_components)
pca_features = pca.fit_transform(filtered_features)
variance_explained = pca.explained_variance_ratio_.sum()

log(f"PCA variance preserved: {variance_explained:.4f} ({variance_explained*100:.2f}%)")
log(f"PCA output shape: {pca_features.shape}")
RESULTS['pca_components'] = n_components
RESULTS['pca_variance_preserved'] = float(variance_explained)

# ═══════════════════════════════════════════════════════════
# STEP 4: CHRONOLOGICAL SPLIT
# ═══════════════════════════════════════════════════════════

log("")
log("=" * 60)
log("STEP 4: CHRONOLOGICAL TRAIN/TEST SPLIT")
log("=" * 60)

years = company_df['year'].values
sorted_years = np.sort(np.unique(years))
split_idx = int(len(sorted_years) * 0.6)
split_year = sorted_years[split_idx]

train_mask = years <= split_year
test_mask = years > split_year

log(f"Split year: {split_year}")
log(f"Training set: {train_mask.sum()} documents (years <= {split_year})")
log(f"Test set: {test_mask.sum()} documents (years > {split_year})")
RESULTS['split_year'] = int(split_year)
RESULTS['n_train'] = int(train_mask.sum())
RESULTS['n_test'] = int(test_mask.sum())

# ═══════════════════════════════════════════════════════════
# STEP 5: BUILD RETURN CORRELATION MATRIX
# ═══════════════════════════════════════════════════════════

log("")
log("=" * 60)
log("STEP 5: BUILDING RETURN CORRELATION MATRIX")
log("=" * 60)

# Build return series per company-year from the logged monthly returns
log("Processing logged monthly return matrices...")
n_docs = len(returns_matrices)
return_series = {}
for i in range(n_docs):
    ret = returns_matrices[i]
    if ret is not None and len(ret) > 0:
        # Each row is a year of monthly returns (12 values)
        flat = np.array(ret).flatten()
        # Remove NaN/inf
        flat = flat[np.isfinite(flat)]
        if len(flat) >= 12:
            ticker = company_df.iloc[i]['ticker']
            year = company_df.iloc[i]['year']
            if ticker not in return_series:
                return_series[ticker] = {}
            return_series[ticker][year] = flat

log(f"Tickers with return data: {len(return_series)}")

# Build correlation matrix for companies with sufficient data
# Use all available returns per ticker (concatenate across years)
ticker_returns = {}
for ticker, yearly in return_series.items():
    all_returns = []
    for y in sorted(yearly.keys()):
        all_returns.extend(yearly[y].tolist())
    if len(all_returns) >= 36:  # Minimum 36 months
        ticker_returns[ticker] = np.array(all_returns)

log(f"Tickers with >= 36 months of returns: {len(ticker_returns)}")
RESULTS['n_tickers_with_returns'] = len(ticker_returns)

# Compute pairwise correlations
tickers_list = sorted(ticker_returns.keys())
n_tickers = len(tickers_list)
log(f"Computing {n_tickers * (n_tickers - 1) // 2} pairwise correlations...")

# Align returns by truncating to common length per pair
corr_matrix = np.full((n_tickers, n_tickers), np.nan)
for i in range(n_tickers):
    for j in range(i, n_tickers):
        if i == j:
            corr_matrix[i, j] = 1.0
            continue
        r1 = ticker_returns[tickers_list[i]]
        r2 = ticker_returns[tickers_list[j]]
        min_len = min(len(r1), len(r2))
        if min_len >= 12:
            corr_val = np.corrcoef(r1[:min_len], r2[:min_len])[0, 1]
            if np.isfinite(corr_val):
                corr_matrix[i, j] = corr_val
                corr_matrix[j, i] = corr_val

# Population baseline
valid_corrs = corr_matrix[np.triu_indices(n_tickers, k=1)]
valid_corrs = valid_corrs[np.isfinite(valid_corrs)]
population_baseline = np.mean(valid_corrs)
log(f"Population baseline MC: {population_baseline:.4f}")
RESULTS['population_baseline_mc'] = float(population_baseline)

# ═══════════════════════════════════════════════════════════
# STEP 6: CLUSTERING (MST + THRESHOLD)
# ═══════════════════════════════════════════════════════════

log("")
log("=" * 60)
log("STEP 6: MST CLUSTERING")
log("=" * 60)

from scipy.spatial.distance import pdist, squareform
from scipy.sparse.csgraph import minimum_spanning_tree

# Build document-level similarity for clustering
# Average PCA vectors per ticker for the most recent filing
log("Building per-ticker feature vectors (latest filing per ticker)...")
ticker_to_idx = {t: i for i, t in enumerate(tickers_list)}
ticker_features = {}

for i, row in company_df.iterrows():
    ticker = row['ticker']
    if ticker in ticker_to_idx and i < len(pca_features):
        year = row['year']
        if ticker not in ticker_features or year > ticker_features[ticker][1]:
            ticker_features[ticker] = (pca_features[i], year)

# Build feature matrix for tickers with both features and returns
valid_tickers = [t for t in tickers_list if t in ticker_features]
feature_matrix = np.array([ticker_features[t][0] for t in valid_tickers])
log(f"Valid tickers (features + returns): {len(valid_tickers)}")

# Compute cosine distance matrix
log("Computing cosine distance matrix...")
dist_vector = pdist(feature_matrix, metric='cosine')
dist_matrix = squareform(dist_vector)

# Build MST
log("Building minimum spanning tree...")
mst = minimum_spanning_tree(dist_matrix)
mst_array = mst.toarray()
# Make symmetric
mst_symmetric = mst_array + mst_array.T

# Get all MST edge weights for threshold sweep
edges = []
rows, cols = mst_symmetric.nonzero()
for r, c in zip(rows, cols):
    if r < c:
        edges.append(mst_symmetric[r, c])
edges = np.array(edges)
log(f"MST edges: {len(edges)}, range: [{edges.min():.4f}, {edges.max():.4f}]")

# ═══════════════════════════════════════════════════════════
# STEP 7: THETA OPTIMIZATION + MC COMPUTATION
# ═══════════════════════════════════════════════════════════

log("")
log("=" * 60)
log("STEP 7: THETA OPTIMIZATION")
log("=" * 60)

def get_clusters(dist_mat_sym, threshold):
    """Cut MST at threshold, return connected components."""
    from scipy.sparse import csr_matrix
    from scipy.sparse.csgraph import connected_components
    adj = (dist_mat_sym > 0) & (dist_mat_sym <= threshold)
    adj_sparse = csr_matrix(adj.astype(float))
    n_components, labels = connected_components(adj_sparse, directed=False)
    return labels, n_components

def compute_mc(cluster_labels, ticker_list, corr_mat, ticker_to_idx_map):
    """Compute mean intra-cluster correlation."""
    unique_labels = np.unique(cluster_labels)
    all_corrs = []
    for label in unique_labels:
        members = [ticker_list[i] for i in range(len(ticker_list)) if cluster_labels[i] == label]
        if len(members) < 2:
            continue
        for ii in range(len(members)):
            for jj in range(ii + 1, len(members)):
                t1_idx = ticker_to_idx_map.get(members[ii])
                t2_idx = ticker_to_idx_map.get(members[jj])
                if t1_idx is not None and t2_idx is not None:
                    c = corr_mat[t1_idx, t2_idx]
                    if np.isfinite(c):
                        all_corrs.append(c)
    if len(all_corrs) == 0:
        return np.nan, 0
    return np.mean(all_corrs), len(all_corrs)

# Sweep theta
thresholds = np.linspace(np.percentile(edges, 10), np.percentile(edges, 90), 50)
mc_by_theta = []

valid_ticker_idx = {t: i for i, t in enumerate(tickers_list)}

log(f"Sweeping {len(thresholds)} threshold values...")
for theta in thresholds:
    labels, n_clusters = get_clusters(mst_symmetric, theta)
    mc_val, n_pairs = compute_mc(labels, valid_tickers, corr_matrix, valid_ticker_idx)
    mc_by_theta.append({
        'theta': float(theta),
        'mc': float(mc_val) if not np.isnan(mc_val) else None,
        'n_clusters': int(n_clusters),
        'n_pairs': int(n_pairs)
    })

# Find optimal theta
valid_mc = [(m['theta'], m['mc']) for m in mc_by_theta if m['mc'] is not None]
if valid_mc:
    best_theta, best_mc = max(valid_mc, key=lambda x: x[1])
    log(f"Optimal theta: {best_theta:.4f}")
    log(f"Best MC: {best_mc:.4f}")
else:
    log("WARNING: No valid MC values found. Check data.")
    best_theta = np.median(edges)
    best_mc = np.nan

RESULTS['optimal_theta'] = float(best_theta)
RESULTS['best_mc_training'] = float(best_mc) if not np.isnan(best_mc) else None
RESULTS['theta_sweep'] = mc_by_theta

# Get clusters at optimal theta
best_labels, best_n_clusters = get_clusters(mst_symmetric, best_theta)
log(f"Clusters at optimal theta: {best_n_clusters}")
RESULTS['n_clusters'] = int(best_n_clusters)

# Cluster size distribution
cluster_sizes = pd.Series(best_labels).value_counts()
log(f"Cluster sizes: min={cluster_sizes.min()}, max={cluster_sizes.max()}, median={cluster_sizes.median():.0f}")

# ═══════════════════════════════════════════════════════════
# EXPERIMENT 1A: MC WITH BOOTSTRAP CIs
# ═══════════════════════════════════════════════════════════

log("")
log("=" * 60)
log("EXPERIMENT 1A: MEAN INTRA-CLUSTER CORRELATION")
log("=" * 60)

# SAE MC at optimal theta (already computed)
sae_mc = best_mc
log(f"SAE MC: {sae_mc:.4f}")

# --- GICS/SIC Baseline ---
# Map SIC codes to broad sectors (first 2 digits)
log("Computing SIC baseline...")
sic_codes = {}
for _, row in company_df.iterrows():
    if row['ticker'] in valid_ticker_idx:
        sic = str(row['sic_code'])[:2] if pd.notna(row['sic_code']) else '00'
        sic_codes[row['ticker']] = sic

sic_labels = np.array([sic_codes.get(t, '00') for t in valid_tickers])
# Convert to numeric labels
unique_sics = np.unique(sic_labels)
sic_label_map = {s: i for i, s in enumerate(unique_sics)}
sic_numeric = np.array([sic_label_map[s] for s in sic_labels])

sic_mc, sic_n_pairs = compute_mc(sic_numeric, valid_tickers, corr_matrix, valid_ticker_idx)
log(f"SIC MC: {sic_mc:.4f} ({sic_n_pairs} pairs)")
RESULTS['sic_mc'] = float(sic_mc)

# --- Random Baseline ---
log("Computing random baseline (1000 shuffles)...")
random_mcs = []
for _ in range(1000):
    shuffled = np.random.permutation(best_labels)
    rmc, _ = compute_mc(shuffled, valid_tickers, corr_matrix, valid_ticker_idx)
    if not np.isnan(rmc):
        random_mcs.append(rmc)

random_mc_mean = np.mean(random_mcs)
random_mc_std = np.std(random_mcs)
log(f"Random MC: {random_mc_mean:.4f} +/- {random_mc_std:.4f}")
RESULTS['random_mc_mean'] = float(random_mc_mean)
RESULTS['random_mc_std'] = float(random_mc_std)

# --- Bootstrap CIs ---
log("Bootstrapping SAE MC (10000 resamples)...")
bootstrap_mcs = []
n_valid = len(valid_tickers)
for b in range(10000):
    if b % 2000 == 0:
        log(f"  Bootstrap {b}/10000...")
    # Resample company indices
    idx = np.random.choice(n_valid, size=n_valid, replace=True)
    resampled_tickers = [valid_tickers[i] for i in idx]
    resampled_labels = best_labels[idx]
    # Compute MC on resampled
    bmc, _ = compute_mc(resampled_labels, resampled_tickers, corr_matrix, valid_ticker_idx)
    if not np.isnan(bmc):
        bootstrap_mcs.append(bmc)

bootstrap_mcs = np.array(bootstrap_mcs)
ci_lower = np.percentile(bootstrap_mcs, 2.5)
ci_upper = np.percentile(bootstrap_mcs, 97.5)
log(f"SAE MC: {sae_mc:.4f} [{ci_lower:.4f}, {ci_upper:.4f}] (95% CI)")

RESULTS['experiment_1a'] = {
    'sae_mc': float(sae_mc),
    'sae_mc_ci_lower': float(ci_lower),
    'sae_mc_ci_upper': float(ci_upper),
    'sic_mc': float(sic_mc),
    'random_mc_mean': float(random_mc_mean),
    'random_mc_std': float(random_mc_std),
    'population_baseline': float(population_baseline),
    'n_bootstrap': 10000,
    'signal_above_baseline': float(sae_mc - population_baseline),
    'signal_above_sic': float(sae_mc - sic_mc),
    'ci_excludes_baseline': bool(ci_lower > population_baseline),
    'ci_excludes_sic': bool(ci_lower > sic_mc),
}

# --- Theta sensitivity ---
valid_thetas = [m for m in mc_by_theta if m['mc'] is not None]
if len(valid_thetas) > 5:
    mc_values = [m['mc'] for m in valid_thetas]
    theta_cv = np.std(mc_values) / np.mean(mc_values) if np.mean(mc_values) > 0 else float('inf')
    log(f"Theta sensitivity CV: {theta_cv:.4f}")
    RESULTS['experiment_1a']['theta_sensitivity_cv'] = float(theta_cv)

# ═══════════════════════════════════════════════════════════
# EXPERIMENT 1B: FACTOR-ADJUSTED MC
# ═══════════════════════════════════════════════════════════

log("")
log("=" * 60)
log("EXPERIMENT 1B: FACTOR-ADJUSTED MC")
log("=" * 60)

log("Downloading Fama-French 5 factors...")
# Try to download FF5 factors
try:
    import io
    import zipfile
    import urllib.request

    ff_url = "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/F-F_Research_Data_5_Factors_2x3_CSV.zip"
    response = urllib.request.urlopen(ff_url)
    zip_data = zipfile.ZipFile(io.BytesIO(response.read()))
    csv_name = zip_data.namelist()[0]

    with zip_data.open(csv_name) as f:
        lines = f.read().decode('utf-8').split('\n')

    # Find the header line
    start_idx = None
    for i, line in enumerate(lines):
        if 'Mkt-RF' in line:
            start_idx = i
            break

    if start_idx is not None:
        # Read until blank line (monthly data ends, annual begins)
        data_lines = [lines[start_idx]]  # header
        for line in lines[start_idx + 1:]:
            stripped = line.strip()
            if stripped == '' or not stripped[0].isdigit():
                break
            data_lines.append(stripped)

        ff_df = pd.read_csv(io.StringIO('\n'.join(data_lines)))
        ff_df.columns = [c.strip() for c in ff_df.columns]
        ff_df = ff_df.rename(columns={ff_df.columns[0]: 'date'})
        ff_df['date'] = ff_df['date'].astype(str)
        # Convert to decimals (FF data is in percentage)
        for col in ['Mkt-RF', 'SMB', 'HML', 'RMW', 'CMA', 'RF']:
            if col in ff_df.columns:
                ff_df[col] = pd.to_numeric(ff_df[col], errors='coerce') / 100.0

        log(f"Fama-French factors loaded: {len(ff_df)} months, {ff_df['date'].iloc[0]} to {ff_df['date'].iloc[-1]}")
        RESULTS['ff_factors_loaded'] = True
        RESULTS['ff_months'] = len(ff_df)

        # Compute residual returns for each ticker
        log("Computing factor-adjusted residuals per ticker...")
        residual_returns = {}
        ff_factors = ff_df[['Mkt-RF', 'SMB', 'HML', 'RMW', 'CMA']].values

        for ticker in ticker_returns:
            raw = ticker_returns[ticker]
            # Align lengths (use min of ticker returns and ff factors)
            n = min(len(raw), len(ff_factors))
            if n < 36:
                continue
            y = raw[:n]
            X = ff_factors[:n]
            # Add constant
            X_with_const = np.column_stack([np.ones(n), X])
            try:
                # OLS regression
                beta = np.linalg.lstsq(X_with_const, y, rcond=None)[0]
                predicted = X_with_const @ beta
                resid = y - predicted
                residual_returns[ticker] = resid
            except:
                continue

        log(f"Tickers with residual returns: {len(residual_returns)}")

        # Compute factor-adjusted correlation matrix
        adj_tickers = [t for t in tickers_list if t in residual_returns]
        n_adj = len(adj_tickers)
        adj_corr_matrix = np.full((n_adj, n_adj), np.nan)
        adj_ticker_idx = {t: i for i, t in enumerate(adj_tickers)}

        for i in range(n_adj):
            for j in range(i, n_adj):
                if i == j:
                    adj_corr_matrix[i, j] = 1.0
                    continue
                r1 = residual_returns[adj_tickers[i]]
                r2 = residual_returns[adj_tickers[j]]
                min_len = min(len(r1), len(r2))
                if min_len >= 12:
                    c = np.corrcoef(r1[:min_len], r2[:min_len])[0, 1]
                    if np.isfinite(c):
                        adj_corr_matrix[i, j] = c
                        adj_corr_matrix[j, i] = c

        # Adjusted population baseline
        adj_valid = adj_corr_matrix[np.triu_indices(n_adj, k=1)]
        adj_valid = adj_valid[np.isfinite(adj_valid)]
        adj_baseline = np.mean(adj_valid)
        log(f"Adjusted population baseline: {adj_baseline:.4f}")

        # Adjusted MC for SAE clusters
        adj_valid_tickers = [t for t in valid_tickers if t in adj_ticker_idx]
        adj_cluster_labels = np.array([best_labels[valid_tickers.index(t)] for t in adj_valid_tickers])
        adj_sae_mc, adj_n_pairs = compute_mc(adj_cluster_labels, adj_valid_tickers, adj_corr_matrix, adj_ticker_idx)
        log(f"Factor-adjusted SAE MC: {adj_sae_mc:.4f}")
        log(f"Raw MC: {sae_mc:.4f} -> Adjusted MC: {adj_sae_mc:.4f}")
        log(f"Signal from factor exposure: {sae_mc - adj_sae_mc:.4f} ({(sae_mc - adj_sae_mc) / sae_mc * 100:.1f}%)")
        log(f"Signal from company-specific: {adj_sae_mc - adj_baseline:.4f}")

        RESULTS['experiment_1b'] = {
            'adjusted_sae_mc': float(adj_sae_mc),
            'adjusted_baseline': float(adj_baseline),
            'raw_mc': float(sae_mc),
            'signal_from_factors': float(sae_mc - adj_sae_mc),
            'signal_from_structure': float(adj_sae_mc - adj_baseline),
            'pct_from_factors': float((sae_mc - adj_sae_mc) / sae_mc * 100) if sae_mc > 0 else None,
            'pct_from_structure': float((adj_sae_mc - adj_baseline) / (sae_mc - population_baseline) * 100) if (sae_mc - population_baseline) > 0 else None,
            'adjusted_signal_survives': bool(adj_sae_mc > adj_baseline),
            'n_tickers_adjusted': len(residual_returns),
        }

    else:
        log("ERROR: Could not parse Fama-French CSV")
        RESULTS['experiment_1b'] = {'error': 'Could not parse FF data'}

except Exception as e:
    log(f"ERROR in Experiment 1B: {e}")
    import traceback
    traceback.print_exc()
    RESULTS['experiment_1b'] = {'error': str(e)}

# ═══════════════════════════════════════════════════════════
# EXPERIMENT 1C: PERMUTATION TEST
# ═══════════════════════════════════════════════════════════

log("")
log("=" * 60)
log("EXPERIMENT 1C: PERMUTATION TEST")
log("=" * 60)

N_PERMUTATIONS = 10000
log(f"Running {N_PERMUTATIONS} permutations...")

permuted_mcs = []
for p in range(N_PERMUTATIONS):
    if p % 1000 == 0:
        log(f"  Permutation {p}/{N_PERMUTATIONS}...")
    # Shuffle feature-to-company mapping
    shuffled_idx = np.random.permutation(len(valid_tickers))
    shuffled_features = feature_matrix[shuffled_idx]

    # Recompute distances on shuffled features
    shuffled_dist = pdist(shuffled_features, metric='cosine')
    shuffled_dist_mat = squareform(shuffled_dist)

    # Build MST on shuffled
    shuffled_mst = minimum_spanning_tree(shuffled_dist_mat)
    shuffled_mst_sym = shuffled_mst.toarray()
    shuffled_mst_sym = shuffled_mst_sym + shuffled_mst_sym.T

    # Apply same theta
    shuffled_labels, _ = get_clusters(shuffled_mst_sym, best_theta)
    pmc, _ = compute_mc(shuffled_labels, valid_tickers, corr_matrix, valid_ticker_idx)
    if not np.isnan(pmc):
        permuted_mcs.append(pmc)

permuted_mcs = np.array(permuted_mcs)
empirical_p = np.mean(permuted_mcs >= sae_mc)

log(f"Permuted MC: {np.mean(permuted_mcs):.4f} +/- {np.std(permuted_mcs):.4f}")
log(f"Real MC: {sae_mc:.4f}")
log(f"Empirical p-value: {empirical_p:.6f}")
log(f"Signal is {'feature-driven' if empirical_p < 0.001 else 'POSSIBLY ALGORITHMIC'}")

RESULTS['experiment_1c'] = {
    'permuted_mc_mean': float(np.mean(permuted_mcs)),
    'permuted_mc_std': float(np.std(permuted_mcs)),
    'real_mc': float(sae_mc),
    'empirical_p_value': float(empirical_p),
    'n_permutations': N_PERMUTATIONS,
    'signal_feature_driven': bool(empirical_p < 0.001),
    'permuted_mc_95th_pctile': float(np.percentile(permuted_mcs, 95)),
    'permuted_mc_99th_pctile': float(np.percentile(permuted_mcs, 99)),
}

# ═══════════════════════════════════════════════════════════
# SUMMARY
# ═══════════════════════════════════════════════════════════

log("")
log("=" * 60)
log("PHASE 1 RESULTS SUMMARY")
log("=" * 60)

log("")
log("EXPERIMENT 1A — Signal Existence:")
log(f"  SAE MC:              {RESULTS['experiment_1a']['sae_mc']:.4f} [{RESULTS['experiment_1a']['sae_mc_ci_lower']:.4f}, {RESULTS['experiment_1a']['sae_mc_ci_upper']:.4f}]")
log(f"  SIC MC:              {RESULTS['experiment_1a']['sic_mc']:.4f}")
log(f"  Population baseline: {RESULTS['experiment_1a']['population_baseline']:.4f}")
log(f"  Random MC:           {RESULTS['experiment_1a']['random_mc_mean']:.4f} +/- {RESULTS['experiment_1a']['random_mc_std']:.4f}")
log(f"  CI excludes baseline: {RESULTS['experiment_1a']['ci_excludes_baseline']}")
log(f"  CI excludes SIC:      {RESULTS['experiment_1a']['ci_excludes_sic']}")

if 'adjusted_sae_mc' in RESULTS.get('experiment_1b', {}):
    log("")
    log("EXPERIMENT 1B — Factor Adjustment:")
    log(f"  Raw MC:              {RESULTS['experiment_1b']['raw_mc']:.4f}")
    log(f"  Adjusted MC:         {RESULTS['experiment_1b']['adjusted_sae_mc']:.4f}")
    log(f"  Adjusted baseline:   {RESULTS['experiment_1b']['adjusted_baseline']:.4f}")
    log(f"  Signal from factors: {RESULTS['experiment_1b']['pct_from_factors']:.1f}%")
    log(f"  Signal survives:     {RESULTS['experiment_1b']['adjusted_signal_survives']}")

log("")
log("EXPERIMENT 1C — Permutation Test:")
log(f"  Real MC:             {RESULTS['experiment_1c']['real_mc']:.4f}")
log(f"  Permuted MC:         {RESULTS['experiment_1c']['permuted_mc_mean']:.4f} +/- {RESULTS['experiment_1c']['permuted_mc_std']:.4f}")
log(f"  Empirical p-value:   {RESULTS['experiment_1c']['empirical_p_value']:.6f}")
log(f"  Feature-driven:      {RESULTS['experiment_1c']['signal_feature_driven']}")

# ═══════════════════════════════════════════════════════════
# CONVERGENCE STATEMENT
# ═══════════════════════════════════════════════════════════

log("")
log("=" * 60)
log("CONVERGENCE ASSESSMENT")
log("=" * 60)

passes = []
fails = []

if RESULTS['experiment_1a']['ci_excludes_baseline']:
    passes.append("1A: SAE MC significantly exceeds population baseline")
else:
    fails.append("1A: SAE MC CI includes population baseline")

if 'adjusted_signal_survives' in RESULTS.get('experiment_1b', {}) and RESULTS['experiment_1b']['adjusted_signal_survives']:
    passes.append("1B: Factor-adjusted signal survives")
else:
    fails.append("1B: Factor-adjusted signal does not survive or was not computed")

if RESULTS['experiment_1c']['signal_feature_driven']:
    passes.append("1C: Signal is feature-driven (p < 0.001)")
else:
    fails.append(f"1C: Permutation p-value = {RESULTS['experiment_1c']['empirical_p_value']:.4f}")

for p in passes:
    log(f"  PASS: {p}")
for f in fails:
    log(f"  FAIL: {f}")

log("")
if len(fails) == 0:
    log("PHASE 1 PASSES: All three experiments converge.")
else:
    log(f"PHASE 1 ISSUES: {len(fails)} experiment(s) did not pass.")

RESULTS['convergence'] = {
    'passes': passes,
    'fails': fails,
    'phase1_passes': len(fails) == 0,
}

# Save results
elapsed = time.time() - START_TIME
RESULTS['total_runtime_seconds'] = elapsed
RESULTS['timestamp'] = datetime.now().isoformat()

output_path = 'phase1_results.json'
with open(output_path, 'w') as f:
    json.dump(RESULTS, f, indent=2, default=str)

log(f"")
log(f"Results saved to {output_path}")
log(f"Total runtime: {elapsed/60:.1f} minutes")
log("Done.")

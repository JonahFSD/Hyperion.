"""
2c_01_event_prediction.py — Do SAE structural changes predict corporate events?

Phase 2C: Event Prediction Backtest
====================================
Tests whether companies undergoing large SAE fingerprint changes
subsequently experience corporate events at higher rates than controls.

Go/no-go: AUC > 0.60 on holdout (2011-2020) for any event type.

See experiments/prompts/2c_01_event_prediction.md for full methodology.

Data sources:
  - HuggingFace: marco-molinari/company_reports_with_features (SAE features)
  - HuggingFace: Mateusz1017/annual_reports_tokenized_llama3_logged_returns_no_null_returns_and_incomplete_descriptions_24k
  - (Optional) SEC data.gov: Public company bankruptcy cases

Usage:
  python experiments/2c_01_event_prediction.py
"""

import os
import sys
import json
import time
import gc
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

import numpy as np
import pandas as pd
from scipy.spatial.distance import cosine as cosine_dist
from sklearn.decomposition import IncrementalPCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve, precision_recall_curve, average_precision_score
from sklearn.preprocessing import StandardScaler
from datasets import load_dataset

# ─── Configuration ───────────────────────────────────────────────────────────

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ARTIFACTS_DIR = os.path.join(BASE_DIR, "artifacts")
os.makedirs(ARTIFACTS_DIR, exist_ok=True)

# Output files
OUT_MAGNITUDES = os.path.join(ARTIFACTS_DIR, "2c_01_change_magnitudes.parquet")
OUT_EVENTS = os.path.join(ARTIFACTS_DIR, "2c_01_events.parquet")
OUT_SUMMARY = os.path.join(ARTIFACTS_DIR, "2c_01_summary.json")
OUT_REPORT = os.path.join(ARTIFACTS_DIR, "2c_01_report.md")
OUT_PCA_FEATURES = os.path.join(ARTIFACTS_DIR, "2c_01_pca_features.parquet")

# HuggingFace datasets
HF_FEATURES = "marco-molinari/company_reports_with_features"
HF_COMPANY = "Mateusz1017/annual_reports_tokenized_llama3_logged_returns_no_null_returns_and_incomplete_descriptions_24k"

# PCA parameters — reduced from Phase 1's 4000 to avoid OOM on laptop.
# 500 components captures ~70% variance and is sufficient for change-magnitude features.
# Batch matrix: 600 × 131072 × float32 ≈ 315 MB (vs 2.4 GB at batch=4500).
N_PCA_COMPONENTS = 500
BATCH_SIZE_PCA = 600  # Must be >= N_PCA_COMPONENTS

# Event thresholds
EXTREME_RETURN_THRESHOLD = -0.693      # log return ≈ 50% decline
CUMULATIVE_DISTRESS_THRESHOLD = -0.693  # annual cumulative log return ≈ 50% decline

# Train/test split
TRAIN_END_YEAR = 2010  # Train: 1996-2010, Test: 2011-2020
RIGHT_CENSOR_YEARS = 2  # Exclude last N years for disappearance events

# Statistical
SEED = 42
SIGNIFICANCE_LEVEL = 0.05

def log(msg):
    print(f"[2c_01] {msg}", flush=True)


# ─── Step 0: Data Inspection ────────────────────────────────────────────────

def inspect_datasets():
    """Load small samples and report schemas."""
    log("Step 0: Inspecting dataset schemas...")
    t0 = time.time()

    # Features dataset
    log("  Loading features dataset (first 5 rows)...")
    ds_feat = load_dataset(HF_FEATURES, split="train", streaming=True)
    feat_sample = []
    for i, row in enumerate(ds_feat):
        feat_sample.append(row)
        if i >= 4:
            break
    feat_cols = list(feat_sample[0].keys())
    log(f"  Features dataset columns: {feat_cols}")

    # Check features field structure
    for col in feat_cols:
        val = feat_sample[0][col]
        if isinstance(val, (list, np.ndarray)):
            if isinstance(val[0], (list, np.ndarray)):
                log(f"  Column '{col}': nested list, outer len={len(val)}, inner len={len(val[0])}")
            else:
                log(f"  Column '{col}': flat list, len={len(val)}")
        else:
            log(f"  Column '{col}': {type(val).__name__} = {val}")

    # Company dataset
    log("  Loading company dataset (first 5 rows)...")
    ds_comp = load_dataset(HF_COMPANY, split="train", streaming=True)
    comp_sample = []
    for i, row in enumerate(ds_comp):
        comp_sample.append(row)
        if i >= 4:
            break
    comp_cols = list(comp_sample[0].keys())
    log(f"  Company dataset columns: {comp_cols}")
    for col in comp_cols:
        val = comp_sample[0][col]
        if isinstance(val, (list, np.ndarray)):
            log(f"  Column '{col}': list, len={len(val)}")
        else:
            log(f"  Column '{col}': {type(val).__name__} = {val}")

    log(f"  Schema inspection complete ({time.time()-t0:.1f}s)")
    return feat_cols, comp_cols, feat_sample, comp_sample


# ─── Step 1: Load Company Metadata ──────────────────────────────────────────

def load_company_metadata():
    """Load company metadata with returns."""
    log("Step 1: Loading company metadata...")
    t0 = time.time()

    ds = load_dataset(HF_COMPANY, split="train")
    df = ds.to_pandas()
    log(f"  Loaded {len(df):,} rows, columns: {list(df.columns)}")

    # Identify key columns (names may vary)
    # Expected: cik, year, ticker, sic_code, and monthly return columns
    col_map = {}
    for col in df.columns:
        cl = col.lower().strip()
        if cl in ("cik", "cik_number", "cik_code"):
            col_map["cik"] = col
        elif cl in ("year", "fiscal_year"):
            col_map["year"] = col
        elif cl in ("ticker", "symbol"):
            col_map["ticker"] = col
        elif cl in ("sic_code", "sic", "sic_number"):
            col_map["sic"] = col

    log(f"  Identified columns: {col_map}")

    # Find return columns (12 monthly logged returns)
    # Prefer a single list-typed matrix column (e.g. logged_monthly_returns_matrix)
    # over mixing it with scalar return columns (e.g. `returns`).
    matrix_col = None
    scalar_return_cols = []
    for col in df.columns:
        cl = col.lower()
        if "return" in cl or "log_ret" in cl or "logged_return" in cl:
            sample_val = df[col].iloc[0]
            if isinstance(sample_val, (list, np.ndarray)):
                matrix_col = col
            else:
                scalar_return_cols.append(col)

    return_cols = []
    if matrix_col is not None:
        # Expand list column into 12 flat monthly return columns
        n_months = len(df[matrix_col].iloc[0])
        monthly_cols = [f"monthly_return_{i}" for i in range(n_months)]
        expanded = pd.DataFrame(df[matrix_col].tolist(), columns=monthly_cols, index=df.index)
        df = pd.concat([df, expanded], axis=1)
        return_cols = monthly_cols
        log(f"  Expanded '{matrix_col}' into {n_months} monthly return columns")
    elif scalar_return_cols:
        return_cols = scalar_return_cols
        log(f"  Using {len(return_cols)} scalar return columns")

    if not return_cols:
        # Try numbered columns that could be returns
        for col in df.columns:
            if col.startswith("month_") or col.startswith("m_") or col.startswith("ret_"):
                return_cols.append(col)
    log(f"  Found {len(return_cols)} return columns")

    # Standardize column names
    rename = {}
    if "cik" in col_map:
        rename[col_map["cik"]] = "cik"
    if "year" in col_map:
        rename[col_map["year"]] = "year"
    if "ticker" in col_map:
        rename[col_map["ticker"]] = "ticker"
    if "sic" in col_map:
        rename[col_map["sic"]] = "sic_code"
    df = df.rename(columns=rename)

    # Compute annual cumulative return and extreme monthly return
    if return_cols:
        returns_matrix = df[return_cols].values.astype(float)
        df["cum_log_return"] = np.nansum(returns_matrix, axis=1)
        df["min_monthly_return"] = np.nanmin(returns_matrix, axis=1)
        df["max_monthly_return"] = np.nanmax(returns_matrix, axis=1)
        df["n_extreme_months"] = np.sum(returns_matrix < EXTREME_RETURN_THRESHOLD, axis=1)
        log(f"  Computed return stats: cum_log_return range [{df['cum_log_return'].min():.3f}, {df['cum_log_return'].max():.3f}]")
    else:
        log("  WARNING: No return columns found. Return-based events will be unavailable.")

    df["cik"] = df["cik"].astype(int)
    df["year"] = df["year"].astype(int)
    if "sic_code" in df.columns:
        sic_numeric = pd.to_numeric(df["sic_code"], errors="coerce")
        df["sic_2digit"] = (sic_numeric // 100).astype("Int64")

    log(f"  Company metadata ready: {len(df):,} rows, years {df['year'].min()}-{df['year'].max()} ({time.time()-t0:.1f}s)")
    return df, return_cols


# ─── Step 2: Load Features & Compute PCA ────────────────────────────────────

def load_features_and_pca():
    """Load SAE features, apply PCA, return reduced feature matrix."""
    log("Step 2: Loading features and computing PCA...")
    t0 = time.time()

    # Check for cached PCA features
    if os.path.exists(OUT_PCA_FEATURES):
        log(f"  Found cached PCA features at {OUT_PCA_FEATURES}")
        pca_df = pd.read_parquet(OUT_PCA_FEATURES)
        log(f"  Loaded {len(pca_df):,} rows with {len([c for c in pca_df.columns if c.startswith('pc_')])} PCA components ({time.time()-t0:.1f}s)")
        return pca_df

    log("  Loading full features dataset from HuggingFace (this may take several minutes)...")
    ds = load_dataset(HF_FEATURES, split="train")

    # Extract feature vectors and metadata
    log(f"  Dataset has {len(ds):,} rows. Extracting features...")
    t1 = time.time()

    # Identify the features column and metadata columns
    sample = ds[0]
    features_col = None
    meta_cols = {}
    for key, val in sample.items():
        if isinstance(val, (list, np.ndarray)):
            if isinstance(val, list) and len(val) > 0 and isinstance(val[0], (list, np.ndarray)):
                features_col = key
                log(f"  Features column: '{key}' (nested list, inner dim={len(val[0])})")
            elif isinstance(val, list) and len(val) > 1000:
                features_col = key
                log(f"  Features column: '{key}' (flat list, dim={len(val)})")
        else:
            meta_cols[key] = type(val).__name__

    if features_col is None:
        log("  ERROR: Could not identify features column. Columns and types:")
        for key, val in sample.items():
            log(f"    {key}: {type(val).__name__}, len={len(val) if hasattr(val, '__len__') else 'N/A'}")
        sys.exit(1)

    log(f"  Metadata columns: {meta_cols}")

    # Extract metadata into DataFrame
    df_meta = ds.to_pandas().drop(columns=[features_col])
    log(f"  Metadata extracted: {len(df_meta):,} rows ({time.time()-t1:.1f}s)")

    # Determine expected feature dimensionality from first valid row
    n_rows = len(ds)
    expected_dim = None
    for i in range(min(10, n_rows)):
        raw = ds[i][features_col]
        if isinstance(raw, list) and len(raw) > 0 and isinstance(raw[0], list):
            candidate = raw[0]
        else:
            candidate = raw
        if len(candidate) > 1:
            expected_dim = len(candidate)
            break
    if expected_dim is None:
        log("  ERROR: Could not determine feature dimensionality from first 10 rows.")
        sys.exit(1)
    log(f"  Expected feature dim: {expected_dim}")

    def extract_vec(raw):
        """Return float32 vector or None if malformed."""
        if isinstance(raw, list) and len(raw) > 0 and isinstance(raw[0], list):
            vec = np.array(raw[0], dtype=np.float32)
        else:
            vec = np.array(raw, dtype=np.float32)
        if vec.ndim != 1 or vec.shape[0] != expected_dim:
            return None
        return vec

    # Fit IncrementalPCA on feature vectors
    log(f"  Fitting IncrementalPCA (n_components={N_PCA_COMPONENTS}, batch_size={BATCH_SIZE_PCA})...")
    t2 = time.time()
    ipca = IncrementalPCA(n_components=N_PCA_COMPONENTS, batch_size=BATCH_SIZE_PCA)

    batch_features = []
    batch_indices = []  # track which dataset indices are valid
    valid_indices = []
    skipped = 0

    for i in range(n_rows):
        raw = ds[i][features_col]
        vec = extract_vec(raw)
        if vec is None:
            skipped += 1
            continue
        batch_features.append(vec)
        batch_indices.append(i)

        if len(batch_features) >= BATCH_SIZE_PCA:
            batch_matrix = np.vstack(batch_features)
            ipca.partial_fit(batch_matrix)
            valid_indices.extend(batch_indices)
            n_fit = len(valid_indices)
            batch_features = []
            batch_indices = []
            if n_fit % (BATCH_SIZE_PCA * 2) < BATCH_SIZE_PCA:
                log(f"    PCA fit progress: {n_fit:,}/{n_rows:,} rows ({n_fit/n_rows*100:.0f}%)")

    # Fit remaining
    if len(batch_features) >= N_PCA_COMPONENTS:
        batch_matrix = np.vstack(batch_features)
        ipca.partial_fit(batch_matrix)
        valid_indices.extend(batch_indices)
    elif len(batch_features) > 0:
        log(f"  WARNING: {len(batch_features)} leftover rows < n_components ({N_PCA_COMPONENTS}), skipping final partial_fit")
        valid_indices.extend(batch_indices)

    if skipped:
        log(f"  Skipped {skipped} malformed rows (wrong feature dim)")

    variance_explained = ipca.explained_variance_ratio_.sum()
    log(f"  PCA fit complete: {variance_explained:.4f} variance explained ({time.time()-t2:.1f}s)")

    # Transform valid rows only
    log("  Transforming features through PCA...")
    t3 = time.time()
    pca_vectors = []
    batch_features = []

    for i in valid_indices:
        raw = ds[i][features_col]
        vec = extract_vec(raw)
        if vec is None:
            continue
        batch_features.append(vec)

        if len(batch_features) >= BATCH_SIZE_PCA:
            batch_matrix = np.vstack(batch_features)
            transformed = ipca.transform(batch_matrix)
            pca_vectors.append(transformed)
            batch_features = []

    if batch_features:
        batch_matrix = np.vstack(batch_features)
        transformed = ipca.transform(batch_matrix)
        pca_vectors.append(transformed)

    pca_matrix = np.vstack(pca_vectors)
    log(f"  PCA transform complete: shape {pca_matrix.shape} ({time.time()-t3:.1f}s)")

    # Build output DataFrame aligned to valid rows
    pca_cols = [f"pc_{i}" for i in range(N_PCA_COMPONENTS)]
    pca_df = pd.DataFrame(pca_matrix, columns=pca_cols)
    for col in df_meta.columns:
        pca_df[col] = df_meta[col].iloc[valid_indices].values

    # Cache to parquet
    pca_df.to_parquet(OUT_PCA_FEATURES, index=False)
    log(f"  Cached PCA features to {OUT_PCA_FEATURES} ({time.time()-t0:.1f}s)")

    del ds, pca_matrix, pca_vectors, batch_features
    gc.collect()

    return pca_df


# ─── Step 3: Compute Change Magnitudes ──────────────────────────────────────

def compute_change_magnitudes(pca_df):
    """Compute year-over-year SAE change magnitude per company."""
    log("Step 3: Computing year-over-year change magnitudes...")
    t0 = time.time()

    # Check for cached magnitudes
    if os.path.exists(OUT_MAGNITUDES):
        log(f"  Found cached magnitudes at {OUT_MAGNITUDES}")
        mag_df = pd.read_parquet(OUT_MAGNITUDES)
        log(f"  Loaded {len(mag_df):,} change magnitudes ({time.time()-t0:.1f}s)")
        return mag_df

    pca_cols = [c for c in pca_df.columns if c.startswith("pc_")]

    # Identify CIK and year columns in PCA DataFrame
    cik_col = None
    year_col = None
    for col in pca_df.columns:
        cl = col.lower().strip()
        if cl in ("cik", "cik_number", "cik_code"):
            cik_col = col
        elif cl in ("year", "fiscal_year"):
            year_col = col
    if cik_col is None or year_col is None:
        log(f"  ERROR: Cannot find CIK/year columns in PCA DataFrame. Columns: {list(pca_df.columns)}")
        log(f"  Non-PC columns: {[c for c in pca_df.columns if not c.startswith('pc_')]}")
        sys.exit(1)

    pca_df[cik_col] = pca_df[cik_col].astype(int)
    pca_df[year_col] = pca_df[year_col].astype(int)

    log(f"  Computing cosine distance for consecutive years per company...")
    results = []
    companies = pca_df.groupby(cik_col)
    n_companies = len(companies)
    n_processed = 0

    for cik, group in companies:
        group = group.sort_values(year_col)
        years = group[year_col].values
        features = group[pca_cols].values

        for i in range(1, len(years)):
            if years[i] - years[i-1] <= 2:  # Allow gap of 1 year (filing skip)
                vec_prev = features[i-1]
                vec_curr = features[i]

                # Cosine distance (1 - cosine_similarity)
                norm_prev = np.linalg.norm(vec_prev)
                norm_curr = np.linalg.norm(vec_curr)
                if norm_prev > 0 and norm_curr > 0:
                    cos_sim = np.dot(vec_prev, vec_curr) / (norm_prev * norm_curr)
                    change_mag = 1.0 - cos_sim
                else:
                    change_mag = np.nan

                results.append({
                    "cik": cik,
                    "year": years[i],
                    "year_prev": years[i-1],
                    "change_magnitude": change_mag,
                    "year_gap": int(years[i] - years[i-1]),
                })

        n_processed += 1
        if n_processed % 500 == 0:
            log(f"    Progress: {n_processed:,}/{n_companies:,} companies ({n_processed/n_companies*100:.0f}%)")

    mag_df = pd.DataFrame(results)
    log(f"  Computed {len(mag_df):,} change magnitudes across {mag_df['cik'].nunique():,} companies")
    log(f"  Change magnitude stats: mean={mag_df['change_magnitude'].mean():.4f}, "
        f"median={mag_df['change_magnitude'].median():.4f}, "
        f"std={mag_df['change_magnitude'].std():.4f}, "
        f"min={mag_df['change_magnitude'].min():.4f}, "
        f"max={mag_df['change_magnitude'].max():.4f}")

    # Distribution by year gap
    for gap in sorted(mag_df["year_gap"].unique()):
        subset = mag_df[mag_df["year_gap"] == gap]
        log(f"  Year gap {gap}: {len(subset):,} pairs, mean change={subset['change_magnitude'].mean():.4f}")

    mag_df.to_parquet(OUT_MAGNITUDES, index=False)
    log(f"  Saved to {OUT_MAGNITUDES} ({time.time()-t0:.1f}s)")
    return mag_df


# ─── Step 4: Define Events ──────────────────────────────────────────────────

def define_events(company_df, mag_df):
    """Define event flags from existing data."""
    log("Step 4: Defining events from existing data...")
    t0 = time.time()

    # Check cache
    if os.path.exists(OUT_EVENTS):
        log(f"  Found cached events at {OUT_EVENTS}")
        events_df = pd.read_parquet(OUT_EVENTS)
        log(f"  Loaded {len(events_df):,} rows ({time.time()-t0:.1f}s)")
        return events_df

    max_year = company_df["year"].max()
    min_year = company_df["year"].min()
    log(f"  Data range: {min_year}-{max_year}")

    # Build company-year presence set
    company_years = set(zip(company_df["cik"].values, company_df["year"].values))

    # ── Event Type A: Disappearance ──
    log("  Computing disappearance events...")
    disappearance = {}
    for cik in company_df["cik"].unique():
        cik_years = sorted(company_df[company_df["cik"] == cik]["year"].values)
        last_year = cik_years[-1]

        # Right-censored: skip if last year is within RIGHT_CENSOR_YEARS of max
        if last_year >= max_year - RIGHT_CENSOR_YEARS + 1:
            continue

        # Company disappeared if it has no filing in last_year+1 AND last_year+2
        has_next = (cik, last_year + 1) in company_years
        has_next2 = (cik, last_year + 2) in company_years
        if not has_next and not has_next2:
            disappearance[cik] = last_year  # Last observed year

    log(f"  Found {len(disappearance):,} companies that disappeared "
        f"(out of {company_df['cik'].nunique():,} total, "
        f"excluding {RIGHT_CENSOR_YEARS} right-censored years)")

    # ── Event Type B: Extreme Monthly Return ──
    log("  Computing extreme return events...")
    has_returns = "min_monthly_return" in company_df.columns
    if has_returns:
        extreme_mask = company_df["min_monthly_return"] < EXTREME_RETURN_THRESHOLD
        log(f"  Found {extreme_mask.sum():,} company-years with extreme monthly return "
            f"(< {EXTREME_RETURN_THRESHOLD:.3f} log return)")
    else:
        log("  WARNING: No return data available. Skipping extreme return events.")

    # ── Event Type C: Cumulative Distress ──
    log("  Computing cumulative distress events...")
    if has_returns:
        distress_mask = company_df["cum_log_return"] < CUMULATIVE_DISTRESS_THRESHOLD
        log(f"  Found {distress_mask.sum():,} company-years with cumulative distress "
            f"(annual log return < {CUMULATIVE_DISTRESS_THRESHOLD:.3f})")

    # ── Merge events with change magnitudes ──
    log("  Merging events with change magnitudes...")

    # Start with change magnitudes as base
    events_df = mag_df[["cik", "year", "change_magnitude", "year_gap"]].copy()

    # Merge company metadata
    company_cols = ["cik", "year"]
    if "sic_2digit" in company_df.columns:
        company_cols.append("sic_2digit")
    if "sic_code" in company_df.columns:
        company_cols.append("sic_code")
    # ticker is a list column (unhashable) — skip it to avoid drop_duplicates failure
    if has_returns:
        company_cols.extend(["cum_log_return", "min_monthly_return", "n_extreme_months"])

    events_df = events_df.merge(
        company_df[company_cols].drop_duplicates(),
        on=["cik", "year"],
        how="left"
    )

    # Event A: Disappearance in year+1 or year+2
    # For each company-year, check if this company disappears within 2 years
    events_df["event_disappear"] = 0
    for idx, row in events_df.iterrows():
        cik = row["cik"]
        year = row["year"]
        if cik in disappearance:
            last_year = disappearance[cik]
            # Signal year should be 1-2 years before disappearance
            if year >= last_year - 1 and year <= last_year:
                events_df.at[idx, "event_disappear"] = 1

    # Event B: Extreme return in the NEXT year (year+1)
    if has_returns:
        # Build lookup: (cik, year) -> has extreme return
        extreme_lookup = set()
        for _, row in company_df[extreme_mask].iterrows():
            extreme_lookup.add((int(row["cik"]), int(row["year"])))

        events_df["event_extreme_return"] = events_df.apply(
            lambda r: 1 if (r["cik"], r["year"] + 1) in extreme_lookup else 0,
            axis=1
        )
    else:
        events_df["event_extreme_return"] = np.nan

    # Event C: Cumulative distress in the NEXT year (year+1)
    if has_returns:
        distress_lookup = set()
        for _, row in company_df[distress_mask].iterrows():
            distress_lookup.add((int(row["cik"]), int(row["year"])))

        events_df["event_distress"] = events_df.apply(
            lambda r: 1 if (r["cik"], r["year"] + 1) in distress_lookup else 0,
            axis=1
        )
    else:
        events_df["event_distress"] = np.nan

    # Prior year returns (control variable)
    if has_returns:
        return_lookup = dict(zip(
            zip(company_df["cik"].values, company_df["year"].values),
            company_df["cum_log_return"].values
        ))
        events_df["prior_cum_return"] = events_df.apply(
            lambda r: return_lookup.get((r["cik"], r["year"]), np.nan),
            axis=1
        )

    # Summary
    event_cols = [c for c in events_df.columns if c.startswith("event_")]
    log(f"\n  Event summary:")
    for col in event_cols:
        valid = events_df[col].dropna()
        n_events = (valid == 1).sum()
        rate = n_events / len(valid) * 100 if len(valid) > 0 else 0
        log(f"    {col}: {n_events:,} events / {len(valid):,} obs ({rate:.2f}%)")

    events_df.to_parquet(OUT_EVENTS, index=False)
    log(f"  Saved to {OUT_EVENTS} ({time.time()-t0:.1f}s)")
    return events_df


# ─── Step 5: Statistical Tests ──────────────────────────────────────────────

def run_statistical_tests(events_df):
    """Logistic regression, AUC-ROC, quartile analysis."""
    log("Step 5: Running statistical tests...")
    t0 = time.time()

    results = {
        "tests": {},
        "quartile_analysis": {},
        "verdict": None,
        "go_no_go": None,
    }

    event_cols = [c for c in events_df.columns if c.startswith("event_") and events_df[c].notna().any()]
    if not event_cols:
        log("  ERROR: No event columns with valid data.")
        return results

    for event_col in event_cols:
        log(f"\n  ── Testing: {event_col} ──")

        # Prepare data
        df = events_df.dropna(subset=[event_col, "change_magnitude"]).copy()
        df[event_col] = df[event_col].astype(int)

        n_events = df[event_col].sum()
        n_total = len(df)
        event_rate = n_events / n_total * 100
        log(f"  {n_events:,} events / {n_total:,} obs ({event_rate:.2f}%)")

        if n_events < 20:
            log(f"  SKIP: Too few events ({n_events}) for reliable statistics")
            results["tests"][event_col] = {"skipped": True, "reason": f"Only {n_events} events"}
            continue

        # ── Quartile Analysis (non-parametric, always valid) ──
        df["change_quartile"] = pd.qcut(df["change_magnitude"], q=4, labels=["Q1_low", "Q2", "Q3", "Q4_high"])
        quartile_rates = df.groupby("change_quartile")[event_col].agg(["mean", "sum", "count"])
        quartile_rates.columns = ["event_rate", "n_events", "n_total"]
        quartile_rates["event_rate_pct"] = quartile_rates["event_rate"] * 100

        log(f"  Quartile event rates:")
        for q, row in quartile_rates.iterrows():
            log(f"    {q}: {row['event_rate_pct']:.2f}% ({int(row['n_events'])}/{int(row['n_total'])})")

        q1_rate = quartile_rates.loc["Q1_low", "event_rate"]
        q4_rate = quartile_rates.loc["Q4_high", "event_rate"]
        monotonic = all(
            quartile_rates.iloc[i]["event_rate"] <= quartile_rates.iloc[i+1]["event_rate"]
            for i in range(len(quartile_rates)-1)
        )
        log(f"  Q4/Q1 ratio: {q4_rate/q1_rate:.2f}x" if q1_rate > 0 else "  Q1 rate is zero")
        log(f"  Monotonic increasing: {monotonic}")

        results["quartile_analysis"][event_col] = {
            "quartiles": quartile_rates.to_dict(orient="index"),
            "q4_q1_ratio": float(q4_rate / q1_rate) if q1_rate > 0 else None,
            "monotonic": monotonic,
        }

        # ── Train/Test Split ──
        train = df[df["year"] <= TRAIN_END_YEAR].copy()
        test = df[df["year"] > TRAIN_END_YEAR].copy()
        log(f"  Train: {len(train):,} obs ({train[event_col].sum():,} events), "
            f"Test: {len(test):,} obs ({test[event_col].sum():,} events)")

        if test[event_col].sum() < 10 or train[event_col].sum() < 10:
            log(f"  SKIP: Too few events in train ({train[event_col].sum()}) or test ({test[event_col].sum()})")
            results["tests"][event_col] = {"skipped": True, "reason": "Too few events in train/test split"}
            continue

        # ── Feature Matrix ──
        feature_cols = ["change_magnitude"]

        # Add controls if available
        if "sic_2digit" in df.columns and df["sic_2digit"].notna().any():
            # One-hot encode top SIC codes, leave rare ones as "other"
            sic_counts = df["sic_2digit"].value_counts()
            top_sics = sic_counts[sic_counts >= 20].index.tolist()
            for sic in top_sics[:20]:  # Cap at 20 dummies
                col_name = f"sic_{sic}"
                df[col_name] = (df["sic_2digit"] == sic).fillna(False).astype(int)
                train[col_name] = (train["sic_2digit"] == sic).fillna(False).astype(int)
                test[col_name] = (test["sic_2digit"] == sic).fillna(False).astype(int)
                feature_cols.append(col_name)

        if "prior_cum_return" in df.columns and df["prior_cum_return"].notna().any():
            feature_cols.append("prior_cum_return")
            train["prior_cum_return"] = train["prior_cum_return"].fillna(0)
            test["prior_cum_return"] = test["prior_cum_return"].fillna(0)

        X_train = train[feature_cols].values
        y_train = train[event_col].values
        X_test = test[feature_cols].values
        y_test = test[event_col].values

        # Handle NaN
        X_train = np.nan_to_num(X_train, nan=0.0)
        X_test = np.nan_to_num(X_test, nan=0.0)

        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # ── Logistic Regression ──
        log(f"  Fitting logistic regression ({len(feature_cols)} features)...")
        model = LogisticRegression(
            max_iter=1000,
            class_weight="balanced",  # Handle class imbalance
            random_state=SEED,
            solver="lbfgs",
        )
        model.fit(X_train_scaled, y_train)

        # Predictions
        y_pred_proba_train = model.predict_proba(X_train_scaled)[:, 1]
        y_pred_proba_test = model.predict_proba(X_test_scaled)[:, 1]

        # ── AUC-ROC ──
        auc_train = roc_auc_score(y_train, y_pred_proba_train)
        auc_test = roc_auc_score(y_test, y_pred_proba_test)
        log(f"  AUC-ROC: train={auc_train:.4f}, test={auc_test:.4f}")

        # ── Average Precision (Precision-Recall) ──
        ap_test = average_precision_score(y_test, y_pred_proba_test)
        log(f"  Average Precision (test): {ap_test:.4f}")

        # ── Change Magnitude Coefficient ──
        cm_coef = model.coef_[0][0]  # First feature is change_magnitude
        cm_direction = "positive" if cm_coef > 0 else "negative"
        log(f"  Change magnitude coefficient: {cm_coef:.4f} ({cm_direction})")

        # ── Change-Magnitude-Only Model (ablation) ──
        model_simple = LogisticRegression(
            max_iter=1000, class_weight="balanced", random_state=SEED
        )
        model_simple.fit(X_train_scaled[:, 0:1], y_train)
        y_simple_test = model_simple.predict_proba(X_test_scaled[:, 0:1])[:, 1]
        auc_simple = roc_auc_score(y_test, y_simple_test)
        log(f"  AUC (change_magnitude only, no controls): {auc_simple:.4f}")

        # ── Random Baseline ──
        rng = np.random.default_rng(SEED)
        auc_random_trials = []
        for trial in range(100):
            y_random = rng.permutation(y_test)
            try:
                auc_r = roc_auc_score(y_random, y_pred_proba_test)
                auc_random_trials.append(auc_r)
            except ValueError:
                pass
        auc_random_mean = np.mean(auc_random_trials) if auc_random_trials else 0.5
        auc_random_p95 = np.percentile(auc_random_trials, 95) if auc_random_trials else 0.5
        log(f"  Random baseline AUC: mean={auc_random_mean:.4f}, p95={auc_random_p95:.4f}")

        results["tests"][event_col] = {
            "n_events": int(n_events),
            "n_total": int(n_total),
            "event_rate_pct": float(event_rate),
            "auc_train": float(auc_train),
            "auc_test": float(auc_test),
            "auc_simple_test": float(auc_simple),
            "average_precision_test": float(ap_test),
            "change_magnitude_coef": float(cm_coef),
            "coef_direction": cm_direction,
            "random_baseline_auc_mean": float(auc_random_mean),
            "random_baseline_auc_p95": float(auc_random_p95),
            "beats_random_p95": bool(auc_test > auc_random_p95),
            "n_train": int(len(train)),
            "n_test": int(len(test)),
            "n_train_events": int(train[event_col].sum()),
            "n_test_events": int(test[event_col].sum()),
            "feature_cols": feature_cols,
        }

    # ── Verdict ──
    log(f"\n  ── VERDICT ──")
    best_auc = 0
    best_event = None
    for event_col, test_result in results["tests"].items():
        if "skipped" in test_result and test_result["skipped"]:
            continue
        auc = test_result["auc_test"]
        if auc > best_auc:
            best_auc = auc
            best_event = event_col

    if best_auc > 0.60:
        verdict = "GO"
        explanation = (f"AUC={best_auc:.4f} on {best_event} exceeds 0.60 threshold. "
                      f"SAE structural changes predict corporate events. "
                      f"Proceed to Tier 2 validation with real event data.")
    elif best_auc > 0.55:
        # Check for monotonic quartile pattern
        has_monotonic = any(
            qa.get("monotonic", False)
            for qa in results["quartile_analysis"].values()
        )
        if has_monotonic:
            verdict = "QUALIFIED_GO"
            explanation = (f"AUC={best_auc:.4f} on {best_event} is moderate (0.55-0.60) "
                          f"but quartile analysis shows monotonic relationship. "
                          f"Investigate feature-level patterns before deciding.")
        else:
            verdict = "QUALIFIED_GO"
            explanation = (f"AUC={best_auc:.4f} on {best_event} is moderate (0.55-0.60). "
                          f"Signal exists but weak. Investigate feature-level patterns.")
    else:
        verdict = "NO_GO"
        explanation = (f"Best AUC={best_auc:.4f} on {best_event}. "
                      f"SAE structural changes do not predict corporate events above threshold. "
                      f"Market research verdict stands.")

    results["verdict"] = verdict
    results["verdict_explanation"] = explanation
    results["best_auc"] = float(best_auc)
    results["best_event_type"] = best_event

    log(f"  Verdict: {verdict}")
    log(f"  {explanation}")

    log(f"  ({time.time()-t0:.1f}s)")
    return results


# ─── Step 6: Feature Attribution ─────────────────────────────────────────────

def feature_attribution(events_df, pca_df, results):
    """For event companies, identify which features changed most pre-event."""
    if results.get("verdict") == "NO_GO":
        log("Step 6: Skipping feature attribution (NO_GO verdict)")
        return results

    log("Step 6: Feature attribution for event companies...")
    t0 = time.time()

    pca_cols = [c for c in pca_df.columns if c.startswith("pc_")]

    # Identify CIK and year columns in PCA DataFrame
    cik_col_pca = None
    year_col_pca = None
    for col in pca_df.columns:
        cl = col.lower().strip()
        if cl in ("cik", "cik_number", "cik_code"):
            cik_col_pca = col
        elif cl in ("year", "fiscal_year"):
            year_col_pca = col

    if cik_col_pca is None or year_col_pca is None:
        log("  Cannot identify CIK/year in PCA data. Skipping attribution.")
        return results

    # Build feature lookup: (cik, year) -> PCA vector
    feature_lookup = {}
    for _, row in pca_df.iterrows():
        key = (int(row[cik_col_pca]), int(row[year_col_pca]))
        feature_lookup[key] = row[pca_cols].values.astype(float)

    # For the best-performing event type, analyze feature changes
    best_event = results.get("best_event_type")
    if best_event is None or best_event not in events_df.columns:
        log("  No valid event type for attribution. Skipping.")
        return results

    log(f"  Analyzing feature changes for {best_event}...")

    # Event companies: those with event=1
    event_rows = events_df[events_df[best_event] == 1].copy()
    control_rows = events_df[events_df[best_event] == 0].copy()

    # For each event company-year, compute per-feature change (year vs year-1)
    event_deltas = []
    for _, row in event_rows.iterrows():
        cik = int(row["cik"])
        year = int(row["year"])
        vec_curr = feature_lookup.get((cik, year))
        vec_prev = feature_lookup.get((cik, year - 1))
        if vec_prev is None:
            vec_prev = feature_lookup.get((cik, year - 2))  # Allow 1-year gap
        if vec_curr is not None and vec_prev is not None:
            delta = np.abs(vec_curr - vec_prev)
            event_deltas.append(delta)

    # Same for control companies (sample to match size)
    rng = np.random.default_rng(SEED)
    control_sample = control_rows.sample(n=min(len(control_rows), len(event_deltas) * 5), random_state=SEED)
    control_deltas = []
    for _, row in control_sample.iterrows():
        cik = int(row["cik"])
        year = int(row["year"])
        vec_curr = feature_lookup.get((cik, year))
        vec_prev = feature_lookup.get((cik, year - 1))
        if vec_prev is None:
            vec_prev = feature_lookup.get((cik, year - 2))
        if vec_curr is not None and vec_prev is not None:
            control_deltas.append(np.abs(vec_curr - vec_prev))

    if not event_deltas or not control_deltas:
        log("  Insufficient feature data for attribution. Skipping.")
        return results

    event_mean = np.mean(event_deltas, axis=0)
    control_mean = np.mean(control_deltas, axis=0)
    diff = event_mean - control_mean

    # Top features where event companies changed MORE than controls
    top_indices = np.argsort(diff)[::-1][:20]
    log(f"\n  Top 20 PCA components with largest event vs. control difference:")
    log(f"  {'PC':>6} | {'Event Mean':>12} | {'Control Mean':>12} | {'Difference':>12}")
    log(f"  {'-'*6}-+-{'-'*12}-+-{'-'*12}-+-{'-'*12}")
    for idx in top_indices:
        log(f"  pc_{idx:>4} | {event_mean[idx]:>12.6f} | {control_mean[idx]:>12.6f} | {diff[idx]:>12.6f}")

    results["feature_attribution"] = {
        "n_event_companies_with_features": len(event_deltas),
        "n_control_companies_with_features": len(control_deltas),
        "top_20_pca_components": [int(i) for i in top_indices],
        "top_20_event_mean": [float(event_mean[i]) for i in top_indices],
        "top_20_control_mean": [float(control_mean[i]) for i in top_indices],
        "top_20_diff": [float(diff[i]) for i in top_indices],
        "overall_event_mean_change": float(np.mean(event_mean)),
        "overall_control_mean_change": float(np.mean(control_mean)),
    }

    log(f"\n  Overall mean absolute change: events={np.mean(event_mean):.6f}, controls={np.mean(control_mean):.6f}")
    log(f"  Ratio: {np.mean(event_mean)/np.mean(control_mean):.2f}x")
    log(f"  ({time.time()-t0:.1f}s)")

    return results


# ─── Step 7: Write Report ───────────────────────────────────────────────────

def write_report(results):
    """Write human-readable report."""
    log("Step 7: Writing report...")

    lines = []
    lines.append("# Experiment 2C-01: SAE Structural Changes → Corporate Event Prediction\n")
    lines.append(f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")

    lines.append(f"## Verdict: {results.get('verdict', 'UNKNOWN')}\n")
    lines.append(f"{results.get('verdict_explanation', 'No explanation available.')}\n")

    lines.append("## Test Results\n")
    for event_col, test_result in results.get("tests", {}).items():
        lines.append(f"### {event_col}\n")
        if test_result.get("skipped"):
            lines.append(f"Skipped: {test_result.get('reason', 'Unknown')}\n")
            continue

        lines.append(f"- Events: {test_result['n_events']:,} / {test_result['n_total']:,} ({test_result['event_rate_pct']:.2f}%)")
        lines.append(f"- **AUC-ROC (test): {test_result['auc_test']:.4f}**")
        lines.append(f"- AUC-ROC (train): {test_result['auc_train']:.4f}")
        lines.append(f"- AUC (change_magnitude only): {test_result['auc_simple_test']:.4f}")
        lines.append(f"- Average Precision: {test_result['average_precision_test']:.4f}")
        lines.append(f"- Change magnitude coefficient: {test_result['change_magnitude_coef']:.4f} ({test_result['coef_direction']})")
        lines.append(f"- Beats random p95: {test_result['beats_random_p95']}")
        lines.append(f"- Train: {test_result['n_train']:,} ({test_result['n_train_events']:,} events)")
        lines.append(f"- Test: {test_result['n_test']:,} ({test_result['n_test_events']:,} events)\n")

    if results.get("quartile_analysis"):
        lines.append("## Quartile Analysis\n")
        for event_col, qa in results["quartile_analysis"].items():
            lines.append(f"### {event_col}\n")
            lines.append(f"- Q4/Q1 ratio: {qa['q4_q1_ratio']:.2f}x" if qa['q4_q1_ratio'] else "- Q4/Q1 ratio: N/A (Q1 rate is zero)")
            lines.append(f"- Monotonic: {qa['monotonic']}\n")
            lines.append("| Quartile | Event Rate | Events | Total |")
            lines.append("|----------|-----------|--------|-------|")
            for q_name, q_data in qa["quartiles"].items():
                lines.append(f"| {q_name} | {q_data['event_rate_pct']:.2f}% | {int(q_data['n_events'])} | {int(q_data['n_total'])} |")
            lines.append("")

    if results.get("feature_attribution"):
        fa = results["feature_attribution"]
        lines.append("## Feature Attribution\n")
        lines.append(f"- Event companies with features: {fa['n_event_companies_with_features']}")
        lines.append(f"- Control companies with features: {fa['n_control_companies_with_features']}")
        lines.append(f"- Overall mean change: events={fa['overall_event_mean_change']:.6f}, controls={fa['overall_control_mean_change']:.6f}")
        lines.append(f"- Ratio: {fa['overall_event_mean_change']/fa['overall_control_mean_change']:.2f}x\n")

    lines.append("## Interpretation\n")
    verdict = results.get("verdict", "UNKNOWN")
    if verdict == "GO":
        lines.append("SAE structural changes predict corporate events. The product thesis shifts from ")
        lines.append("'interesting intelligence' to 'early warning system.' Next step: validate with ")
        lines.append("real event data (Tier 2: NT filings, AAER, bankruptcies from SEC datasets).\n")
    elif verdict == "QUALIFIED_GO":
        lines.append("Signal exists but is moderate. Feature-level investigation needed to determine ")
        lines.append("whether specific SAE features carry the predictive weight. If specific features ")
        lines.append("reliably shift before events, the interpretability angle becomes the product.\n")
    else:
        lines.append("SAE structural changes do not predict corporate events at a useful level. ")
        lines.append("Combined with the null return backtest (Phase 2A/2B), this means the SAE signal ")
        lines.append("captures real structural relationships but does not predict actionable outcomes. ")
        lines.append("The market research verdict — 'vitamin, not painkiller' — stands.\n")

    report_text = "\n".join(lines)
    with open(OUT_REPORT, "w") as f:
        f.write(report_text)

    log(f"  Report written to {OUT_REPORT}")


# ─── Main ────────────────────────────────────────────────────────────────────

def main():
    t_start = time.time()
    log("=" * 70)
    log("Phase 2C: Do SAE Structural Changes Predict Corporate Events?")
    log("=" * 70)

    # Step 0: Inspect data
    feat_cols, comp_cols, feat_sample, comp_sample = inspect_datasets()

    # Step 1: Load company metadata
    company_df, return_cols = load_company_metadata()

    # Step 2: Load features and compute PCA
    pca_df = load_features_and_pca()

    # Join PCA features with company metadata via __index_level_0__
    # The features dataset has no CIK/year — align by shared index column.
    if "cik" not in pca_df.columns and "__index_level_0__" in pca_df.columns and "__index_level_0__" in company_df.columns:
        meta_cols_to_join = [c for c in ["__index_level_0__", "cik", "year", "sic_code", "sic_2digit"] if c in company_df.columns]
        pca_df = pca_df.merge(company_df[meta_cols_to_join], on="__index_level_0__", how="inner")
        log(f"  Joined PCA features with company metadata: {len(pca_df):,} rows")
    elif "cik" not in pca_df.columns:
        log("  WARNING: Could not join CIK/year onto PCA DataFrame — missing __index_level_0__ key")

    # Step 3: Compute change magnitudes
    mag_df = compute_change_magnitudes(pca_df)

    # Step 4: Define events
    events_df = define_events(company_df, mag_df)

    # Step 5: Statistical tests
    results = run_statistical_tests(events_df)

    # Step 6: Feature attribution (conditional on verdict)
    results = feature_attribution(events_df, pca_df, results)

    # Step 7: Save results
    with open(OUT_SUMMARY, "w") as f:
        json.dump(results, f, indent=2, default=str)
    log(f"Summary saved to {OUT_SUMMARY}")

    write_report(results)

    log(f"\nTotal time: {time.time()-t_start:.1f}s")
    log(f"Verdict: {results.get('verdict', 'UNKNOWN')}")
    log("=" * 70)


if __name__ == "__main__":
    main()

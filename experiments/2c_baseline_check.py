"""
Baseline check: prior_cum_return alone predicting next-year extreme events.
Does NOT need SAE features — just company return data.
Comparison:
  - prior_cum_return alone      (this script)
  - SAE change_magnitude alone  = 0.5598  (from 2c_01_summary.json)
  - both + SIC controls         = 0.7111  (from 2c_01_summary.json)
"""
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")
from datasets import load_dataset
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

HF_COMPANY     = "Mateusz1017/annual_reports_tokenized_llama3_logged_returns_no_null_returns_and_incomplete_descriptions_24k"
TRAIN_END      = 2010
EXTREME_THRESH = -0.693
SEED           = 42

def fit_eval(X_tr, y_tr, X_te, y_te, label):
    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(np.nan_to_num(X_tr, nan=0.0))
    X_te_s = scaler.transform(np.nan_to_num(X_te, nan=0.0))
    m = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=SEED)
    m.fit(X_tr_s, y_tr)
    a = roc_auc_score(y_te, m.predict_proba(X_te_s)[:, 1])
    print(f"  {label:45s}  AUC={a:.4f}")
    return a

print("Loading company metadata...")
ds = load_dataset(HF_COMPANY, split="train")
comp = ds.to_pandas()[["cik", "year", "logged_monthly_returns_matrix"]]
comp["cik"]  = comp["cik"].astype(int)
comp["year"] = comp["year"].astype(int)
monthly = pd.DataFrame(comp["logged_monthly_returns_matrix"].tolist(), index=comp.index)
comp["cum_log_return"]     = monthly.sum(axis=1)
comp["min_monthly_return"] = monthly.min(axis=1)
print(f"  {len(comp):,} company-years, {comp['year'].min()}-{comp['year'].max()}")

# Forward-looking label: event at year+1, feature at year t
# Merge comp with itself shifted by one year
comp_curr = comp[["cik", "year", "cum_log_return"]].copy()
comp_next = comp[["cik", "year", "min_monthly_return", "cum_log_return"]].copy()
comp_next = comp_next.rename(columns={
    "year":             "year_curr",
    "min_monthly_return": "next_min_return",
    "cum_log_return":   "next_cum_return",
})
comp_next["year_curr"] = comp_next["year_curr"] - 1  # join on signal year

df = comp_curr.merge(
    comp_next.rename(columns={"year_curr": "year"}),
    on=["cik", "year"], how="inner"
)
df["event_extreme"]  = (df["next_min_return"] < EXTREME_THRESH).astype(int)
df["event_distress"] = (df["next_cum_return"] < EXTREME_THRESH).astype(int)
df = df.dropna(subset=["cum_log_return", "next_min_return"])

print(f"  {len(df):,} obs after year+1 alignment")
print(f"  event_extreme rate:  {df['event_extreme'].mean()*100:.1f}%")
print(f"  event_distress rate: {df['event_distress'].mean()*100:.1f}%\n")

train = df[df["year"] <= TRAIN_END]
test  = df[df["year"] >  TRAIN_END]

for event_col in ["event_extreme", "event_distress"]:
    y_tr = train[event_col].values
    y_te = test[event_col].values
    print(f"── {event_col}  (train events={y_tr.sum()}, test events={y_te.sum()}) ──")
    fit_eval(train[["cum_log_return"]].values, y_tr, test[["cum_log_return"]].values, y_te,
             "prior_cum_return only")
    print(f"  {'SAE change_magnitude only (from summary.json)':45s}  AUC=0.5598  [event_extreme]")
    print(f"  {'both + SIC controls (from summary.json)':45s}  AUC=0.7111  [event_extreme]")
    print()

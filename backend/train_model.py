import os, json, joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.metrics import (precision_score, recall_score, f1_score,
                             roc_auc_score, average_precision_score,
                             confusion_matrix)
from sklearn.model_selection import train_test_split

PROC = "data/processed"
os.makedirs("models", exist_ok=True)

pq = f"{PROC}/features.parquet"
df = pd.read_parquet(pq) if os.path.exists(pq) else pd.read_csv(f"{PROC}/features.csv")
print("Loaded:", df.shape)
print("Columns:", list(df.columns))

def find(cands, contains=False):
    for c in df.columns:
        n = c.lower()
        if contains and any(k in n for k in cands):
            return c
        if not contains and n in cands:
            return c
    return None

user_col = find(["user", "user_id"])
date_col = find(["date", "day"])
label_col = find(["label", "malicious", "insider", "target"], contains=True)

if not (user_col and date_col and label_col):
    raise SystemExit(f"Could not detect columns. user={user_col}, date={date_col}, "
                     f"label={label_col}. Send me the column list printed above.")

print(f"Detected -> user: {user_col}, date: {date_col}, label: {label_col}")

df[date_col] = pd.to_datetime(df[date_col])
df = df.sort_values(date_col).reset_index(drop=True)
y = df[label_col].astype(int).clip(0, 1)

feat_cols = [c for c in df.select_dtypes(include=[np.number]).columns
             if c not in (label_col, user_col, date_col)]
X = df[feat_cols].fillna(0)
print("Features used:", len(feat_cols))
print("Malicious rows:", int(y.sum()), "of", len(y))

# Time-based split (first 70% of dates = train, rest = test)
cut = df[date_col].quantile(0.7)
tr = (df[date_col] <= cut).values
te = ~tr
if y[tr].sum() == 0 or y[te].sum() == 0:
    print("WARNING: time split has no malicious rows on one side; using stratified random split instead.")
    idx_tr, idx_te = train_test_split(np.arange(len(df)), test_size=0.3,
                                      stratify=y, random_state=42)
    tr = np.zeros(len(df), bool); tr[idx_tr] = True
    te = ~tr

Xtr, ytr, Xte, yte = X[tr], y[tr], X[te], y[te]

iso = IsolationForest(n_estimators=200, random_state=42, n_jobs=-1).fit(Xtr)
rf = RandomForestClassifier(n_estimators=300, class_weight="balanced_subsample",
                            min_samples_leaf=2, random_state=42, n_jobs=-1).fit(Xtr, ytr)

# Evaluation on the held-out test period
proba = rf.predict_proba(Xte)[:, 1]
pred = (proba >= 0.5).astype(int)
metrics = {
    "precision": float(precision_score(yte, pred, zero_division=0)),
    "recall": float(recall_score(yte, pred, zero_division=0)),
    "f1": float(f1_score(yte, pred, zero_division=0)),
    "roc_auc": float(roc_auc_score(yte, proba)),
    "avg_precision": float(average_precision_score(yte, proba)),
    "confusion_matrix": confusion_matrix(yte, pred).tolist(),
    "train_rows": int(tr.sum()), "test_rows": int(te.sum()),
    "features": feat_cols,
}
json.dump(metrics, open("models/metrics.json", "w"), indent=2)
print("\nMETRICS:", json.dumps({k: v for k, v in metrics.items() if k != "features"}, indent=2))

# Risk score 0-100 for every user-day (supervised + anomaly blend)
def mm(a):
    return (a - a.min()) / (a.max() - a.min() + 1e-9)

rf_all = rf.predict_proba(X)[:, 1]
iso_all = -iso.score_samples(X)
risk = 100 * (0.7 * mm(rf_all) + 0.3 * mm(iso_all))

# Severity by risk percentile
q85, q95, q99 = np.quantile(risk, [0.85, 0.95, 0.99])
sev = np.where(risk >= q99, "Critical",
      np.where(risk >= q95, "High",
      np.where(risk >= q85, "Medium", "Low")))

# Top 3 reasons per row: how far above average, weighted by importance
mu, sd = X.mean(), X.std().replace(0, 1)
z = ((X - mu) / sd).clip(lower=0).values
contrib = z * rf.feature_importances_
top = np.argsort(-contrib, axis=1)[:, :3]
names = np.array(feat_cols)
reasons = [", ".join(f"{names[j]} (z={z[i, j]:.1f})" for j in top[i] if z[i, j] > 0)
           for i in range(len(df))]

out = pd.DataFrame({
    "user": df[user_col], "date": df[date_col].dt.strftime("%Y-%m-%d"),
    "label": y, "risk_score": risk.round(1), "severity": sev, "reasons": reasons,
})
out.to_csv(f"{PROC}/risk_scores.csv", index=False)
joblib.dump({"rf": rf, "iso": iso, "features": feat_cols}, "models/model.joblib")

print("\nSeverity counts:\n", out["severity"].value_counts())
top_users = out.groupby("user").agg(max_risk=("risk_score", "max"),
                                    is_insider=("label", "max")) \
               .sort_values("max_risk", ascending=False).head(10)
print("\nTop 10 riskiest users:\n", top_users)
print("\nSaved: data/processed/risk_scores.csv, models/model.joblib, models/metrics.json")

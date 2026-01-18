#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import shap
import joblib
import json
import os

from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler

# ---------------------------
# Χρήσιμες μετρικές
# ---------------------------
def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))

def mape(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    eps = 1e-8
    return np.mean(np.abs((y_true - y_pred) / (y_true + eps))) * 100.0

def safe_mape(y_true, y_pred, min_denom=1.0):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    denom = np.maximum(np.abs(y_true), min_denom)
    return np.mean(np.abs((y_true - y_pred) / denom)) * 100.0

# ---------------------------
# Ρυθμίσεις
# ---------------------------
DATA_PATH = "Enriched_BDI_Dataset.csv"  # ✅ ΔΙΟΡΘΩΜΕΝΟ
TARGET = "BDI"
HORIZON = 1
OUTPUT_DIR = "artifacts_xgb_bdi"
TEST_SIZE = 0.2
RANDOM_STATE = 42

# ---------------------------
# Paper-worthy checklist (για τεκμηρίωση σε paper)
# ---------------------------
# 1) Baseline σύγκριση: random-walk/persistence στο ίδιο split & horizon.
# 2) Έλεγχος διαθεσιμότητας features: αν κάποιο feature έχει καθυστέρηση,
#    χρησιμοποίησε lag/shift για να μην υπάρχει leakage.
# 3) Walk-forward validation: αξιολόγηση σε πολλαπλά χρονικά παράθυρα.
# 4) Στατιστική σημαντικότητα: π.χ. Diebold-Mariano test έναντι baseline.

# ---------------------------
# 1. Φόρτωση Δεδομένων
# ---------------------------
df = pd.read_csv(DATA_PATH, parse_dates=["date"])
df = df.sort_values("date").reset_index(drop=True)

# ---------------------------
# 2. Feature engineering (lags/rolling) + στόχος forecast (ΔBDI)
# ---------------------------
# Χρησιμοποιούμε μόνο παρελθοντικές τιμές για lag/rolling ώστε να αποφύγουμε leakage.
df["BDI_lag1"] = df[TARGET].shift(1)
df["BDI_lag3"] = df[TARGET].shift(3)
df["BDI_lag5"] = df[TARGET].shift(5)
df["BDI_roll3"] = df[TARGET].shift(1).rolling(window=3).mean()
df["BDI_roll7"] = df[TARGET].shift(1).rolling(window=7).mean()
df["BDI_today"] = df[TARGET]

base_feature_cols = [c for c in df.columns if c not in ["date", "BDI"]]
lagged_feature_cols = []
lagged_frames = []
for col in base_feature_cols:
    if col not in ["BDI_lag1", "BDI_lag3", "BDI_lag5", "BDI_roll3", "BDI_roll7", "BDI_today"]:
        lagged_frames.append(df[col].shift(1).rename(f"{col}_lag1"))
        lagged_frames.append(df[col].shift(3).rename(f"{col}_lag3"))
        lagged_feature_cols.extend([f"{col}_lag1", f"{col}_lag3"])
if lagged_frames:
    df = pd.concat([df] + lagged_frames, axis=1)

df["y"] = df[TARGET].shift(-HORIZON) - df[TARGET]

required_cols = [
    "y",
    "BDI_lag1",
    "BDI_lag3",
    "BDI_lag5",
    "BDI_roll3",
    "BDI_roll7",
    "BDI_today",
]
required_cols.extend(lagged_feature_cols)
df = df.dropna(subset=required_cols).reset_index(drop=True)

X = df.drop(columns=["date", "BDI", "y"])
y = df["y"]

# Χρονικό split (όχι τυχαίο) για αποφυγή leakage
split_idx = int(len(df) * (1 - TEST_SIZE))
X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
dates_train, dates_test = df["date"].iloc[:split_idx], df["date"].iloc[split_idx:]

print(f"Train: {dates_train.min().date()} → {dates_train.max().date()} ({len(X_train)} δείγματα)")
print(f"Test : {dates_test.min().date()} → {dates_test.max().date()} ({len(X_test)} δείγματα)")

# ---------------------------
# 3. Προεπεξεργασία (scaling)
# ---------------------------
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ---------------------------
# 4. XGBoost Εκπαίδευση
# ---------------------------
model = XGBRegressor(
    n_estimators=500,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=RANDOM_STATE,
    n_jobs=-1,
    verbosity=1
)

model.fit(X_train_scaled, y_train)

# ---------------------------
# 5. Baseline (Random Walk / Persistence) + Αξιολόγηση
# ---------------------------
y_pred_rw_delta = np.zeros_like(y_test)
rw_metrics_delta = {
    "RMSE": rmse(y_test, y_pred_rw_delta),
    "MAE": mean_absolute_error(y_test, y_pred_rw_delta),
    "SMAPE": safe_mape(y_test, y_pred_rw_delta),
    "R2": r2_score(y_test, y_pred_rw_delta),
}

print("\nRandom Walk / Persistence (Baseline) - ΔBDI:")
for k, v in rw_metrics_delta.items():
    print(f"{k}: {v:.4f}")

y_pred = model.predict(X_test_scaled)

metrics_delta = {
    "RMSE": rmse(y_test, y_pred),
    "MAE": mean_absolute_error(y_test, y_pred),
    "SMAPE": safe_mape(y_test, y_pred),
    "R2": r2_score(y_test, y_pred),
}

print("\nΑποτελέσματα στο Test Set (XGBoost) - ΔBDI:")
for k, v in metrics_delta.items():
    print(f"{k}: {v:.4f}")

y_test_level = df[TARGET].iloc[split_idx:split_idx + len(y_test)] + y_test
y_pred_level = df[TARGET].iloc[split_idx:split_idx + len(y_test)] + y_pred
y_pred_rw_level = df[TARGET].iloc[split_idx:split_idx + len(y_test)]

rw_metrics_level = {
    "RMSE": rmse(y_test_level, y_pred_rw_level),
    "MAE": mean_absolute_error(y_test_level, y_pred_rw_level),
    "MAPE": mape(y_test_level, y_pred_rw_level),
    "R2": r2_score(y_test_level, y_pred_rw_level),
}
metrics_level = {
    "RMSE": rmse(y_test_level, y_pred_level),
    "MAE": mean_absolute_error(y_test_level, y_pred_level),
    "MAPE": mape(y_test_level, y_pred_level),
    "R2": r2_score(y_test_level, y_pred_level),
}

print("\nRandom Walk / Persistence (Baseline) - Level:")
for k, v in rw_metrics_level.items():
    print(f"{k}: {v:.4f}")

print("\nΑποτελέσματα στο Test Set (XGBoost) - Level:")
for k, v in metrics_level.items():
    print(f"{k}: {v:.4f}")

# ---------------------------
# 6. Γράφημα πραγματικών vs προβλεπόμενων
# ---------------------------
os.makedirs(OUTPUT_DIR, exist_ok=True)

plt.figure(figsize=(12, 5))
plt.plot(dates_test, y_test_level, label="Πραγματικό BDI", linewidth=2)
plt.plot(dates_test, y_pred_level, label="Πρόβλεψη XGBoost", linewidth=2)
plt.xlabel("Ημερομηνία")
plt.ylabel("BDI")
plt.title(f"Πρόβλεψη Baltic Dry Index (t+{HORIZON}) με XGBoost")
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "actual_vs_pred_xgb.png"), dpi=200)
plt.close()

# ---------------------------
# 7. SHAP Explainability
# ---------------------------
explainer = shap.Explainer(model)
shap_values = explainer(X_test_scaled)

# Beeswarm plot
plt.figure()
shap.summary_plot(shap_values, X_test, show=False)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "shap_beeswarm.png"), dpi=200)
plt.close()

# Bar plot
plt.figure()
shap.summary_plot(shap_values, X_test, show=False, plot_type="bar")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "shap_bar.png"), dpi=200)
plt.close()

# ---------------------------
# 8. Αποθήκευση Μοντέλου και Αποτελεσμάτων
# ---------------------------
joblib.dump(model, os.path.join(OUTPUT_DIR, "xgb_bdi_model.joblib"))
joblib.dump(scaler, os.path.join(OUTPUT_DIR, "scaler.joblib"))

with open(os.path.join(OUTPUT_DIR, "test_metrics.json"), "w") as f:
    json.dump(
        {
            "random_walk_delta": rw_metrics_delta,
            "xgboost_delta": metrics_delta,
            "random_walk_level": rw_metrics_level,
            "xgboost_level": metrics_level,
        },
        f,
        indent=2,
    )

print("\n✅ Τα αποτελέσματα αποθηκεύτηκαν στον φάκελο:", OUTPUT_DIR)

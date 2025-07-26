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
from sklearn.model_selection import train_test_split
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

# ---------------------------
# Ρυθμίσεις
# ---------------------------
DATA_PATH = "enriched_bdi_dataset.csv"
TARGET = "BDI"
OUTPUT_DIR = "artifacts_xgb_bdi"
TEST_SIZE = 0.2
RANDOM_STATE = 42

# ---------------------------
# 1. Φόρτωση Δεδομένων
# ---------------------------
df = pd.read_csv(DATA_PATH, parse_dates=["date"])
df = df.sort_values("date").reset_index(drop=True)

# ---------------------------
# 2. Ορισμός Χ και y
# ---------------------------
X = df.drop(columns=["date", "BDI"])
y = df[TARGET]

# Χρονικό split (όχι τυχαίο) για αποφυγή leakage
split_idx = int(len(df) * (1 - TEST_SIZE))
X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
dates_train, dates_test = df["date"].iloc[:split_idx], df["date"].iloc[split_idx:]

print(f"Train: {dates_train.min().date()} → {dates_train.max().date()} ({len(X_train)} δείγματα)")
print(f"Test : {dates_test.min().date()} → {dates_test.max().date()} ({len(X_test)} δείγματα)")

# ---------------------------
# 3. Προεπεξεργασία (π.χ. scaling)
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
# 5. Αξιολόγηση
# ---------------------------
y_pred = model.predict(X_test_scaled)

metrics = {
    "RMSE": rmse(y_test, y_pred),
    "MAE": mean_absolute_error(y_test, y_pred),
    "MAPE": mape(y_test, y_pred),
    "R2": r2_score(y_test, y_pred)
}

print("\nΑποτελέσματα στο Test Set:")
for k, v in metrics.items():
    print(f"{k}: {v:.4f}")

# ---------------------------
# 6. Γραφική παράσταση πραγματικών vs προβλεπόμενων
# ---------------------------
os.makedirs(OUTPUT_DIR, exist_ok=True)

plt.figure(figsize=(12, 5))
plt.plot(dates_test, y_test, label="Πραγματικό BDI", linewidth=2)
plt.plot(dates_test, y_pred, label="Πρόβλεψη XGBoost", linewidth=2)
plt.xlabel("Ημερομηνία")
plt.ylabel("BDI")
plt.title("Πρόβλεψη Baltic Dry Index με XGBoost")
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
# 8. Αποθήκευση Μοντέλου και αποτελεσμάτων
# ---------------------------
joblib.dump(model, os.path.join(OUTPUT_DIR, "xgb_bdi_model.joblib"))
joblib.dump(scaler, os.path.join(OUTPUT_DIR, "scaler.joblib"))

with open(os.path.join(OUTPUT_DIR, "test_metrics.json"), "w") as f:
    json.dump(metrics, f, indent=2)

print("\n✅ Τα αποτελέσματα αποθηκεύτηκαν στον φάκελο:", OUTPUT_DIR)
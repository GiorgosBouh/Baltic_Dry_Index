#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import json
import joblib

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import shap
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


# ---------------------------
# Ρυθμίσεις
# ---------------------------
DATA_PATH = "Enriched_BDI_Dataset.csv"
DATE_COL = "date"
TARGET = "BDI"
HORIZON = 1
OUTPUT_DIR = "artifacts_xgb_bdi"
TEST_SIZE = 0.2
RANDOM_STATE = 42


# ---------------------------
# 1. Φόρτωση Δεδομένων
# ---------------------------
if not os.path.exists(DATA_PATH):
    raise FileNotFoundError(f"Δεν βρέθηκε το αρχείο: {DATA_PATH}")

df = pd.read_csv(DATA_PATH)

# Έλεγχος βασικών στηλών
missing = [c for c in [DATE_COL, TARGET] if c not in df.columns]
if missing:
    raise ValueError(f"Λείπουν στήλες από το dataset: {missing}. Διαθέσιμες: {list(df.columns)}")

df[DATE_COL] = pd.to_datetime(df[DATE_COL], errors="coerce")
df = df.dropna(subset=[DATE_COL]).copy()
df = df.sort_values(DATE_COL).reset_index(drop=True)

# Βεβαιώσου ότι ο στόχος είναι numeric
df[TARGET] = pd.to_numeric(df[TARGET], errors="coerce")
df = df.dropna(subset=[TARGET]).reset_index(drop=True)


# ---------------------------
# 2. Feature engineering (lags/rolling) + στόχος forecast (ΔBDI)
# ---------------------------
# Lag/rolling του BDI (με shift(1) στα rolling για να αποφεύγεται leakage)
df["BDI_lag1"] = df[TARGET].shift(1)
df["BDI_lag3"] = df[TARGET].shift(3)
df["BDI_lag5"] = df[TARGET].shift(5)
df["BDI_roll3"] = df[TARGET].shift(1).rolling(window=3, min_periods=3).mean()
df["BDI_roll7"] = df[TARGET].shift(1).rolling(window=7, min_periods=7).mean()
df["BDI_today"] = df[TARGET]

# Θα κάνουμε lags ΜΟΝΟ για numeric base-features (εκτός date/BDI)
base_feature_cols = [c for c in df.columns if c not in [DATE_COL, TARGET]]
numeric_base_cols = df[base_feature_cols].select_dtypes(include=[np.number]).columns.tolist()

# Μην ξανα-lag-άρεις τα ήδη φτιαγμένα features
already_bdi_feats = {"BDI_lag1", "BDI_lag3", "BDI_lag5", "BDI_roll3", "BDI_roll7", "BDI_today"}

lagged_feature_cols = []
for col in numeric_base_cols:
    if col in already_bdi_feats:
        continue
    df[f"{col}_lag1"] = df[col].shift(1)
    df[f"{col}_lag3"] = df[col].shift(3)
    lagged_feature_cols.extend([f"{col}_lag1", f"{col}_lag3"])

# Στόχος: ΔBDI = BDI(t+H) - BDI(t)
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

# ---------------------------
# 3. Prepare X, y (μόνο numeric features)
# ---------------------------
X = df.drop(columns=[DATE_COL, TARGET, "y"]).copy()
X = X.select_dtypes(include=[np.number])  # safety
y = df["y"].copy()

if X.shape[1] == 0:
    raise ValueError("Δεν βρέθηκαν numeric features για εκπαίδευση. Έλεγξε το dataset.")

# Χρονικό split
split_idx = int(len(df) * (1 - TEST_SIZE))
if split_idx <= 0 or split_idx >= len(df):
    raise ValueError("Άκυρο split. Ρύθμισε το TEST_SIZE ή έλεγξε το μέγεθος δεδομένων.")

X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
dates_train, dates_test = df[DATE_COL].iloc[:split_idx], df[DATE_COL].iloc[split_idx:]

print(f"Train: {dates_train.min().date()} → {dates_train.max().date()} ({len(X_train)} δείγματα)")
print(f"Test : {dates_test.min().date()} → {dates_test.max().date()} ({len(X_test)} δείγματα)")


# ---------------------------
# 4. Scaling
# ---------------------------
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Κράτα DataFrame εκδοχή για feature names (για SHAP/plots)
X_train_scaled_df = pd.DataFrame(X_train_scaled, columns=X.columns, index=X_train.index)
X_test_scaled_df = pd.DataFrame(X_test_scaled, columns=X.columns, index=X_test.index)


# ---------------------------
# 5. XGBoost Training
# ---------------------------
model = XGBRegressor(
    n_estimators=500,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=RANDOM_STATE,
    n_jobs=-1,
    verbosity=1,
)

model.fit(X_train_scaled_df, y_train)


# ---------------------------
# 6. Baseline + Αξιολόγηση
# ---------------------------
# Baseline για ΔBDI: persistence στο delta-space => 0 (δηλ. "δεν αλλάζει")
y_pred_rw_delta = np.zeros_like(y_test.to_numpy())

rw_metrics_delta = {
    "RMSE": float(rmse(y_test, y_pred_rw_delta)),
    "MAE": float(mean_absolute_error(y_test, y_pred_rw_delta)),
    "MAPE": float(mape(y_test, y_pred_rw_delta)),
    "R2": float(r2_score(y_test, y_pred_rw_delta)),
}

print("\nRandom Walk / Persistence (Baseline) - ΔBDI:")
for k, v in rw_metrics_delta.items():
    print(f"{k}: {v:.4f}")

# Πρόβλεψη XGBoost για ΔBDI
y_pred_delta = model.predict(X_test_scaled_df)

metrics_delta = {
    "RMSE": float(rmse(y_test, y_pred_delta)),
    "MAE": float(mean_absolute_error(y_test, y_pred_delta)),
    "MAPE": float(mape(y_test, y_pred_delta)),
    "R2": float(r2_score(y_test, y_pred_delta)),
}

print("\nΑποτελέσματα στο Test Set (XGBoost) - ΔBDI:")
for k, v in metrics_delta.items():
    print(f"{k}: {v:.4f}")

# Μετατροπή σε επίπεδο (level): BDI_pred(t+H) = BDI(t) + Δ_pred
base_level = df[TARGET].iloc[split_idx : split_idx + len(y_test)].to_numpy()
y_test_level = base_level + y_test.to_numpy()
y_pred_level = base_level + y_pred_delta
y_pred_rw_level = base_level  # persistence σε level

rw_metrics_level = {
    "RMSE": float(rmse(y_test_level, y_pred_rw_level)),
    "MAE": float(mean_absolute_error(y_test_level, y_pred_rw_level)),
    "MAPE": float(mape(y_test_level, y_pred_rw_level)),
    "R2": float(r2_score(y_test_level, y_pred_rw_level)),
}

metrics_level = {
    "RMSE": float(rmse(y_test_level, y_pred_level)),
    "MAE": float(mean_absolute_error(y_test_level, y_pred_level)),
    "MAPE": float(mape(y_test_level, y_pred_level)),
    "R2": float(r2_score(y_test_level, y_pred_level)),
}

print("\nRandom Walk / Persistence (Baseline) - Level:")
for k, v in rw_metrics_level.items():
    print(f"{k}: {v:.4f}")

print("\nΑποτελέσματα στο Test Set (XGBoost) - Level:")
for k, v in metrics_level.items():
    print(f"{k}: {v:.4f}")


# ---------------------------
# 7. Plot πραγματικό vs πρόβλεψη (Level)
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
# 8. SHAP Explainability (robust)
# ---------------------------
# Προτιμάμε TreeExplainer για XGBoost + κρατάμε feature names.
try:
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test_scaled_df)

    plt.figure()
    shap.summary_plot(shap_values, X_test_scaled_df, show=False)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "shap_beeswarm.png"), dpi=200)
    plt.close()

    plt.figure()
    shap.summary_plot(shap_values, X_test_scaled_df, show=False, plot_type="bar")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "shap_bar.png"), dpi=200)
    plt.close()
except Exception as e:
    print(f"\n⚠️ SHAP απέτυχε: {e}")
    print("   (Το training/metrics ολοκληρώθηκαν κανονικά.)")


# ---------------------------
# 9. Save artifacts
# ---------------------------
joblib.dump(model, os.path.join(OUTPUT_DIR, "xgb_bdi_model.joblib"))
joblib.dump(scaler, os.path.join(OUTPUT_DIR, "scaler.joblib"))

with open(os.path.join(OUTPUT_DIR, "test_metrics.json"), "w", encoding="utf-8") as f:
    json.dump(
        {
            "random_walk_delta": rw_metrics_delta,
            "xgboost_delta": metrics_delta,
            "random_walk_level": rw_metrics_level,
            "xgboost_level": metrics_level,
            "horizon": HORIZON,
            "test_size": TEST_SIZE,
        },
        f,
        indent=2,
        ensure_ascii=False,
    )

print("\n✅ Τα αποτελέσματα αποθηκεύτηκαν στον φάκελο:", OUTPUT_DIR)
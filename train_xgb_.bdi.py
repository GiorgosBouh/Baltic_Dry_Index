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

# Βεβαιώσου ότι υπάρχουν βασικές στήλες
missing = [c for c in [DATE_COL, TARGET] if c not in df.columns]
if missing:
    raise ValueError(f"Λείπουν στήλες από το dataset: {missing}. Διαθέσιμες: {list(df.columns)}")

df[DATE_COL] = pd.to_datetime(df[DATE_COL], errors="coerce")
df = df.dropna(subset=[DATE_COL]).copy()

df = df.sort_values(DATE_COL).reset_index(drop=True)

# Μετατροπή στόχου σε numeric (σε περίπτωση που είναι string)
df[TARGET] = pd.to_numeric(df[TARGET], errors="coerce")
df = df.dropna(subset=[TARGET]).reset_index(drop=True)


# ---------------------------
# 2. Feature engineering (lags/rolling) + στόχος forecast
# ---------------------------
# Χρησιμοποιούμε μόνο παρελθοντικές τιμές για lag/rolling ώστε να αποφύγουμε leakage.
df["BDI_lag1"] = df[TARGET].shift(1)
df["BDI_lag3"] = df[TARGET].shift(3)
df["BDI_lag5"] = df[TARGET].shift(5)
df["BDI_roll3"] = df[TARGET].shift(1).rolling(window=3, min_periods=3).mean()
df["BDI_roll7"] = df[TARGET].shift(1).rolling(window=7, min_periods=7).mean()

# (Προαιρετικό) κρατάμε και το "σήμερα" ως feature - δεν είναι leakage γιατί προβλέπουμε t+HORIZON
# αλλά αν το horizon=1 και τα features προέρχονται από την ίδια μέρα, είναι ok μόνο αν
# οι υπόλοιπες features είναι διαθέσιμες στο τέλος της μέρας. Εδώ το αφήνουμε αλλά μπορείς να το αφαιρέσεις.
df["BDI_today"] = df[TARGET]

# Στόχος πρόβλεψης: t + HORIZON
df["y"] = df[TARGET].shift(-HORIZON)

# Κράτα μόνο γραμμές με πλήρη features + στόχο
req_cols = ["y", "BDI_lag1", "BDI_lag3", "BDI_lag5", "BDI_roll3", "BDI_roll7", "BDI_today"]
df = df.dropna(subset=req_cols).reset_index(drop=True)

# ---------------------------
# 3. Prepare X, y (μόνο αριθμητικά features)
# ---------------------------
# Αφαίρεση date/target/y
X = df.drop(columns=[DATE_COL, TARGET, "y"]).copy()
y = df["y"].copy()

# Κράτα μόνο numeric columns (για να μην σκάσει ο StandardScaler)
X = X.select_dtypes(include=[np.number])

if X.shape[1] == 0:
    raise ValueError("Δεν βρέθηκαν αριθμητικά features μετά το preprocessing. Έλεγξε το dataset.")

# Χρονικό split (όχι τυχαίο) για αποφυγή leakage
split_idx = int(len(df) * (1 - TEST_SIZE))
if split_idx <= 0 or split_idx >= len(df):
    raise ValueError("Το TEST_SIZE οδηγεί σε άκυρο split. Ρύθμισε το TEST_SIZE.")

X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
dates_train = df[DATE_COL].iloc[:split_idx]
dates_test = df[DATE_COL].iloc[split_idx:]

print(f"Train: {dates_train.min().date()} → {dates_train.max().date()} ({len(X_train)} δείγματα)")
print(f"Test : {dates_test.min().date()} → {dates_test.max().date()} ({len(X_test)} δείγματα)")


# ---------------------------
# 4. Προεπεξεργασία (scaling)
# ---------------------------
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Για SHAP/plots: κράτα DataFrames με feature names
X_train_scaled_df = pd.DataFrame(X_train_scaled, columns=X.columns, index=X_train.index)
X_test_scaled_df = pd.DataFrame(X_test_scaled, columns=X.columns, index=X_test.index)


# ---------------------------
# 5. XGBoost Εκπαίδευση
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

model.fit(X_train_scaled_df, y_train)


# ---------------------------
# 6. Baseline (Random Walk / Persistence) + Αξιολόγηση
# ---------------------------
# Baseline: πρόβλεψη y(t+H) = BDI(t)
# Μετά τα dropna, το df είναι ευθυγραμμισμένο και το "BDI_today" είναι η BDI(t)
y_pred_rw = df["BDI_today"].iloc[split_idx:split_idx + len(y_test)].to_numpy()

rw_metrics = {
    "RMSE": float(rmse(y_test, y_pred_rw)),
    "MAE": float(mean_absolute_error(y_test, y_pred_rw)),
    "MAPE": float(mape(y_test, y_pred_rw)),
    "R2": float(r2_score(y_test, y_pred_rw)),
}

print("\nRandom Walk / Persistence (Baseline):")
for k, v in rw_metrics.items():
    print(f"{k}: {v:.4f}")

# Πρόβλεψη XGBoost
y_pred = model.predict(X_test_scaled_df)

metrics = {
    "RMSE": float(rmse(y_test, y_pred)),
    "MAE": float(mean_absolute_error(y_test, y_pred)),
    "MAPE": float(mape(y_test, y_pred)),
    "R2": float(r2_score(y_test, y_pred)),
}

print("\nΑποτελέσματα στο Test Set (XGBoost):")
for k, v in metrics.items():
    print(f"{k}: {v:.4f}")


# ---------------------------
# 7. Γράφημα πραγματικών vs προβλεπόμενων
# ---------------------------
os.makedirs(OUTPUT_DIR, exist_ok=True)

plt.figure(figsize=(12, 5))
plt.plot(dates_test, y_test, label="Πραγματικό BDI", linewidth=2)
plt.plot(dates_test, y_pred, label="Πρόβλεψη XGBoost", linewidth=2)
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
# Για tree models, TreeExplainer είναι πιο σταθερό.
# Υπολογίζουμε SHAP πάνω στα scaled features με ονόματα στηλών.
try:
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test_scaled_df)

    # Beeswarm
    plt.figure()
    shap.summary_plot(shap_values, X_test_scaled_df, show=False)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "shap_beeswarm.png"), dpi=200)
    plt.close()

    # Bar
    plt.figure()
    shap.summary_plot(shap_values, X_test_scaled_df, show=False, plot_type="bar")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "shap_bar.png"), dpi=200)
    plt.close()
except Exception as e:
    print(f"\n⚠️ SHAP απέτυχε να τρέξει: {e}")
    print("   (Το training/metrics ολοκληρώθηκαν κανονικά. Έλεγξε την έκδοση shap/xgboost αν χρειάζεται.)")


# ---------------------------
# 9. Αποθήκευση Μοντέλου και Αποτελεσμάτων
# ---------------------------
joblib.dump(model, os.path.join(OUTPUT_DIR, "xgb_bdi_model.joblib"))
joblib.dump(scaler, os.path.join(OUTPUT_DIR, "scaler.joblib"))

with open(os.path.join(OUTPUT_DIR, "test_metrics.json"), "w", encoding="utf-8") as f:
    json.dump({"random_walk": rw_metrics, "xgboost": metrics}, f, indent=2, ensure_ascii=False)

print("\n✅ Τα αποτελέσματα αποθηκεύτηκαν στον φάκελο:", OUTPUT_DIR)
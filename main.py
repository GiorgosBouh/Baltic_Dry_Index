#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Random Forest για πρόβλεψη Baltic Dry Index (BDI) με SHAP.
- Χωρίς data leakage (χρονική οριοθέτηση, pipeline).
- Πρόβλεψη BDI (y) για horizon h ημέρες μπροστά (default: 1).
- Αξιολόγηση με RMSE, MAE, MAPE, R2 + σύγκριση με naive baseline.
- SHAP (TreeExplainer) για ερμηνευσιμότητα.
"""

import argparse
import os
import json
import joblib
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import TimeSeriesSplit, RandomizedSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

import shap

RANDOM_STATE = 42

# ---------------------------
# Helper metrics
# ---------------------------
def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))

def mape(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    eps = 1e-8
    return np.mean(np.abs((y_true - y_pred) / (y_true + eps))) * 100.0

def print_metrics(y_true, y_pred, prefix=""):
    _rmse = rmse(y_true, y_pred)
    _mae = mean_absolute_error(y_true, y_pred)
    _mape = mape(y_true, y_pred)
    _r2 = r2_score(y_true, y_pred)
    print(f"{prefix}RMSE: { _rmse:,.4f}")
    print(f"{prefix}MAE : { _mae:,.4f}")
    print(f"{prefix}MAPE: { _mape:,.2f}%")
    print(f"{prefix}R2  : { _r2:,.4f}")
    return {"rmse": _rmse, "mae": _mae, "mape": _mape, "r2": _r2}

# ---------------------------
# Main
# ---------------------------
def main(args):
    # ---------------------------
    # 1) Load data
    # ---------------------------
    df = pd.read_excel(args.data_path, sheet_name="Sheet1", engine="openpyxl")
    # Βεβαιώσου ότι η ημερομηνία είναι datetime
    if not np.issubdtype(df['date'].dtype, np.datetime64):
        df['date'] = pd.to_datetime(df['date'])

    df = df.sort_values('date').reset_index(drop=True)

    # ---------------------------
    # 2) Define target & features
    # ---------------------------
    target_col = "BDI"
    feature_cols = [
        "Port Call - Number",
        "global_trade",
        "Total Bulkcarrier Bosphorus Strait Transits - DWT million",
        "Total Bulkcarrier Suez Canal Transits - DWT million",
        "Total Bulkcarrier Panama Canal Transits - DWT million",
        "Commodity Index",
        "Bulkcarrier Average Speed - Knots",
        "daily_policy_index",
        "sp500",
    ]

    # Επεξεργασία κενών τιμών (π.χ. Commodity Index) με forward/backward fill πριν το split
    # (Αυτό ΔΕΝ δημιουργεί leakage γιατί δεν χρησιμοποιεί πληροφορία από το μέλλον στις μεταβλητές-χαρακτηριστικά;
    # forward fill χρησιμοποιεί παρελθόν/τρέχον. Για ασφάλεια μπορούμε να κάνουμε ffill → bfill.)
    df[feature_cols] = df[feature_cols].ffill().bfill()

    # ---------------------------
    # 3) Create next-day (or next-h) target to avoid same-day leakage
    # ---------------------------
    # y_t = BDI_(t + horizon)
    horizon = args.horizon
    df["y"] = df[target_col].shift(-horizon)

    # Drop last h rows where y is NaN
    df = df.dropna(subset=["y"]).reset_index(drop=True)

    X = df[feature_cols].copy()
    y = df["y"].copy()
    dates = df["date"].copy()
    n_samples = len(df)

    # ---------------------------
    # 4) Train/test split preserving time order (last test_size%)
    # ---------------------------
    test_size = int(n_samples * args.test_size)
    if test_size == 0:
        raise ValueError("Test size is 0 samples. Increase dataset or test_size fraction.")

    split_idx = n_samples - test_size
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
    dates_train, dates_test = dates.iloc[:split_idx], dates.iloc[split_idx:]

    print(f"Train samples: {len(X_train)}, Test samples: {len(X_test)}")
    print(f"Train period: {dates_train.min().date()} → {dates_train.max().date()}")
    print(f"Test  period: {dates_test.min().date()} → {dates_test.max().date()}")

    # ---------------------------
    # 5) Baseline (naive) model
    # Προβλέπουμε ότι y_t ≈ BDI_t (δηλ. η τιμή της ίδιας μέρας, άρα για horizon=1 γίνεται previous-day BDI του στόχου)
    # Για δίκαιη σύγκριση, ο naive predictor για y_{t} (BDI_{t+h}) είναι απλά y_{t-1} (shifted)
    # ---------------------------
    y_test_naive = df[target_col].shift(-(horizon+1)).iloc[split_idx:]  # (BDI_{t+h-1}) -> λίγο tricky
    # Πιο απλά: baseline: predict y(t) = last available y_train's last value (πολύ φτωχή baseline)
    # Θα δώσουμε και την απλούστερη baseline: y_pred_naive = previous available y value shifted by 1 in test
    y_pred_naive = y_test.shift(1)
    # Επειδή το πρώτο γίνεται NaN, συμπληρώνουμε με το πρώτο διαθέσιμο
    y_pred_naive.iloc[0] = y_train.iloc[-1]
    baseline_metrics = print_metrics(y_test.values, y_pred_naive.values, prefix="[Baseline] ")

    # ---------------------------
    # 6) Build pipeline (Imputer -> Scaler -> RF)
    # ---------------------------
    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())  # RF δεν το χρειάζεται, αλλά δεν βλάπτει
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, feature_cols),
        ],
        remainder="drop"
    )

    rf = RandomForestRegressor(random_state=RANDOM_STATE, n_jobs=-1)

    pipe = Pipeline(steps=[
        ("preprocess", preprocessor),
        ("model", rf)
    ])

    # ---------------------------
    # 7) Hyperparameter tuning (TimeSeriesSplit)
    # ---------------------------
    tscv = TimeSeriesSplit(n_splits=args.cv_splits)
    param_distributions = {
        "model__n_estimators": [300, 500, 800, 1000],
        "model__max_depth": [None, 5, 10, 15, 25, 35],
        "model__min_samples_split": [2, 5, 10, 20],
        "model__min_samples_leaf": [1, 2, 4, 8],
        "model__max_features": ["sqrt", "log2", 0.5, 0.7, 1.0],
    }

    search = RandomizedSearchCV(
        pipe,
        param_distributions=param_distributions,
        n_iter=args.n_iter,
        cv=tscv,
        scoring="neg_mean_squared_error",
        n_jobs=-1,
        random_state=RANDOM_STATE,
        verbose=1
    )

    search.fit(X_train, y_train)
    best_model = search.best_estimator_
    print("\nBest params:")
    print(search.best_params_)

    # ---------------------------
    # 8) Evaluate on test
    # ---------------------------
    y_pred = best_model.predict(X_test)
    test_metrics = print_metrics(y_test.values, y_pred, prefix="[Test] ")

    # ---------------------------
    # 9) Plot Actual vs Predicted
    # ---------------------------
    os.makedirs(args.output_dir, exist_ok=True)

    plt.figure(figsize=(12, 5))
    plt.plot(dates_test, y_test.values, label="Actual", linewidth=2)
    plt.plot(dates_test, y_pred, label="Predicted", linewidth=2, alpha=0.8)
    plt.title(f"BDI Prediction (horizon={horizon}) - Random Forest")
    plt.xlabel("Date")
    plt.ylabel("BDI")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(args.output_dir, "actual_vs_pred.png"), dpi=200)
    plt.close()

    # ---------------------------
    # 10) SHAP Explainability
    # Θα κάνουμε explain τον τελικό RF (όχι όλο το pipeline).
    # Παίρνουμε το random forest από το pipeline και τα transformed features.
    # ---------------------------
    # Transform train features με το ίδο preprocessor για να ταΐσουμε τον explainer
    X_train_transformed = best_model.named_steps["preprocess"].transform(X_train)
    rf_model = best_model.named_steps["model"]

    # Επιλέγουμε ένα υποσύνολο για ταχύτητα
    nsamples = min(args.shap_samples, X_train_transformed.shape[0])
    idx_sample = np.random.RandomState(RANDOM_STATE).choice(X_train_transformed.shape[0], nsamples, replace=False)
    X_shap = X_train_transformed[idx_sample]

    explainer = shap.TreeExplainer(rf_model)
    shap_values = explainer.shap_values(X_shap)

    # SHAP summary plot (bar)
    shap_fig = plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_values, X_shap, feature_names=feature_cols, show=False, plot_type="bar")
    plt.tight_layout()
    shap_fig.savefig(os.path.join(args.output_dir, "shap_summary_bar.png"), dpi=200)
    plt.close(shap_fig)

    # SHAP summary plot (beeswarm)
    shap_fig2 = plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_values, X_shap, feature_names=feature_cols, show=False)
    plt.tight_layout()
    shap_fig2.savefig(os.path.join(args.output_dir, "shap_summary_beeswarm.png"), dpi=200)
    plt.close(shap_fig2)

    # ---------------------------
    # 11) Feature Importances (από RF)
    # ---------------------------
    fi = rf_model.feature_importances_
    fi_df = pd.DataFrame({"feature": feature_cols, "importance": fi}).sort_values("importance", ascending=False)
    fi_df.to_csv(os.path.join(args.output_dir, "feature_importances.csv"), index=False)

    plt.figure(figsize=(8, 5))
    plt.barh(fi_df["feature"], fi_df["importance"])
    plt.gca().invert_yaxis()
    plt.title("Random Forest Feature Importances")
    plt.tight_layout()
    plt.savefig(os.path.join(args.output_dir, "rf_feature_importances.png"), dpi=200)
    plt.close()

    # ---------------------------
    # 12) Save artifacts
    # ---------------------------
    joblib.dump(best_model, os.path.join(args.output_dir, "rf_bdi_model.joblib"))

    results = {
        "horizon": horizon,
        "train_start": str(dates_train.min().date()),
        "train_end": str(dates_train.max().date()),
        "test_start": str(dates_test.min().date()),
        "test_end": str(dates_test.max().date()),
        "best_params": search.best_params_,
        "baseline": baseline_metrics,
        "test_metrics": test_metrics
    }
    with open(os.path.join(args.output_dir, "results.json"), "w") as f:
        json.dump(results, f, indent=2)

    print("\nArtifacts saved to:", args.output_dir)
    print(" - rf_bdi_model.joblib")
    print(" - results.json")
    print(" - actual_vs_pred.png")
    print(" - shap_summary_bar.png")
    print(" - shap_summary_beeswarm.png")
    print(" - rf_feature_importances.png")
    print(" - feature_importances.csv")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predict BDI with Random Forest + SHAP (no leakage, time-series safe).")
    parser.add_argument("--data_path", type=str, default="Final Data File.xlsx",
                        help="Path to the Excel file with the data.")
    parser.add_argument("--horizon", type=int, default=1,
                        help="Forecast horizon in days (predict BDI t+h).")
    parser.add_argument("--test_size", type=float, default=0.2,
                        help="Fraction for test split (chronological).")
    parser.add_argument("--cv_splits", type=int, default=5,
                        help="Number of TimeSeriesSplit folds for CV.")
    parser.add_argument("--n_iter", type=int, default=40,
                        help="RandomizedSearchCV iterations.")
    parser.add_argument("--output_dir", type=str, default="artifacts_rf_bdi",
                        help="Directory to save artifacts (model, plots, metrics).")
    parser.add_argument("--shap_samples", type=int, default=1000,
                        help="Number of training samples to use for SHAP calculation (for speed).")

    args = parser.parse_args()
    main(args)
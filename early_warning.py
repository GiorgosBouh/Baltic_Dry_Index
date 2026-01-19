#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Early Warning / Event Forecasting for Baltic Dry Index (BDI)

Question:
- NOT: "What will be the BDI value at t+H?"
- BUT: "Is a large move likely within the next H steps?"

The model outputs a probability of an upcoming large movement.
"""

import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from sklearn.metrics import average_precision_score, brier_score_loss
from sklearn.preprocessing import StandardScaler

# =========================
# Configuration
# =========================
DATA_PATH = "Enriched_BDI_Dataset.csv"
DATE_COL = "date"
TARGET_COL = "BDI"

HORIZON = 10                 # forecasting horizon (steps)
EVENT_QUANTILE = 0.80        # top 20% absolute returns = event
MIN_EXOG_LAG = 14            # realistic delay for exogenous data
TEST_WINDOW = 120
MIN_TRAIN_SIZE = 365
PURGE_GAP = HORIZON
N_SPLITS = 5
RANDOM_STATE = 42

# =========================
# Utility functions
# =========================
def compute_future_return(series, h):
    return np.log(series.shift(-h) / series)

def directional_accuracy(y_true, prob, threshold=0.5):
    pred = (prob >= threshold).astype(int)
    return (pred == y_true).mean()

# =========================
# Load & prepare data
# =========================
df = pd.read_csv(DATA_PATH, parse_dates=[DATE_COL])
df = df.sort_values(DATE_COL).reset_index(drop=True)

# Future return
df["ret_H"] = compute_future_return(df[TARGET_COL], HORIZON)

# =========================
# Event definition (TRAIN-ONLY threshold)
# =========================
abs_ret = df["ret_H"].abs()
event_threshold = abs_ret.quantile(EVENT_QUANTILE)
df["event"] = (abs_ret >= event_threshold).astype(int)

# =========================
# Feature selection
# =========================
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

feature_cols = []
for col in numeric_cols:
    if col in ["ret_H", "event"]:
        continue
    if "lag" in col.lower() or "roll" in col.lower() or col == TARGET_COL:
        # enforce realistic delay for exogenous variables
        if col != TARGET_COL:
            if any(f"lag{l}" in col.lower() for l in range(MIN_EXOG_LAG)):
                feature_cols.append(col)
        else:
            feature_cols.append(col)

X_all = df[feature_cols]
y_all = df["event"]

# Drop NaNs from lagging
mask = X_all.notna().all(axis=1)
X_all = X_all[mask]
y_all = y_all[mask]
dates = df.loc[X_all.index, DATE_COL]

print(f"Rows: {len(X_all)}, Features: {len(feature_cols)}")
print(f"Event rate: {y_all.mean():.3f}")

# =========================
# Walk-forward evaluation
# =========================
start = MIN_TRAIN_SIZE
split = 1

results = []

while start + TEST_WINDOW <= len(X_all) and split <= N_SPLITS:
    train_end = start
    test_start = train_end + PURGE_GAP
    test_end = test_start + TEST_WINDOW

    if test_end > len(X_all):
        break

    X_train = X_all.iloc[:train_end]
    y_train = y_all.iloc[:train_end]

    X_test = X_all.iloc[test_start:test_end]
    y_test = y_all.iloc[test_start:test_end]

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    model = XGBClassifier(
        n_estimators=500,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="binary:logistic",
        eval_metric="logloss",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )

    model.fit(X_train_s, y_train)

    prob = model.predict_proba(X_test_s)[:, 1]

    ap = average_precision_score(y_test, prob)
    brier = brier_score_loss(y_test, prob)
    dir_acc = directional_accuracy(y_test, prob)

    print(f"\nSplit {split}")
    print(f"Event rate (test): {y_test.mean():.3f}")
    print(f"AUPRC: {ap:.3f}")
    print(f"Brier score: {brier:.3f}")
    print(f"Direction accuracy: {dir_acc:.3f}")

    results.append({
        "split": split,
        "AUPRC": ap,
        "Brier": brier,
        "DirAcc": dir_acc
    })

    start += TEST_WINDOW
    split += 1

# =========================
# Summary
# =========================
res_df = pd.DataFrame(results)
print("\n===== SUMMARY =====")
print(res_df.mean())
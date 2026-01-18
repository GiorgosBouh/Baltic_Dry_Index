#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
BDI Forecasting v3 (STEP-based horizon) — "Meaningful forecasting"
Implements ALL three ideas:

A) Event-only KPI (big moves):
   - Define events when |true_return| > threshold (per split, computed on TRAIN only)
   - Evaluate model vs persistence ONLY on those events (and report improvement)

B) Two-stage model (regime + forecast):
   1) Classifier predicts probability of a "big move" event at t+H
   2) Regressor predicts delta_level = BDI(t+H) - BDI(t) only for high-prob events
      Otherwise output persistence (BDI(t))

C) Reduce exogenous + realistic delay:
   - Exogenous features only as LAGS (no exog rolling)
   - min_exog_lag = 14 supported
   - top_k_features = 50/100 supported
   - BDI-only mode as reference

Run:
  python train_xgb_.bdi.py

Output:
  artifacts_bdi_v3/.../metrics.json
  artifacts_bdi_v3/.../predictions_last_window.csv
  artifacts_bdi_v3/.../summary_runs.csv
"""

import json
import os
from dataclasses import dataclass
from typing import Dict, List, Tuple

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from xgboost import XGBClassifier, XGBRegressor


# =========================
# Config
# =========================
@dataclass(frozen=True)
class Config:
    data_path: str = "Enriched_BDI_Dataset.csv"
    target_col: str = "BDI"
    date_col: str = "date"

    # Step-based horizon (rows)
    horizon_steps: int = 10

    # Walk-forward CV
    n_splits: int = 5
    test_window: int = 120
    min_train_size: int = 365
    step: int = 120

    # Purge gap to reduce overlap leakage in multi-step settings
    purge_gap: int = -1  # if -1 => horizon_steps

    # Feature engineering (keep compact)
    lags: Tuple[int, ...] = (1, 2, 3, 5, 7, 10, 14, 20, 30)
    roll_windows: Tuple[int, ...] = (7, 14, 30)

    # Two-stage / selection
    val_fraction: float = 0.15
    random_state: int = 42

    # Event definition (computed on TRAIN only per split)
    event_method: str = "std"  # "std" or "quantile"
    event_std_mult: float = 1.0          # if event_method=="std"
    event_quantile: float = 0.80         # if event_method=="quantile" (top 20% |returns|)

    # Gating: only regress when p(event) >= this threshold
    event_prob_threshold: float = 0.55

    # Exogenous realism + feature selection
    min_exog_lag_grid: Tuple[int, ...] = (7, 14)
    top_k_grid: Tuple[int, ...] = (50, 100)
    bdi_only_grid: Tuple[bool, ...] = (True, False)

    # Output
    out_root: str = "artifacts_bdi_v3"


# =========================
# Utils / Metrics
# =========================
EPS = 1e-8


def rmse(y_true, y_pred) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def mae(y_true, y_pred) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return float(np.mean(np.abs(y_true - y_pred)))


def mape(y_true, y_pred) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    denom = np.maximum(np.abs(y_true), EPS)
    return float(np.mean(np.abs((y_true - y_pred) / denom)) * 100.0)


def smape(y_true, y_pred) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    denom = np.maximum(np.abs(y_true) + np.abs(y_pred), EPS)
    return float(np.mean(2.0 * np.abs(y_true - y_pred) / denom) * 100.0)


def r2_score(y_true, y_pred) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return float(1.0 - ss_res / ss_tot) if ss_tot > 0 else float("nan")


def directional_accuracy(y_true_return, y_pred_return) -> float:
    yt = np.asarray(y_true_return, dtype=float)
    yp = np.asarray(y_pred_return, dtype=float)
    return float(np.mean(np.sign(yt) == np.sign(yp)))


def level_metrics(y_true_level, y_pred_level) -> Dict[str, float]:
    return {
        "RMSE": rmse(y_true_level, y_pred_level),
        "MAE": mae(y_true_level, y_pred_level),
        "MAPE": mape(y_true_level, y_pred_level),
        "sMAPE": smape(y_true_level, y_pred_level),
        "R2": r2_score(y_true_level, y_pred_level),
    }


def return_metrics(y_true_return, y_pred_return) -> Dict[str, float]:
    # Note: sMAPE on returns can be unstable near 0; keep but don't over-interpret.
    return {
        "RMSE": rmse(y_true_return, y_pred_return),
        "MAE": mae(y_true_return, y_pred_return),
        "sMAPE": smape(y_true_return, y_pred_return),
        "R2": r2_score(y_true_return, y_pred_return),
        "DirectionAcc": directional_accuracy(y_true_return, y_pred_return),
    }


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


# =========================
# Data / Features
# =========================
def load_data(path: str, date_col: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Dataset not found: {path}")
    df = pd.read_csv(path)
    if date_col not in df.columns:
        raise ValueError(f"Missing date column '{date_col}'.")
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    df = df.dropna(subset=[date_col]).sort_values(date_col).reset_index(drop=True)
    return df


def to_numeric_df(df: pd.DataFrame, exclude: List[str]) -> pd.DataFrame:
    out = df.copy()
    for c in out.columns:
        if c in exclude:
            continue
        if out[c].dtype == object:
            out[c] = pd.to_numeric(out[c], errors="coerce")
    return out


def build_features_targets(
    df_raw: pd.DataFrame,
    cfg: Config,
    min_exog_lag: int,
    bdi_only: bool,
) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns:
      X (features at time t),
      y_level = BDI(t+H),
      y_return = log(BDI(t+H)) - log(BDI(t)),
      dates_origin (t),
      bdi_t (BDI at t)
    """
    if cfg.target_col not in df_raw.columns:
        raise ValueError(f"Missing target column '{cfg.target_col}'")

    df = to_numeric_df(df_raw, exclude=[cfg.date_col])
    df[cfg.target_col] = pd.to_numeric(df[cfg.target_col], errors="coerce")
    df = df.dropna(subset=[cfg.target_col]).reset_index(drop=True)

    bdi = pd.Series(df[cfg.target_col].astype(float).to_numpy())
    bdi_fut = pd.Series(df[cfg.target_col].shift(-cfg.horizon_steps).astype(float).to_numpy())
    dates = df[cfg.date_col].to_numpy()

    keep = np.isfinite(bdi_fut.to_numpy())
    bdi = bdi.loc[keep].reset_index(drop=True)
    bdi_fut = bdi_fut.loc[keep].reset_index(drop=True)
    dates = dates[: len(bdi)]

    pos = (bdi.to_numpy() > 0) & (bdi_fut.to_numpy() > 0)
    bdi = bdi.loc[pos].reset_index(drop=True)
    bdi_fut = bdi_fut.loc[pos].reset_index(drop=True)
    dates = dates[pos]

    y_level = bdi_fut.to_numpy(dtype=float)
    bdi_t = bdi.to_numpy(dtype=float)
    y_return = np.log(np.maximum(y_level, EPS)) - np.log(np.maximum(bdi_t, EPS))

    # Feature candidates
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    exog_cols = [c for c in numeric_cols if c != cfg.target_col]

    features = {}

    # Core BDI features
    features["bdi_t"] = bdi
    for L in cfg.lags:
        features[f"BDI_lag{L}"] = bdi.shift(L)

    r1 = np.log(np.maximum(bdi, EPS)).diff(1)
    features["ret1"] = r1
    for w in cfg.roll_windows:
        features[f"ret1_roll_mean_{w}"] = r1.shift(1).rolling(w).mean()
        features[f"ret1_roll_std_{w}"] = r1.shift(1).rolling(w).std()
        features[f"BDI_roll_mean_{w}"] = bdi.shift(1).rolling(w).mean()
        features[f"BDI_roll_std_{w}"] = bdi.shift(1).rolling(w).std()

    # Exogenous: only lagged, only if not bdi_only
    if not bdi_only:
        # Align exog with current trimmed indices: easiest is to rebuild from trimmed df
        # Create a temp df aligned to bdi indices
        # We built bdi from df after dropping future NaNs and pos mask; use same masks:
        df2 = df.copy()
        df2 = df2.iloc[: len(df2) - cfg.horizon_steps].reset_index(drop=True)
        df2 = df2.loc[pos].reset_index(drop=True)

        for col in exog_cols:
            s = pd.Series(pd.to_numeric(df2[col], errors="coerce").astype(float).to_numpy())
            for L in cfg.lags:
                if L < min_exog_lag:
                    continue
                features[f"{col}_lag{L}"] = s.shift(L)

    X = pd.DataFrame(features)

    # Drop rows with NaNs (caused by lags/rolling)
    mask = np.isfinite(X.to_numpy(dtype=float)).all(axis=1)
    X = X.loc[mask].reset_index(drop=True)
    y_level = y_level[mask]
    y_return = y_return[mask]
    dates = dates[mask]
    bdi_t = bdi_t[mask]

    return X, y_level, y_return, dates, bdi_t


# =========================
# Walk-forward splits
# =========================
def walk_forward_splits(
    n_samples: int,
    cfg: Config,
    purge_gap: int,
) -> List[Tuple[np.ndarray, np.ndarray]]:
    splits: List[Tuple[np.ndarray, np.ndarray]] = []
    train_end = cfg.min_train_size

    for _ in range(cfg.n_splits):
        train_effective_end = max(0, train_end - purge_gap)
        test_start = train_end
        test_end = test_start + cfg.test_window
        if train_effective_end <= 0:
            break
        if test_end > n_samples:
            break
        train_idx = np.arange(0, train_effective_end)
        test_idx = np.arange(test_start, test_end)
        splits.append((train_idx, test_idx))
        train_end += cfg.step

    if not splits:
        raise ValueError("No splits generated. Reduce min_train_size/test_window or increase data length.")
    return splits


# =========================
# Feature selection
# =========================
def top_k_by_gain(model, feature_names: List[str], k: int) -> List[str]:
    booster = model.get_booster()
    score = booster.get_score(importance_type="gain")  # f0->gain
    gains = []
    for i, name in enumerate(feature_names):
        gains.append((name, float(score.get(f"f{i}", 0.0))))
    gains.sort(key=lambda x: x[1], reverse=True)
    top = [n for n, g in gains[:k] if g > 0]
    if not top:
        top = feature_names[:k]
    return top


# =========================
# Baselines
# =========================
def persistence_level(bdi_t_all: np.ndarray, idx: np.ndarray) -> np.ndarray:
    return np.asarray(bdi_t_all[idx], dtype=float)


def persistence_return(y_pred_level: np.ndarray, bdi_origin: np.ndarray) -> np.ndarray:
    return np.log(np.maximum(y_pred_level, EPS)) - np.log(np.maximum(bdi_origin, EPS))


# =========================
# Event thresholding (TRAIN only)
# =========================
def compute_event_threshold(train_returns: np.ndarray, cfg: Config) -> float:
    abs_r = np.abs(train_returns)
    abs_r = abs_r[np.isfinite(abs_r)]
    if abs_r.size == 0:
        return float("inf")
    if cfg.event_method == "quantile":
        return float(np.quantile(abs_r, cfg.event_quantile))
    # default: std
    return float(np.std(abs_r) * cfg.event_std_mult)


def make_event_labels(returns: np.ndarray, thr: float) -> np.ndarray:
    return (np.abs(returns) > thr).astype(int)


# =========================
# Two-stage training for one split
# =========================
def train_two_stage_one_split(
    X: pd.DataFrame,
    y_level: np.ndarray,
    y_return: np.ndarray,
    dates: np.ndarray,
    bdi_t_all: np.ndarray,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    cfg: Config,
    top_k: int,
) -> Dict[str, object]:
    # Targets
    y_delta = y_level - bdi_t_all  # residual vs persistence
    bdi_origin_test = bdi_t_all[test_idx].astype(float)
    base_pred_test = bdi_origin_test.copy()

    # Event threshold from TRAIN returns only
    thr = compute_event_threshold(y_return[train_idx], cfg)
    y_event_train = make_event_labels(y_return[train_idx], thr)
    y_event_test = make_event_labels(y_return[test_idx], thr)

    # Train/Val split on TRAIN only (tail)
    X_train_full = X.iloc[train_idx]
    y_delta_train_full = y_delta[train_idx]
    y_event_train_full = y_event_train

    val_size = max(1, int(len(train_idx) * cfg.val_fraction))
    X_tr = X_train_full.iloc[:-val_size]
    X_val = X_train_full.iloc[-val_size:]
    y_delta_tr = y_delta_train_full[:-val_size]
    y_delta_val = y_delta_train_full[-val_size:]
    y_evt_tr = y_event_train_full[:-val_size]
    y_evt_val = y_event_train_full[-val_size:]

    # ---------- Stage A: quick regressor for feature ranking ----------
    reg_rank = XGBRegressor(
        n_estimators=4000,
        learning_rate=0.02,
        max_depth=6,
        min_child_weight=3,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=2.0,
        objective="reg:pseudohubererror",
        eval_metric="rmse",
        early_stopping_rounds=200,
        random_state=cfg.random_state,
        n_jobs=-1,
    )
    reg_rank.fit(X_tr, y_delta_tr, eval_set=[(X_val, y_delta_val)], verbose=False)
    top_feats = top_k_by_gain(reg_rank, X.columns.tolist(), top_k)

    # Reduce matrices
    X_tr_k = X_tr[top_feats]
    X_val_k = X_val[top_feats]
    X_train_full_k = X_train_full[top_feats]
    X_test_k = X.iloc[test_idx][top_feats]

    # ---------- Model 1: event classifier ----------
    # Handle class imbalance (events can be rare)
    pos = float(np.sum(y_evt_tr == 1))
    neg = float(np.sum(y_evt_tr == 0))
    scale_pos_weight = (neg / max(pos, 1.0)) if pos > 0 else 1.0

    clf = XGBClassifier(
        n_estimators=4000,
        learning_rate=0.02,
        max_depth=4,
        min_child_weight=5,
        subsample=0.9,
        colsample_bytree=0.9,
        reg_alpha=0.2,
        reg_lambda=2.0,
        objective="binary:logistic",
        eval_metric="logloss",
        early_stopping_rounds=200,
        random_state=cfg.random_state,
        n_jobs=-1,
        scale_pos_weight=scale_pos_weight,
    )
    clf.fit(X_tr_k, y_evt_tr, eval_set=[(X_val_k, y_evt_val)], verbose=False)

    p_event_test = clf.predict_proba(X_test_k)[:, 1]
    gate = (p_event_test >= cfg.event_prob_threshold).astype(int)

    # ---------- Model 2: regressor for delta (trained on TRAIN EVENTS only) ----------
    # Train regression only on event samples to focus magnitude where it matters
    event_mask_train = (y_event_train_full == 1)
    # Fallback if too few events:
    min_events = 30
    if int(event_mask_train.sum()) < min_events:
        # If too few events, train on all samples (still gated at inference)
        X_reg_full = X_train_full_k
        y_reg_full = y_delta_train_full
    else:
        X_reg_full = X_train_full_k.loc[event_mask_train]
        y_reg_full = y_delta_train_full[event_mask_train]

    # Simple val split for regressor
    val_size_reg = max(1, int(len(X_reg_full) * cfg.val_fraction))
    Xr_tr = X_reg_full.iloc[:-val_size_reg]
    Xr_val = X_reg_full.iloc[-val_size_reg:]
    yr_tr = y_reg_full[:-val_size_reg]
    yr_val = y_reg_full[-val_size_reg:]

    reg = XGBRegressor(
        n_estimators=6000,
        learning_rate=0.02,
        max_depth=6,
        min_child_weight=3,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=2.0,
        objective="reg:pseudohubererror",
        eval_metric="rmse",
        early_stopping_rounds=250,
        random_state=cfg.random_state,
        n_jobs=-1,
    )
    reg.fit(Xr_tr, yr_tr, eval_set=[(Xr_val, yr_val)], verbose=False)

    delta_pred_test = reg.predict(X_test_k)

    # Apply gating: if not event, output persistence; else persistence + delta
    y_pred_level = base_pred_test.copy()
    y_pred_level = y_pred_level + delta_pred_test * gate

    # Returns
    y_pred_return = persistence_return(y_pred_level, bdi_origin_test)

    # ---------- Metrics (overall) ----------
    y_true_level = y_level[test_idx]
    y_true_return = y_return[test_idx]

    model_level = level_metrics(y_true_level, y_pred_level)
    model_ret = return_metrics(y_true_return, y_pred_return)

    # Persistence baseline
    pers_level = base_pred_test
    pers_ret = persistence_return(pers_level, bdi_origin_test)
    pers_level_m = level_metrics(y_true_level, pers_level)
    pers_ret_m = return_metrics(y_true_return, pers_ret)

    # ---------- Event-only evaluation ----------
    event_mask_test = (np.abs(y_true_return) > thr)
    # If no events, event metrics are NaN
    def safe_metrics_level(mask):
        if int(mask.sum()) == 0:
            return {"RMSE": float("nan"), "MAE": float("nan"), "MAPE": float("nan"), "sMAPE": float("nan"), "R2": float("nan")}
        return level_metrics(y_true_level[mask], y_pred_level[mask])

    def safe_metrics_level_pers(mask):
        if int(mask.sum()) == 0:
            return {"RMSE": float("nan"), "MAE": float("nan"), "MAPE": float("nan"), "sMAPE": float("nan"), "R2": float("nan")}
        return level_metrics(y_true_level[mask], pers_level[mask])

    def safe_metrics_return(mask):
        if int(mask.sum()) == 0:
            return {"RMSE": float("nan"), "MAE": float("nan"), "sMAPE": float("nan"), "R2": float("nan"), "DirectionAcc": float("nan")}
        return return_metrics(y_true_return[mask], y_pred_return[mask])

    model_level_evt = safe_metrics_level(event_mask_test)
    pers_level_evt = safe_metrics_level_pers(event_mask_test)
    model_ret_evt = safe_metrics_return(event_mask_test)

    # Improvement vs persistence (overall + events)
    imp_overall_rmse = pers_level_m["RMSE"] - model_level["RMSE"]
    imp_event_rmse = pers_level_evt["RMSE"] - model_level_evt["RMSE"] if np.isfinite(pers_level_evt["RMSE"]) else float("nan")

    return {
        "top_features": top_feats,
        "event_threshold": thr,
        "event_rate_test": float(np.mean(event_mask_test)),

        "metrics_overall": {"level": model_level, "return": model_ret},
        "metrics_persistence": {"level": pers_level_m, "return": pers_ret_m},
        "metrics_event_only": {
            "model_level": model_level_evt,
            "pers_level": pers_level_evt,
            "model_return": model_ret_evt,
            "event_count": int(event_mask_test.sum()),
        },

        "improvement_vs_persistence": {
            "level_RMSE_overall": float(imp_overall_rmse),
            "level_RMSE_event_only": float(imp_event_rmse),
        },

        "pred": {
            "dates_test": dates[test_idx],
            "bdi_origin": bdi_origin_test,
            "y_true_level": y_true_level,
            "y_pred_level": y_pred_level,
            "pers_level": pers_level,
            "y_true_return": y_true_return,
            "y_pred_return": y_pred_return,
            "p_event": p_event_test,
            "gate": gate,
            "event_true": y_event_test,
        },

        "models": {"clf": clf, "reg": reg},
    }


# =========================
# Saving artifacts
# =========================
def save_split_artifacts(outdir: str, split_id: int, res: Dict[str, object]) -> None:
    ensure_dir(outdir)

    # Save models
    joblib.dump(res["models"]["clf"], os.path.join(outdir, f"split{split_id}_event_clf.joblib"))
    joblib.dump(res["models"]["reg"], os.path.join(outdir, f"split{split_id}_delta_reg.joblib"))

    # Save top features
    with open(os.path.join(outdir, f"split{split_id}_top_features.txt"), "w") as f:
        for feat in res["top_features"]:
            f.write(feat + "\n")

    # Save predictions
    p = res["pred"]
    pred_df = pd.DataFrame(
        {
            "date_origin": pd.to_datetime(p["dates_test"]),
            "bdi_origin": p["bdi_origin"],
            "y_true_level": p["y_true_level"],
            "y_pred_level": p["y_pred_level"],
            "persistence_level": p["pers_level"],
            "y_true_return": p["y_true_return"],
            "y_pred_return": p["y_pred_return"],
            "p_event": p["p_event"],
            "gate": p["gate"],
            "event_true": p["event_true"],
        }
    )
    pred_df.to_csv(os.path.join(outdir, f"split{split_id}_predictions.csv"), index=False)

    # Plot (level)
    plt.figure(figsize=(12, 5))
    d = pd.to_datetime(p["dates_test"])
    plt.plot(d, p["y_true_level"], label="Actual BDI(t+H)", linewidth=2)
    plt.plot(d, p["y_pred_level"], label="Two-stage forecast", linewidth=2)
    plt.plot(d, p["pers_level"], label="Persistence", linewidth=1.5, alpha=0.8)
    plt.title(f"Split {split_id}: Forecast vs Actual (H steps)")
    plt.xlabel("Origin date t")
    plt.ylabel("BDI at t+H")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, f"split{split_id}_forecast_plot.png"), dpi=200)
    plt.close()

    # Save split metrics
    with open(os.path.join(outdir, f"split{split_id}_metrics.json"), "w") as f:
        json.dump(
            {
                "event_threshold": res["event_threshold"],
                "event_rate_test": res["event_rate_test"],
                "metrics_overall": res["metrics_overall"],
                "metrics_persistence": res["metrics_persistence"],
                "metrics_event_only": res["metrics_event_only"],
                "improvement_vs_persistence": res["improvement_vs_persistence"],
            },
            f,
            indent=2,
        )


# =========================
# Main grid runner
# =========================
def main() -> None:
    cfg = Config()
    purge_gap = cfg.horizon_steps if cfg.purge_gap == -1 else cfg.purge_gap

    df_raw = load_data(cfg.data_path, cfg.date_col)

    all_runs_rows = []

    for bdi_only in cfg.bdi_only_grid:
        for min_exog_lag in cfg.min_exog_lag_grid:
            for top_k in cfg.top_k_grid:
                run_name = f"h{cfg.horizon_steps}_bdiOnly{int(bdi_only)}_exogLag{min_exog_lag}_topK{top_k}"
                outdir = os.path.join(cfg.out_root, run_name)
                ensure_dir(outdir)

                # Build features/targets
                X, y_level, y_return, dates, bdi_t_all = build_features_targets(
                    df_raw=df_raw,
                    cfg=cfg,
                    min_exog_lag=min_exog_lag,
                    bdi_only=bdi_only,
                )

                print("\n" + "=" * 90)
                print(f"RUN: {run_name}")
                print(f"Rows: {len(X)}, Features: {X.shape[1]}")
                print(f"Origin date range: {pd.to_datetime(dates).min().date()} → {pd.to_datetime(dates).max().date()}")
                print("Feature head:", X.columns[:15].tolist())
                print(f"Event method: {cfg.event_method} | prob_threshold={cfg.event_prob_threshold} | purge_gap={purge_gap}")

                splits = walk_forward_splits(len(X), cfg, purge_gap)

                split_summaries = []
                # For a "last window" file later:
                last_split_pred_df = None

                for sidx, (train_idx, test_idx) in enumerate(splits, start=1):
                    print(f"\nSplit {sidx}/{len(splits)}: train={len(train_idx)} test={len(test_idx)}")

                    res = train_two_stage_one_split(
                        X=X,
                        y_level=y_level,
                        y_return=y_return,
                        dates=dates,
                        bdi_t_all=bdi_t_all,
                        train_idx=train_idx,
                        test_idx=test_idx,
                        cfg=cfg,
                        top_k=top_k,
                    )

                    # Print compact summary
                    m = res["metrics_overall"]["level"]
                    p = res["metrics_persistence"]["level"]
                    evt = res["metrics_event_only"]
                    imp_all = res["improvement_vs_persistence"]["level_RMSE_overall"]
                    imp_evt = res["improvement_vs_persistence"]["level_RMSE_event_only"]

                    print("Overall LEVEL RMSE (model vs pers):", f"{m['RMSE']:.3f}", "vs", f"{p['RMSE']:.3f}",
                          f"| IMP={imp_all:+.4f}")
                    print("Event-only count:", evt["event_count"],
                          "| Event-only LEVEL RMSE (model vs pers):",
                          f"{evt['model_level']['RMSE']:.3f}" if np.isfinite(evt["model_level"]["RMSE"]) else "nan",
                          "vs",
                          f"{evt['pers_level']['RMSE']:.3f}" if np.isfinite(evt["pers_level"]["RMSE"]) else "nan",
                          f"| IMP_EVT={imp_evt:+.4f}" if np.isfinite(imp_evt) else "| IMP_EVT=nan")
                    print("Event threshold | event_rate_test:",
                          f"{res['event_threshold']:.6f}", f"{res['event_rate_test']:.3f}")

                    # Save per-split artifacts
                    save_split_artifacts(outdir, sidx, res)

                    # Keep split summary
                    split_summaries.append(
                        {
                            "split": sidx,
                            "train_n": int(len(train_idx)),
                            "test_n": int(len(test_idx)),
                            "features_total": int(X.shape[1]),
                            "top_k": int(top_k),
                            "bdi_only": int(bdi_only),
                            "min_exog_lag": int(min_exog_lag),
                            "event_threshold": float(res["event_threshold"]),
                            "event_rate_test": float(res["event_rate_test"]),
                            "event_count": int(res["metrics_event_only"]["event_count"]),
                            "overall_rmse_model": float(m["RMSE"]),
                            "overall_rmse_pers": float(p["RMSE"]),
                            "imp_rmse_overall": float(imp_all),
                            "event_rmse_model": float(res["metrics_event_only"]["model_level"]["RMSE"]),
                            "event_rmse_pers": float(res["metrics_event_only"]["pers_level"]["RMSE"]),
                            "imp_rmse_event_only": float(imp_evt),
                            "diracc_return_overall": float(res["metrics_overall"]["return"]["DirectionAcc"]),
                            "diracc_return_event_only": float(res["metrics_event_only"]["model_return"]["DirectionAcc"]),
                        }
                    )

                    # Save "last window" predictions for quick inspection
                    if sidx == len(splits):
                        pdat = res["pred"]
                        last_split_pred_df = pd.DataFrame(
                            {
                                "date_origin": pd.to_datetime(pdat["dates_test"]),
                                "bdi_origin": pdat["bdi_origin"],
                                "y_true_level": pdat["y_true_level"],
                                "y_pred_level": pdat["y_pred_level"],
                                "persistence_level": pdat["pers_level"],
                                "y_true_return": pdat["y_true_return"],
                                "y_pred_return": pdat["y_pred_return"],
                                "p_event": pdat["p_event"],
                                "gate": pdat["gate"],
                                "event_true": pdat["event_true"],
                            }
                        )

                # Save split summary table
                split_df = pd.DataFrame(split_summaries)
                split_df.to_csv(os.path.join(outdir, "splits_summary.csv"), index=False)

                # Save last window predictions
                if last_split_pred_df is not None:
                    last_split_pred_df.to_csv(os.path.join(outdir, "predictions_last_window.csv"), index=False)

                # Aggregate run summary (means)
                mean_imp_overall = float(split_df["imp_rmse_overall"].mean())
                mean_imp_event = float(split_df["imp_rmse_event_only"].mean(skipna=True))
                mean_diracc = float(split_df["diracc_return_overall"].mean())
                mean_event_diracc = float(split_df["diracc_return_event_only"].mean(skipna=True))

                run_row = {
                    "run": run_name,
                    "bdi_only": int(bdi_only),
                    "min_exog_lag": int(min_exog_lag),
                    "top_k": int(top_k),
                    "rows": int(len(X)),
                    "features_total": int(X.shape[1]),
                    "splits": int(len(split_df)),
                    "mean_imp_rmse_overall": mean_imp_overall,
                    "mean_imp_rmse_event_only": mean_imp_event,
                    "mean_diracc_overall": mean_diracc,
                    "mean_diracc_event_only": mean_event_diracc,
                }
                all_runs_rows.append(run_row)

                with open(os.path.join(outdir, "run_summary.json"), "w") as f:
                    json.dump(
                        {
                            "config": {
                                "horizon_steps": cfg.horizon_steps,
                                "event_method": cfg.event_method,
                                "event_std_mult": cfg.event_std_mult,
                                "event_quantile": cfg.event_quantile,
                                "event_prob_threshold": cfg.event_prob_threshold,
                                "purge_gap": purge_gap,
                                "lags": list(cfg.lags),
                                "roll_windows": list(cfg.roll_windows),
                            },
                            "run": run_row,
                        },
                        f,
                        indent=2,
                    )

                print("\nRUN SUMMARY:")
                print("Mean IMP RMSE overall:", f"{mean_imp_overall:+.4f}")
                print("Mean IMP RMSE event-only:", f"{mean_imp_event:+.4f}" if np.isfinite(mean_imp_event) else "nan")
                print("Mean DirectionAcc overall:", f"{mean_diracc:.3f}")
                print("Mean DirectionAcc event-only:", f"{mean_event_diracc:.3f}" if np.isfinite(mean_event_diracc) else "nan")

    # Save all-runs comparison table
    out_root = cfg.out_root
    ensure_dir(out_root)
    runs_df = pd.DataFrame(all_runs_rows)
    runs_df = runs_df.sort_values(["mean_imp_rmse_event_only", "mean_imp_rmse_overall"], ascending=False)
    runs_df.to_csv(os.path.join(out_root, "summary_runs.csv"), index=False)

    print("\n" + "=" * 90)
    print("✅ Finished. Overall runs summary saved to:", os.path.join(out_root, "summary_runs.csv"))
    print("Top rows (best by event-only improvement):")
    print(runs_df.head(10).to_string(index=False))


if __name__ == "__main__":
    main()
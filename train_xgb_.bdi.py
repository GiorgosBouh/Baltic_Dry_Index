#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
BDI forecasting (STEP-BASED horizon) — v2
Goal: REAL forecasting signal (beat persistence), not just "copy baseline".

What this version adds:
✅ Residual learning vs persistence: predict delta = BDI(t+H) - BDI(t)
✅ Strong time-series features from BDI (lags/roll/volatility)
✅ Exogenous features only as LAGS (NO exogenous rolling to avoid feature explosion)
✅ Production realism: min_exog_lag (simulate reporting delay)
✅ Two-stage feature selection:
   - Stage A: fit model on train, rank features by gain
   - Stage B: retrain using only top-K features
✅ Walk-forward CV with purge gap
✅ Clear reporting: improvement over persistence per split + summary
✅ Artifacts: predictions CSV, plots, metrics JSON, top_features.txt, SHAP

Run:
  python train_xgb_.bdi.py

If you want more "production realism", set min_exog_lag=7 or 14 in Config.
"""

import json
import os
from dataclasses import dataclass
from typing import Dict, List, Tuple

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from xgboost import XGBRegressor


# =========================
# Config
# =========================
@dataclass(frozen=True)
class Config:
    data_path: str = "Enriched_BDI_Dataset.csv"
    target_col: str = "BDI"
    date_col: str = "date"

    # Forecast horizon as steps (rows)
    horizon_steps: int = 10

    # Walk-forward evaluation
    n_splits: int = 5
    test_window: int = 120
    min_train_size: int = 365
    step: int = 120
    val_fraction: float = 0.15
    random_state: int = 42

    # Purge gap (steps) to reduce leakage overlap for multi-step
    purge_gap: int = -1  # if -1 => horizon_steps

    # Feature engineering
    # Keep it compact. Too many features = overfit & no edge.
    lags: Tuple[int, ...] = (1, 2, 3, 5, 7, 10, 14, 20, 30)
    roll_windows: Tuple[int, ...] = (7, 14, 30)

    # Production realism for exogenous features (publication delay)
    # Recommended runs: 1 (optimistic), 7 (weekly delay), 14 (biweekly)
    min_exog_lag: int = 7

    # Two-stage selection
    top_k_features: int = 150

    # Model search
    use_param_grid: bool = False

    # Output
    output_dir_template: str = "artifacts_xgb_bdi_v2_h{h}steps_k{k}_exoglag{elag}"


# =========================
# Metrics
# =========================
def rmse(y_true, y_pred) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def mae(y_true, y_pred) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return float(np.mean(np.abs(y_true - y_pred)))


def mape(y_true, y_pred, eps=1e-8) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    denom = np.maximum(np.abs(y_true), eps)
    return float(np.mean(np.abs((y_true - y_pred) / denom)) * 100.0)


def smape(y_true, y_pred, eps=1e-8) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    denom = np.maximum(np.abs(y_true) + np.abs(y_pred), eps)
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


def evaluate_level_metrics(y_true_level, y_pred_level) -> Dict[str, float]:
    return {
        "RMSE": rmse(y_true_level, y_pred_level),
        "MAE": mae(y_true_level, y_pred_level),
        "MAPE": mape(y_true_level, y_pred_level),
        "sMAPE": smape(y_true_level, y_pred_level),
        "R2": r2_score(y_true_level, y_pred_level),
    }


def evaluate_return_metrics(y_true_return, y_pred_return) -> Dict[str, float]:
    # sMAPE on returns can be unstable (returns near 0); keep it but do not over-interpret
    return {
        "RMSE": rmse(y_true_return, y_pred_return),
        "MAE": mae(y_true_return, y_pred_return),
        "sMAPE": smape(y_true_return, y_pred_return),
        "R2": r2_score(y_true_return, y_pred_return),
        "DirectionAcc": directional_accuracy(y_true_return, y_pred_return),
    }


# =========================
# Data
# =========================
def load_data(path: str, date_col: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Dataset not found: {path}")
    df = pd.read_csv(path)
    if date_col not in df.columns:
        raise ValueError(f"Missing date column '{date_col}' in dataset.")
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    df = df.dropna(subset=[date_col]).copy()
    df = df.sort_values(date_col).reset_index(drop=True)
    return df


def _to_numeric_df(df: pd.DataFrame, exclude: List[str]) -> pd.DataFrame:
    out = df.copy()
    for c in out.columns:
        if c in exclude:
            continue
        if out[c].dtype == object:
            out[c] = pd.to_numeric(out[c], errors="coerce")
    return out


# =========================
# Baselines (STEP-BASED)
# =========================
def compute_step_baselines(
    bdi_t_all: np.ndarray,
    test_idx: np.ndarray,
    seasonal_lag_steps: int = 7,
    ma_window_steps: int = 7,
) -> Dict[str, Dict[str, np.ndarray]]:
    """
    Baselines predict LEVEL at t+H, compared against y_level = BDI_{t+H}.

    persistence_level(t) = BDI(t)
    seasonal_level(t) = BDI(t - seasonal_lag_steps)
    moving_avg_level(t) = mean(BDI(t-ma_window..t-1))

    Also returns predicted returns relative to BDI(t).
    """
    bdi_t_all = np.asarray(bdi_t_all, dtype=float)
    persistence_level = bdi_t_all[test_idx].astype(float)

    seasonal_level = np.full(len(test_idx), np.nan, dtype=float)
    for j, i in enumerate(test_idx):
        k = i - seasonal_lag_steps
        if k >= 0:
            seasonal_level[j] = float(bdi_t_all[k])

    ma_level = np.full(len(test_idx), np.nan, dtype=float)
    for j, i in enumerate(test_idx):
        start = i - ma_window_steps
        end = i
        if start >= 0:
            ma_level[j] = float(np.mean(bdi_t_all[start:end]))

    eps = 1e-8
    origin = np.maximum(persistence_level, eps)

    def to_return(level_pred: np.ndarray) -> np.ndarray:
        out = np.full_like(level_pred, np.nan, dtype=float)
        ok = np.isfinite(level_pred) & (level_pred > 0) & (origin > 0)
        out[ok] = np.log(level_pred[ok]) - np.log(origin[ok])
        return out

    return {
        "persistence": {"level": persistence_level, "return": to_return(persistence_level)},
        "seasonal_naive": {"level": seasonal_level, "return": to_return(seasonal_level)},
        "moving_average": {"level": ma_level, "return": to_return(ma_level)},
    }


# =========================
# Feature Engineering (compact)
# =========================
def build_features_and_targets(df_raw: pd.DataFrame, cfg: Config):
    """
    Returns:
      X (features at time t),
      y_level = BDI_{t+H},
      y_return = log(BDI_{t+H})-log(BDI_t),
      dates (origin dates),
      bdi_t (origin BDI)
    """
    if cfg.target_col not in df_raw.columns:
        raise ValueError(f"Missing target column '{cfg.target_col}'")

    df = df_raw.copy()
    df = _to_numeric_df(df, exclude=[cfg.date_col])
    df[cfg.target_col] = pd.to_numeric(df[cfg.target_col], errors="coerce")
    df = df.dropna(subset=[cfg.target_col]).reset_index(drop=True)

    # Targets (STEP-based)
    bdi_t = df[cfg.target_col].astype(float)
    bdi_future = df[cfg.target_col].shift(-cfg.horizon_steps).astype(float)
    dates = df[cfg.date_col].to_numpy()

    keep = bdi_future.notna().to_numpy()
    df = df.loc[keep].reset_index(drop=True)
    bdi_t = bdi_t.loc[keep].reset_index(drop=True).to_numpy(dtype=float)
    bdi_future = bdi_future.loc[keep].reset_index(drop=True).to_numpy(dtype=float)
    dates = dates[: len(df)]

    eps = 1e-8
    pos_mask = (bdi_t > 0) & (bdi_future > 0)
    df = df.loc[pos_mask].reset_index(drop=True)
    bdi_t = bdi_t[pos_mask]
    bdi_future = bdi_future[pos_mask]
    dates = dates[pos_mask]

    y_level = bdi_future
    y_return = np.log(np.maximum(bdi_future, eps)) - np.log(np.maximum(bdi_t, eps))

    # Feature candidates: all numeric except target; exogenous will be lagged only
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    exog_cols = [c for c in numeric_cols if c != cfg.target_col]

    features = {}

    # Baseline feature (explicit)
    bdi_series = pd.Series(bdi_t)
    features["bdi_t"] = bdi_series

    # BDI lags
    for L in cfg.lags:
        features[f"BDI_lag{L}"] = bdi_series.shift(L)

    # 1-step log return (momentum & vol)
    r1 = np.log(np.maximum(bdi_series, eps)).diff(1)
    features["ret1"] = r1
    for w in cfg.roll_windows:
        features[f"ret1_roll_mean_{w}"] = r1.shift(1).rolling(w).mean()
        features[f"ret1_roll_std_{w}"] = r1.shift(1).rolling(w).std()

    # BDI rolling stats (level)
    for w in cfg.roll_windows:
        features[f"BDI_roll_mean_{w}"] = bdi_series.shift(1).rolling(w).mean()
        features[f"BDI_roll_std_{w}"] = bdi_series.shift(1).rolling(w).std()
        features[f"BDI_roll_min_{w}"] = bdi_series.shift(1).rolling(w).min()
        features[f"BDI_roll_max_{w}"] = bdi_series.shift(1).rolling(w).max()

    # Exogenous lags ONLY (no rolling) to prevent feature explosion
    for col in exog_cols:
        s = pd.Series(df[col].to_numpy(dtype=float))
        for L in cfg.lags:
            if L < cfg.min_exog_lag:
                continue
            features[f"{col}_lag{L}"] = s.shift(L)

    X = pd.DataFrame(features)

    # Drop rows with NaN
    mask = np.isfinite(X.to_numpy(dtype=float)).all(axis=1)
    X = X.loc[mask].reset_index(drop=True)
    y_level = y_level[mask]
    y_return = y_return[mask]
    dates = dates[mask]
    bdi_t = bdi_t[mask]

    return X, y_level, y_return, dates, bdi_t


# =========================
# Walk-forward splits with purge gap
# =========================
def walk_forward_splits(
    n_samples: int,
    n_splits: int,
    test_window: int,
    min_train_size: int,
    step: int,
    purge_gap: int,
) -> List[Tuple[np.ndarray, np.ndarray]]:
    splits: List[Tuple[np.ndarray, np.ndarray]] = []
    train_end = min_train_size

    for _ in range(n_splits):
        train_effective_end = max(0, train_end - purge_gap)
        test_start = train_end
        test_end = test_start + test_window
        if train_effective_end <= 0:
            break
        if test_end > n_samples:
            break
        train_idx = np.arange(0, train_effective_end)
        test_idx = np.arange(test_start, test_end)
        splits.append((train_idx, test_idx))
        train_end += step

    if not splits:
        raise ValueError("No splits generated. Reduce min_train_size/test_window or increase dataset length.")
    return splits


# =========================
# Model factory + feature selection
# =========================
def make_model(cfg: Config, params: Dict[str, object]) -> XGBRegressor:
    return XGBRegressor(
        n_estimators=6000,
        learning_rate=0.02,
        objective="reg:pseudohubererror",  # robust
        random_state=cfg.random_state,
        n_jobs=-1,
        eval_metric="rmse",
        early_stopping_rounds=250,
        **params,
    )


def get_param_grid(cfg: Config) -> List[Dict[str, object]]:
    grid = [
        {"max_depth": 4, "min_child_weight": 5, "subsample": 0.9, "colsample_bytree": 0.9, "reg_alpha": 0.2, "reg_lambda": 2.0},
        {"max_depth": 6, "min_child_weight": 3, "subsample": 0.8, "colsample_bytree": 0.8, "reg_alpha": 0.1, "reg_lambda": 2.0},
        {"max_depth": 8, "min_child_weight": 2, "subsample": 0.7, "colsample_bytree": 0.7, "reg_alpha": 0.0, "reg_lambda": 3.0},
    ]
    if not cfg.use_param_grid:
        return [grid[1]]
    return grid


def select_top_k_features_by_gain(model: XGBRegressor, feature_names: List[str], k: int) -> List[str]:
    booster = model.get_booster()
    score = booster.get_score(importance_type="gain")  # dict: f0->gain, etc.

    # Map f{idx} to feature name
    gains = []
    for i, name in enumerate(feature_names):
        key = f"f{i}"
        gains.append((name, float(score.get(key, 0.0))))
    gains.sort(key=lambda x: x[1], reverse=True)

    top = [n for n, g in gains[:k] if g > 0]
    # If all gains are 0 (rare), fallback to first k
    if not top:
        top = feature_names[:k]
    return top


# =========================
# Train / Eval split (2-stage selection)
# =========================
def train_eval_one_split(
    X: pd.DataFrame,
    y_level: np.ndarray,
    y_return: np.ndarray,
    dates: np.ndarray,
    bdi_t_all: np.ndarray,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    cfg: Config,
) -> Dict[str, object]:
    # Baseline (persistence)
    base_pred_level_test = bdi_t_all[test_idx].astype(float)

    # Residual target: delta over persistence
    y_delta = y_level - bdi_t_all

    X_train_full = X.iloc[train_idx]
    y_train_full = y_delta[train_idx]

    X_test_full = X.iloc[test_idx]
    y_test_level = y_level[test_idx]
    y_test_return = y_return[test_idx]
    dates_test = dates[test_idx]
    bdi_origin_test = bdi_t_all[test_idx]

    # Validation split (tail)
    val_size = max(1, int(len(X_train_full) * cfg.val_fraction))
    X_train = X_train_full.iloc[:-val_size]
    y_train = y_train_full[:-val_size]
    X_val = X_train_full.iloc[-val_size:]
    y_val = y_train_full[-val_size:]

    # ---- Stage A: fit model for feature ranking ----
    best_stageA = None
    best_score = float("inf")
    best_params = None

    for params in get_param_grid(cfg):
        m = make_model(cfg, params)
        m.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
        pred_val = m.predict(X_val)
        score_rmse = rmse(y_val, pred_val)
        if score_rmse < best_score:
            best_score = score_rmse
            best_stageA = m
            best_params = params

    # Rank features
    feature_names = X_train_full.columns.tolist()
    top_feats = select_top_k_features_by_gain(best_stageA, feature_names, cfg.top_k_features)

    # ---- Stage B: retrain on top-K features only ----
    X_train_full_k = X_train_full[top_feats]
    X_test_k = X_test_full[top_feats]

    # Recreate train/val on top feats
    X_train_k = X_train_full_k.iloc[:-val_size]
    X_val_k = X_train_full_k.iloc[-val_size:]

    best_model = None
    best_score_B = float("inf")

    for params in get_param_grid(cfg):
        m = make_model(cfg, params)
        m.fit(X_train_k, y_train, eval_set=[(X_val_k, y_val)], verbose=False)
        pred_val = m.predict(X_val_k)
        score_rmse = rmse(y_val, pred_val)
        if score_rmse < best_score_B:
            best_score_B = score_rmse
            best_model = m

    # Predict residual delta and reconstruct level forecast
    delta_pred = best_model.predict(X_test_k)
    y_pred_level = base_pred_level_test + delta_pred

    # returns
    eps = 1e-8
    y_pred_return = np.log(np.maximum(y_pred_level, eps)) - np.log(np.maximum(bdi_origin_test, eps))

    # model metrics
    model_metrics_level = evaluate_level_metrics(y_test_level, y_pred_level)
    model_metrics_return = evaluate_return_metrics(y_test_return, y_pred_return)

    # baselines
    baselines = compute_step_baselines(bdi_t_all, test_idx, seasonal_lag_steps=7, ma_window_steps=7)
    baseline_metrics = {}
    for name, preds in baselines.items():
        lvl = preds["level"]
        m_lvl = np.isfinite(lvl)
        ret = preds["return"]
        m_ret = np.isfinite(ret)
        baseline_metrics[name] = {
            "level": evaluate_level_metrics(y_test_level[m_lvl], lvl[m_lvl]),
            "return": evaluate_return_metrics(y_test_return[m_ret], ret[m_ret]),
        }

    # Improvement vs persistence (level RMSE)
    pers_rmse = baseline_metrics["persistence"]["level"]["RMSE"]
    imp_rmse = pers_rmse - model_metrics_level["RMSE"]  # positive = improvement

    return {
        "model": best_model,
        "best_params": best_params,
        "top_features": top_feats,
        "imp_vs_persistence_rmse": imp_rmse,

        "dates_test": dates_test,
        "bdi_origin_test": bdi_origin_test,
        "y_true_level": y_test_level,
        "y_pred_level": y_pred_level,
        "y_true_return": y_test_return,
        "y_pred_return": y_pred_return,

        "baselines": baselines,
        "metrics": {"level": model_metrics_level, "return": model_metrics_return},
        "baseline_metrics": baseline_metrics,

        "X_test_k": X_test_k,
        "base_pred_level_test": base_pred_level_test,
    }


# =========================
# Aggregation + artifacts
# =========================
def aggregate_results(split_results: List[Dict[str, object]]) -> Dict[str, Dict[str, float]]:
    def summarize(dicts: List[Dict[str, float]]) -> Dict[str, float]:
        keys = list(dicts[0].keys())
        out = {}
        for k in keys:
            vals = [d[k] for d in dicts]
            out[f"{k}_mean"] = float(np.mean(vals))
            out[f"{k}_std"] = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
        return out

    model_level = [r["metrics"]["level"] for r in split_results]
    model_return = [r["metrics"]["return"] for r in split_results]
    summary = {
        "model_level": summarize(model_level),
        "model_return": summarize(model_return),
        "imp_vs_persistence_rmse_mean": float(np.mean([r["imp_vs_persistence_rmse"] for r in split_results])),
        "imp_vs_persistence_rmse_std": float(np.std([r["imp_vs_persistence_rmse"] for r in split_results], ddof=1)) if len(split_results) > 1 else 0.0,
    }

    baseline_names = list(split_results[0]["baseline_metrics"].keys())
    for bn in baseline_names:
        blev = [r["baseline_metrics"][bn]["level"] for r in split_results]
        bret = [r["baseline_metrics"][bn]["return"] for r in split_results]
        summary[f"{bn}_level"] = summarize(blev)
        summary[f"{bn}_return"] = summarize(bret)

    return summary


def save_artifacts(
    outdir: str,
    final_result: Dict[str, object],
    split_results: List[Dict[str, object]],
    summary: Dict[str, Dict[str, float]],
    cfg: Config,
) -> None:
    os.makedirs(outdir, exist_ok=True)

    # Save final model
    joblib.dump(final_result["model"], os.path.join(outdir, "xgb_bdi_model.joblib"))

    # Save top features used (final split)
    with open(os.path.join(outdir, "top_features_last_split.txt"), "w") as f:
        for name in final_result["top_features"]:
            f.write(name + "\n")

    # Save union of top features across splits (for stability check)
    union = set()
    for res in split_results:
        union.update(res["top_features"])
    with open(os.path.join(outdir, "top_features_union.txt"), "w") as f:
        for name in sorted(union):
            f.write(name + "\n")

    # predictions CSV (final window)
    pred_df = pd.DataFrame(
        {
            "date_origin": pd.to_datetime(final_result["dates_test"]),
            "bdi_origin": final_result["bdi_origin_test"],
            "baseline_persistence_level": final_result["base_pred_level_test"],
            "y_true_level": final_result["y_true_level"],
            "y_pred_level": final_result["y_pred_level"],
            "y_true_return": final_result["y_true_return"],
            "y_pred_return": final_result["y_pred_return"],
            "baseline_seasonal_level": final_result["baselines"]["seasonal_naive"]["level"],
            "baseline_movingavg_level": final_result["baselines"]["moving_average"]["level"],
        }
    )
    pred_df.to_csv(os.path.join(outdir, "predictions_last_window.csv"), index=False)

    # plot level
    dates = pd.to_datetime(final_result["dates_test"])
    plt.figure(figsize=(12, 5))
    plt.plot(dates, final_result["y_true_level"], label="Actual BDI(t+H)", linewidth=2)
    plt.plot(dates, final_result["y_pred_level"], label="XGB v2 (residual + top-K)", linewidth=2)
    plt.plot(dates, final_result["base_pred_level_test"], label="Persistence", linewidth=1.5, alpha=0.8)
    plt.title(f"BDI Forecast (H={cfg.horizon_steps} steps) — Final Window")
    plt.xlabel("Origin date t")
    plt.ylabel("BDI at t+H")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "actual_vs_pred_final.png"), dpi=200)
    plt.close()

    # SHAP on top-K features (final split)
    try:
        explainer = shap.TreeExplainer(final_result["model"])
        shap_values = explainer.shap_values(final_result["X_test_k"])

        plt.figure()
        shap.summary_plot(shap_values, final_result["X_test_k"], show=False)
        plt.tight_layout()
        plt.savefig(os.path.join(outdir, "shap_beeswarm.png"), dpi=200)
        plt.close()

        plt.figure()
        shap.summary_plot(shap_values, final_result["X_test_k"], show=False, plot_type="bar")
        plt.tight_layout()
        plt.savefig(os.path.join(outdir, "shap_bar.png"), dpi=200)
        plt.close()
    except Exception as e:
        with open(os.path.join(outdir, "shap_error.txt"), "w") as f:
            f.write(str(e))

    # metrics
    with open(os.path.join(outdir, "metrics.json"), "w") as f:
        json.dump(
            {
                "config": {
                    "horizon_steps": cfg.horizon_steps,
                    "n_splits": cfg.n_splits,
                    "test_window": cfg.test_window,
                    "min_train_size": cfg.min_train_size,
                    "step": cfg.step,
                    "val_fraction": cfg.val_fraction,
                    "purge_gap": cfg.purge_gap if cfg.purge_gap != -1 else cfg.horizon_steps,
                    "lags": list(cfg.lags),
                    "roll_windows": list(cfg.roll_windows),
                    "min_exog_lag": cfg.min_exog_lag,
                    "top_k_features": cfg.top_k_features,
                    "use_param_grid": cfg.use_param_grid,
                    "objective": "reg:pseudohubererror",
                    "residual_learning": "delta_level = y_level - persistence",
                    "feature_selection": "two-stage gain-based top-K per split",
                },
                "per_split": [
                    {
                        "imp_vs_persistence_rmse": res["imp_vs_persistence_rmse"],
                        "metrics": res["metrics"],
                        "baseline_metrics": res["baseline_metrics"],
                        "best_params": res["best_params"],
                        "top_features_head": res["top_features"][:30],
                        "top_features_count": len(res["top_features"]),
                    }
                    for res in split_results
                ],
                "summary": summary,
            },
            f,
            indent=2,
        )


# =========================
# Main
# =========================
def main() -> None:
    cfg = Config()
    purge_gap = cfg.horizon_steps if cfg.purge_gap == -1 else cfg.purge_gap

    outdir = cfg.output_dir_template.format(h=cfg.horizon_steps, k=cfg.top_k_features, elag=cfg.min_exog_lag)

    df_raw = load_data(cfg.data_path, cfg.date_col)

    # Build features + targets
    X, y_level, y_return, dates, bdi_t_all = build_features_and_targets(df_raw, cfg)

    print(f"Rows: {len(X)}, Features: {X.shape[1]}")
    print(f"Origin date range: {pd.to_datetime(dates).min().date()} → {pd.to_datetime(dates).max().date()}")
    print("Feature sample:", X.columns[:15].tolist())
    print(f"Using min_exog_lag={cfg.min_exog_lag}, top_k_features={cfg.top_k_features}")

    splits = walk_forward_splits(
        n_samples=len(X),
        n_splits=cfg.n_splits,
        test_window=cfg.test_window,
        min_train_size=cfg.min_train_size,
        step=cfg.step,
        purge_gap=purge_gap,
    )

    split_results: List[Dict[str, object]] = []
    for sidx, (train_idx, test_idx) in enumerate(splits, start=1):
        print(f"\nSplit {sidx}/{len(splits)}: train={len(train_idx)} test={len(test_idx)} (purge_gap={purge_gap})")

        res = train_eval_one_split(
            X=X,
            y_level=y_level,
            y_return=y_return,
            dates=dates,
            bdi_t_all=bdi_t_all,
            train_idx=train_idx,
            test_idx=test_idx,
            cfg=cfg,
        )
        split_results.append(res)

        print("Model metrics (LEVEL): ", res["metrics"]["level"])
        print("Model metrics (RETURN):", res["metrics"]["return"])

        # compact baselines
        compact = {}
        for bname, bm in res["baseline_metrics"].items():
            compact[bname] = {
                "level_RMSE": bm["level"]["RMSE"],
                "level_MAE": bm["level"]["MAE"],
                "return_DirAcc": bm["return"]["DirectionAcc"],
            }
        print("Baselines (compact):", compact)

        imp = res["imp_vs_persistence_rmse"]
        print(f"Improvement vs Persistence (LEVEL RMSE): {imp:+.4f}  (positive is better)")

    summary = aggregate_results(split_results)
    final_result = split_results[-1]

    save_artifacts(outdir, final_result, split_results, summary, cfg)

    print(f"\n✅ Artifacts saved to: {outdir}")
    print("\nSummary (model LEVEL RMSE mean/std):",
          summary["model_level"]["RMSE_mean"], summary["model_level"]["RMSE_std"])
    print("Summary (model RETURN DirectionAcc mean/std):",
          summary["model_return"]["DirectionAcc_mean"], summary["model_return"]["DirectionAcc_std"])
    print("Summary improvement vs persistence (LEVEL RMSE mean/std):",
          summary["imp_vs_persistence_rmse_mean"], summary["imp_vs_persistence_rmse_std"])


if __name__ == "__main__":
    main()
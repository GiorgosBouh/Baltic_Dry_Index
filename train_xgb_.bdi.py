#!/usr/bin/env python3
# -*- coding: utf-8 -*-

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

    # Forecast horizon in DAYS (date-aware)
    horizon_days: int = 10

    # Walk-forward evaluation
    n_splits: int = 5
    test_window: int = 120           # number of origin points (rows) per test window
    min_train_size: int = 365
    step: int = 120
    val_fraction: float = 0.15
    random_state: int = 42

    # Purge gap to reduce leakage overlap for multi-step
    purge_gap: int = -1              # if -1 => use horizon_days

    # Feature selection
    include_current_bdi: bool = True # allow BDI(t) as feature at origin
    strict_feature_filter: bool = True  # keep only lag/roll-like features (+ optional current BDI)

    # Model selection
    use_param_grid: bool = False

    # Outputs
    output_dir_template: str = "artifacts_xgb_bdi_h{h}d"


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
    # direction on returns (up/down)
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
    # MAPE is unstable for returns near 0; prefer RMSE/MAE/sMAPE/R2
    return {
        "RMSE": rmse(y_true_return, y_pred_return),
        "MAE": mae(y_true_return, y_pred_return),
        "sMAPE": smape(y_true_return, y_pred_return),
        "R2": r2_score(y_true_return, y_pred_return),
        "DirectionAcc": directional_accuracy(y_true_return, y_pred_return),
    }


# =========================
# Data + targets (DATE-AWARE)
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


def make_date_aware_targets(
    df: pd.DataFrame,
    target_col: str,
    date_col: str,
    horizon_days: int,
) -> pd.DataFrame:
    """
    Builds date-aware:
      - bdi_t: BDI at origin date t
      - bdi_future: BDI at date t + horizon_days (exact match)
      - y_return: log(BDI_{t+H}) - log(BDI_t)
      - y_level: BDI_{t+H}
    If the future date does not exist in the dataset, that row is dropped.
    """
    if target_col not in df.columns:
        raise ValueError(f"Missing target column '{target_col}' in dataset.")

    out = df.copy()
    out[target_col] = pd.to_numeric(out[target_col], errors="coerce")
    out = out.dropna(subset=[target_col]).reset_index(drop=True)

    s = out.set_index(date_col)[target_col].sort_index()
    # Ensure unique index (if duplicates, keep last)
    s = s[~s.index.duplicated(keep="last")]

    horizon_td = pd.Timedelta(days=int(horizon_days))
    origin_dates = out[date_col].values
    future_dates = out[date_col] + horizon_td

    # Date-aware lookup for future
    bdi_t = s.reindex(out[date_col]).to_numpy()
    bdi_future = s.reindex(future_dates).to_numpy()

    out["bdi_t"] = bdi_t
    out["bdi_future"] = bdi_future
    out = out.dropna(subset=["bdi_t", "bdi_future"]).reset_index(drop=True)

    # Safe log-return (BDI should be positive; if not, drop)
    out = out[(out["bdi_t"] > 0) & (out["bdi_future"] > 0)].reset_index(drop=True)
    out["y_level"] = out["bdi_future"]
    out["y_return"] = np.log(out["bdi_future"].to_numpy()) - np.log(out["bdi_t"].to_numpy())

    return out


# =========================
# Feature selection
# =========================
def select_feature_columns(
    df: pd.DataFrame,
    target_col: str,
    date_col: str,
    include_current_bdi: bool,
    strict_filter: bool,
) -> List[str]:
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    # Always drop target/targets
    drop = {date_col, target_col, "bdi_future", "y_level", "y_return"}
    feature_cols = [c for c in numeric_cols if c not in drop]

    # Optionally allow BDI(t) itself as a feature (origin)
    # We use 'bdi_t' (explicit origin BDI) rather than raw 'BDI' to be clear.
    if include_current_bdi and "bdi_t" in df.columns and "bdi_t" not in feature_cols:
        feature_cols.append("bdi_t")

    if not strict_filter:
        if not feature_cols:
            raise ValueError("No numeric features available.")
        return feature_cols

    # Strict: keep lag/rolling-like engineered features + optional bdi_t
    allowed_markers = ("lag", "roll", "moving", "avg", "mean", "_ma")
    filtered = []
    for col in feature_cols:
        if col == "bdi_t":
            filtered.append(col)
            continue
        cl = col.lower()
        if any(m in cl for m in allowed_markers):
            filtered.append(col)

    if not filtered:
        raise ValueError("No usable lag/rolling features found. Check Enriched_BDI_Dataset.csv columns.")
    return filtered


def build_feature_matrix(
    df: pd.DataFrame,
    target_col: str,
    date_col: str,
    include_current_bdi: bool,
    strict_filter: bool,
) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns:
      X: features at origin date t
      y_return: log-return target
      y_level: level target
      dates: origin dates
    """
    cols = select_feature_columns(df, target_col, date_col, include_current_bdi, strict_filter)
    X = df[cols].copy()
    dates = df[date_col].to_numpy()
    y_return = df["y_return"].to_numpy(dtype=float)
    y_level = df["y_level"].to_numpy(dtype=float)

    # Drop rows with NaNs in X (keep alignment)
    mask = np.isfinite(y_return) & np.isfinite(y_level)
    for c in X.columns:
        mask &= np.isfinite(X[c].to_numpy(dtype=float))
    X = X.loc[mask].reset_index(drop=True)
    y_return = y_return[mask]
    y_level = y_level[mask]
    dates = dates[mask]

    return X, y_return, y_level, dates


# =========================
# Walk-forward splits with PURGE GAP
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
        # Purge last 'purge_gap' points from training to avoid overlap leakage
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
        raise ValueError("No splits generated. Check min_train_size/test_window/purge_gap vs dataset length.")
    return splits


# =========================
# Date-aware baselines (LEVEL + returns)
# =========================
def compute_date_aware_baselines(
    origin_dates: np.ndarray,
    bdi_at_origin: np.ndarray,
    date_to_bdi: pd.Series,
    seasonal_days: int = 7,
    ma_days: int = 7,
) -> Dict[str, Dict[str, np.ndarray]]:
    """
    For each origin date t (in test set), produce baseline predictions for level at t+H:
      - persistence_level: BDI(t)
      - seasonal_naive_level: BDI(t - seasonal_days)
      - moving_average_level: mean BDI over (t-ma_days, t) excluding t (time-based window)
    Also returns baseline returns relative to BDI(t):
      return = log(level_pred) - log(BDI(t))
    """
    idx = pd.to_datetime(origin_dates)

    # persistence: level pred = BDI(t)
    persistence_level = np.asarray(bdi_at_origin, dtype=float)

    # seasonal naive: BDI(t - seasonal_days)
    seasonal_dates = idx - pd.Timedelta(days=int(seasonal_days))
    seasonal_level = date_to_bdi.reindex(seasonal_dates).to_numpy(dtype=float)

    # moving average over last ma_days (time-based, excluding t)
    # Use a helper: for each t, take s.loc[(t-ma_days, t)] excluding t itself
    s = date_to_bdi.sort_index()

    ma_level = []
    for t in idx:
        start = t - pd.Timedelta(days=int(ma_days))
        window = s.loc[(s.index > start) & (s.index < t)]
        ma_level.append(float(window.mean()) if len(window) else np.nan)
    ma_level = np.asarray(ma_level, dtype=float)

    # Convert to returns relative to origin BDI(t)
    eps = 1e-8
    bdi_safe = np.maximum(persistence_level, eps)

    def to_return(level_pred: np.ndarray) -> np.ndarray:
        level_pred = np.asarray(level_pred, dtype=float)
        # only valid when both > 0
        out = np.full_like(level_pred, np.nan, dtype=float)
        ok = np.isfinite(level_pred) & (level_pred > 0) & (bdi_safe > 0)
        out[ok] = np.log(level_pred[ok]) - np.log(bdi_safe[ok])
        return out

    return {
        "persistence": {"level": persistence_level, "return": to_return(persistence_level)},
        "seasonal_naive": {"level": seasonal_level, "return": to_return(seasonal_level)},
        "moving_average": {"level": ma_level, "return": to_return(ma_level)},
    }


# =========================
# Train / eval one split (predict RETURNS, reconstruct LEVEL)
# =========================
def train_eval_one_split(
    X: pd.DataFrame,
    y_return: np.ndarray,
    y_level: np.ndarray,
    dates: np.ndarray,
    bdi_t: np.ndarray,  # BDI at origin
    date_to_bdi: pd.Series,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    config: Config,
) -> Dict[str, object]:
    X_train_full = X.iloc[train_idx]
    y_train_full = y_return[train_idx]

    X_test = X.iloc[test_idx]
    y_test_return = y_return[test_idx]
    y_test_level = y_level[test_idx]
    dates_test = dates[test_idx]
    bdi_origin_test = bdi_t[test_idx]

    # Validation split from end of training window
    val_size = max(1, int(len(X_train_full) * config.val_fraction))
    X_train = X_train_full.iloc[:-val_size]
    y_train = y_train_full[:-val_size]
    X_val = X_train_full.iloc[-val_size:]
    y_val = y_train_full[-val_size:]

    # Small param grid (optional)
    param_grid = [
        {
            "max_depth": 6,
            "min_child_weight": 1,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "reg_alpha": 0.0,
            "reg_lambda": 1.0,
        },
        {
            "max_depth": 4,
            "min_child_weight": 5,
            "subsample": 0.9,
            "colsample_bytree": 0.9,
            "reg_alpha": 0.1,
            "reg_lambda": 1.0,
        },
        {
            "max_depth": 8,
            "min_child_weight": 3,
            "subsample": 0.7,
            "colsample_bytree": 0.7,
            "reg_alpha": 0.0,
            "reg_lambda": 2.0,
        },
    ]
    if not config.use_param_grid:
        param_grid = [param_grid[0]]

    best_model = None
    best_score = float("inf")
    best_params = None

    for params in param_grid:
        model = XGBRegressor(
            n_estimators=4000,
            learning_rate=0.03,
            objective="reg:squarederror",
            random_state=config.random_state,
            n_jobs=-1,
            eval_metric="rmse",
            early_stopping_rounds=150,
            **params,
        )
        model.fit(
            X_train,
            y_train,
            eval_set=[(X_val, y_val)],
            verbose=False,
        )
        val_pred = model.predict(X_val)
        val_rmse = rmse(y_val, val_pred)
        if val_rmse < best_score:
            best_score = val_rmse
            best_model = model
            best_params = params

    # Predict RETURNS
    y_pred_return = best_model.predict(X_test)

    # Reconstruct LEVEL: BDI_pred = BDI_t * exp(return_pred)
    eps = 1e-8
    bdi_origin_safe = np.maximum(bdi_origin_test, eps)
    y_pred_level = bdi_origin_safe * np.exp(y_pred_return)

    # Model metrics
    model_metrics_return = evaluate_return_metrics(y_test_return, y_pred_return)
    model_metrics_level = evaluate_level_metrics(y_test_level, y_pred_level)

    # Baselines (date-aware) for LEVEL at target time, plus returns relative to origin
    baselines = compute_date_aware_baselines(
        origin_dates=dates_test,
        bdi_at_origin=bdi_origin_test,
        date_to_bdi=date_to_bdi,
        seasonal_days=7,
        ma_days=7,
    )

    baseline_metrics = {}
    for name, preds in baselines.items():
        # LEVEL metrics (only where finite)
        lvl = preds["level"]
        mask_lvl = np.isfinite(lvl) & np.isfinite(y_test_level)
        baseline_metrics[name] = {
            "level": evaluate_level_metrics(y_test_level[mask_lvl], lvl[mask_lvl]),
        }
        # RETURN metrics
        ret = preds["return"]
        mask_ret = np.isfinite(ret) & np.isfinite(y_test_return)
        baseline_metrics[name]["return"] = evaluate_return_metrics(y_test_return[mask_ret], ret[mask_ret])

    return {
        "model": best_model,
        "best_params": best_params,
        "dates_test": dates_test,
        "bdi_origin_test": bdi_origin_test,
        "y_true_return": y_test_return,
        "y_pred_return": y_pred_return,
        "y_true_level": y_test_level,
        "y_pred_level": y_pred_level,
        "baselines": baselines,
        "metrics": {"return": model_metrics_return, "level": model_metrics_level},
        "baseline_metrics": baseline_metrics,
        "X_test": X_test,
    }


# =========================
# Aggregation + artifacts
# =========================
def aggregate_results(split_results: List[Dict[str, object]]) -> Dict[str, Dict[str, float]]:
    def summarize_dicts(dicts: List[Dict[str, float]]) -> Dict[str, float]:
        keys = list(dicts[0].keys())
        out = {}
        for k in keys:
            vals = [d[k] for d in dicts]
            out[f"{k}_mean"] = float(np.mean(vals))
            out[f"{k}_std"] = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
        return out

    model_return = [r["metrics"]["return"] for r in split_results]
    model_level = [r["metrics"]["level"] for r in split_results]

    summary = {
        "model_return": summarize_dicts(model_return),
        "model_level": summarize_dicts(model_level),
    }

    # Baselines
    baseline_names = list(split_results[0]["baseline_metrics"].keys())
    for bn in baseline_names:
        bret = [r["baseline_metrics"][bn]["return"] for r in split_results]
        blev = [r["baseline_metrics"][bn]["level"] for r in split_results]
        summary[f"{bn}_return"] = summarize_dicts(bret)
        summary[f"{bn}_level"] = summarize_dicts(blev)

    return summary


def save_artifacts(
    output_dir: str,
    final_result: Dict[str, object],
    split_results: List[Dict[str, object]],
    summary: Dict[str, Dict[str, float]],
    feature_names: List[str],
    config: Config,
) -> None:
    os.makedirs(output_dir, exist_ok=True)

    # Save final model
    joblib.dump(final_result["model"], os.path.join(output_dir, "xgb_bdi_model.joblib"))

    # Save predictions on last window (LEVEL)
    pred_df = pd.DataFrame(
        {
            "date_origin": pd.to_datetime(final_result["dates_test"]),
            "bdi_origin": final_result["bdi_origin_test"],
            "y_true_level": final_result["y_true_level"],
            "y_pred_level": final_result["y_pred_level"],
            "y_true_return": final_result["y_true_return"],
            "y_pred_return": final_result["y_pred_return"],
            "baseline_persistence_level": final_result["baselines"]["persistence"]["level"],
            "baseline_seasonal_level": final_result["baselines"]["seasonal_naive"]["level"],
            "baseline_movingavg_level": final_result["baselines"]["moving_average"]["level"],
        }
    )
    pred_df.to_csv(os.path.join(output_dir, "predictions_last_window.csv"), index=False)

    # Plot final window (LEVEL)
    dates = pd.to_datetime(final_result["dates_test"])
    plt.figure(figsize=(12, 5))
    plt.plot(dates, final_result["y_true_level"], label="Actual BDI(t+H)", linewidth=2)
    plt.plot(dates, final_result["y_pred_level"], label="XGBoost Forecast", linewidth=2)
    plt.plot(dates, final_result["baselines"]["persistence"]["level"], label="Persistence", linewidth=1.5, alpha=0.8)
    plt.title(f"BDI Forecast (H={config.horizon_days} days) — Final Window")
    plt.xlabel("Origin date t")
    plt.ylabel("BDI at t+H")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "actual_vs_pred_final.png"), dpi=200)
    plt.close()

    # SHAP (final window)
    try:
        explainer = shap.TreeExplainer(final_result["model"])
        shap_values = explainer.shap_values(final_result["X_test"])
        plt.figure()
        shap.summary_plot(shap_values, final_result["X_test"], show=False)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "shap_beeswarm.png"), dpi=200)
        plt.close()

        plt.figure()
        shap.summary_plot(shap_values, final_result["X_test"], show=False, plot_type="bar")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "shap_bar.png"), dpi=200)
        plt.close()
    except Exception as e:
        with open(os.path.join(output_dir, "shap_error.txt"), "w") as f:
            f.write(str(e))

    # Metrics JSON
    with open(os.path.join(output_dir, "metrics.json"), "w") as f:
        json.dump(
            {
                "config": {
                    "horizon_days": config.horizon_days,
                    "n_splits": config.n_splits,
                    "test_window": config.test_window,
                    "min_train_size": config.min_train_size,
                    "step": config.step,
                    "val_fraction": config.val_fraction,
                    "purge_gap": config.purge_gap if config.purge_gap != -1 else config.horizon_days,
                    "include_current_bdi": config.include_current_bdi,
                    "strict_feature_filter": config.strict_feature_filter,
                    "use_param_grid": config.use_param_grid,
                },
                "per_split": [
                    {
                        "metrics": res["metrics"],
                        "baseline_metrics": res["baseline_metrics"],
                        "best_params": res["best_params"],
                    }
                    for res in split_results
                ],
                "summary": summary,
                "feature_count": len(feature_names),
                "feature_names_head": feature_names[:20],
            },
            f,
            indent=2,
        )


# =========================
# Main
# =========================
def main() -> None:
    config = Config()
    purge_gap = config.horizon_days if config.purge_gap == -1 else config.purge_gap
    output_dir = config.output_dir_template.format(h=config.horizon_days)

    df_raw = load_data(config.data_path, config.date_col)
    df = make_date_aware_targets(df_raw, config.target_col, config.date_col, config.horizon_days)

    # Date->BDI series for date-aware baselines
    date_to_bdi = df.set_index(config.date_col)[config.target_col].sort_index()
    date_to_bdi = date_to_bdi[~date_to_bdi.index.duplicated(keep="last")]

    # Build X and targets
    X, y_return, y_level, dates = build_feature_matrix(
        df=df,
        target_col=config.target_col,
        date_col=config.date_col,
        include_current_bdi=config.include_current_bdi,
        strict_filter=config.strict_feature_filter,
    )
    feature_names = X.columns.tolist()

    # Need BDI(t) at origin for reconstruction & baselines
    if "bdi_t" in df.columns:
        bdi_t = df.loc[df.index[: len(dates)], "bdi_t"].to_numpy(dtype=float)
    else:
        # Fallback: align by dates against date_to_bdi
        bdi_t = date_to_bdi.reindex(pd.to_datetime(dates)).to_numpy(dtype=float)

    # Basic info
    print(f"Rows: {len(X)}, Features: {len(feature_names)}")
    print(f"Origin date range: {pd.to_datetime(dates).min().date()} → {pd.to_datetime(dates).max().date()}")
    print("Feature sample:", feature_names[:15])

    # Splits
    splits = walk_forward_splits(
        n_samples=len(X),
        n_splits=config.n_splits,
        test_window=config.test_window,
        min_train_size=config.min_train_size,
        step=config.step,
        purge_gap=purge_gap,
    )

    split_results: List[Dict[str, object]] = []
    for sidx, (train_idx, test_idx) in enumerate(splits, start=1):
        print(f"\nSplit {sidx}/{len(splits)}: train={len(train_idx)} test={len(test_idx)} (purge_gap={purge_gap})")
        res = train_eval_one_split(
            X=X,
            y_return=y_return,
            y_level=y_level,
            dates=dates,
            bdi_t=bdi_t,
            date_to_bdi=date_to_bdi,
            train_idx=train_idx,
            test_idx=test_idx,
            config=config,
        )
        split_results.append(res)
        print("Model metrics (RETURN):", res["metrics"]["return"])
        print("Model metrics (LEVEL): ", res["metrics"]["level"])
        # Compact baseline print (level RMSE/MAE + return DirectionAcc)
        compact = {}
        for bname, bm in res["baseline_metrics"].items():
            compact[bname] = {
                "level_RMSE": bm["level"]["RMSE"],
                "level_MAE": bm["level"]["MAE"],
                "return_DirAcc": bm["return"]["DirectionAcc"],
            }
        print("Baselines (compact):", compact)

    summary = aggregate_results(split_results)
    final_result = split_results[-1]

    save_artifacts(
        output_dir=output_dir,
        final_result=final_result,
        split_results=split_results,
        summary=summary,
        feature_names=feature_names,
        config=config,
    )

    print(f"\n✅ Artifacts saved to: {output_dir}")
    print("\nSummary (model level RMSE mean/std):",
          summary["model_level"]["RMSE_mean"], summary["model_level"]["RMSE_std"])
    print("Summary (model return DirectionAcc mean/std):",
          summary["model_return"]["DirectionAcc_mean"], summary["model_return"]["DirectionAcc_std"])


if __name__ == "__main__":
    main()
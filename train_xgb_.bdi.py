#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import os
from dataclasses import dataclass
from typing import Dict, Iterable, List, Tuple

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from xgboost import XGBRegressor


@dataclass(frozen=True)
class Config:
    data_path: str = "Enriched_BDI_Dataset.csv"
    target_col: str = "BDI"
    date_col: str = "date"
    horizon: int = 10
    n_splits: int = 5
    test_window: int = 120
    min_train_size: int = 365
    step: int = 120
    val_fraction: float = 0.15
    random_state: int = 42
    output_dir_template: str = "artifacts_xgb_bdi_h{h}"
    include_current_bdi: bool = True
    use_param_grid: bool = False


def rmse(y_true, y_pred) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def mae(y_true, y_pred) -> float:
    return float(np.mean(np.abs(y_true - y_pred)))


def mape(y_true, y_pred, eps=1e-8) -> float:
    denom = np.maximum(np.abs(y_true), eps)
    return float(np.mean(np.abs((y_true - y_pred) / denom)) * 100.0)


def smape(y_true, y_pred, eps=1e-8) -> float:
    denom = np.maximum(np.abs(y_true) + np.abs(y_pred), eps)
    return float(np.mean(2.0 * np.abs(y_true - y_pred) / denom) * 100.0)


def r2_score(y_true, y_pred) -> float:
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return float(1.0 - ss_res / ss_tot) if ss_tot > 0 else float("nan")


def load_data(path: str, date_col: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Dataset not found: {path}")
    df = pd.read_csv(path, parse_dates=[date_col])
    df = df.sort_values(date_col).reset_index(drop=True)
    return df


def make_supervised_target(df: pd.DataFrame, target_col: str, horizon: int) -> pd.DataFrame:
    df = df.copy()
    df["y_target"] = df[target_col].shift(-horizon)
    df = df.dropna(subset=["y_target"]).reset_index(drop=True)
    return df


def _select_feature_columns(
    df: pd.DataFrame,
    target_col: str,
    date_col: str,
    include_current_bdi: bool,
) -> List[str]:
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    drop_cols = {target_col, "y_target"}
    feature_cols = [c for c in numeric_cols if c not in drop_cols]
    if date_col in feature_cols:
        feature_cols.remove(date_col)

    allowed_markers = ("lag", "roll", "moving", "avg", "mean", "_ma")
    filtered_cols = []
    for col in feature_cols:
        if col == target_col and include_current_bdi:
            filtered_cols.append(col)
            continue
        if col.lower() == target_col.lower():
            continue
        col_lower = col.lower()
        if any(marker in col_lower for marker in allowed_markers) or ("ma_" in col_lower):
            filtered_cols.append(col)
    if include_current_bdi and target_col in df.columns and target_col not in filtered_cols:
        filtered_cols.append(target_col)
    if not filtered_cols:
        raise ValueError("No usable lag/rolling features found. Check Enriched_BDI_Dataset.csv columns.")
    return filtered_cols


def build_feature_matrix(
    df: pd.DataFrame,
    target_col: str,
    date_col: str,
    include_current_bdi: bool,
) -> Tuple[pd.DataFrame, pd.Series, pd.Series]:
    feature_cols = _select_feature_columns(df, target_col, date_col, include_current_bdi)
    X = df[feature_cols].copy()
    y = df["y_target"].copy()
    dates = df[date_col].copy()
    if X.isna().any().any():
        X = X.dropna()
        y = y.loc[X.index]
        dates = dates.loc[X.index]
    return X, y, dates


def walk_forward_splits(
    n_samples: int,
    n_splits: int,
    test_window: int,
    min_train_size: int,
    step: int,
) -> List[Tuple[np.ndarray, np.ndarray]]:
    splits = []
    train_end = min_train_size
    for _ in range(n_splits):
        test_start = train_end
        test_end = test_start + test_window
        if test_end > n_samples:
            break
        train_idx = np.arange(0, train_end)
        test_idx = np.arange(test_start, test_end)
        splits.append((train_idx, test_idx))
        train_end += step
    if not splits:
        raise ValueError("No splits generated. Check min_train_size/test_window against dataset length.")
    return splits


def compute_baselines(
    bdi_series: pd.Series,
    test_idx: np.ndarray,
    horizon: int,
    seasonal_lag: int = 7,
    ma_window: int = 7,
) -> Dict[str, np.ndarray]:
    bdi_shift0 = bdi_series
    bdi_shift_seasonal = bdi_series.shift(seasonal_lag)
    bdi_roll = bdi_series.shift(1).rolling(ma_window).mean()
    return {
        "persistence": bdi_shift0.iloc[test_idx].to_numpy(),
        "seasonal_naive": bdi_shift_seasonal.iloc[test_idx].to_numpy(),
        "moving_average": bdi_roll.iloc[test_idx].to_numpy(),
    }


def evaluate_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    return {
        "RMSE": rmse(y_true, y_pred),
        "MAE": mae(y_true, y_pred),
        "MAPE": mape(y_true, y_pred),
        "sMAPE": smape(y_true, y_pred),
        "R2": r2_score(y_true, y_pred),
    }


def train_eval_one_split(
    X: pd.DataFrame,
    y: pd.Series,
    dates: pd.Series,
    bdi_series: pd.Series,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    config: Config,
) -> Dict[str, object]:
    X_train_full = X.iloc[train_idx]
    y_train_full = y.iloc[train_idx]
    X_test = X.iloc[test_idx]
    y_test = y.iloc[test_idx]
    dates_test = dates.iloc[test_idx]

    val_size = max(1, int(len(X_train_full) * config.val_fraction))
    X_train = X_train_full.iloc[:-val_size]
    y_train = y_train_full.iloc[:-val_size]
    X_val = X_train_full.iloc[-val_size:]
    y_val = y_train_full.iloc[-val_size:]

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
            early_stopping_rounds=100,
            **params,
        )
        model.fit(
            X_train,
            y_train,
            eval_set=[(X_val, y_val)],
            verbose=False,
        )
        val_pred = model.predict(X_val)
        val_rmse = rmse(y_val.to_numpy(), val_pred)
        if val_rmse < best_score:
            best_score = val_rmse
            best_model = model
            best_params = params

    y_pred = best_model.predict(X_test)
    model_metrics = evaluate_metrics(y_test.to_numpy(), y_pred)

    baselines = compute_baselines(bdi_series, test_idx, config.horizon)
    baseline_metrics = {}
    for name, preds in baselines.items():
        mask = np.isfinite(preds)
        baseline_metrics[name] = evaluate_metrics(y_test.to_numpy()[mask], preds[mask])

    return {
        "model": best_model,
        "best_params": best_params,
        "y_true": y_test.to_numpy(),
        "y_pred": y_pred,
        "dates_test": dates_test.to_numpy(),
        "baselines": baselines,
        "metrics": model_metrics,
        "baseline_metrics": baseline_metrics,
    }


def aggregate_results(split_results: List[Dict[str, object]]) -> Dict[str, Dict[str, float]]:
    model_metrics = [res["metrics"] for res in split_results]
    baseline_metrics = {}
    for res in split_results:
        for name, metrics in res["baseline_metrics"].items():
            baseline_metrics.setdefault(name, []).append(metrics)

    def summarize(metrics_list: List[Dict[str, float]]) -> Dict[str, float]:
        keys = metrics_list[0].keys()
        summary = {}
        for key in keys:
            values = [m[key] for m in metrics_list]
            summary[f"{key}_mean"] = float(np.mean(values))
            summary[f"{key}_std"] = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
        return summary

    summary = {"model": summarize(model_metrics)}
    for name, metrics_list in baseline_metrics.items():
        summary[name] = summarize(metrics_list)
    return summary


def save_artifacts(
    output_dir: str,
    final_result: Dict[str, object],
    split_results: List[Dict[str, object]],
    summary: Dict[str, Dict[str, float]],
    feature_names: List[str],
) -> None:
    os.makedirs(output_dir, exist_ok=True)
    joblib.dump(final_result["model"], os.path.join(output_dir, "xgb_bdi_model.joblib"))

    dates = final_result["dates_test"]
    y_true = final_result["y_true"]
    y_pred = final_result["y_pred"]
    baselines = final_result["baselines"]

    pred_df = pd.DataFrame(
        {
            "date": dates,
            "y_true_level": y_true,
            "y_pred_level": y_pred,
            "baseline_persistence": baselines["persistence"],
            "baseline_seasonal_naive": baselines["seasonal_naive"],
            "baseline_moving_average": baselines["moving_average"],
        }
    )
    pred_df.to_csv(os.path.join(output_dir, "predictions_last_window.csv"), index=False)

    plt.figure(figsize=(12, 5))
    plt.plot(dates, y_true, label="Actual BDI", linewidth=2)
    plt.plot(dates, y_pred, label="XGBoost Forecast", linewidth=2)
    plt.plot(dates, baselines["persistence"], label="Persistence", linewidth=1.5, alpha=0.8)
    plt.title("BDI Forecast vs Actual (Final Window)")
    plt.xlabel("Date")
    plt.ylabel("BDI")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "actual_vs_pred_final.png"), dpi=200)
    plt.close()

    explainer = shap.TreeExplainer(final_result["model"])
    shap_values = explainer(final_result["X_test_final"])

    plt.figure()
    shap.summary_plot(shap_values, final_result["X_test_final"], show=False)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "shap_beeswarm.png"), dpi=200)
    plt.close()

    plt.figure()
    shap.summary_plot(shap_values, final_result["X_test_final"], show=False, plot_type="bar")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "shap_bar.png"), dpi=200)
    plt.close()

    with open(os.path.join(output_dir, "metrics.json"), "w") as f:
        json.dump(
            {
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
                "feature_names_head": feature_names[:15],
            },
            f,
            indent=2,
        )


def main() -> None:
    config = Config()
    output_dir = config.output_dir_template.format(h=config.horizon)

    df = load_data(config.data_path, config.date_col)
    df = make_supervised_target(df, config.target_col, config.horizon)
    X, y, dates = build_feature_matrix(df, config.target_col, config.date_col, config.include_current_bdi)
    feature_names = X.columns.tolist()

    print(f"Rows: {len(df)}, Features: {len(feature_names)}")
    print(f"Date range: {dates.min().date()} → {dates.max().date()}")
    print("Feature sample:", feature_names[:15])

    splits = walk_forward_splits(
        n_samples=len(X),
        n_splits=config.n_splits,
        test_window=config.test_window,
        min_train_size=config.min_train_size,
        step=config.step,
    )

    split_results = []
    for split_idx, (train_idx, test_idx) in enumerate(splits, start=1):
        print(f"\nSplit {split_idx}/{len(splits)}: train={len(train_idx)} test={len(test_idx)}")
        result = train_eval_one_split(
            X=X,
            y=y,
            dates=dates,
            bdi_series=df[config.target_col],
            train_idx=train_idx,
            test_idx=test_idx,
            config=config,
        )
        split_results.append(result)
        print("Model metrics:", result["metrics"])
        print("Baseline metrics:", result["baseline_metrics"])

    summary = aggregate_results(split_results)
    final_result = split_results[-1]
    final_result["X_test_final"] = X.iloc[splits[-1][1]]
    save_artifacts(output_dir, final_result, split_results, summary, feature_names)
    print(f"\n✅ Artifacts saved to: {output_dir}")


if __name__ == "__main__":
    main()

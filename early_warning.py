#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
EARLY WARNING (Volatility-Regime Event Forecasting) for BDI

Instead of forecasting the BDI value, we forecast whether we will enter a
HIGH-VOLATILITY regime within the next H steps.

Event definition (VOL regime):
- Compute 1-step log returns: ret1[t] = log(BDI[t] / BDI[t-1])
- Compute "future realized volatility" at time t:
    fut_vol[t] = std( ret1[t+1], ret1[t+2], ..., ret1[t+H] )
- Define event = 1 if fut_vol[t] is in the top q quantile (e.g., top 20%)
  (IMPORTANT: the threshold is computed on TRAIN ONLY per split -> no leakage)

Evaluation:
- Purged walk-forward CV (train grows, fixed test window)
- Metrics:
    * AUPRC (Average Precision)
    * Brier score (probability quality)
    * Recall@K (how many true events we catch if we alert on top K% probabilities)
    * Event rate (test)
- Baselines:
    * Random baseline AUPRC ≈ event rate
    * "Past volatility rule": use past rolling std as proxy and threshold it (train-only threshold)

Notes:
- This is a more "meaningful" early warning question for markets like BDI,
  where persistence often beats point forecasts.
"""

import os
import json
from dataclasses import dataclass
from typing import List, Tuple, Dict

import numpy as np
import pandas as pd

from xgboost import XGBClassifier
from sklearn.metrics import average_precision_score
from sklearn.metrics import brier_score_loss


# ---------------------------
# Config
# ---------------------------
@dataclass(frozen=True)
class Config:
    data_path: str = "Enriched_BDI_Dataset.csv"
    date_col: str = "date"
    target_col: str = "BDI"

    horizon: int = 10                 # (2) = 10 "steps" (rows)
    event_quantile: float = 0.80      # top 20% future volatility => event=1

    n_splits: int = 5
    test_window: int = 120
    min_train_size: int = 365
    step: int = 120
    purge_gap: int = 10               # MUST be >= horizon to avoid overlap leakage

    # Alert policy for Recall@K
    alert_top_frac: float = 0.20      # trigger alerts on top 20% highest-risk predictions

    # Feature policy
    include_current_bdi: bool = True  # allow using current BDI level at time t
    allowed_markers: Tuple[str, ...] = ("lag", "roll", "ma", "mean", "std", "ret")

    # Model
    random_state: int = 42
    n_estimators: int = 2000
    learning_rate: float = 0.02
    max_depth: int = 4
    subsample: float = 0.8
    colsample_bytree: float = 0.8
    reg_lambda: float = 1.0
    reg_alpha: float = 0.0

    output_dir: str = "artifacts_early_warning_volreg_h10steps"


# ---------------------------
# Utils
# ---------------------------
def safe_std(x: np.ndarray) -> float:
    if len(x) <= 1:
        return 0.0
    return float(np.std(x, ddof=1))


def recall_at_top_frac(y_true: np.ndarray, y_prob: np.ndarray, top_frac: float) -> float:
    """
    If we can only raise alerts for the top X% highest predicted risks,
    what fraction of true events do we catch?
    """
    n = len(y_true)
    k = int(np.ceil(n * top_frac))
    k = max(1, k)
    idx = np.argsort(-y_prob)[:k]
    tp = int(np.sum(y_true[idx] == 1))
    total_pos = int(np.sum(y_true == 1))
    return float(tp / total_pos) if total_pos > 0 else float("nan")


def walk_forward_splits(n: int, cfg: Config) -> List[Tuple[np.ndarray, np.ndarray]]:
    splits = []
    train_end = cfg.min_train_size
    for _ in range(cfg.n_splits):
        test_start = train_end
        test_end = test_start + cfg.test_window
        if test_end > n:
            break

        train_idx = np.arange(0, train_end)
        test_idx = np.arange(test_start, test_end)

        # purge gap: remove last purge_gap samples from train
        if cfg.purge_gap > 0:
            train_idx = train_idx[:-cfg.purge_gap] if len(train_idx) > cfg.purge_gap else np.array([], dtype=int)

        if len(train_idx) < 50:
            break

        splits.append((train_idx, test_idx))
        train_end += cfg.step

    if not splits:
        raise ValueError("No splits generated. Check min_train_size/test_window/step/purge_gap.")
    return splits


def select_feature_columns(df: pd.DataFrame, cfg: Config) -> List[str]:
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    # exclude target artifacts
    drop = {cfg.target_col, "ret1", "fut_vol", "event"}

    candidates = [c for c in num_cols if c not in drop]

    chosen = []
    for c in candidates:
        lc = c.lower()

        # Keep BDI itself only if asked
        if c == cfg.target_col:
            if cfg.include_current_bdi:
                chosen.append(c)
            continue

        # Keep if looks like lag/rolling/returns feature
        if any(m in lc for m in cfg.allowed_markers):
            chosen.append(c)

    # Fallback: include target_col if nothing
    if cfg.include_current_bdi and cfg.target_col in df.columns and cfg.target_col not in chosen:
        chosen.append(cfg.target_col)

    if not chosen:
        raise ValueError("No usable features found. Check your dataset columns / allowed_markers.")
    return chosen


# ---------------------------
# Build labels (future volatility regime)
# ---------------------------
def build_vol_regime_labels(df: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    df = df.sort_values(cfg.date_col).reset_index(drop=True).copy()

    # 1-step log returns
    df["ret1"] = np.log(df[cfg.target_col] / df[cfg.target_col].shift(1))

    # Future realized volatility over next H steps:
    # fut_vol[t] = std(ret1[t+1:t+H+1])
    # We compute by shifting ret1 backward and using rolling std.
    # ret1_lead = ret1 shifted -1 means at index t we have ret1[t+1]
    ret1_lead = df["ret1"].shift(-1)
    df["fut_vol"] = ret1_lead.rolling(window=cfg.horizon).apply(lambda x: safe_std(x.to_numpy()), raw=False)

    # Drop rows without fut_vol and ret1 (edges)
    df = df.dropna(subset=["fut_vol"]).reset_index(drop=True)

    return df


# ---------------------------
# Baseline: Past volatility rule
# ---------------------------
def past_vol_rule_probs(df: pd.DataFrame, cfg: Config) -> np.ndarray:
    """
    Simple rule feature: past rolling volatility at time t (based only on past returns)
    past_vol[t] = std(ret1[t], ret1[t-1], ..., ret1[t-H+1]) shifted by 1 (so strictly past)
    We'll use it as a "score" (higher vol -> higher risk).
    """
    past_vol = df["ret1"].shift(1).rolling(window=cfg.horizon).apply(lambda x: safe_std(x.to_numpy()), raw=False)
    return past_vol.to_numpy()


# ---------------------------
# Train/Eval per split
# ---------------------------
def train_eval_split(df: pd.DataFrame, cfg: Config, train_idx: np.ndarray, test_idx: np.ndarray) -> Dict[str, float]:
    # threshold for event computed on TRAIN ONLY
    train_fut_vol = df["fut_vol"].iloc[train_idx].to_numpy()
    thr = float(np.quantile(train_fut_vol, cfg.event_quantile))

    # create event labels (binary) using this split-specific threshold
    event = (df["fut_vol"].to_numpy() >= thr).astype(int)
    y_train = event[train_idx]
    y_test = event[test_idx]

    # Features
    feat_cols = select_feature_columns(df, cfg)
    X = df[feat_cols].copy()

    X_train = X.iloc[train_idx].to_numpy()
    X_test = X.iloc[test_idx].to_numpy()

    # class imbalance handling
    pos = max(1, int(np.sum(y_train == 1)))
    neg = max(1, int(np.sum(y_train == 0)))
    scale_pos_weight = neg / pos

    model = XGBClassifier(
        n_estimators=cfg.n_estimators,
        learning_rate=cfg.learning_rate,
        max_depth=cfg.max_depth,
        subsample=cfg.subsample,
        colsample_bytree=cfg.colsample_bytree,
        reg_lambda=cfg.reg_lambda,
        reg_alpha=cfg.reg_alpha,
        objective="binary:logistic",
        eval_metric="aucpr",
        random_state=cfg.random_state,
        n_jobs=-1,
        scale_pos_weight=scale_pos_weight,
    )

    model.fit(X_train, y_train)

    prob = model.predict_proba(X_test)[:, 1]

    # Metrics
    auprc = float(average_precision_score(y_test, prob)) if len(np.unique(y_test)) > 1 else float("nan")
    brier = float(brier_score_loss(y_test, prob))

    rec_k = recall_at_top_frac(y_test, prob, cfg.alert_top_frac)

    # Baseline: random AUPRC ≈ event rate in test
    event_rate_test = float(np.mean(y_test))

    # Baseline: Past-volatility rule (score -> probability-like via rank/normalization)
    pv = past_vol_rule_probs(df, cfg)
    pv_test = pv[test_idx]

    # Convert past-vol score to [0,1] via percentile rank (stable, no fitting needed)
    # (NaNs can happen at the start; handle by filling with median)
    pv_test_clean = pv_test.copy()
    if np.any(~np.isfinite(pv_test_clean)):
        med = np.nanmedian(pv_test_clean)
        pv_test_clean[~np.isfinite(pv_test_clean)] = med

    # rank-based probability proxy
    ranks = pd.Series(pv_test_clean).rank(pct=True).to_numpy()
    pv_prob = ranks

    pv_auprc = float(average_precision_score(y_test, pv_prob)) if len(np.unique(y_test)) > 1 else float("nan")
    pv_brier = float(brier_score_loss(y_test, pv_prob))
    pv_rec_k = recall_at_top_frac(y_test, pv_prob, cfg.alert_top_frac)

    return {
        "event_rate_test": event_rate_test,
        "AUPRC_model": auprc,
        "Brier_model": brier,
        "Recall@topFrac_model": rec_k,
        "AUPRC_pastVolRule": pv_auprc,
        "Brier_pastVolRule": pv_brier,
        "Recall@topFrac_pastVolRule": pv_rec_k,
        "threshold_fut_vol": thr,
        "feature_count": float(len(feat_cols)),
    }


# ---------------------------
# Main
# ---------------------------
def main():
    cfg = Config()
    os.makedirs(cfg.output_dir, exist_ok=True)

    df = pd.read_csv(cfg.data_path, parse_dates=[cfg.date_col])
    df = df.sort_values(cfg.date_col).reset_index(drop=True)

    # Build label frame
    df = build_vol_regime_labels(df, cfg)

    # Drop remaining NaNs in numeric features (light cleanup)
    df = df.dropna(subset=[cfg.target_col, "ret1", "fut_vol"]).reset_index(drop=True)

    # Pre-select features once (to report)
    feat_cols = select_feature_columns(df, cfg)

    # Global event rate preview (using global threshold is only for reporting)
    global_thr = float(np.quantile(df["fut_vol"].to_numpy(), cfg.event_quantile))
    global_event = (df["fut_vol"].to_numpy() >= global_thr).astype(int)

    print(f"Rows: {len(df)}, Features: {len(feat_cols)}")
    print(f"Date range: {df[cfg.date_col].min().date()} → {df[cfg.date_col].max().date()}")
    print(f"Global event rate (report-only): {float(np.mean(global_event)):.3f}")
    print(f"horizon={cfg.horizon} steps | event_quantile={cfg.event_quantile} | purge_gap={cfg.purge_gap}")
    print("Feature head:", feat_cols[:15])

    splits = walk_forward_splits(len(df), cfg)

    split_rows = []
    for i, (train_idx, test_idx) in enumerate(splits, start=1):
        print(f"\nSplit {i}/{len(splits)}: train={len(train_idx)} test={len(test_idx)}")
        res = train_eval_split(df, cfg, train_idx, test_idx)
        split_rows.append(res)

        print(f"Event rate (test): {res['event_rate_test']:.3f}")
        print(f"AUPRC  model: {res['AUPRC_model']:.3f} | pastVolRule: {res['AUPRC_pastVolRule']:.3f} | random≈{res['event_rate_test']:.3f}")
        print(f"Brier  model: {res['Brier_model']:.3f} | pastVolRule: {res['Brier_pastVolRule']:.3f}")
        print(f"Recall@top{int(cfg.alert_top_frac*100)}%  model: {res['Recall@topFrac_model']:.3f} | pastVolRule: {res['Recall@topFrac_pastVolRule']:.3f}")

    out_df = pd.DataFrame(split_rows)
    summary = out_df.mean(numeric_only=True)

    print("\n===== SUMMARY (mean over splits) =====")
    print(summary[["event_rate_test", "AUPRC_model", "AUPRC_pastVolRule", "Brier_model", "Brier_pastVolRule", "Recall@topFrac_model", "Recall@topFrac_pastVolRule"]])

    # Save artifacts
    out_df.to_csv(os.path.join(cfg.output_dir, "per_split_metrics.csv"), index=False)
    with open(os.path.join(cfg.output_dir, "summary.json"), "w") as f:
        json.dump(
            {
                "config": cfg.__dict__,
                "summary_mean": {k: float(v) for k, v in summary.to_dict().items()},
                "feature_count": len(feat_cols),
                "feature_head": feat_cols[:25],
            },
            f,
            indent=2,
        )

    print(f"\n✅ Saved to: {cfg.output_dir}")


if __name__ == "__main__":
    main()
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
early_warning_transition_v3.py

Early-warning transition forecasting for BDI:
- Not "will volatility be high in next H steps?" (easy; volatility persists)
- But "will we TRANSITION from low-vol NOW to high-vol within next H steps?"

This removes the trivial advantage of a past-volatility rule and tests whether
exogenous + lag features add real predictive value.

Key ideas:
1) Define a future-risk label based on FUTURE volatility (forward-looking),
   e.g., forward rolling std of 1-step returns over a window.
2) Define "CURRENT regime" based on PAST volatility (backward-looking).
3) Positive samples are those where:
   current regime is LOW, and within next H steps the future-risk becomes HIGH.
4) Time-series CV with purge_gap to avoid leakage (embargo around horizon).

Outputs:
- AUPRC, Brier score, Recall@topFrac for model vs baselines
- Saves artifacts to artifacts_early_warning_transition_h{H}steps

Requirements:
pip install pandas numpy scikit-learn xgboost matplotlib

(If you use shap, add shap, but not needed for v3 baseline.)
"""

import os
import json
import math
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from xgboost import XGBClassifier
from sklearn.metrics import average_precision_score
from sklearn.metrics import brier_score_loss


# -----------------------------
# Config
# -----------------------------
@dataclass(frozen=True)
class Config:
    data_path: str = "Enriched_BDI_Dataset.csv"
    date_col: str = "date"
    target_col: str = "BDI"

    # Forecast horizon in "steps" (rows)
    horizon: int = 10

    # Regime definition windows
    past_vol_window: int = 14      # for "current low-vol" filter
    future_vol_window: int = 10    # for "future high-vol" label

    # Quantiles for regimes (computed on TRAIN ONLY per split)
    low_vol_quantile: float = 0.50     # below median = "low"
    high_future_quantile: float = 0.80 # top 20% future vol = "high risk"

    # Walk-forward
    n_splits: int = 5
    test_window: int = 120
    min_train_size: int = 365
    step: int = 120
    purge_gap: int = 10

    # Feature delays / selection
    min_exog_lag: int = 14            # realistic delay for exogenous
    include_current_bdi: bool = True  # allow using BDI_t level itself

    # Model
    random_state: int = 42
    n_estimators: int = 2000
    learning_rate: float = 0.03
    max_depth: int = 4
    subsample: float = 0.8
    colsample_bytree: float = 0.8
    reg_lambda: float = 1.0
    reg_alpha: float = 0.0
    min_child_weight: int = 5
    early_stopping_rounds: int = 100

    # Decision-oriented metric
    top_frac_alerts: float = 0.20  # Recall@top20% alerts

    # Output
    output_dir: str = "artifacts_early_warning_transition_h{h}steps"


# -----------------------------
# Utilities
# -----------------------------
def safe_mkdir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


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
        splits.append((train_idx, test_idx))
        train_end += cfg.step
    if not splits:
        raise ValueError("No splits created; adjust min_train_size/test_window/step.")
    return splits


def recall_at_top_frac(y_true: np.ndarray, y_prob: np.ndarray, frac: float) -> float:
    """Recall if we alert only on top 'frac' highest predicted probabilities."""
    n = len(y_true)
    k = max(1, int(math.ceil(frac * n)))
    idx = np.argsort(-y_prob)[:k]
    positives = np.sum(y_true == 1)
    if positives == 0:
        return float("nan")
    return float(np.sum(y_true[idx] == 1) / positives)


def select_feature_columns(df: pd.DataFrame, cfg: Config) -> List[str]:
    """
    Keep numeric columns that look like lag/rolling features.
    Also filter exogenous features to have lag >= cfg.min_exog_lag.
    """
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    # drop raw target and helper cols
    drop = {
        cfg.target_col,
        "ret1",
        "past_vol",
        "future_vol",
        "future_high",
        "current_low",
        "y_transition",
    }
    candidates = [c for c in numeric_cols if c not in drop]

    # heuristics
    keep = []
    for c in candidates:
        cl = c.lower()

        # allow current level if requested (often called bdi_t or BDI or similar)
        if cfg.include_current_bdi and c in ["bdi_t", cfg.target_col]:
            keep.append(c)
            continue

        # typical engineered columns
        if ("lag" in cl) or ("roll" in cl) or ("ma" in cl) or ("mean" in cl) or ("std" in cl):
            # enforce exogenous lag delay if the name looks like "..._lag{K}"
            if "lag" in cl:
                # try to parse trailing lag number
                # examples: X_lag14, X_lag_14, Xlag14
                digits = "".join(ch for ch in cl[::-1] if ch.isdigit())
                lag_k = None
                if digits:
                    lag_k = int(digits[::-1])
                if lag_k is not None and lag_k < cfg.min_exog_lag:
                    # BUT: never drop BDI_lag* (autoregressive)
                    if "bdi" not in cl:
                        continue
            keep.append(c)

    if not keep:
        raise ValueError("No usable features found. Check dataset column naming.")
    return sorted(set(keep))


def compute_returns_and_vol(df: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    df = df.copy()

    # 1-step log return
    df["ret1"] = np.log(df[cfg.target_col]).diff()

    # Past vol: backward-looking rolling std of ret1, uses only history
    df["past_vol"] = df["ret1"].rolling(cfg.past_vol_window).std()

    # Future vol proxy: forward-looking rolling std of ret1 over future window
    # We compute it as rolling std on ret1 then shift backward to align at time t:
    # future_vol[t] = std(ret1[t+1 : t+future_window])
    df["future_vol"] = (
        df["ret1"]
        .shift(-1)
        .rolling(cfg.future_vol_window)
        .std()
        .shift(-(cfg.future_vol_window - 1))
    )

    return df


def build_transition_labels(df: pd.DataFrame, train_idx: np.ndarray, cfg: Config) -> pd.DataFrame:
    """
    Build labels using TRAIN-only thresholds (no leakage):
    - current_low: past_vol <= q_low (train)
    - future_high: future_vol >= q_high (train)
    - y_transition: current_low==1 AND (within next H steps future_high becomes 1)
    """
    df = df.copy()

    # thresholds from TRAIN portion only
    train_past = df.loc[train_idx, "past_vol"].dropna()
    train_future = df.loc[train_idx, "future_vol"].dropna()

    q_low = train_past.quantile(cfg.low_vol_quantile)
    q_high = train_future.quantile(cfg.high_future_quantile)

    df["current_low"] = (df["past_vol"] <= q_low).astype(int)
    df["future_high"] = (df["future_vol"] >= q_high).astype(int)

    # Transition: low now, and high appears within next H steps
    # i.e., max(future_high[t+1 : t+H]) == 1
    future_high_fwdmax = (
        df["future_high"]
        .shift(-1)
        .rolling(cfg.horizon)
        .max()
        .shift(-(cfg.horizon - 1))
    )
    df["y_transition"] = ((df["current_low"] == 1) & (future_high_fwdmax == 1)).astype(int)

    return df


def purge_indices(train_idx: np.ndarray, test_idx: np.ndarray, purge_gap: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Purge overlap around the boundary to reduce leakage from rolling windows/horizon.
    Removes last purge_gap samples from train and first purge_gap samples from test.
    """
    if purge_gap <= 0:
        return train_idx, test_idx
    train_idx_p = train_idx[:-purge_gap] if len(train_idx) > purge_gap else train_idx[:0]
    test_idx_p = test_idx[purge_gap:] if len(test_idx) > purge_gap else test_idx[:0]
    return train_idx_p, test_idx_p


def baseline_always_no_event(y_test: np.ndarray) -> Dict[str, float]:
    # Prob=0 for all
    prob = np.zeros_like(y_test, dtype=float)
    auprc = average_precision_score(y_test, prob) if np.any(y_test == 1) else float("nan")
    brier = brier_score_loss(y_test, prob)
    return {"AUPRC": float(auprc), "Brier": float(brier)}


def baseline_past_vol_rule(df: pd.DataFrame, idx: np.ndarray) -> np.ndarray:
    """
    Simple rule baseline: use normalized past_vol as "risk probability".
    This is fair now because we're predicting *transitions* (low->high),
    where past_vol alone should be weaker than in regime persistence.
    """
    pv = df.loc[idx, "past_vol"].to_numpy()
    pv = np.nan_to_num(pv, nan=np.nanmedian(pv))
    # min-max normalize
    mn, mx = np.min(pv), np.max(pv)
    if mx - mn < 1e-12:
        return np.zeros_like(pv)
    return (pv - mn) / (mx - mn)


# -----------------------------
# Main training / eval
# -----------------------------
def main():
    cfg = Config()
    out_dir = cfg.output_dir.format(h=cfg.horizon)
    safe_mkdir(out_dir)

    df = pd.read_csv(cfg.data_path, parse_dates=[cfg.date_col])
    df = df.sort_values(cfg.date_col).reset_index(drop=True)

    df = compute_returns_and_vol(df, cfg)

    # Remove early NaNs due to rolling
    df = df.dropna(subset=["ret1", "past_vol", "future_vol"]).reset_index(drop=True)

    # Prepare walk-forward splits
    splits = walk_forward_splits(len(df), cfg)

    # Feature selection (based on columns present AFTER drops)
    feature_cols = select_feature_columns(df, cfg)

    print(f"Rows: {len(df)}, Features: {len(feature_cols)}")
    print(f"Date range: {df[cfg.date_col].min().date()} → {df[cfg.date_col].max().date()}")
    print(f"horizon={cfg.horizon} steps | past_vol_window={cfg.past_vol_window} | future_vol_window={cfg.future_vol_window}")
    print(f"Transition definition: current_low (<=q{int(cfg.low_vol_quantile*100)}) -> future_high (>=q{int(cfg.high_future_quantile*100)}) within next H")
    print("Feature head:", feature_cols[:15])

    rows_summary = []

    for s, (train_idx, test_idx) in enumerate(splits, start=1):
        # Build labels using TRAIN-only thresholds
        df_l = build_transition_labels(df, train_idx, cfg)

        # Purge boundary
        tr_idx, te_idx = purge_indices(train_idx, test_idx, cfg.purge_gap)
        if len(tr_idx) == 0 or len(te_idx) == 0:
            print(f"\nSplit {s}: skipped (empty after purge)")
            continue

        # Filter to "current_low == 1" ONLY (we only care about warnings when we're currently low-vol)
        tr_mask = df_l.loc[tr_idx, "current_low"].to_numpy() == 1
        te_mask = df_l.loc[te_idx, "current_low"].to_numpy() == 1

        tr_idx2 = tr_idx[tr_mask]
        te_idx2 = te_idx[te_mask]

        # If too few samples remain, skip
        if len(tr_idx2) < 100 or len(te_idx2) < 30:
            print(f"\nSplit {s}: skipped (too few low-vol samples) train={len(tr_idx2)} test={len(te_idx2)}")
            continue

        X_train = df_l.loc[tr_idx2, feature_cols]
        y_train = df_l.loc[tr_idx2, "y_transition"].astype(int)
        X_test = df_l.loc[te_idx2, feature_cols]
        y_test = df_l.loc[te_idx2, "y_transition"].astype(int)

        # Basic cleanup: drop columns with NaN in train (rare but safe)
        bad_cols = X_train.columns[X_train.isna().any()].tolist()
        if bad_cols:
            X_train = X_train.drop(columns=bad_cols)
            X_test = X_test.drop(columns=bad_cols)

        # Pos/neg weights
        pos = int((y_train == 1).sum())
        neg = int((y_train == 0).sum())
        scale_pos_weight = (neg / max(pos, 1)) if pos > 0 else 1.0

        # Train/val split (time-respecting)
        val_size = max(1, int(0.15 * len(X_train)))
        X_tr = X_train.iloc[:-val_size]
        y_tr = y_train.iloc[:-val_size]
        X_val = X_train.iloc[-val_size:]
        y_val = y_train.iloc[-val_size:]

        model = XGBClassifier(
            n_estimators=cfg.n_estimators,
            learning_rate=cfg.learning_rate,
            max_depth=cfg.max_depth,
            subsample=cfg.subsample,
            colsample_bytree=cfg.colsample_bytree,
            reg_lambda=cfg.reg_lambda,
            reg_alpha=cfg.reg_alpha,
            min_child_weight=cfg.min_child_weight,
            objective="binary:logistic",
            eval_metric="aucpr",
            n_jobs=-1,
            random_state=cfg.random_state,
            scale_pos_weight=scale_pos_weight,
        )

        model.fit(
            X_tr, y_tr,
            eval_set=[(X_val, y_val)],
            verbose=False,
            early_stopping_rounds=cfg.early_stopping_rounds,
        )

        prob = model.predict_proba(X_test)[:, 1]

        # Metrics
        event_rate_test = float(np.mean(y_test))
        auprc_model = float(average_precision_score(y_test, prob)) if np.any(y_test == 1) else float("nan")
        brier_model = float(brier_score_loss(y_test, prob))
        recall_top = recall_at_top_frac(y_test.to_numpy(), prob, cfg.top_frac_alerts)

        # Baselines
        # 1) Always no-event
        base0 = baseline_always_no_event(y_test.to_numpy())

        # 2) past-vol rule (normalized past_vol) on the same filtered test indices
        pv_prob = baseline_past_vol_rule(df_l, te_idx2)
        auprc_pv = float(average_precision_score(y_test, pv_prob)) if np.any(y_test == 1) else float("nan")
        brier_pv = float(brier_score_loss(y_test, pv_prob))
        recall_pv = recall_at_top_frac(y_test.to_numpy(), pv_prob, cfg.top_frac_alerts)

        print(f"\nSplit {s}/{len(splits)}: train(low-vol)={len(tr_idx2)} test(low-vol)={len(te_idx2)} (purge_gap={cfg.purge_gap})")
        print(f"Transition event rate (test): {event_rate_test:.3f}")
        print(f"AUPRC  model: {auprc_model:.3f} | pastVolRule: {auprc_pv:.3f} | alwaysNoEvent: {base0['AUPRC']:.3f} | random≈{event_rate_test:.3f}")
        print(f"Brier  model: {brier_model:.3f} | pastVolRule: {brier_pv:.3f} | alwaysNoEvent: {base0['Brier']:.3f}")
        print(f"Recall@top{int(cfg.top_frac_alerts*100)}%  model: {recall_top:.3f} | pastVolRule: {recall_pv:.3f}")

        rows_summary.append({
            "split": s,
            "train_n": int(len(tr_idx2)),
            "test_n": int(len(te_idx2)),
            "event_rate_test": event_rate_test,
            "AUPRC_model": auprc_model,
            "AUPRC_pastVolRule": auprc_pv,
            "AUPRC_alwaysNoEvent": base0["AUPRC"],
            "Brier_model": brier_model,
            "Brier_pastVolRule": brier_pv,
            "Recall@topFrac_model": recall_top,
            "Recall@topFrac_pastVolRule": recall_pv,
            "best_iteration": int(getattr(model, "best_iteration", 0)),
            "scale_pos_weight": float(scale_pos_weight),
            "features_used": int(X_train.shape[1]),
        })

        # Save last split artifacts
        if s == len(splits):
            split_dir = os.path.join(out_dir, "last_split")
            safe_mkdir(split_dir)
            # model
            try:
                import joblib
                joblib.dump(model, os.path.join(split_dir, "xgb_transition_model.joblib"))
            except Exception:
                pass

            # predictions
            pred_df = pd.DataFrame({
                "date": df_l.loc[te_idx2, cfg.date_col].to_numpy(),
                "y_true": y_test.to_numpy(),
                "prob_model": prob,
                "prob_pastVolRule": pv_prob,
                "past_vol": df_l.loc[te_idx2, "past_vol"].to_numpy(),
                "future_vol": df_l.loc[te_idx2, "future_vol"].to_numpy(),
            })
            pred_df.to_csv(os.path.join(split_dir, "predictions.csv"), index=False)

            # config snapshot
            with open(os.path.join(split_dir, "config.json"), "w") as f:
                json.dump(cfg.__dict__, f, indent=2)

    # Summary
    if rows_summary:
        summary_df = pd.DataFrame(rows_summary)
        summary_path = os.path.join(out_dir, "summary.csv")
        summary_df.to_csv(summary_path, index=False)

        means = summary_df[[
            "event_rate_test",
            "AUPRC_model", "AUPRC_pastVolRule", "AUPRC_alwaysNoEvent",
            "Brier_model", "Brier_pastVolRule",
            "Recall@topFrac_model", "Recall@topFrac_pastVolRule"
        ]].mean(numeric_only=True)

        print("\n===== SUMMARY (mean over splits) =====")
        print(means)
        print(f"\n✅ Saved to: {out_dir}")
    else:
        print("\nNo valid splits produced results. Consider reducing purge_gap or adjusting windows.")


if __name__ == "__main__":
    main()
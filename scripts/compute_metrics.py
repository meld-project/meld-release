#!/usr/bin/env python3
"""
Compute evaluation metrics for malware detection experiments.

Supported metrics:
 - Macro-F1 (primary)
 - Accuracy
 - AUROC (requires probability scores)
 - AUPR  (requires probability scores)
 - TPR@FPR=1% (deployment metric; requires probability scores)

Input formats (CSV):
 - Binary classification (recommended):
     label,pred,score
     1,1,0.91
     0,0,0.02
     ...
   - label: 0/1 ground truth (1=malicious, 0=benign)
   - pred:  0/1 predicted label (optional if score is provided; threshold=0.5 used if pred absent)
   - score: probability/logit for positive class (optional but required for AUROC/AUPR/TPR@FPR)

Usage:
  python scripts/compute_metrics.py \
    --pred_file results/predictions.csv \
    --out_json results/metrics.json

"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from typing import Optional, Dict, Any

import numpy as np
import pandas as pd
from sklearn.metrics import (
    f1_score,
    accuracy_score,
    roc_auc_score,
    average_precision_score,
    roc_curve,
)


@dataclass
class MetricsResult:
    macro_f1: float
    accuracy: float
    auroc: Optional[float]
    aupr: Optional[float]
    tpr_at_fpr_001: Optional[float]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "macro_f1": self.macro_f1,
            "accuracy": self.accuracy,
            "auroc": self.auroc,
            "aupr": self.aupr,
            "tpr_at_fpr_001": self.tpr_at_fpr_001,
        }


def compute_tpr_at_fpr(scores: np.ndarray, labels: np.ndarray, target_fpr: float = 0.01) -> float:
    fpr, tpr, _ = roc_curve(labels, scores)
    # Interpolate TPR at or below target FPR
    mask = fpr <= target_fpr
    if not np.any(mask):
        return float(0.0)
    return float(np.max(tpr[mask]))


def compute_metrics(df: pd.DataFrame) -> MetricsResult:
    if "label" not in df.columns:
        raise ValueError("Input CSV must contain a 'label' column (0/1).")

    y_true = df["label"].astype(int).to_numpy()

    y_score = None
    if "score" in df.columns and df["score"].notna().any():
        y_score = df["score"].astype(float).to_numpy()

    if "pred" in df.columns and df["pred"].notna().any():
        y_pred = df["pred"].astype(int).to_numpy()
    else:
        if y_score is None:
            raise ValueError("Either 'pred' or 'score' must be provided to compute metrics.")
        y_pred = (y_score >= 0.5).astype(int)

    macro_f1 = float(f1_score(y_true, y_pred, average="macro"))
    acc = float(accuracy_score(y_true, y_pred))

    auroc = float(roc_auc_score(y_true, y_score)) if y_score is not None else None
    aupr = float(average_precision_score(y_true, y_score)) if y_score is not None else None
    tpr_fpr001 = compute_tpr_at_fpr(y_score, y_true, 0.01) if y_score is not None else None

    return MetricsResult(
        macro_f1=macro_f1,
        accuracy=acc,
        auroc=auroc,
        aupr=aupr,
        tpr_at_fpr_001=tpr_fpr001,
    )


def main():
    ap = argparse.ArgumentParser(description="Compute Malware Detection Metrics")
    ap.add_argument("--pred_file", required=True, help="CSV file with columns: label,[pred],[score]")
    ap.add_argument("--out_json", required=False, help="Path to save metrics JSON")
    args = ap.parse_args()

    df = pd.read_csv(args.pred_file)
    res = compute_metrics(df)
    metrics = res.to_dict()
    print(json.dumps(metrics, indent=2))
    if args.out_json:
        with open(args.out_json, "w", encoding="utf-8") as f:
            json.dump(metrics, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()




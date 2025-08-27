#!/usr/bin/env python3
"""
MELD 家族多分类（按 manifest 的 family_id/family）
- 读取恶意 manifest（CSV），筛选 family 作为多分类标签
- 从 md 根目录加载 Markdown 文本
- 使用 Qwen3-0.6B 提取第 until_layer 层特征（默认15）
- 训练多分类线性分类器（LogReg multinomial/Ridge），评估 holdout 性能
- 导出：整体/每类指标、混淆矩阵、逐样本预测CSV
"""

import os
import csv
import json
import argparse
from pathlib import Path
from typing import List, Tuple, Dict, Optional

os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
os.environ.setdefault("TRANSFORMERS_NO_FLAX", "1")
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

import numpy as np
import torch
import pandas as pd
from tqdm import tqdm

from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    accuracy_score,
)

# 导入特征提取器（包内相对导入）
from .feature_extractor import LayerwiseFeatureExtractor


def read_manifest(manifest_csv: str) -> pd.DataFrame:
    df = pd.read_csv(manifest_csv)
    # 需要列: sha256,label,family,family_id
    required = {"sha256", "label", "family", "family_id"}
    if not required.issubset(set(df.columns)):
        raise RuntimeError(f"manifest 缺少列: {required - set(df.columns)}")
    # 仅恶意
    df = df[df["label"] == "malicious"].copy()
    return df


def filter_families(df: pd.DataFrame, min_count: int = 50, top_k: Optional[int] = None) -> pd.DataFrame:
    # 统计家族频次
    counts = df.groupby("family_id").size().sort_values(ascending=False)
    if top_k is not None and top_k > 0:
        keep_ids = set(counts.head(top_k).index.tolist())
    else:
        keep_ids = set(counts[counts >= min_count].index.tolist())
    kept = df[df["family_id"].isin(keep_ids)].copy()
    return kept


def build_samples(df: pd.DataFrame, md_root: str, limit: Optional[int]) -> Tuple[List[str], List[str], List[str], List[int]]:
    texts: List[str] = []
    sha_list: List[str] = []
    fam_names: List[str] = []
    fam_ids: List[int] = []
    md_root_p = Path(md_root)
    for _, row in df.iterrows():
        sha = str(row["sha256"]) 
        fam = str(row["family"]) if not pd.isna(row["family"]) else "unknown"
        fam_id = int(row["family_id"]) if not pd.isna(row["family_id"]) else -1
        md_path = md_root_p / f"{sha}.md"
        if not md_path.exists():
            continue
        try:
            with open(md_path, "r", encoding="utf-8", errors="ignore") as f:
                texts.append(f.read())
            sha_list.append(sha)
            fam_names.append(fam)
            fam_ids.append(fam_id)
        except Exception:
            continue
        if limit is not None and len(texts) >= limit:
            break
    return texts, sha_list, fam_names, fam_ids


def encode_layer(texts: List[str], model_dir: str, device: str, until_layer: int, max_tokens: int, stride: int) -> np.ndarray:
    extractor = LayerwiseFeatureExtractor(model_dir=model_dir, device=device)
    feats: List[np.ndarray] = []
    iterator = enumerate(texts)
    for _, text in tqdm(iterator, total=len(texts), desc="Encoding (layer)"):
        X = extractor.encode_document_layers(text=text, max_tokens=max_tokens, stride=stride, until_layer=until_layer)
        feats.append(X.numpy()[until_layer - 1])  # 取指定层
    return np.stack(feats, axis=0)


def stratified_holdout(y: np.ndarray, val_ratio: float, test_ratio: float, seed: int = 42):
    idx = np.arange(len(y))
    sss1 = StratifiedShuffleSplit(n_splits=1, test_size=test_ratio, random_state=seed)
    train_val_idx, test_idx = next(sss1.split(idx, y))
    y_trv = y[train_val_idx]
    val_size_adjusted = val_ratio / (1 - test_ratio)
    sss2 = StratifiedShuffleSplit(n_splits=1, test_size=val_size_adjusted, random_state=seed)
    tr_idx_rel, va_idx_rel = next(sss2.split(train_val_idx, y_trv))
    train_idx = train_val_idx[tr_idx_rel]
    val_idx = train_val_idx[va_idx_rel]
    return train_idx, val_idx, test_idx


def save_confusion_csv(cm: np.ndarray, labels: List[str], out_csv: str):
    # 行: true, 列: pred
    with open(out_csv, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["true\\pred"] + labels)
        for i, lab in enumerate(labels):
            w.writerow([lab] + cm[i].tolist())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model_dir', required=True)
    ap.add_argument('--manifest', required=True)
    ap.add_argument('--md_dir', required=True)
    ap.add_argument('--min_count', type=int, default=50)
    ap.add_argument('--top_k', type=int, default=None)
    ap.add_argument('--until_layer', type=int, default=15)
    ap.add_argument('--max_tokens', type=int, default=1024)
    ap.add_argument('--stride', type=int, default=256)
    ap.add_argument('--val_ratio', type=float, default=0.1)
    ap.add_argument('--test_ratio', type=float, default=0.2)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--clf', choices=['logreg', 'ridge'], default='logreg')
    ap.add_argument('--gpu', type=int, default=0)
    ap.add_argument('--limit', type=int, default=None)
    ap.add_argument('--out_dir', type=str, default='/tmp/')
    ap.add_argument('--run_tag', type=str, default='MELD-family')
    ap.add_argument('--save_raw', action='store_true')
    args = ap.parse_args()

    device = f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu"
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("读取 manifest …")
    df = read_manifest(args.manifest)
    df = filter_families(df, min_count=args.min_count, top_k=args.top_k)
    print(f"筛选后家族数: {df['family_id'].nunique()}, 样本数: {len(df)}")

    print("构建文本样本 …")
    texts, sha_list, fam_names, fam_ids = build_samples(df, args.md_dir, args.limit)
    if not texts:
        raise RuntimeError("未能读取到有效的Markdown文本样本")

    # 标签编码（family_id -> 索引）
    le = LabelEncoder()
    y = le.fit_transform(fam_ids)
    classes = le.classes_.tolist()  # family_id 列表
    class_labels = []
    # 建立 family_id -> family_name 的映射（选 df 中第一个）
    id_to_name: Dict[int, str] = {}
    for fid, g in df.groupby('family_id'):
        name = str(g['family'].iloc[0]) if len(g) > 0 else str(fid)
        id_to_name[int(fid)] = name
    for fid in classes:
        class_labels.append(f"{int(fid)}:{id_to_name.get(int(fid), 'unknown')}")

    print("编码层特征 …")
    X = encode_layer(texts, args.model_dir, device, args.until_layer, args.max_tokens, args.stride)

    print("划分 train/val/test …")
    tr_idx, va_idx, te_idx = stratified_holdout(y, args.val_ratio, args.test_ratio, args.seed)
    X_tr, X_va, X_te = X[tr_idx], X[va_idx], X[te_idx]
    y_tr, y_va, y_te = y[tr_idx], y[va_idx], y[te_idx]

    print("训练分类器 …")
    if args.clf == 'logreg':
        clf = make_pipeline(
            StandardScaler(with_mean=True, with_std=True),
            LogisticRegression(max_iter=3000, class_weight='balanced', multi_class='multinomial')
        )
    else:
        clf = make_pipeline(
            StandardScaler(with_mean=True, with_std=True),
            RidgeClassifier(class_weight='balanced')
        )
    clf.fit(X_tr, y_tr)

    print("验证与测试评估 …")
    y_va_pred = clf.predict(X_va)
    y_te_pred = clf.predict(X_te)

    # 指标
    va_macro_f1 = f1_score(y_va, y_va_pred, average='macro')
    te_macro_f1 = f1_score(y_te, y_te_pred, average='macro')
    te_micro_f1 = f1_score(y_te, y_te_pred, average='micro')
    te_acc = accuracy_score(y_te, y_te_pred)

    # 混淆矩阵
    cm = confusion_matrix(y_te, y_te_pred)
    save_confusion_csv(cm, class_labels, str(out_dir / 'confusion_matrix_family.csv'))

    # 分类报告（测试）
    cls_rep = classification_report(y_te, y_te_pred, target_names=class_labels, output_dict=True)
    with open(out_dir / 'classification_report.json', 'w', encoding='utf-8') as f:
        json.dump(cls_rep, f, ensure_ascii=False, indent=2)

    # 逐样本预测（测试）
    rows = []
    for idx in te_idx:
        rows.append({
            'sha256': sha_list[idx],
            'true_family_id': int(fam_ids[idx]),
            'true_family_name': id_to_name.get(int(fam_ids[idx]), 'unknown'),
            'pred_family_id': int(classes[y_te_pred[list(te_idx).index(idx)]]),
            'pred_family_name': id_to_name.get(int(classes[y_te_pred[list(te_idx).index(idx)]]), 'unknown')
        })
    pd.DataFrame(rows).to_csv(out_dir / 'test_predictions.csv', index=False)

    summary = {
        'mode': 'family-holdout',
        'num_samples': int(len(y)),
        'num_classes': int(len(classes)),
        'until_layer': int(args.until_layer),
        'clf': args.clf,
        'val_macro_f1': float(va_macro_f1),
        'test_macro_f1': float(te_macro_f1),
        'test_micro_f1': float(te_micro_f1),
        'test_accuracy': float(te_acc),
        'class_labels': class_labels,
        'run_tag': args.run_tag,
    }
    with open(out_dir / 'results.json', 'w', encoding='utf-8') as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()





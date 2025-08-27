#!/usr/bin/env python3
"""
最小可运行的 OOD 评测（基于 LEC+PLR）：

功能：
- 家族留出或时间留出，构造 ID（训练分布）与 OOD（留出恶意）
- 选择 LEC 的最佳中间层（用 ID 的 train/val 分类指标）
- 计算三种 OOD 分数：MSP / Energy / Mahalanobis（高=像ID）
- 在验证集（ID_val + OOD_val）上定阈（TPR_ID=95%），测试集上报告 AUROC/AUPR（OOD为正类）和 FPR@95TPR

示例：
python3 src/lec/ood_eval.py \
  --index_csv experiments/lec_qwen3/manifests/dataset_with_family_time.csv \
  --model_dir models/cache/models--Qwen--Qwen3-0.6B/snapshots/c1899de289a04d12100db370d81485cdf75e47ca \
  --mode family --test_family AgentTesla --gpu 1 --progress \
  --out experiments/lec_qwen3/ood_family_AgentTesla_small.json
"""

import os
import csv
import json
import argparse
import sys
from typing import List, Dict, Optional, Tuple
from datetime import datetime

import numpy as np
import torch
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score
from sklearn.covariance import LedoitWolf
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
from tqdm import tqdm

from .feature_extractor import LayerwiseFeatureExtractor


def parse_datetime_safe(s: str) -> Optional[datetime]:
    if not s:
        return None
    s = str(s).strip().strip('"').strip("'")
    for fmt in ["%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%d %H:%M", "%Y-%m-%d"]:
        try:
            return datetime.strptime(s[:len(fmt)], fmt)
        except Exception:
            continue
    return None


def load_index(index_csv: str) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    with open(index_csv, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    return rows


def subsample_indices(idx: np.ndarray, limit: int, ensure_idx: np.ndarray | None = None, seed: int = 42) -> np.ndarray:
    if limit is None or limit <= 0 or len(idx) <= limit:
        return idx
    rng = np.random.default_rng(seed)
    if ensure_idx is not None and len(ensure_idx) > 0:
        ensure_idx = np.intersect1d(idx, ensure_idx)
    else:
        ensure_idx = np.array([], dtype=int)
    remaining = np.setdiff1d(idx, ensure_idx, assume_unique=False)
    need = max(0, limit - len(ensure_idx))
    if need > 0 and len(remaining) > 0:
        pick = rng.choice(remaining, size=min(need, len(remaining)), replace=False)
        chosen = np.concatenate([ensure_idx, pick])
    else:
        chosen = ensure_idx
    return chosen


def family_ood_split(rows: List[Dict[str, str]], test_family: str, limit: int | None, seed: int = 42) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    families = np.array([r.get('family', '') for r in rows])
    labels = np.array([int(r['label']) for r in rows], dtype=int)
    idx_all = np.arange(len(rows))

    # OOD 正类候选：目标家族的恶意
    ood_pos_all = idx_all[(labels == 1) & (families == test_family)]
    if len(ood_pos_all) == 0:
        raise RuntimeError(f"目标家族 {test_family} 无恶意样本，无法构造OOD。")

    # ID 池：其余样本
    id_pool = idx_all[families != test_family]

    # 若设置 limit，则从 (ID池 ∪ 目标家族恶意 ∪ 全体良性) 中子采样，保证包含目标家族恶意
    if limit is not None and limit > 0:
        keep = subsample_indices(idx_all, limit, ensure_idx=ood_pos_all, seed=seed)
        mask = np.isin(idx_all, keep)
        idx_all = idx_all[mask]
        families = families[mask]
        labels = labels[mask]
        # 重新计算集合
        ood_pos_all = idx_all[(labels == 1) & (families == test_family)]
        id_pool = idx_all[families != test_family]

    # 在 OOD 正类中切分 val/test（3:7）
    rng = np.random.default_rng(seed)
    rng.shuffle(ood_pos_all)
    n_val = max(1, int(round(len(ood_pos_all) * 0.3)))
    ood_val = ood_pos_all[:n_val]
    ood_test = ood_pos_all[n_val:]
    if len(ood_test) == 0:
        ood_test = ood_val
        ood_val = ood_pos_all[:1]

    # 从 ID 池中做 train / id_val / id_test：先取 ID 池的标签
    y_id = labels[np.isin(np.arange(len(labels)), np.where(np.isin(idx_all, id_pool))[0])]
    # 为方便，用绝对索引集合切分
    id_pool_idx = id_pool
    y_id_pool = labels[np.isin(idx_all, id_pool_idx)]
    sss1 = StratifiedShuffleSplit(n_splits=1, test_size=0.15, random_state=seed)  # 15% -> id_val
    tr_rel, id_val_rel = next(sss1.split(id_pool_idx, y_id_pool))
    id_train = id_pool_idx[tr_rel]
    id_val = id_pool_idx[id_val_rel]

    # 从剩余 ID 中切 id_test（约 20% of remaining）
    rem = np.setdiff1d(id_pool_idx, np.concatenate([id_train, id_val]))
    y_rem = labels[np.isin(idx_all, rem)]
    if len(rem) > 10 and len(np.unique(y_rem)) == 2:
        sss2 = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=seed)
        id_tr2_rel, id_test_rel = next(sss2.split(rem, y_rem))
        # 将 id_tr2_rel 加回训练，以增大可用训练量
        id_train = np.concatenate([id_train, rem[id_tr2_rel]])
        id_test = rem[id_test_rel]
    else:
        id_test = rem

    # 保障 id_test 非空；若为空则从 id_val 或 id_train 中挪少量样本
    if len(id_test) == 0:
        rng = np.random.default_rng(seed)
        if len(id_val) > 1:
            take = max(1, int(round(len(id_val) * 0.3)))
            take = min(take, len(id_val) - 1)
            moved = rng.choice(id_val, size=take, replace=False)
            id_val = np.setdiff1d(id_val, moved)
            id_test = moved
        elif len(id_train) > 1:
            take = max(1, int(round(len(id_train) * 0.1)))
            take = min(take, len(id_train) - 1)
            moved = rng.choice(id_train, size=take, replace=False)
            id_train = np.setdiff1d(id_train, moved)
            id_test = moved

    return id_train, id_val, id_test, ood_val, ood_test, labels


def time_ood_split(rows: List[Dict[str, str]], threshold: datetime, limit: int | None, seed: int = 42) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    times: List[Optional[datetime]] = []
    for r in rows:
        t = parse_datetime_safe(r.get('first_seen', '')) or parse_datetime_safe(r.get('mtime', ''))
        times.append(t)
    idx_all = np.arange(len(rows))
    labels = np.array([int(r['label']) for r in rows], dtype=int)

    before = idx_all[[i for i, t in enumerate(times) if (t is not None and t < threshold)]]
    after = idx_all[[i for i, t in enumerate(times) if (t is not None and t >= threshold)]]
    ood_pos_all = after[labels[after] == 1]
    if len(before) == 0 or len(ood_pos_all) == 0:
        raise RuntimeError('时间留出划分失败：before/after 为空或无恶意样本。')

    if limit is not None and limit > 0:
        keep = subsample_indices(idx_all, limit, ensure_idx=ood_pos_all, seed=seed)
        mask = np.isin(idx_all, keep)
        idx_all = idx_all[mask]
        labels = labels[mask]
        before = np.intersect1d(before, idx_all)
        after = np.intersect1d(after, idx_all)
        ood_pos_all = after[labels[np.searchsorted(idx_all, after)] == 1] if len(after) > 0 else after
        # 简化：重新按过滤后定义
        ood_pos_all = after[labels[np.isin(idx_all, after)] == 1]

    rng = np.random.default_rng(seed)
    rng.shuffle(ood_pos_all)
    n_val = max(1, int(round(len(ood_pos_all) * 0.3)))
    ood_val = ood_pos_all[:n_val]
    ood_test = ood_pos_all[n_val:]
    if len(ood_test) == 0:
        ood_test = ood_val
        ood_val = ood_pos_all[:1]

    # ID 池 = before（恶意+良性）
    y_id_pool = labels[np.isin(idx_all, before)]
    sss1 = StratifiedShuffleSplit(n_splits=1, test_size=0.15, random_state=seed)
    tr_rel, id_val_rel = next(sss1.split(before, y_id_pool))
    id_train = before[tr_rel]
    id_val = before[id_val_rel]

    rem = np.setdiff1d(before, np.concatenate([id_train, id_val]))
    y_rem = labels[np.isin(idx_all, rem)]
    if len(rem) > 10 and len(np.unique(y_rem)) == 2:
        sss2 = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=seed)
        id_tr2_rel, id_test_rel = next(sss2.split(rem, y_rem))
        id_train = np.concatenate([id_train, rem[id_tr2_rel]])
        id_test = rem[id_test_rel]
    else:
        id_test = rem

    # 保障 id_test 非空；若为空则从 id_val 或 id_train 中挪少量样本
    if len(id_test) == 0:
        rng = np.random.default_rng(seed)
        if len(id_val) > 1:
            take = max(1, int(round(len(id_val) * 0.3)))
            take = min(take, len(id_val) - 1)
            moved = rng.choice(id_val, size=take, replace=False)
            id_val = np.setdiff1d(id_val, moved)
            id_test = moved
        elif len(id_train) > 1:
            take = max(1, int(round(len(id_train) * 0.1)))
            take = min(take, len(id_train) - 1)
            moved = rng.choice(id_train, size=take, replace=False)
            id_train = np.setdiff1d(id_train, moved)
            id_test = moved

    return id_train, id_val, id_test, ood_val, ood_test, labels


def encode_rows(rows: List[Dict[str, str]], indices: np.ndarray, extractor: LayerwiseFeatureExtractor, max_tokens: int, stride: int, progress: bool) -> Tuple[np.ndarray, np.ndarray]:
    texts: List[str] = []
    labels: List[int] = []
    it = indices
    if progress:
        it = tqdm(indices, desc='Encoding docs (subset)')
    for i in it:
        path = rows[int(i)]['path']
        try:
            with open(path, 'r', encoding='utf-8', errors='ignore') as f:
                txt = f.read()
        except Exception:
            txt = ''
        texts.append(txt)
        labels.append(int(rows[int(i)]['label']))
    feats: List[np.ndarray] = []
    it2 = texts
    if progress:
        it2 = tqdm(texts, desc='Forward LEC layers')
    for t in it2:
        x = extractor.encode_document_layers(text=t, max_tokens=max_tokens, stride=stride)
        feats.append(x.numpy())
    X_layers = np.stack(feats, axis=0)
    y = np.array(labels, dtype=int)
    return X_layers, y


def pick_best_layer_plr(X_tr: np.ndarray, y_tr: np.ndarray, X_va: np.ndarray, y_va: np.ndarray, th_step: float = 0.01) -> Tuple[int, LogisticRegression, float]:
    best_layer = 1
    best_f1 = -1.0
    best_clf = None
    L = X_tr.shape[1]
    for li in range(L):
        clf = make_pipeline(
            StandardScaler(with_mean=True, with_std=True),
            LogisticRegression(max_iter=2000, class_weight='balanced')
        )
        clf.fit(X_tr[:, li, :], y_tr)
        prob = clf.predict_proba(X_va[:, li, :])[:, 1]
        # 用 Macro-F1 选层（阈值在val上扫描）
        best_th = 0.5
        best_local_f1 = -1.0
        for th in np.arange(0.0, 1.0 + 1e-9, th_step):
            pred = (prob >= th).astype(int)
            # 简单 macro-F1：正负各一类
            tp = ((pred == 1) & (y_va == 1)).sum()
            fp = ((pred == 1) & (y_va == 0)).sum()
            fn = ((pred == 0) & (y_va == 1)).sum()
            tn = ((pred == 0) & (y_va == 0)).sum()
            def f1(p, r):
                return 0.0 if (p + r) == 0 else (2*p*r/(p+r+1e-12))
            p_pos = 0.0 if (tp+fp)==0 else tp/(tp+fp+1e-12)
            r_pos = 0.0 if (tp+fn)==0 else tp/(tp+fn+1e-12)
            f1_pos = f1(p_pos, r_pos)
            p_neg = 0.0 if (tn+fn)==0 else tn/(tn+fn+1e-12)
            r_neg = 0.0 if (tn+fp)==0 else tn/(tn+fp+1e-12)
            f1_neg = f1(p_neg, r_neg)
            macro_f1 = 0.5*(f1_pos + f1_neg)
            if macro_f1 > best_local_f1:
                best_local_f1 = macro_f1
                best_th = float(th)
        if best_local_f1 > best_f1:
            best_f1 = best_local_f1
            best_layer = li + 1
            best_clf = clf
    return best_layer, best_clf, float(best_f1)


def compute_scores(clf_pipeline, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    # 返回 (msp, energy) with high=ID
    # logistic 回归：decision_function 作为单logit z
    z = clf_pipeline.decision_function(X)
    # 概率
    p = 1.0 / (1.0 + np.exp(-z))
    msp = np.maximum(p, 1.0 - p)
    # 能量：logsumexp([0, z]) = softplus(z)
    # 数值稳定：softplus
    energy = np.log1p(np.exp(-np.abs(z))) + np.maximum(z, 0.0)
    return msp.astype(float), energy.astype(float)


def compute_mahalanobis_scores(X_tr: np.ndarray, y_tr: np.ndarray, X_eval: np.ndarray) -> np.ndarray:
    # 在训练ID上估计 scaler -> 特征标准化 -> 均值/共享协方差（LedoitWolf）
    scaler = StandardScaler(with_mean=True, with_std=True)
    Xtr = scaler.fit_transform(X_tr)
    Xev = scaler.transform(X_eval)
    classes = np.unique(y_tr)
    mu: Dict[int, np.ndarray] = {}
    for c in classes:
        mu[c] = Xtr[y_tr == c].mean(axis=0)
    lw = LedoitWolf().fit(Xtr)
    prec = lw.precision_
    # d_c(x) = (x-mu)^T Σ^{-1} (x-mu)
    dists = []
    for c in classes:
        diff = Xev - mu[c]
        # (diff @ prec) * diff, sum over dim
        m = np.einsum('ij,jk,ik->i', diff, prec, diff)
        dists.append(m)
    d_min = np.min(np.stack(dists, axis=1), axis=1)
    s_maha = -d_min  # 高=像ID
    return s_maha.astype(float)


def compute_mahalanobis_scores_pca(X_tr: np.ndarray, y_tr: np.ndarray, X_eval: np.ndarray, pca_dim: int) -> np.ndarray:
    # 标准化 -> PCA -> LedoitWolf 共享协方差 -> Mahalanobis
    scaler = StandardScaler(with_mean=True, with_std=True)
    Xtr = scaler.fit_transform(X_tr)
    Xev = scaler.transform(X_eval)
    pca = PCA(n_components=min(pca_dim, Xtr.shape[1]), svd_solver='auto', random_state=42)
    Ztr = pca.fit_transform(Xtr)
    Zev = pca.transform(Xev)
    classes = np.unique(y_tr)
    mu: Dict[int, np.ndarray] = {}
    for c in classes:
        mu[c] = Ztr[y_tr == c].mean(axis=0)
    lw = LedoitWolf().fit(Ztr)
    prec = lw.precision_
    dists = []
    for c in classes:
        diff = Zev - mu[c]
        m = np.einsum('ij,jk,ik->i', diff, prec, diff)
        dists.append(m)
    d_min = np.min(np.stack(dists, axis=1), axis=1)
    s_maha = -d_min
    return s_maha.astype(float)


def compute_knn_scores_pca(X_tr: np.ndarray, X_eval: np.ndarray, pca_dim: int, k_neighbors: int = 5) -> np.ndarray:
    # 基于ID训练集的近邻距离（均值距离），距离越大越像OOD -> 返回高=ID的分数（取负）
    scaler = StandardScaler(with_mean=True, with_std=True)
    Xtr = scaler.fit_transform(X_tr)
    Xev = scaler.transform(X_eval)
    pca = PCA(n_components=min(pca_dim, Xtr.shape[1]), svd_solver='auto', random_state=42)
    Ztr = pca.fit_transform(Xtr)
    Zev = pca.transform(Xev)
    n_neighbors = max(1, min(k_neighbors, max(1, Ztr.shape[0] - 1)))
    nn = NearestNeighbors(n_neighbors=n_neighbors, algorithm='auto', n_jobs=-1)
    nn.fit(Ztr)
    dists, _ = nn.kneighbors(Zev, return_distance=True)
    avg_dist = dists.mean(axis=1)
    s_knn = -avg_dist
    return s_knn.astype(float)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--index_csv', required=True, type=str)
    parser.add_argument('--model_dir', required=True, type=str)
    parser.add_argument('--mode', choices=['family', 'time'], required=True)
    parser.add_argument('--test_family', type=str, default=None)
    parser.add_argument('--time_threshold', type=str, default=None)
    parser.add_argument('--max_tokens', type=int, default=1024)
    parser.add_argument('--stride', type=int, default=256)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--progress', action='store_true')
    parser.add_argument('--limit', type=int, default=None)
    parser.add_argument('--min_family_size', type=int, default=20)
    parser.add_argument('--tau_mode', type=str, choices=['tpr95', 'max_f1_ood'], default='tpr95', help='阈值选择策略：tpr95 或 max_f1_ood')
    parser.add_argument('--save_raw', action='store_true', help='保存最原始分数、配置与指标，便于复现实验')
    parser.add_argument('--run_tag', type=str, default='', help='自定义run标签，便于区分不同实验')
    parser.add_argument('--artifacts_dir', type=str, default='experiments/lec_qwen3/runs', help='实验产物根目录')
    parser.add_argument('--out', type=str, required=True)
    args = parser.parse_args()

    rows = load_index(args.index_csv)
    if not rows:
        raise RuntimeError('index_csv 为空或无法读取')

    seed = 42
    if args.mode == 'family':
        if not args.test_family:
            raise RuntimeError('--mode family 需提供 --test_family')
        id_train, id_val, id_test, ood_val, ood_test, labels = family_ood_split(rows, args.test_family, args.limit, seed=seed)
        split_meta = {'mode': 'family', 'test_family': args.test_family}
    else:
        if not args.time_threshold:
            raise RuntimeError('--mode time 需提供 --time_threshold')
        th_input = (args.time_threshold or '').strip().lower()
        th: Optional[datetime] = None
        q_val: Optional[float] = None
        # 支持 auto/quantile:x 写法
        if th_input in ('auto', 'automatic'):
            q_val = 0.6
        elif th_input.startswith('quantile:'):
            try:
                q_val = float(th_input.split(':', 1)[1])
                q_val = min(max(q_val, 0.05), 0.95)
            except Exception:
                q_val = 0.6
        else:
            th = parse_datetime_safe(args.time_threshold)
        if q_val is not None:
            # 解析时间字段，失败则回退到文件 mtime
            parsed_times: List[Optional[datetime]] = []
            for r in rows:
                t = (
                    parse_datetime_safe(r.get('last_seen', '')) or
                    parse_datetime_safe(r.get('last see', '')) or
                    parse_datetime_safe(r.get('first_seen', '')) or
                    parse_datetime_safe(r.get('first see', '')) or
                    parse_datetime_safe(r.get('mtime', ''))
                )
                if t is None:
                    p = r.get('path', '')
                    try:
                        if p and os.path.exists(p):
                            t = datetime.fromtimestamp(os.path.getmtime(p))
                    except Exception:
                        t = None
                parsed_times.append(t)
            ts = [int(t.timestamp()) for t in parsed_times if t is not None]
            if not ts:
                raise RuntimeError('无法解析 --time_threshold（时间字段缺失/格式不一致）')
            import numpy as _np
            # 带约束搜索分位点，确保 before/after 与恶意/良性分布不退化
            # 优先使用给定分位点，其次在 [0.4, 0.7] 网格内搜索可用阈值
            candidate_qs = [q_val] + [x/100.0 for x in range(40, 71, 5) if abs(x/100.0 - q_val) > 1e-6]
            labels_arr = _np.array([int(r['label']) for r in rows], dtype=int)
            best_th: Optional[datetime] = None
            for q in candidate_qs:
                cutoff = int(_np.quantile(_np.array(ts), q))
                th_try = datetime.fromtimestamp(cutoff)
                # 统计 before/after 类别情况
                before_idx = [i for i, t in enumerate(parsed_times) if (t is not None and t < th_try)]
                after_idx = [i for i, t in enumerate(parsed_times) if (t is not None and t >= th_try)]
                if not before_idx or not after_idx:
                    continue
                y_before = labels_arr[before_idx]
                y_after = labels_arr[after_idx]
                # 约束：before 同时包含恶/良；after 至少包含恶
                if len(_np.unique(y_before)) == 2 and (y_after == 1).sum() >= 1:
                    best_th = th_try
                    break
            if best_th is None:
                # 兜底：用 0.5 分位点
                cutoff = int(_np.quantile(_np.array(ts), 0.5))
                best_th = datetime.fromtimestamp(cutoff)
            th = best_th
            split_meta = {'mode': 'time', 'time_threshold': f'auto(q={q_val})', 'resolved_datetime': th.strftime('%Y-%m-%d %H:%M:%S')}
        else:
            if th is None:
                raise RuntimeError('无法解析 --time_threshold')
            split_meta = {'mode': 'time', 'time_threshold': args.time_threshold}
        id_train, id_val, id_test, ood_val, ood_test, labels = time_ood_split(rows, th, args.limit, seed=seed)

    device = f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu"
    extractor = LayerwiseFeatureExtractor(model_dir=args.model_dir, device=device)

    # 编码特征
    X_tr_layers, y_tr = encode_rows(rows, id_train, extractor, args.max_tokens, args.stride, args.progress)
    X_idval_layers, y_idval = encode_rows(rows, id_val, extractor, args.max_tokens, args.stride, args.progress)
    X_idtest_layers, y_idtest = encode_rows(rows, id_test, extractor, args.max_tokens, args.stride, args.progress)
    X_oodval_layers, _ = encode_rows(rows, ood_val, extractor, args.max_tokens, args.stride, args.progress)
    X_oodtest_layers, _ = encode_rows(rows, ood_test, extractor, args.max_tokens, args.stride, args.progress)

    # 选最佳层（用 ID 的 train/val 分类）
    best_layer, clf_pipeline, best_val_macro_f1 = pick_best_layer_plr(X_tr_layers, y_tr, X_idval_layers, y_idval)

    # 拿到该层的矩阵
    bl = best_layer - 1
    Xtr = X_tr_layers[:, bl, :]
    Xidv = X_idval_layers[:, bl, :]
    Xidt = X_idtest_layers[:, bl, :]
    Xodv = X_oodval_layers[:, bl, :]
    Xodt = X_oodtest_layers[:, bl, :]

    # MSP / Energy on val & test（高=ID）
    msp_id_val, energy_id_val = compute_scores(clf_pipeline, Xidv)
    msp_ood_val, energy_ood_val = compute_scores(clf_pipeline, Xodv)
    msp_id_test, energy_id_test = compute_scores(clf_pipeline, Xidt)
    msp_ood_test, energy_ood_test = compute_scores(clf_pipeline, Xodt)

    # Mahalanobis（高=ID）
    maha_id_val = compute_mahalanobis_scores(Xtr, y_tr, Xidv)
    maha_ood_val = compute_mahalanobis_scores(Xtr, y_tr, Xodv)
    maha_id_test = compute_mahalanobis_scores(Xtr, y_tr, Xidt)
    maha_ood_test = compute_mahalanobis_scores(Xtr, y_tr, Xodt)

    def pick_tau_at_tpr95(id_scores: np.ndarray, ood_scores: np.ndarray) -> float:
        # 二分或扫描，这里扫描以简洁（取所有分数并加小扰动）
        all_scores = np.concatenate([id_scores, ood_scores])
        taus = np.unique(all_scores)
        best_tau = np.min(taus)
        # TPR_ID = P(s >= tau | ID)
        target = 0.95
        for tau in np.sort(taus):
            tpr = (id_scores >= tau).mean() if len(id_scores) > 0 else 0.0
            if tpr >= target:
                best_tau = float(tau)
                break
        return best_tau

    def pick_tau_max_f1_ood(id_scores: np.ndarray, ood_scores: np.ndarray) -> float:
        # OOD为正类；预测OOD当 s_id < tau
        all_scores = np.concatenate([id_scores, ood_scores])
        taus = np.unique(all_scores)
        best_tau = np.min(taus)
        best_f1 = -1.0
        for tau in np.sort(taus):
            tp = float((ood_scores < tau).sum())
            fp = float((id_scores < tau).sum())
            fn = float((ood_scores >= tau).sum())
            precision = 0.0 if tp + fp == 0 else tp / (tp + fp)
            recall = 0.0 if tp + fn == 0 else tp / (tp + fn)
            f1 = 0.0 if precision + recall == 0 else (2 * precision * recall) / (precision + recall)
            if f1 > best_f1:
                best_f1 = f1
                best_tau = float(tau)
        return best_tau

    def eval_with_scores(id_scores_val: np.ndarray, ood_scores_val: np.ndarray, id_scores_test: np.ndarray, ood_scores_test: np.ndarray, tau_mode: str) -> Dict[str, float]:
        if tau_mode == 'max_f1_ood':
            tau = pick_tau_max_f1_ood(id_scores_val, ood_scores_val)
        else:
            tau = pick_tau_at_tpr95(id_scores_val, ood_scores_val)
        # FPR@95TPR：在测试集用同一 tau，OOD 被误判为ID的比例
        fpr95 = float((ood_scores_test >= tau).mean()) if len(ood_scores_test) > 0 else 1.0
        # AUROC/AUPR（OOD为正类），用 s_ood = -s_id 方向
        s_test = np.concatenate([-id_scores_test, -ood_scores_test])
        y_test = np.concatenate([np.zeros_like(id_scores_test), np.ones_like(ood_scores_test)])
        auroc = float(roc_auc_score(y_test, s_test)) if len(np.unique(y_test)) == 2 else float('nan')
        aupr = float(average_precision_score(y_test, s_test)) if len(np.unique(y_test)) == 2 else float('nan')
        
        # 添加F1指标计算（OOD为正类）
        y_pred_ood = np.concatenate([
            (id_scores_test < tau).astype(int),  # ID样本被预测为OOD
            (ood_scores_test < tau).astype(int)  # OOD样本被预测为OOD
        ])
        f1_ood = float(f1_score(y_test, y_pred_ood, average='binary')) if len(np.unique(y_test)) == 2 else float('nan')
        
        # 也计算最佳F1对应的阈值和分数
        tau_best_f1 = pick_tau_max_f1_ood(id_scores_val, ood_scores_val)
        y_pred_best_f1 = np.concatenate([
            (id_scores_test < tau_best_f1).astype(int),
            (ood_scores_test < tau_best_f1).astype(int)
        ])
        f1_ood_best = float(f1_score(y_test, y_pred_best_f1, average='binary')) if len(np.unique(y_test)) == 2 else float('nan')
        
        return {
            'tau': float(tau),
            'tau_mode': tau_mode,
            'fpr': fpr95,
            'fpr_label': 'FPR@95TPR',
            'auroc': auroc,
            'aupr': aupr,
            'f1_ood': f1_ood,  # 使用当前阈值的F1
            'f1_ood_best': f1_ood_best,  # 使用最佳F1阈值的F1
            'tau_best_f1': float(tau_best_f1)
        }

    res_msp = eval_with_scores(msp_id_val, msp_ood_val, msp_id_test, msp_ood_test, args.tau_mode)
    res_energy = eval_with_scores(energy_id_val, energy_ood_val, energy_id_test, energy_ood_test, args.tau_mode)
    res_maha = eval_with_scores(maha_id_val, maha_ood_val, maha_id_test, maha_ood_test, args.tau_mode)

    # 改进分数：PCA+Mahalanobis、层集成、kNN
    pca_dim = 256
    # 用训练ID（最佳层）特征拟合PCA与协方差
    maha_pca_id_val = compute_mahalanobis_scores_pca(Xtr, y_tr, Xidv, pca_dim)
    maha_pca_ood_val = compute_mahalanobis_scores_pca(Xtr, y_tr, Xodv, pca_dim)
    maha_pca_id_test = compute_mahalanobis_scores_pca(Xtr, y_tr, Xidt, pca_dim)
    maha_pca_ood_test = compute_mahalanobis_scores_pca(Xtr, y_tr, Xodt, pca_dim)
    res_maha_pca = eval_with_scores(maha_pca_id_val, maha_pca_ood_val, maha_pca_id_test, maha_pca_ood_test, args.tau_mode)

    # 层集成（L9–L16）：对该区间的马氏距离取最大“像ID”分数（即各层 s_maha 的最大值）
    li_start, li_end = 9, 16
    li_start = max(1, li_start)
    li_end = min(X_tr_layers.shape[1], li_end)
    def maha_layerwise_max(X_layers_eval: np.ndarray) -> np.ndarray:
        scores_layers = []
        for li in range(li_start-1, li_end):
            s = compute_mahalanobis_scores_pca(X_tr_layers[:, li, :], y_tr, X_layers_eval[:, li, :], pca_dim)
            scores_layers.append(s)
        return np.max(np.stack(scores_layers, axis=1), axis=1)
    mahaL_id_val = maha_layerwise_max(X_idval_layers)
    mahaL_ood_val = maha_layerwise_max(X_oodval_layers)
    mahaL_id_test = maha_layerwise_max(X_idtest_layers)
    mahaL_ood_test = maha_layerwise_max(X_oodtest_layers)
    res_maha_layer = eval_with_scores(mahaL_id_val, mahaL_ood_val, mahaL_id_test, mahaL_ood_test, args.tau_mode)

    # kNN距离（PCA 256）
    knn_id_val = compute_knn_scores_pca(Xtr, Xidv, pca_dim, k_neighbors=5)
    knn_ood_val = compute_knn_scores_pca(Xtr, Xodv, pca_dim, k_neighbors=5)
    knn_id_test = compute_knn_scores_pca(Xtr, Xidt, pca_dim, k_neighbors=5)
    knn_ood_test = compute_knn_scores_pca(Xtr, Xodt, pca_dim, k_neighbors=5)
    res_knn = eval_with_scores(knn_id_val, knn_ood_val, knn_id_test, knn_ood_test, args.tau_mode)

    # 分组指标：根据家族规模划分 OOD 测试为 main(≥min_family_size) 与 rare(<min)
    def eval_grouped(id_scores_test: np.ndarray, ood_scores_test: np.ndarray, ood_indices: np.ndarray, tau: float) -> Dict[str, Dict[str, float]]:
        # 统计全局家族规模（按恶意样本计数）
        fam_counts: Dict[str, int] = {}
        for r in rows:
            if int(r['label']) == 1:
                fam = r.get('family', '') or ''
                fam_counts[fam] = fam_counts.get(fam, 0) + 1
        ood_fams = [rows[int(i)].get('family', '') or '' for i in ood_indices]
        mask_main = np.array([fam_counts.get(f, 0) >= args.min_family_size for f in ood_fams], dtype=bool)
        mask_rare = ~mask_main
        def eval_mask(mask: np.ndarray) -> Dict[str, float]:
            if mask.sum() == 0:
                return {'tau_tpr95': float(tau), 'fpr_at_95tpr': float('nan'), 'auroc': float('nan'), 'aupr': float('nan'), 'f1_ood': float('nan'), 'num_ood': 0}
            fpr = float((ood_scores_test[mask] >= tau).mean())
            s_test = np.concatenate([-id_scores_test, -ood_scores_test[mask]])
            y_test = np.concatenate([np.zeros_like(id_scores_test), np.ones(mask.sum(), dtype=int)])
            auroc = float(roc_auc_score(y_test, s_test)) if len(np.unique(y_test)) == 2 else float('nan')
            aupr = float(average_precision_score(y_test, s_test)) if len(np.unique(y_test)) == 2 else float('nan')
            # 计算F1分数（OOD为正类）
            y_pred = np.concatenate([
                (id_scores_test < tau).astype(int),
                (ood_scores_test[mask] < tau).astype(int)
            ])
            f1_ood = float(f1_score(y_test, y_pred, average='binary')) if len(np.unique(y_test)) == 2 else float('nan')
            return {'tau_tpr95': float(tau), 'fpr_at_95tpr': fpr, 'auroc': auroc, 'aupr': aupr, 'f1_ood': f1_ood, 'num_ood': int(mask.sum())}
        return {'main': eval_mask(mask_main), 'rare': eval_mask(mask_rare)}

    grouped = {
        'msp': eval_grouped(msp_id_test, msp_ood_test, ood_test, res_msp['tau']),
        'energy': eval_grouped(energy_id_test, energy_ood_test, ood_test, res_energy['tau']),
        'mahalanobis': eval_grouped(maha_id_test, maha_ood_test, ood_test, res_maha['tau']),
        'mahalanobis_pca256': eval_grouped(maha_pca_id_test, maha_pca_ood_test, ood_test, res_maha_pca['tau']),
        'mahalanobis_layer_max_L9_L16_pca256': eval_grouped(mahaL_id_test, mahaL_ood_test, ood_test, res_maha_layer['tau']),
        'knn_pca256_k5': eval_grouped(knn_id_test, knn_ood_test, ood_test, res_knn['tau']),
    }

    summary = {
        'split': split_meta,
        'limit': int(args.limit) if args.limit is not None else None,
        'best_layer': int(best_layer),
        'val_macro_f1_id_cls': float(best_val_macro_f1),
        'scores': {
            'msp': res_msp,
            'energy': res_energy,
            'mahalanobis': res_maha,
            'mahalanobis_pca256': res_maha_pca,
            'mahalanobis_layer_max_L9_L16_pca256': res_maha_layer,
            'knn_pca256_k5': res_knn,
        },
        'grouped_by_family_size': grouped,
        'tau_mode': args.tau_mode,
        'min_family_size': int(args.min_family_size),
    }
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, 'w', encoding='utf-8') as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(json.dumps(summary, ensure_ascii=False, indent=2))

    # 保存原始产物（可复现）
    if args.save_raw:
        ts = datetime.now().strftime('%Y%m%d_%H%M%S')
        tag = args.run_tag.strip().replace(' ', '_')
        ident = split_meta.get('test_family', split_meta.get('time_threshold', ''))
        run_name = f"{ts}_ood_eval_{split_meta['mode']}_{ident}"
        if tag:
            run_name += f"_{tag}"
        run_dir = os.path.join(args.artifacts_dir, run_name)
        os.makedirs(run_dir, exist_ok=True)
        # configs
        cfg_dir = os.path.join(run_dir, 'configs'); os.makedirs(cfg_dir, exist_ok=True)
        with open(os.path.join(cfg_dir, 'config.json'), 'w', encoding='utf-8') as f:
            json.dump({k: (v if isinstance(v, (int, float, str, bool, type(None))) else str(v)) for k, v in vars(args).items()}, f, ensure_ascii=False, indent=2)
        env = {
            'python': sys.version,
            'numpy': np.__version__,
            'torch': getattr(torch, '__version__', 'n/a'),
            'sklearn': __import__('sklearn').__version__,
            'model_dir': args.model_dir,
        }
        with open(os.path.join(cfg_dir, 'env.json'), 'w', encoding='utf-8') as f:
            json.dump(env, f, ensure_ascii=False, indent=2)
        with open(os.path.join(cfg_dir, 'cmd.txt'), 'w', encoding='utf-8') as f:
            f.write(' '.join(sys.argv))
        # manifests/meta
        man_dir = os.path.join(run_dir, 'manifests'); os.makedirs(man_dir, exist_ok=True)
        with open(os.path.join(man_dir, 'split_meta.json'), 'w', encoding='utf-8') as f:
            json.dump(split_meta, f, ensure_ascii=False, indent=2)
        # raw scores
        raw_dir = os.path.join(run_dir, 'raw'); os.makedirs(raw_dir, exist_ok=True)
        np.save(os.path.join(raw_dir, 'msp_id_val.npy'), msp_id_val)
        np.save(os.path.join(raw_dir, 'msp_ood_val.npy'), msp_ood_val)
        np.save(os.path.join(raw_dir, 'msp_id_test.npy'), msp_id_test)
        np.save(os.path.join(raw_dir, 'msp_ood_test.npy'), msp_ood_test)
        np.save(os.path.join(raw_dir, 'energy_id_val.npy'), energy_id_val)
        np.save(os.path.join(raw_dir, 'energy_ood_val.npy'), energy_ood_val)
        np.save(os.path.join(raw_dir, 'energy_id_test.npy'), energy_id_test)
        np.save(os.path.join(raw_dir, 'energy_ood_test.npy'), energy_ood_test)
        np.save(os.path.join(raw_dir, 'maha_id_val.npy'), maha_id_val)
        np.save(os.path.join(raw_dir, 'maha_ood_val.npy'), maha_ood_val)
        np.save(os.path.join(raw_dir, 'maha_id_test.npy'), maha_id_test)
        np.save(os.path.join(raw_dir, 'maha_ood_test.npy'), maha_ood_test)
        np.save(os.path.join(raw_dir, 'maha_pca_id_val.npy'), maha_pca_id_val)
        np.save(os.path.join(raw_dir, 'maha_pca_ood_val.npy'), maha_pca_ood_val)
        np.save(os.path.join(raw_dir, 'maha_pca_id_test.npy'), maha_pca_id_test)
        np.save(os.path.join(raw_dir, 'maha_pca_ood_test.npy'), maha_pca_ood_test)
        np.save(os.path.join(raw_dir, 'mahaL_id_val.npy'), mahaL_id_val)
        np.save(os.path.join(raw_dir, 'mahaL_ood_val.npy'), mahaL_ood_val)
        np.save(os.path.join(raw_dir, 'mahaL_id_test.npy'), mahaL_id_test)
        np.save(os.path.join(raw_dir, 'mahaL_ood_test.npy'), mahaL_ood_test)
        np.save(os.path.join(raw_dir, 'knn_id_val.npy'), knn_id_val)
        np.save(os.path.join(raw_dir, 'knn_ood_val.npy'), knn_ood_val)
        np.save(os.path.join(raw_dir, 'knn_id_test.npy'), knn_id_test)
        np.save(os.path.join(raw_dir, 'knn_ood_test.npy'), knn_ood_test)
        # metrics
        met_dir = os.path.join(run_dir, 'metrics'); os.makedirs(met_dir, exist_ok=True)
        with open(os.path.join(met_dir, 'metrics.json'), 'w', encoding='utf-8') as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)


if __name__ == '__main__':
    main()



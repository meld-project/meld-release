#!/usr/bin/env python3
"""
逐层 LEC 训练脚本（不集成 Venn-Abers）
- 从本地 Qwen3-0.6B 提取每层文档向量，训练 L2 逻辑回归/岭分类
- 通过 K 折交叉验证选最佳中间层
- 在验证集扫描阈值（步长可与 mdreport_eval.py 一致，如 0.01）
"""

import os
import json
import argparse
from typing import List, Tuple, Optional, Dict

# 设置环境变量避免TensorFlow兼容性问题（必须在导入transformers相关库之前）
os.environ.setdefault("TRANSFORMERS_NO_TF", "1") 
os.environ.setdefault("TRANSFORMERS_NO_FLAX", "1")

import numpy as np
import torch
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.metrics import f1_score, average_precision_score, roc_auc_score, accuracy_score
from tqdm import tqdm

# 在设置环境变量后再导入feature_extractor
from .feature_extractor import LayerwiseFeatureExtractor


def load_md_reports(md_dir: str, limit: Optional[int] = None) -> Tuple[List[str], List[int]]:
    """
    读取 Markdown 文本与二分类标签。
    优先规则：
      1) 若存在同名 .label 文件（内容为0/1），使用该标签；
      2) 否则从父目录名推断：'black'->1, 'white'->0；'unknown' 跳过（不参与训练）。
    """
    texts: List[str] = []
    labels: List[int] = []
    files: List[str] = []
    for root, _, fnames in os.walk(md_dir, followlinks=True):
        for fn in fnames:
            if fn.endswith('.md'):
                files.append(os.path.join(root, fn))
    files.sort()

    skipped_unknown = 0
    for md_path in files:
        label_path = os.path.splitext(md_path)[0] + '.label'
        label: Optional[int] = None
        if os.path.exists(label_path):
            try:
                with open(label_path, 'r', encoding='utf-8', errors='ignore') as f:
                    lab = f.read().strip()
                    label = int(lab)
            except Exception:
                label = None
        if label is None:
            # 从父目录名推断
            parts = os.path.normpath(md_path).split(os.sep)
            parts_lower = [p.lower() for p in parts]
            if ('black' in parts_lower) or ('_all_malicious_md' in parts_lower) or any('malicious' in p for p in parts_lower):
                label = 1
            elif ('white' in parts_lower) or ('_all_benign_md' in parts_lower) or any('benign' in p for p in parts_lower):
                label = 0
            elif 'unknown' in parts_lower:
                skipped_unknown += 1
                continue
            else:
                # 无法推断标签，跳过
                continue
        with open(md_path, 'r', encoding='utf-8', errors='ignore') as f:
            texts.append(f.read())
        labels.append(label)
    if skipped_unknown > 0:
        print(f"提示: 跳过 unknown 样本 {skipped_unknown} 个（未参与训练）。")

    # 若需要限制样本数，则做分层抽样，确保两类均存在
    if limit is not None and len(labels) > limit and limit >= 2:
        y_arr = np.array(labels, dtype=int)
        idx_all = np.arange(len(labels))
        pos_idx = idx_all[y_arr == 1]
        neg_idx = idx_all[y_arr == 0]
        if len(pos_idx) == 0 or len(neg_idx) == 0:
            # 无法分层，直接截断
            keep = idx_all[:limit]
        else:
            # 按总体比例近似分配，并确保每类至少1个
            pos_ratio = len(pos_idx) / len(idx_all)
            n_pos = max(1, int(round(limit * pos_ratio)))
            n_pos = min(n_pos, len(pos_idx) - 0)
            n_neg = max(1, limit - n_pos)
            n_neg = min(n_neg, len(neg_idx) - 0)
            # 若由于上限导致总数不足，再补齐
            while n_pos + n_neg < limit:
                if len(pos_idx) - n_pos > len(neg_idx) - n_neg and n_pos < len(pos_idx):
                    n_pos += 1
                elif n_neg < len(neg_idx):
                    n_neg += 1
                else:
                    break
            rng = np.random.default_rng(42)
            keep_pos = rng.choice(pos_idx, size=n_pos, replace=False)
            keep_neg = rng.choice(neg_idx, size=n_neg, replace=False)
            keep = np.concatenate([keep_pos, keep_neg])
            rng.shuffle(keep)
        texts = [texts[i] for i in keep]
        labels = [labels[i] for i in keep]

    return texts, labels


def scan_threshold(y_true: np.ndarray, y_prob: np.ndarray, step: float = 0.01) -> Tuple[float, float, float]:
    """
    在 [0,1] 扫描阈值，返回最佳阈值、对应F1与AUPR。
    """
    best_f1 = -1.0
    best_th = 0.5
    for th in np.arange(0.0, 1.0 + 1e-9, step):
        y_pred = (y_prob >= th).astype(int)
        f1 = f1_score(y_true, y_pred, average='macro')
        if f1 > best_f1:
            best_f1 = f1
            best_th = float(th)
    aupr = average_precision_score(y_true, y_prob)
    return best_th, best_f1, aupr


def kfold_eval_layer(X: np.ndarray, y: np.ndarray, clf_name: str, n_splits: int = 10, th_step: float = 0.01) -> dict:
    classes, counts = np.unique(y, return_counts=True)
    if len(classes) < 2:
        raise ValueError("数据仅包含单一类别，无法进行二分类训练/评估。")
    effective_splits = max(2, min(n_splits, int(counts.min())))
    skf = StratifiedKFold(n_splits=effective_splits, shuffle=True, random_state=42)
    preds = np.zeros_like(y, dtype=float)
    idx_all = np.arange(len(y))
    for train_idx, val_idx in skf.split(X, y):
        X_tr, X_va = X[train_idx], X[val_idx]
        y_tr, y_va = y[train_idx], y[val_idx]
        if clf_name == 'logreg':
            clf = make_pipeline(
                StandardScaler(with_mean=True, with_std=True),
                LogisticRegression(max_iter=2000, class_weight='balanced')
            )
        else:
            clf = make_pipeline(
                StandardScaler(with_mean=True, with_std=True),
                RidgeClassifier(class_weight='balanced')
            )
        clf.fit(X_tr, y_tr)
        # 使用 decision_function 或 predict_proba
        if hasattr(clf[-1], 'predict_proba'):
            prob = clf.predict_proba(X_va)[:, 1]
        else:
            # 将 decision_function 标准化到 [0,1]
            score = clf.decision_function(X_va)
            prob = (score - score.min()) / (score.max() - score.min() + 1e-8)
        preds[val_idx] = prob
    # 全部折叠后的评估
    best_th, best_f1, aupr = scan_threshold(y, preds, step=th_step)
    auroc = roc_auc_score(y, preds)
    return {
        'best_threshold': best_th,
        'macro_f1': float(best_f1),
        'aupr': float(aupr),
        'auroc': float(auroc)
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_dir', type=str, required=True, help='本地 Qwen3-0.6B 路径')
    parser.add_argument('--md_dir', type=str, required=True, help='Markdown 报告目录（含 black/white/unknown）')
    parser.add_argument('--split_mode', type=str, choices=['cv', 'holdout', 'dir'], default='cv', help='数据划分方式')
    parser.add_argument('--val_ratio', type=float, default=0.1, help='holdout 模式验证集占比')
    parser.add_argument('--test_ratio', type=float, default=0.2, help='holdout 模式测试集占比')
    parser.add_argument('--train_dir', type=str, default=None, help='dir 模式下训练目录')
    parser.add_argument('--val_dir', type=str, default=None, help='dir 模式下验证目录')
    parser.add_argument('--test_dir', type=str, default=None, help='dir 模式下测试目录')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--limit', type=int, default=None)
    parser.add_argument('--max_tokens', type=int, default=1024)
    parser.add_argument('--stride', type=int, default=256)
    parser.add_argument('--clf', type=str, choices=['logreg', 'ridge'], default='logreg')
    parser.add_argument('--n_splits', type=int, default=10)
    parser.add_argument('--th_step', type=float, default=0.01)
    parser.add_argument('--until_layer', type=int, default=None, help='仅前向到该层（1-based）；None为全部层')
    parser.add_argument('--out', type=str, default=None, help='结果 JSON 输出路径')
    parser.add_argument('--gpu', type=int, default=0, help='GPU 编号，如使用 1 号显卡则置为 1')
    parser.add_argument('--progress', action='store_true', help='显示进度条')
    parser.add_argument('--save_raw', action='store_true', help='保存原始实验数据和详细结果')
    parser.add_argument('--run_tag', type=str, default='', help='实验标记，用于标识不同的实验运行')
    args = parser.parse_args()

    def encode_texts(texts_list: List[str]) -> np.ndarray:
        device = f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu"
        extractor = LayerwiseFeatureExtractor(model_dir=args.model_dir, device=device)
        features: List[np.ndarray] = []
        iterator = enumerate(texts_list)
        if args.progress:
            iterator = tqdm(iterator, total=len(texts_list), desc='Encoding docs')
        for _, text in iterator:
            feats = extractor.encode_document_layers(
                text=text,
                max_tokens=args.max_tokens,
                stride=args.stride,
                until_layer=args.until_layer,
            )
            features.append(feats.numpy())
        X_layers_local = np.stack(features, axis=0)
        return X_layers_local

    def evaluate_cv(X_layers: np.ndarray, labels_arr: np.ndarray) -> Dict:
        L = X_layers.shape[1]
        results = []
        best = None
        layer_iter = range(L)
        if args.progress:
            layer_iter = tqdm(layer_iter, desc='Evaluating layers (CV)')
        for li in layer_iter:
            X = X_layers[:, li, :]
            metrics = kfold_eval_layer(X, labels_arr, clf_name=args.clf, n_splits=args.n_splits, th_step=args.th_step)
            metrics['layer_index'] = int(li + 1)
            results.append(metrics)
            if (best is None) or (metrics['macro_f1'] > best['macro_f1']):
                best = metrics
        H = X_layers.shape[2]
        result = {
            'num_samples': int(len(labels_arr)),
            'num_layers': int(L),
            'hidden_size': int(H),
            'clf': args.clf,
            'cv_splits': args.n_splits,
            'threshold_step': args.th_step,
            'best': best,
            'all_layers': results,
        }
        if args.save_raw:
            result['save_raw'] = True
        if args.run_tag:
            result['run_tag'] = args.run_tag
        return result

    def stratified_holdout_indices(y_arr: np.ndarray, val_ratio: float, test_ratio: float, seed: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        idx = np.arange(len(y_arr))
        classes, counts = np.unique(y_arr, return_counts=True)
        min_count = counts.min()
        n_total = len(y_arr)
        
        # 确保每个划分至少有每类1个样本
        min_test = max(len(classes), int(n_total * test_ratio))
        min_val = max(len(classes), int(n_total * val_ratio))
        
        if min_test + min_val >= n_total:
            # 样本太少，简单按类均分
            pos_idx = idx[y_arr == 1]
            neg_idx = idx[y_arr == 0]
            rng = np.random.default_rng(seed)
            rng.shuffle(pos_idx)
            rng.shuffle(neg_idx)
            n_pos_test = max(1, len(pos_idx) // 3)
            n_neg_test = max(1, len(neg_idx) // 3)
            test_idx = np.concatenate([pos_idx[:n_pos_test], neg_idx[:n_neg_test]])
            remaining_pos = pos_idx[n_pos_test:]
            remaining_neg = neg_idx[n_neg_test:]
            n_pos_val = max(1, len(remaining_pos) // 2) if len(remaining_pos) > 1 else len(remaining_pos)
            n_neg_val = max(1, len(remaining_neg) // 2) if len(remaining_neg) > 1 else len(remaining_neg)
            val_idx = np.concatenate([remaining_pos[:n_pos_val], remaining_neg[:n_neg_val]])
            train_idx = np.concatenate([remaining_pos[n_pos_val:], remaining_neg[n_neg_val:]])
            return train_idx, val_idx, test_idx
        
        # 正常情况：先拆出测试集
        sss1 = StratifiedShuffleSplit(n_splits=1, test_size=test_ratio, random_state=seed)
        train_val_idx, test_idx = next(sss1.split(idx, y_arr))
        # 再从 train_val 中拆出验证集
        y_train_val = y_arr[train_val_idx]
        val_size_adjusted = val_ratio / (1 - test_ratio)
        val_size_adjusted = max(len(classes) / len(train_val_idx), min(val_size_adjusted, 0.8))
        sss2 = StratifiedShuffleSplit(n_splits=1, test_size=val_size_adjusted, random_state=seed)
        train_idx_rel, val_idx_rel = next(sss2.split(train_val_idx, y_train_val))
        train_idx = train_val_idx[train_idx_rel]
        val_idx = train_val_idx[val_idx_rel]
        return train_idx, val_idx, test_idx

    # 载入数据
    if args.split_mode == 'dir':
        if not (args.train_dir and args.val_dir and args.test_dir):
            raise RuntimeError('dir 模式需要同时提供 --train_dir/--val_dir/--test_dir')
        tr_texts, tr_labels = load_md_reports(args.train_dir, None)
        va_texts, va_labels = load_md_reports(args.val_dir, None)
        te_texts, te_labels = load_md_reports(args.test_dir, None)
        if not tr_texts or not va_texts or not te_texts:
            raise RuntimeError('dir 模式下某个子集为空，请检查目录结构。')
        X_tr = encode_texts(tr_texts)
        X_va = encode_texts(va_texts)
        X_te = encode_texts(te_texts)
        y_tr = np.array(tr_labels, dtype=int)
        y_va = np.array(va_labels, dtype=int)
        y_te = np.array(te_labels, dtype=int)
        # 简单逐层在 val 上选阈值，在 test 上报告指标
        L = X_tr.shape[1]
        results = []
        best = None
        layer_iter = range(L)
        if args.progress:
            layer_iter = tqdm(layer_iter, desc='Evaluating layers (holdout)')
        for li in layer_iter:
            # 拟合
            if args.clf == 'logreg':
                clf = make_pipeline(
                    StandardScaler(with_mean=True, with_std=True),
                    LogisticRegression(max_iter=2000, class_weight='balanced')
                )
            else:
                clf = make_pipeline(
                    StandardScaler(with_mean=True, with_std=True),
                    RidgeClassifier(class_weight='balanced')
                )
            clf.fit(X_tr[:, li, :], y_tr)
            # val 上选阈值
            if hasattr(clf[-1], 'predict_proba'):
                val_prob = clf.predict_proba(X_va[:, li, :])[:, 1]
            else:
                val_score = clf.decision_function(X_va[:, li, :])
                val_prob = (val_score - val_score.min()) / (val_score.max() - val_score.min() + 1e-8)
            best_th, _, _ = scan_threshold(y_va, val_prob, step=args.th_step)
            # test 上评估
            if hasattr(clf[-1], 'predict_proba'):
                te_prob = clf.predict_proba(X_te[:, li, :])[:, 1]
            else:
                te_score = clf.decision_function(X_te[:, li, :])
                te_prob = (te_score - te_score.min()) / (te_score.max() - te_score.min() + 1e-8)
            te_pred = (te_prob >= best_th).astype(int)
            m = {
                'layer_index': int(li + 1),
                'best_threshold': float(best_th),
                'macro_f1': float(f1_score(y_te, te_pred, average='macro')),
                'aupr': float(average_precision_score(y_te, te_prob)),
                'auroc': float(roc_auc_score(y_te, te_prob)),
                'accuracy': float(accuracy_score(y_te, te_pred)),
            }
            results.append(m)
            if (best is None) or (m['macro_f1'] > best['macro_f1']):
                best = m
        H = X_tr.shape[2]
        summary = {
            'mode': 'holdout-dir',
            'num_layers': int(L),
            'hidden_size': int(H),
            'clf': args.clf,
            'threshold_step': args.th_step,
            'until_layer': int(args.until_layer) if args.until_layer is not None else None,
            'best': best,
            'all_layers': results,
        }
        if args.save_raw:
            summary['save_raw'] = True
        if args.run_tag:
            summary['run_tag'] = args.run_tag
    else:
        texts, labels = load_md_reports(args.md_dir, args.limit)
        if not texts:
            raise RuntimeError('未读取到有效样本，请检查数据目录。')
        y = np.array(labels, dtype=int)
        X_layers = encode_texts(texts)
        if args.split_mode == 'cv':
            summary = evaluate_cv(X_layers, y)
        else:
            # holdout: 按比例划分 train/val/test
            tr_idx, va_idx, te_idx = stratified_holdout_indices(y, args.val_ratio, args.test_ratio, args.seed)
            # 编码已完成，按索引切分
            X_tr, X_va, X_te = X_layers[tr_idx], X_layers[va_idx], X_layers[te_idx]
            y_tr, y_va, y_te = y[tr_idx], y[va_idx], y[te_idx]
            # 与 dir 模式共用评估逻辑
            L = X_tr.shape[1]
            results = []
            best = None
            layer_iter = range(L)
            if args.progress:
                layer_iter = tqdm(layer_iter, desc='Evaluating layers (holdout)')
            for li in layer_iter:
                if args.clf == 'logreg':
                    clf = make_pipeline(
                        StandardScaler(with_mean=True, with_std=True),
                        LogisticRegression(max_iter=2000, class_weight='balanced')
                    )
                else:
                    clf = make_pipeline(
                        StandardScaler(with_mean=True, with_std=True),
                        RidgeClassifier(class_weight='balanced')
                    )
                clf.fit(X_tr[:, li, :], y_tr)
                if hasattr(clf[-1], 'predict_proba'):
                    val_prob = clf.predict_proba(X_va[:, li, :])[:, 1]
                else:
                    val_score = clf.decision_function(X_va[:, li, :])
                    val_prob = (val_score - val_score.min()) / (val_score.max() - val_score.min() + 1e-8)
                best_th, _, _ = scan_threshold(y_va, val_prob, step=args.th_step)
                if hasattr(clf[-1], 'predict_proba'):
                    te_prob = clf.predict_proba(X_te[:, li, :])[:, 1]
                else:
                    te_score = clf.decision_function(X_te[:, li, :])
                    te_prob = (te_score - te_score.min()) / (te_score.max() - te_score.min() + 1e-8)
                te_pred = (te_prob >= best_th).astype(int)
                m = {
                    'layer_index': int(li + 1),
                    'best_threshold': float(best_th),
                    'macro_f1': float(f1_score(y_te, te_pred, average='macro')),
                    'aupr': float(average_precision_score(y_te, te_prob)),
                    'auroc': float(roc_auc_score(y_te, te_prob)),
                    'accuracy': float(accuracy_score(y_te, te_pred)),
                }
                results.append(m)
                if (best is None) or (m['macro_f1'] > best['macro_f1']):
                    best = m
            H = X_tr.shape[2]
            summary = {
                'mode': 'holdout',
                'num_samples': int(len(y)),
                'num_layers': int(L),
                'hidden_size': int(H),
                'clf': args.clf,
                'val_ratio': args.val_ratio,
                'test_ratio': args.test_ratio,
                'threshold_step': args.th_step,
                'until_layer': int(args.until_layer) if args.until_layer is not None else None,
                'best': best,
                'all_layers': results,
            }
            if args.save_raw:
                summary['save_raw'] = True
            if args.run_tag:
                summary['run_tag'] = args.run_tag

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if args.out:
        os.makedirs(os.path.dirname(args.out), exist_ok=True)
        with open(args.out, 'w', encoding='utf-8') as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)


if __name__ == '__main__':
    main()



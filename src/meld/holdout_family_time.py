#!/usr/bin/env python3
"""
基于家族留出 / 时间留出 的 LEC 评测：
- 输入：由 build_dataset_index.py 生成的索引CSV（含 sha256/label/path/first_seen/family；不使用 mtime/last_seen）
- 过程：按索引读取文本 -> 分层编码 -> 逐层训练线性头（PLR或Ridge）
- 策略：
  * 家族留出：指定 test 家族，训练集/验证集不含该家族
  * 时间留出：仅按 first_seen 时间阈值划分 train/val/test（不使用 mtime/last_seen）

示例：
python3 src/lec/holdout_family_time.py \
  --index_csv experiments/lec_qwen3/manifests/dataset_with_family_time.csv \
  --model_dir models/cache/models--Qwen--Qwen3-0.6B/snapshots/c1899de289a04d12100db370d81485cdf75e47ca \
  --mode family --test_family AgentTesla \
  --clf logreg --gpu 1 --progress \
  --out experiments/lec_qwen3/holdout_family_AgentTesla.json

python3 src/lec/holdout_family_time.py \
  --index_csv experiments/lec_qwen3/manifests/dataset_with_family_time.csv \
  --model_dir models/cache/models--Qwen--Qwen3-0.6B/snapshots/c1899de289a04d12100db370d81485cdf75e47ca \
  --mode time --time_threshold 2024-01-01 \
  --clf logreg --gpu 1 --progress \
  --out experiments/lec_qwen3/holdout_time_2024-01-01.json
"""

import os
import json
import csv
import argparse
import sys
from typing import List, Dict, Optional, Tuple
from datetime import datetime

import numpy as np
import torch
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.metrics import f1_score, average_precision_score, roc_auc_score, accuracy_score
from tqdm import tqdm

from .feature_extractor import LayerwiseFeatureExtractor


def parse_datetime_safe(s: str) -> Optional[datetime]:
    if not s:
        return None
    s = str(s).strip().strip('"').strip("'")
    # 支持常见日期格式
    for fmt in ["%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%d %H:%M", "%Y-%m-%d"]:
        try:
            return datetime.strptime(s, fmt)
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


def select_family_holdout(rows: List[Dict[str, str]], test_family: str) -> Tuple[List[int], List[int], List[int]]:
    """
    家族留一法（Leave-One-Family-Out，LOFO）划分规则（RULES）：
    - 白样本不具备时间属性与家族属性，严禁基于这些字段做任何划分或筛选
    - 测试集由两部分组成：
      1) 目标家族的全部恶意样本（positive）
      2) 从全体白样本池中按比例随机抽样，与positive数量1:1匹配（negative），且与训练/验证不重叠
    - 训练/验证集由其余样本组成：所有非目标家族的恶意样本 + 其余白样本
    - 验证集从训练候选集中按5%做分层抽样（若类别不足则退化为简单切分）
    - 全流程不使用白样本的时间/家族字段，严格避免分布泄漏
    """
    families = [r.get('family', '') for r in rows]
    labels = np.array([int(r['label']) for r in rows], dtype=int)
    idx = np.arange(len(rows))

    # 目标家族恶意样本作为测试集positive
    test_pos_idx = idx[(labels == 1) & (np.array(families) == test_family)]
    if len(test_pos_idx) == 0:
        print(f"Warning: 家族 '{test_family}' 不存在或无恶意样本，跳过该实验")
        return [], [], []

    # 白样本池：不依据时间/家族做任何筛选，只做随机按比例抽样
    neg_idx_all = idx[labels == 0]
    n_test_neg = min(len(test_pos_idx), len(neg_idx_all))
    rng = np.random.default_rng(42)
    test_neg_idx = rng.choice(neg_idx_all, size=n_test_neg, replace=False) if n_test_neg > 0 else np.array([], dtype=int)

    # 测试集：目标家族恶意 + 随机白样本（1:1）
    test_idx = np.concatenate([test_pos_idx, test_neg_idx])

    # 训练/验证候选：其余样本（自动确保与测试集不重叠）
    remaining = np.setdiff1d(idx, test_idx, assume_unique=False)
    y_remaining = labels[remaining]

    # 分层抽样划分验证集（5%）；若类别不足则退化为简单切分
    if len(remaining) < 10:
        train_idx = remaining
        val_idx = np.array([], dtype=int)
    else:
        if len(np.unique(y_remaining)) < 2:
            n_val = max(1, len(remaining) // 20)  # 约5%
            val_idx = remaining[:n_val]
            train_idx = remaining[n_val:]
        else:
            sss = StratifiedShuffleSplit(n_splits=1, test_size=0.05, random_state=42)
            tr_rel, va_rel = next(sss.split(remaining, y_remaining))
            train_idx = remaining[tr_rel]
            val_idx = remaining[va_rel]

    return train_idx.tolist(), val_idx.tolist(), test_idx.tolist()


def select_time_holdout(rows: List[Dict[str, str]], threshold: datetime) -> Tuple[List[int], List[int], List[int]]:
    """
    时间留出实验修正版：解决白样本无时间戳导致的单类别问题
    - 恶意样本按first_seen时间分割
    - 白样本按比例人工分配到训练/测试集，确保两个时期都有恶/良样本
    """
    labels = np.array([int(r['label']) for r in rows], dtype=int)
    idx = np.arange(len(rows))
    
    # 恶意样本按时间分割
    malicious_times = []
    malicious_idx = []
    for i, r in enumerate(rows):
        if int(r['label']) == 1:  # 恶意样本
            t = (
                parse_datetime_safe(r.get('first_seen', '')) or
                parse_datetime_safe(r.get('first seen', ''))
            )
            if t is not None:
                malicious_times.append(t)
                malicious_idx.append(i)
    
    if not malicious_times:
        raise RuntimeError('没有有效的恶意样本时间戳')
        
    # 按时间阈值分割恶意样本
    malicious_before = [i for i, t in zip(malicious_idx, malicious_times) if t < threshold]
    malicious_after = [i for i, t in zip(malicious_idx, malicious_times) if t >= threshold]
    
    if not malicious_before or not malicious_after:
        raise RuntimeError('时间阈值导致恶意样本极端不平衡')
    
    # 白样本按比例分配（因为无时间戳）
    benign_idx = idx[labels == 0].tolist()
    if not benign_idx:
        raise RuntimeError('没有白样本用于平衡')
    
    # 按恶意样本比例分配白样本
    ratio_before = len(malicious_before) / (len(malicious_before) + len(malicious_after))
    ratio_after = 1 - ratio_before
    
    rng = np.random.default_rng(42)
    n_benign_before = int(len(benign_idx) * ratio_before)
    n_benign_after = len(benign_idx) - n_benign_before
    
    shuffled_benign = rng.permutation(benign_idx)
    benign_before = shuffled_benign[:n_benign_before].tolist()
    benign_after = shuffled_benign[n_benign_before:].tolist()
    
    # 组合训练/测试集
    before_all = malicious_before + benign_before
    after_all = malicious_after + benign_after
    
    # 从before_all中分出验证集
    y_before = np.array([int(rows[i]['label']) for i in before_all], dtype=int)
    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.05, random_state=42)
    tr_rel, va_rel = next(sss.split(np.array(before_all), y_before))
    
    train_idx = [before_all[i] for i in tr_rel]
    val_idx = [before_all[i] for i in va_rel]
    test_idx = after_all
    
    print(f"时间留出数据分割:")
    print(f"  训练集: {len(train_idx)}个 (恶意:{sum(1 for i in train_idx if int(rows[i]['label'])==1)}, 良性:{sum(1 for i in train_idx if int(rows[i]['label'])==0)})")
    print(f"  验证集: {len(val_idx)}个 (恶意:{sum(1 for i in val_idx if int(rows[i]['label'])==1)}, 良性:{sum(1 for i in val_idx if int(rows[i]['label'])==0)})")
    print(f"  测试集: {len(test_idx)}个 (恶意:{sum(1 for i in test_idx if int(rows[i]['label'])==1)}, 良性:{sum(1 for i in test_idx if int(rows[i]['label'])==0)})")
    
    return train_idx, val_idx, test_idx


def encode_rows(rows: List[Dict[str, str]], indices: List[int], extractor: LayerwiseFeatureExtractor, max_tokens: int, stride: int, progress: bool) -> Tuple[np.ndarray, np.ndarray]:
    texts: List[str] = []
    labels: List[int] = []
    it = indices
    if progress:
        it = tqdm(indices, desc='Encoding docs (subset)')
    for i in it:
        path = rows[i]['path']
        try:
            with open(path, 'r', encoding='utf-8', errors='ignore') as f:
                txt = f.read()
        except Exception:
            txt = ''
        texts.append(txt)
        labels.append(int(rows[i]['label']))
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


def scan_threshold(y_true: np.ndarray, y_prob: np.ndarray, step: float = 0.01) -> Tuple[float, float, float]:
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--index_csv', required=True, type=str)
    parser.add_argument('--model_dir', required=True, type=str)
    parser.add_argument('--mode', choices=['family', 'time'], required=True)
    parser.add_argument('--test_family', type=str, default=None)
    parser.add_argument('--time_threshold', type=str, default=None, help='如 2024-01-01 或 "YYYY-MM-DD HH:MM:SS"；也可用 auto 或 quantile:0.6')
    parser.add_argument('--clf', choices=['logreg', 'ridge'], default='logreg')
    parser.add_argument('--max_tokens', type=int, default=1024)
    parser.add_argument('--stride', type=int, default=256)
    parser.add_argument('--th_step', type=float, default=0.01)
    # 快速测试可选限额：>0 时对各 split 子采样
    parser.add_argument('--train_limit', type=int, default=0, help='快速测试：限制训练集最大样本数（0为不限）')
    parser.add_argument('--val_limit', type=int, default=0, help='快速测试：限制验证集最大样本数（0为不限）')
    parser.add_argument('--test_limit', type=int, default=0, help='快速测试：限制测试集最大样本数（0为不限，测试集尽量按1:1保持平衡）')
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--progress', action='store_true')
    parser.add_argument('--out', type=str, required=True)
    parser.add_argument('--save_raw', action='store_true', help='保存配置/环境/命令与结果到runs目录')
    parser.add_argument('--run_tag', type=str, default='', help='自定义run标签')
    parser.add_argument('--artifacts_dir', type=str, default='experiments/lec_qwen3/runs', help='产物根目录')
    args = parser.parse_args()

    rows = load_index(args.index_csv)
    if not rows:
        raise RuntimeError('index_csv 为空或无法读取')

    if args.mode == 'family':
        if not args.test_family:
            raise RuntimeError('--mode family 需提供 --test_family')
        train_idx, val_idx, test_idx = select_family_holdout(rows, args.test_family)
        split_meta = {'mode': 'family', 'test_family': args.test_family}
    else:
        if not args.time_threshold:
            raise RuntimeError('--mode time 需提供 --time_threshold')
        th_input = (args.time_threshold or '').strip().lower()
        th_dt: Optional[datetime] = None
        q_val: Optional[float] = None
        if th_input in ('auto', 'automatic'):
            q_val = 0.6
        elif th_input.startswith('quantile:'):
            try:
                q_val = float(th_input.split(':', 1)[1])
                q_val = min(max(q_val, 0.05), 0.95)
            except Exception:
                q_val = 0.6
        else:
            th_dt = parse_datetime_safe(args.time_threshold)
        if q_val is not None:
            # 仅基于 first_seen 计算分位点阈值
            parsed_times: List[Optional[datetime]] = []
            for r in rows:
                t = (
                    parse_datetime_safe(r.get('first_seen', '')) or
                    parse_datetime_safe(r.get('first seen', ''))
                )
                parsed_times.append(t)
            ts = [int(t.timestamp()) for t in parsed_times if t is not None]
            if not ts:
                raise RuntimeError('无法解析 --time_threshold（时间字段缺失/格式不一致）')
            import numpy as _np
            cutoff = int(_np.quantile(_np.array(ts), q_val))
            th_dt = datetime.fromtimestamp(cutoff)
            split_meta = {'mode': 'time', 'time_threshold': f'auto(q={q_val})', 'resolved_datetime': th_dt.strftime('%Y-%m-%d %H:%M:%S')}
        else:
            if th_dt is None:
                raise RuntimeError('无法解析 --time_threshold')
            split_meta = {'mode': 'time', 'time_threshold': args.time_threshold}
        train_idx, val_idx, test_idx = select_time_holdout(rows, th_dt)

    # 快速测试：如设置了限制，对索引进行子采样（测试集尽量保持恶/良1:1）
    rng = np.random.default_rng(42)
    def _random_limit(idxs: List[int], limit: int) -> List[int]:
        if limit and len(idxs) > limit:
            return rng.choice(np.array(idxs), size=limit, replace=False).tolist()
        return list(idxs)
    def _balanced_limit_test(idxs: List[int], limit: int) -> List[int]:
        if not limit or len(idxs) <= limit:
            return list(idxs)
        y_test = np.array([int(rows[i]['label']) for i in idxs], dtype=int)
        idxs_np = np.array(idxs)
        pos = idxs_np[y_test == 1]
        neg = idxs_np[y_test == 0]
        n_per = min(limit // 2, len(pos), len(neg))
        if n_per <= 0:
            return rng.choice(idxs_np, size=limit, replace=False).tolist()
        s_pos = rng.choice(pos, size=n_per, replace=False)
        s_neg = rng.choice(neg, size=n_per, replace=False)
        return np.concatenate([s_pos, s_neg]).tolist()

    if args.train_limit:
        train_idx = _random_limit(train_idx, args.train_limit)
    if args.val_limit:
        val_idx = _random_limit(val_idx, args.val_limit)
    if args.test_limit:
        test_idx = _balanced_limit_test(test_idx, args.test_limit)

    device = f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu"
    extractor = LayerwiseFeatureExtractor(model_dir=args.model_dir, device=device)

    X_tr, y_tr = encode_rows(rows, train_idx, extractor, args.max_tokens, args.stride, args.progress)
    X_va, y_va = encode_rows(rows, val_idx, extractor, args.max_tokens, args.stride, args.progress)
    X_te, y_te = encode_rows(rows, test_idx, extractor, args.max_tokens, args.stride, args.progress)

    L = X_tr.shape[1]
    results = []
    best: Optional[Dict] = None
    layer_iter = range(L)
    if args.progress:
        layer_iter = tqdm(layer_iter, desc='Evaluating layers (holdout-generalization)')
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
        # val 选阈值
        if hasattr(clf[-1], 'predict_proba'):
            val_prob = clf.predict_proba(X_va[:, li, :])[:, 1]
        else:
            val_score = clf.decision_function(X_va[:, li, :])
            val_prob = (val_score - val_score.min()) / (val_score.max() - val_score.min() + 1e-8)
        best_th, _, _ = scan_threshold(y_va, val_prob, step=args.th_step)
        # test 评估
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

    summary = {
        'split': split_meta,
        'num_layers': int(L),
        'hidden_size': int(X_tr.shape[2]),
        'clf': args.clf,
        'threshold_step': args.th_step,
        'until_layer': None,
        'sizes': {
            'train': int(len(train_idx)),
            'val': int(len(val_idx)),
            'test': int(len(test_idx)),
        },
        'best': best,
        'all_layers': results,
    }
    out_dir = os.path.dirname(args.out)
    if out_dir:  # 只有当输出文件有目录部分时才创建目录
        os.makedirs(out_dir, exist_ok=True)
    with open(args.out, 'w', encoding='utf-8') as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(json.dumps(summary, ensure_ascii=False, indent=2))

    # 可选：保存原始产物（配置/环境/命令与结果）
    if args.save_raw:
        ts = datetime.now().strftime('%Y%m%d_%H%M%S')
        ident = args.test_family if args.mode == 'family' else args.time_threshold
        tag = (args.run_tag or '').strip().replace(' ', '_')
        run_name = f"{ts}_holdout_{args.mode}_{ident}"
        if tag:
            run_name += f"_{tag}"
        run_dir = os.path.join(args.artifacts_dir, run_name)
        os.makedirs(run_dir, exist_ok=True)
        cfg_dir = os.path.join(run_dir, 'configs'); os.makedirs(cfg_dir, exist_ok=True)
        with open(os.path.join(cfg_dir, 'config.json'), 'w', encoding='utf-8') as f:
            json.dump({k: (v if isinstance(v, (int, float, str, bool, type(None))) else str(v)) for k, v in vars(args).items()}, f, ensure_ascii=False, indent=2)
        env = {
            'python': sys.version,
            'sklearn': __import__('sklearn').__version__,
            'torch': getattr(torch, '__version__', 'n/a'),
        }
        with open(os.path.join(cfg_dir, 'env.json'), 'w', encoding='utf-8') as f:
            json.dump(env, f, ensure_ascii=False, indent=2)
        with open(os.path.join(cfg_dir, 'cmd.txt'), 'w', encoding='utf-8') as f:
            f.write(' '.join(sys.argv))
        met_dir = os.path.join(run_dir, 'metrics'); os.makedirs(met_dir, exist_ok=True)
        with open(os.path.join(met_dir, 'metrics.json'), 'w', encoding='utf-8') as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)


if __name__ == '__main__':
    main()



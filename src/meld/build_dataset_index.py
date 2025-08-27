#!/usr/bin/env python3
import os
import csv
import argparse
from typing import Dict, Optional
from datetime import datetime


def scan_md_reports(md_dir: str) -> Dict[str, dict]:
    index: Dict[str, dict] = {}
    for root, _, files in os.walk(md_dir):
        for fn in files:
            if not fn.endswith('.md'):
                continue
            path = os.path.join(root, fn)
            name = os.path.splitext(fn)[0]
            # 约定：文件名为 sha256（64位hex）
            sha256 = name.lower()
            # 标签来源：父目录名
            parts = os.path.normpath(path).split(os.sep)
            label: Optional[int] = None
            if 'black' in parts:
                label = 1
            elif 'white' in parts:
                label = 0
            # unknown 或其他目录不纳入索引
            if label is None:
                continue
            try:
                stat = os.stat(path)
                size = stat.st_size
            except Exception:
                size = -1
            index[sha256] = {
                'sha256': sha256,
                'label': label,
                'path': path,
                'size_bytes': size,
                'family': '',
                'first_seen': '',
            }
    return index


def enrich_with_full_csv(index: Dict[str, dict], full_csv_path: str) -> None:
    if not os.path.exists(full_csv_path):
        print(f"警告：{full_csv_path} 不存在，跳过家族/时间补充。")
        return
    wanted = set(index.keys())
    # MalwareBazaar full.csv 无表头，含注释行；字段为带引号CSV
    # 经验字段位次（可能随版本变动）：
    #  0:first_seen 1:sha256 2:md5 3:sha1 4:reporter 5:file_name 6:file_type 7:mime_type 8:signature(family)
    with open(full_csv_path, 'r', encoding='utf-8', errors='ignore', newline='') as f:
        reader = csv.reader((line for line in f if line and not line.startswith('#')))
        for row in reader:
            if not row:
                continue
            # 容错：去除两端空白与引号
            row = [col.strip().strip('"') for col in row]
            if len(row) < 9:
                continue
            sha256 = row[1].lower()
            if sha256 in wanted:
                index[sha256]['first_seen'] = row[0]
                index[sha256]['family'] = row[8]


def write_csv(index: Dict[str, dict], out_csv: str) -> None:
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    fields = ['sha256', 'label', 'path', 'size_bytes', 'first_seen', 'family']
    with open(out_csv, 'w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for sha256, rec in index.items():
            writer.writerow({k: rec.get(k, '') for k in fields})
    print(f"Saved dataset index: {out_csv}  (rows={len(index)})")


def write_split_csv(index: Dict[str, dict], out_black: str, out_white: str) -> None:
    blacks = {k: v for k, v in index.items() if v.get('label') == 1}
    whites = {k: v for k, v in index.items() if v.get('label') == 0}
    if out_black:
        write_csv(blacks, out_black)
    if out_white:
        write_csv(whites, out_white)


def _parse_dt(s: str) -> Optional[datetime]:
    if not s:
        return None
    s = str(s).strip().strip('"').strip("'")
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%d %H:%M", "%Y-%m-%d"):
        try:
            return datetime.strptime(s[:len(fmt)], fmt)
        except Exception:
            continue
    return None


def write_family_stats(index: Dict[str, dict], out_csv: str) -> None:
    by_family: Dict[str, Dict[str, int]] = {}
    fam_times: Dict[str, list] = {}
    for rec in index.values():
        fam = rec.get('family', '') or ''
        if fam not in by_family:
            by_family[fam] = {'num_black': 0, 'num_white': 0}
            fam_times[fam] = []
        if rec.get('label') == 1:
            by_family[fam]['num_black'] += 1
        elif rec.get('label') == 0:
            by_family[fam]['num_white'] += 1
        dt = _parse_dt(rec.get('first_seen', ''))
        if dt is not None:
            fam_times[fam].append(dt)

    rows = []
    for fam, cnts in by_family.items():
        times = fam_times.get(fam, [])
        if times:
            t_min = min(times).strftime('%Y-%m-%d %H:%M:%S')
            t_max = max(times).strftime('%Y-%m-%d %H:%M:%S')
        else:
            t_min = ''
            t_max = ''
        rows.append({
            'family': fam,
            'num_black': cnts['num_black'],
            'num_white': cnts['num_white'],
            'num_total': cnts['num_black'] + cnts['num_white'],
            'first_seen_min': t_min,
            'last_seen_approx': t_max,
        })

    # 排序：按 num_total 降序，其次 family 名称
    rows.sort(key=lambda r: (-int(r['num_total']), r['family']))

    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    fields = ['family', 'num_black', 'num_white', 'num_total', 'first_seen_min', 'last_seen_approx']
    with open(out_csv, 'w', encoding='utf-8', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"Saved family stats: {out_csv}  (rows={len(rows)})")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--md_dir', required=True, type=str, help='md 报告根目录（含 black/white）')
    parser.add_argument('--full_csv', default=None, type=str, help='MalwareBazaar full.csv 路径（可选）')
    parser.add_argument('--out_csv', required=True, type=str, help='输出CSV路径')
    parser.add_argument('--out_csv_black', default=None, type=str, help='可选：单独导出黑样本CSV')
    parser.add_argument('--out_csv_white', default=None, type=str, help='可选：单独导出白样本CSV')
    parser.add_argument('--out_family_stats', default=None, type=str, help='可选：按 family 汇总统计并计算 last_seen_approx')
    args = parser.parse_args()

    idx = scan_md_reports(args.md_dir)
    if args.full_csv:
        enrich_with_full_csv(idx, args.full_csv)
    write_csv(idx, args.out_csv)
    if args.out_csv_black or args.out_csv_white:
        write_split_csv(idx, args.out_csv_black or '', args.out_csv_white or '')
    if args.out_family_stats:
        write_family_stats(idx, args.out_family_stats)


if __name__ == '__main__':
    main()



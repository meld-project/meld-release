#!/usr/bin/env python3
"""
从CSV清单中选取样本并从正确的源目录复制CAPE JSON文件

功能：
1. 从malicious_dataset_manifest.csv中选取前8大家族的恶意样本（各250个，共2000个）
2. 选取2000个良性样本
3. 从源目录复制对应的{sha256}.json文件到目标目录
4. 恶意样本按家族创建子目录

使用方法：
python scripts/copy_correct_cape_samples.py
"""

import os
import csv
import shutil
from pathlib import Path
from collections import Counter, defaultdict
import argparse

def load_manifest(manifest_path):
    """加载CSV清单"""
    print(f"📖 读取清单文件: {manifest_path}")
    
    malicious_samples = defaultdict(list)
    benign_samples = []
    
    with open(manifest_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            sha256 = row['sha256']
            label = row['label']
            family = row.get('family', '')
            
            if label == 'malicious':
                malicious_samples[family].append({
                    'sha256': sha256,
                    'family': family,
                    'size_bytes': row.get('size_bytes', ''),
                    'first_seen': row.get('first_seen', '')
                })
            else:
                benign_samples.append({
                    'sha256': sha256,
                    'size_bytes': row.get('size_bytes', ''),
                    'first_seen': row.get('first_seen', '')
                })
    
    print(f"✅ 加载完成:")
    print(f"   恶意家族数: {len(malicious_samples)}")
    print(f"   良性样本数: {len(benign_samples)}")
    
    return malicious_samples, benign_samples

def get_top_families(malicious_samples, top_n=8):
    """获取前N大家族（排除unknown等无效家族）"""
    # 排除的无效家族名称
    excluded_families = {'unknown', '', 'Unknown', 'UNKNOWN', 'N/A', 'n/a', 'null', 'NULL'}
    
    # 过滤有效家族
    valid_families = {family: samples for family, samples in malicious_samples.items() 
                     if family not in excluded_families}
    
    family_counts = {family: len(samples) for family, samples in valid_families.items()}
    
    # 按样本数量排序
    sorted_families = sorted(family_counts.items(), key=lambda x: x[1], reverse=True)
    
    print(f"\n📊 有效家族样本数量排序 (前{top_n}个，已排除unknown等无效家族):")
    for i, (family, count) in enumerate(sorted_families[:top_n], 1):
        print(f"   {i:2d}. {family:20s}: {count:,} 个样本")
    
    if len(sorted_families) < top_n:
        print(f"\n⚠️  警告：只找到 {len(sorted_families)} 个有效家族，少于请求的 {top_n} 个")
    
    top_families = [family for family, count in sorted_families[:top_n]]
    return top_families

def select_samples(malicious_samples, benign_samples, top_families, samples_per_family=250, benign_count=2000):
    """选择样本"""
    print(f"\n🎯 选择样本:")
    print(f"   每个家族: {samples_per_family} 个")
    print(f"   良性样本: {benign_count} 个")
    
    selected_malicious = {}
    total_malicious = 0
    
    # 选择恶意样本
    for family in top_families:
        available = malicious_samples[family]
        selected_count = min(samples_per_family, len(available))
        selected_malicious[family] = available[:selected_count]
        total_malicious += selected_count
        print(f"   ✅ {family}: 选择了 {selected_count}/{len(available)} 个样本")
    
    # 选择良性样本 - 如果CSV中没有良性样本，从源目录直接获取
    if not benign_samples:
        print(f"   ⚠️  CSV中无良性样本，将从源目录直接选择 {benign_count} 个")
        selected_benign = []  # 标记需要从源目录复制
        actual_benign_count = benign_count
    else:
        selected_benign = benign_samples[:benign_count]
        actual_benign_count = len(selected_benign)
    
    print(f"\n📋 选择汇总:")
    print(f"   恶意样本: {total_malicious} 个 (来自 {len(top_families)} 个家族)")
    print(f"   良性样本: {actual_benign_count} 个")
    print(f"   总计: {total_malicious + actual_benign_count} 个样本")
    
    return selected_malicious, selected_benign

def copy_samples(selected_malicious, selected_benign, source_malicious_dir, source_benign_dir, 
                target_malicious_dir, target_benign_dir):
    """复制样本文件"""
    print(f"\n📁 开始复制文件...")
    print(f"   源恶意目录: {source_malicious_dir}")
    print(f"   源良性目录: {source_benign_dir}")
    print(f"   目标恶意目录: {target_malicious_dir}")
    print(f"   目标良性目录: {target_benign_dir}")
    
    # 创建目标目录
    Path(target_malicious_dir).mkdir(parents=True, exist_ok=True)
    Path(target_benign_dir).mkdir(parents=True, exist_ok=True)
    
    copied_files = 0
    missing_files = []
    
    # 复制恶意样本
    print(f"\n🦠 复制恶意样本:")
    for family, samples in selected_malicious.items():
        family_dir = Path(target_malicious_dir) / family
        family_dir.mkdir(parents=True, exist_ok=True)
        
        family_copied = 0
        family_missing = 0
        
        for sample in samples:
            sha256 = sample['sha256']
            source_file = Path(source_malicious_dir) / f"{sha256}.json"
            target_file = family_dir / f"{sha256}.json"
            
            if source_file.exists():
                try:
                    shutil.copy2(source_file, target_file)
                    family_copied += 1
                    copied_files += 1
                except Exception as e:
                    print(f"     ❌ 复制失败 {sha256}: {e}")
                    missing_files.append(f"{family}/{sha256}.json")
                    family_missing += 1
            else:
                missing_files.append(f"{family}/{sha256}.json")
                family_missing += 1
        
        print(f"   {family:20s}: ✅ {family_copied:3d} 个, ❌ {family_missing:3d} 个缺失")
    
    # 复制良性样本
    print(f"\n✅ 复制良性样本:")
    benign_copied = 0
    benign_missing = 0
    
    if not selected_benign:
        # 如果CSV中没有良性样本，直接从源目录复制前N个
        print(f"   从源目录直接复制前2000个良性样本...")
        source_benign_path = Path(source_benign_dir)
        if source_benign_path.exists():
            benign_files = list(source_benign_path.glob("*.json"))[:2000]
            for source_file in benign_files:
                target_file = Path(target_benign_dir) / source_file.name
                try:
                    shutil.copy2(source_file, target_file)
                    benign_copied += 1
                    copied_files += 1
                except Exception as e:
                    print(f"     ❌ 复制失败 {source_file.name}: {e}")
                    missing_files.append(f"benign/{source_file.name}")
                    benign_missing += 1
        else:
            print(f"   ❌ 源良性目录不存在: {source_benign_dir}")
            benign_missing = 2000
    else:
        # 从CSV选择的良性样本复制
        for sample in selected_benign:
            sha256 = sample['sha256']
            source_file = Path(source_benign_dir) / f"{sha256}.json"
            target_file = Path(target_benign_dir) / f"{sha256}.json"
            
            if source_file.exists():
                try:
                    shutil.copy2(source_file, target_file)
                    benign_copied += 1
                    copied_files += 1
                except Exception as e:
                    print(f"     ❌ 复制失败 {sha256}: {e}")
                    missing_files.append(f"benign/{sha256}.json")
                    benign_missing += 1
            else:
                missing_files.append(f"benign/{sha256}.json")
                benign_missing += 1
    
    print(f"   良性样本: ✅ {benign_copied} 个, ❌ {benign_missing} 个缺失")
    
    # 汇总结果
    print(f"\n📊 复制结果汇总:")
    print(f"   ✅ 成功复制: {copied_files:,} 个文件")
    print(f"   ❌ 缺失文件: {len(missing_files):,} 个")
    print(f"   📈 成功率: {copied_files/(copied_files+len(missing_files))*100:.1f}%")
    
    if missing_files and len(missing_files) <= 20:
        print(f"\n⚠️  缺失文件列表:")
        for missing in missing_files:
            print(f"     - {missing}")
    elif missing_files:
        print(f"\n⚠️  缺失文件过多，仅显示前20个:")
        for missing in missing_files[:20]:
            print(f"     - {missing}")
        print(f"     ... 还有 {len(missing_files)-20} 个缺失文件")
    
    return copied_files, len(missing_files)

def verify_copied_files(target_malicious_dir, target_benign_dir):
    """验证复制的文件"""
    print(f"\n🔍 验证复制的文件:")
    
    # 统计恶意样本
    malicious_stats = {}
    total_malicious = 0
    
    for family_dir in Path(target_malicious_dir).iterdir():
        if family_dir.is_dir():
            family_name = family_dir.name
            json_files = list(family_dir.glob("*.json"))
            count = len(json_files)
            malicious_stats[family_name] = count
            total_malicious += count
    
    # 统计良性样本
    benign_files = list(Path(target_benign_dir).glob("*.json"))
    total_benign = len(benign_files)
    
    print(f"   恶意样本分布:")
    for family, count in sorted(malicious_stats.items()):
        print(f"     {family:20s}: {count:3d} 个文件")
    
    print(f"   良性样本: {total_benign} 个文件")
    print(f"   总计: {total_malicious + total_benign} 个文件")
    
    return total_malicious, total_benign

def main():
    parser = argparse.ArgumentParser(description="从CSV选取样本并复制正确的CAPE JSON文件")
    parser.add_argument("--manifest", default="malicious_dataset_manifest.csv",
                       help="CSV清单文件路径")
    parser.add_argument("--source-malicious", default="dataset/meld-data/cape_reports_malicious",
                       help="源恶意样本目录")
    parser.add_argument("--source-benign", default="dataset/meld-data/cape_reports_benign",
                       help="源良性样本目录")
    parser.add_argument("--target-malicious", default="input/cape_behavior_malicious_train",
                       help="目标恶意样本目录")
    parser.add_argument("--target-benign", default="input/cape_behavior_benign_train",
                       help="目标良性样本目录")
    parser.add_argument("--samples-per-family", type=int, default=250,
                       help="每个家族的样本数量")
    parser.add_argument("--benign-count", type=int, default=2000,
                       help="良性样本总数")
    parser.add_argument("--top-families", type=int, default=8,
                       help="选择前N大家族")
    
    args = parser.parse_args()
    
    print("🚀 CAPE样本复制工具")
    print("=" * 60)
    
    # 检查源目录
    if not Path(args.source_malicious).exists():
        print(f"❌ 源恶意样本目录不存在: {args.source_malicious}")
        return 1
    
    if not Path(args.source_benign).exists():
        print(f"❌ 源良性样本目录不存在: {args.source_benign}")
        return 1
    
    if not Path(args.manifest).exists():
        print(f"❌ CSV清单文件不存在: {args.manifest}")
        return 1
    
    try:
        # 1. 加载清单
        malicious_samples, benign_samples = load_manifest(args.manifest)
        
        # 2. 获取前N大家族
        top_families = get_top_families(malicious_samples, args.top_families)
        
        # 3. 选择样本
        selected_malicious, selected_benign = select_samples(
            malicious_samples, benign_samples, top_families, 
            args.samples_per_family, args.benign_count
        )
        
        # 4. 复制文件
        copied_files, missing_files = copy_samples(
            selected_malicious, selected_benign,
            args.source_malicious, args.source_benign,
            args.target_malicious, args.target_benign
        )
        
        # 5. 验证结果
        total_malicious, total_benign = verify_copied_files(
            args.target_malicious, args.target_benign
        )
        
        print(f"\n🎉 任务完成!")
        print(f"   成功复制 {copied_files} 个文件")
        print(f"   最终统计: {total_malicious} 个恶意样本, {total_benign} 个良性样本")
        
        return 0
        
    except Exception as e:
        print(f"❌ 执行失败: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    exit(main())

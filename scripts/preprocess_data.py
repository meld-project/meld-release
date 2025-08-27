#!/usr/bin/env python3
import os
import sys
import csv
import json
import argparse
from pathlib import Path
from typing import Dict, List, Optional
from datetime import datetime
import logging
import pandas as pd

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

def setup_logging():
    """Setup preprocessing logging"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

def scan_directory(directory: Path, file_extension: str) -> List[Dict]:
    """Scan directory for files with specific extension"""
    logger = logging.getLogger(__name__)
    
    samples = []
    
    if not directory.exists():
        logger.warning(f"Directory not found: {directory}")
        return samples
    
    # For malicious samples, look for family subdirectories
    if "malicious" in directory.name:
        for family_dir in directory.iterdir():
            if family_dir.is_dir() and family_dir.name != "__pycache__":
                family = family_dir.name
                for file_path in family_dir.rglob(f"*.{file_extension}"):
                    sha256 = file_path.stem
                    samples.append({
                        "sha256": sha256,
                        "label": 1,  # malicious
                        "family": family,
                        "path": str(file_path),
                        "size_bytes": file_path.stat().st_size
                    })
    else:
        # For benign samples, no family subdirectory
        for file_path in directory.rglob(f"*.{file_extension}"):
            sha256 = file_path.stem
            samples.append({
                "sha256": sha256,
                "label": 0,  # benign
                "family": "benign",
                "path": str(file_path),
                "size_bytes": file_path.stat().st_size
            })
    
    logger.info(f"Found {len(samples)} samples in {directory}")
    return samples

def extract_time_from_json(json_path: Path) -> Optional[str]:
    """Extract timestamp from JSON file"""
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
        # Look for timestamp in various fields
        time_fields = ['timestamp', 'first_seen', 'analysis_date', 'submitted']
        
        for field in time_fields:
            if field in data and data[field]:
                return data[field]
        
        # If no timestamp found, use file modification time
        return datetime.fromtimestamp(json_path.stat().st_mtime).strftime('%Y-%m-%d %H:%M:%S')
        
    except Exception:
        # Fallback to file modification time
        return datetime.fromtimestamp(json_path.stat().st_mtime).strftime('%Y-%m-%d %H:%M:%S')

def create_dataset_index(input_dir: Path, output_dir: Path):
    """Create comprehensive dataset index"""
    logger = logging.getLogger(__name__)
    logger.info("📊 Creating dataset index...")
    
    # Scan all directories
    all_samples = []
    
    # Malicious JSON behavior data
    mal_json_dir = input_dir / "cape_behavior_malicious_train"
    mal_json_samples = scan_directory(mal_json_dir, "json")
    
    # Benign JSON behavior data  
    ben_json_dir = input_dir / "cape_behavior_benign_train"
    ben_json_samples = scan_directory(ben_json_dir, "json")
    
    # Malicious Markdown reports
    mal_md_dir = input_dir / "cape_reports_malicious_md"
    mal_md_samples = scan_directory(mal_md_dir, "md")
    
    # Benign Markdown reports
    ben_md_dir = input_dir / "cape_reports_benign_md"
    ben_md_samples = scan_directory(ben_md_dir, "md")
    
    # Combine all samples
    all_samples.extend(mal_json_samples)
    all_samples.extend(ben_json_samples)
    all_samples.extend(mal_md_samples)
    all_samples.extend(ben_md_samples)
    
    logger.info(f"Total samples found: {len(all_samples)}")
    
    # Add time information for JSON files
    logger.info("⏰ Extracting time information...")
    for sample in all_samples:
        if sample["path"].endswith(".json"):
            sample["first_seen"] = extract_time_from_json(Path(sample["path"]))
        else:
            # For markdown files, use a simulated time based on SHA256 hash
            # This ensures reproducible time assignments
            hash_int = int(sample["sha256"][:8], 16)
            base_time = datetime(2024, 11, 1)  # Start of training period
            days_offset = hash_int % 365  # Spread across a year
            simulated_time = base_time.replace(day=1) + pd.Timedelta(days=days_offset)
            sample["first_seen"] = simulated_time.strftime('%Y-%m-%d %H:%M:%S')
    
    # Write dataset index
    output_dir.mkdir(parents=True, exist_ok=True)
    index_path = output_dir / "dataset_with_family_time.csv"
    
    with open(index_path, 'w', newline='', encoding='utf-8') as f:
        fieldnames = ["sha256", "label", "family", "path", "size_bytes", "first_seen"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_samples)
    
    logger.info(f"✅ Dataset index created: {index_path}")
    logger.info(f"   Total samples: {len(all_samples)}")
    
    # Create summary statistics
    stats = {
        "total_samples": len(all_samples),
        "malicious_samples": sum(1 for s in all_samples if s["label"] == 1),
        "benign_samples": sum(1 for s in all_samples if s["label"] == 0),
        "json_samples": sum(1 for s in all_samples if s["path"].endswith(".json")),
        "md_samples": sum(1 for s in all_samples if s["path"].endswith(".md")),
        "families": list(set(s["family"] for s in all_samples if s["label"] == 1))
    }
    
    with open(output_dir / "dataset_stats.json", 'w') as f:
        json.dump(stats, f, indent=2)
    
    logger.info("📈 Dataset Statistics:")
    logger.info(f"   Malicious: {stats['malicious_samples']}")
    logger.info(f"   Benign: {stats['benign_samples']}")
    logger.info(f"   JSON files: {stats['json_samples']}")
    logger.info(f"   MD files: {stats['md_samples']}")
    logger.info(f"   Families: {len(stats['families'])}")

def main():
    parser = argparse.ArgumentParser(description="MELD Data Preprocessing")
    parser.add_argument("--input", type=str, default="data/raw",
                       help="Input directory containing raw data")
    parser.add_argument("--output", type=str, default="data/processed", 
                       help="Output directory for processed data")
    
    args = parser.parse_args()
    
    setup_logging()
    logger = logging.getLogger(__name__)
    
    input_dir = Path(args.input)
    output_dir = Path(args.output)
    
    logger.info("🔄 MELD Data Preprocessing Starting")
    logger.info(f"   Input: {input_dir}")
    logger.info(f"   Output: {output_dir}")
    
    if not input_dir.exists():
        logger.error(f"❌ Input directory not found: {input_dir}")
        logger.error("   Run 'bash scripts/download_data.sh' first to download data")
        sys.exit(1)
    
    try:
        create_dataset_index(input_dir, output_dir)
        logger.info("🎉 Data preprocessing completed successfully!")
        
    except Exception as e:
        logger.error(f"❌ Preprocessing failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()

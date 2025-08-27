#!/usr/bin/env python3

import os
import sys
import argparse
import json
from pathlib import Path
from typing import List, Dict
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
import logging

# Add scripts directory to path
sys.path.append(str(Path(__file__).parent))

from cape_to_markdown_converter import DirectCapeConverterEnglish

def setup_logging():
    """Setup batch conversion logging"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

def convert_single_file(args_tuple):
    """Convert a single JSON file to Markdown (for multiprocessing)"""
    json_path, output_dir, family = args_tuple
    
    try:
        converter = DirectCapeConverterEnglish()
        
        # Determine output path
        json_file = Path(json_path)
        sha256 = json_file.stem
        
        if family:
            family_output_dir = Path(output_dir) / family
            family_output_dir.mkdir(parents=True, exist_ok=True)
            output_path = family_output_dir / f"{sha256}.md"
        else:
            output_path = Path(output_dir) / f"{sha256}.md"
        
        # Convert and save
        markdown_content = converter.convert_cape_to_markdown(str(json_path))
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(markdown_content)
        
        return {"success": True, "input": str(json_path), "output": str(output_path)}
        
    except Exception as e:
        return {"success": False, "input": str(json_path), "error": str(e)}

def batch_convert_cape_data(input_dir: Path, output_dir: Path, max_workers: int = 4):
    """Batch convert CAPE JSON files to Markdown reports"""
    logger = logging.getLogger(__name__)
    
    logger.info(f"🔄 Starting batch conversion")
    logger.info(f"   Input: {input_dir}")
    logger.info(f"   Output: {output_dir}")
    logger.info(f"   Workers: {max_workers}")
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Collect all JSON files
    conversion_tasks = []
    
    if input_dir.name.endswith("malicious_train"):
        # Malicious data has family subdirectories
        for family_dir in input_dir.iterdir():
            if family_dir.is_dir() and family_dir.name != "__pycache__":
                family = family_dir.name
                for json_file in family_dir.glob("*.json"):
                    conversion_tasks.append((str(json_file), str(output_dir), family))
    else:
        # Benign data has no family structure
        for json_file in input_dir.glob("*.json"):
            conversion_tasks.append((str(json_file), str(output_dir), None))
    
    logger.info(f"📄 Found {len(conversion_tasks)} JSON files to convert")
    
    # Process in parallel
    successful_conversions = 0
    failed_conversions = 0
    
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        future_to_task = {executor.submit(convert_single_file, task): task for task in conversion_tasks}
        
        # Process results with progress bar
        for future in tqdm(as_completed(future_to_task), total=len(conversion_tasks), desc="Converting"):
            result = future.result()
            
            if result["success"]:
                successful_conversions += 1
            else:
                failed_conversions += 1
                logger.error(f"❌ Conversion failed for {result['input']}: {result['error']}")
    
    logger.info(f"✅ Conversion completed:")
    logger.info(f"   Successful: {successful_conversions}")
    logger.info(f"   Failed: {failed_conversions}")
    logger.info(f"   Success rate: {successful_conversions/(successful_conversions+failed_conversions)*100:.1f}%")

def main():
    parser = argparse.ArgumentParser(description="MELD Batch CAPE Data Converter")
    parser.add_argument("--input", type=str, required=True,
                       help="Input directory containing CAPE JSON files")
    parser.add_argument("--output", type=str, required=True,
                       help="Output directory for Markdown reports")
    parser.add_argument("--workers", type=int, default=4,
                       help="Number of parallel workers (default: 4)")
    
    args = parser.parse_args()
    
    setup_logging()
    logger = logging.getLogger(__name__)
    
    input_dir = Path(args.input)
    output_dir = Path(args.output)
    
    logger.info("🚀 MELD Batch CAPE Data Converter Starting")
    
    if not input_dir.exists():
        logger.error(f"❌ Input directory not found: {input_dir}")
        sys.exit(1)
    
    try:
        batch_convert_cape_data(input_dir, output_dir, args.workers)
        logger.info("🎉 Batch conversion completed successfully!")
        
    except Exception as e:
        logger.error(f"❌ Batch conversion failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()


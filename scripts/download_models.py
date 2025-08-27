#!/usr/bin/env python3

import os
import sys
from pathlib import Path
from huggingface_hub import snapshot_download
import logging

def setup_logging():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def download_qwen_model():
    """Download Qwen3-0.6B model from Hugging Face"""
    logger = logging.getLogger(__name__)
    
    model_name = "Qwen/Qwen3-0.6B"
    local_dir = Path("models/qwen3-0.6b")
    
    logger.info(f"🤖 Downloading {model_name}...")
    logger.info(f"   Target directory: {local_dir}")
    logger.info("   This may take several minutes...")
    
    try:
        snapshot_download(
            repo_id=model_name,
            local_dir=local_dir,
            resume_download=True,
            local_dir_use_symlinks=False
        )
        
        logger.info(f"✅ Model downloaded successfully to {local_dir}")
        return True
        
    except Exception as e:
        logger.error(f"❌ Failed to download model: {e}")
        logger.error("   You may need to:")
        logger.error("   1. Install huggingface_hub: pip install huggingface_hub")
        logger.error("   2. Login to Hugging Face: huggingface-cli login")
        return False

def main():
    setup_logging()
    logger = logging.getLogger(__name__)
    
    logger.info("🚀 MELD Model Download Starting")
    
    # Create models directory
    Path("models").mkdir(exist_ok=True)
    
    success = download_qwen_model()
    
    if success:
        logger.info("🎉 Model download completed!")
        logger.info("   Next steps:")
        logger.info("   1. Preprocess data: python scripts/preprocess_data.py")
        logger.info("   2. Run experiments: python scripts/run_experiments.py")
    else:
        logger.error("❌ Model download failed")
        sys.exit(1)

if __name__ == "__main__":
    main()


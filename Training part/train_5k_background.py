#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
5K Sample Background Training Script
Dedicated script for background training on 5K samples
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from train_fast_local_english import FastTrainer
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('train_5k_background.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def main():
    """5K sample background training main function"""
    
    logger.info("🚀 Starting 5K sample background training")
    logger.info("⚡ Fast mode: 5000 samples, estimated 60 minutes")
    
    # 5K sample training configuration
    trainer = FastTrainer(
        csv_file='/Users/huangxinyue/Multi model distillation/train_10k_results/train_10k_fast_results.csv',
        database_path='/Users/huangxinyue/Downloads/Influencer brand database',
        model_save_dir='/Users/huangxinyue/Multi model distillation/fast_models_5k',
        batch_size=32,
        learning_rate=5e-4,
        num_epochs=6,
        test_size=0.2,
        max_samples=5000  # Fixed 5K samples
    )
    
    # Start training
    trainer.train()
    
    logger.info("🎉 5K sample training completed!")
    logger.info("📁 Model saved in: fast_models_5k/")

if __name__ == "__main__":
    main()

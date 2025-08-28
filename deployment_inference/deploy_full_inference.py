#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Stage 4: Full-Scale Data Inference Deployment
Process 1.6 million Instagram posts using the trained 5K model
"""

import torch
import torch.nn.functional as F
import pandas as pd
import numpy as np
import json
import os
import time
import logging
from datetime import datetime
from PIL import Image
import glob
from tqdm import tqdm
import multiprocessing as mp
from concurrent.futures import ThreadPoolExecutor, as_completed
import gc

# Import model
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from train_fast_local import LightweightGenderBiasModel

# Set environment variables to avoid conflicts
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Configure logging
def setup_logging():
    """Setup logging system"""
    log_format = '%(asctime)s - %(levelname)s - %(message)s'
    logging.basicConfig(
        level=logging.INFO,
        format=log_format,
        handlers=[
            logging.FileHandler('full_inference.log'),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

logger = setup_logging()

class FullInferenceDeployer:
    def __init__(self):
        self.model_path = '/Users/huangxinyue/Multi model distillation/fast_models_5k/fast_best_model.pth'
        self.database_path = '/Users/huangxinyue/Downloads/Influencer brand database'
        self.json_dir = os.path.join(self.database_path, 'json')
        self.post_info_file = os.path.join(self.database_path, 'post_info.txt')
        self.output_dir = '/Users/huangxinyue/Multi model distillation/full_inference_results'
        self.progress_file = os.path.join(self.output_dir, 'inference_progress.json')
        
        # Inference configuration
        self.batch_size = 64  # Larger batch size for efficiency
        self.max_workers = 4  # Number of parallel workers
        self.save_interval = 1000  # Save every 1000 samples
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Model and data
        self.model = None
        self.tokenizer = None
        self.transform = None
        self.post_info_dict = {}
        
        # Ensure output directory exists
        os.makedirs(self.output_dir, exist_ok=True)
        
    def load_model_and_tokenizer(self):
        """Load model and tokenizer"""
        logger.info("🚀 Loading 5K best model...")
        
        # Load model
        self.model = LightweightGenderBiasModel()
        checkpoint = torch.load(self.model_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.device)
        self.model.eval()
        
        # Load tokenizer
        from transformers import DistilBertTokenizer
        self.tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
        
        # Image preprocessing
        from torchvision import transforms
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
        
        logger.info("✅ Model loading completed")
        
    def load_post_info(self):
        """Load post_info.txt mapping"""
        logger.info("📚 Loading post_info mapping...")
        
        try:
            with open(self.post_info_file, 'r', encoding='utf-8') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 2:
                        post_id = parts[0]
                        image_ids = parts[1:]
                        self.post_info_dict[post_id] = image_ids
            
            logger.info(f"✅ Loaded {len(self.post_info_dict):,} post image mappings")
            
        except Exception as e:
            logger.error(f"❌ Failed to load post_info: {e}")
            
    def get_all_json_files(self):
        """Get list of all JSON files"""
        logger.info("📂 Scanning JSON files...")
        
        json_files = glob.glob(os.path.join(self.json_dir, '*.json'))
        json_files.sort()  # Ensure consistent order
        
        logger.info(f"📊 Found {len(json_files):,} JSON files")
        return json_files
        
    def load_progress(self):
        """Load inference progress"""
        if os.path.exists(self.progress_file):
            try:
                with open(self.progress_file, 'r') as f:
                    progress = json.load(f)
                logger.info(f"📈 Resuming inference, already processed {progress.get('processed', 0):,} samples")
                return progress
            except:
                logger.warning("⚠️ Cannot read progress file, starting from beginning")
                
        return {'processed': 0, 'results': []}
        
    def save_progress(self, progress):
        """Save inference progress"""
        try:
            with open(self.progress_file, 'w') as f:
                json.dump(progress, f)
        except Exception as e:
            logger.error(f"Failed to save progress: {e}")
            
    def process_single_post(self, json_file_path):
        """Process a single post"""
        try:
            # Read JSON file
            with open(json_file_path, 'r', encoding='utf-8') as f:
                post_data = json.load(f)
            
            post_id = post_data.get('id', os.path.basename(json_file_path).replace('.json', ''))
            caption = post_data.get('edge_media_to_caption', {}).get('edges', [])
            
            # Extract text
            if caption and len(caption) > 0:
                text = caption[0].get('node', {}).get('text', '')
            else:
                text = ''
            
            # Get images
            image_ids = self.post_info_dict.get(str(post_id), [])
            
            if not image_ids:
                # If no images found, return default score
                return {
                    'post_id': post_id,
                    'gender_bias_score': 5.0,  # Neutral score
                    'image_count': 0,
                    'status': 'no_images'
                }
            
            # Load images (up to 3)
            images = []
            for image_id in image_ids[:3]:
                image_path = self.find_image_path(image_id)
                if image_path and os.path.exists(image_path):
                    try:
                        image = Image.open(image_path).convert('RGB')
                        image_tensor = self.transform(image)
                        images.append(image_tensor)
                    except:
                        continue
            
            if not images:
                return {
                    'post_id': post_id,
                    'gender_bias_score': 5.0,
                    'image_count': 0,
                    'status': 'image_load_failed'
                }
            
            # Prepare input
            # Image: take first one or concatenate multiple
            if len(images) == 1:
                image_input = images[0].unsqueeze(0)
            else:
                # Multiple images: take average
                image_input = torch.stack(images).mean(dim=0).unsqueeze(0)
            
            # Text tokenization
            encoding = self.tokenizer(
                text,
                truncation=True,
                padding='max_length',
                max_length=64,
                return_tensors='pt'
            )
            
            # Move to device
            image_input = image_input.to(self.device)
            input_ids = encoding['input_ids'].to(self.device)
            attention_mask = encoding['attention_mask'].to(self.device)
            
            # Inference
            with torch.no_grad():
                output = self.model(image_input, input_ids, attention_mask)
                score = output.item() * 10.0  # Denormalize to 0-10
                score = max(0.0, min(10.0, score))  # Limit range
            
            return {
                'post_id': post_id,
                'gender_bias_score': round(score, 2),
                'image_count': len(images),
                'status': 'success'
            }
            
        except Exception as e:
            logger.warning(f"Failed to process post {json_file_path}: {e}")
            return {
                'post_id': os.path.basename(json_file_path).replace('.json', ''),
                'gender_bias_score': 5.0,
                'image_count': 0,
                'status': 'error'
            }
    
    def find_image_path(self, image_id):
        """Find image file path"""
        # Search in various img_resized directories
        for i in range(1, 17):  # img_resized_1 to img_resized_16
            dir_path = os.path.join(self.database_path, f'img_resized_{i}')
            image_path = os.path.join(dir_path, f'{image_id}.jpg')
            if os.path.exists(image_path):
                return image_path
        return None
        
    def run_batch_inference(self, json_files, start_idx=0):
        """Run batch inference"""
        logger.info(f"🚀 Starting batch inference from file {start_idx:,}")
        
        # Load progress
        progress = self.load_progress()
        processed_count = progress['processed']
        results = progress['results']
        
        # Start from specified position
        remaining_files = json_files[start_idx:]
        total_files = len(json_files)
        
        logger.info(f"📊 Total files: {total_files:,}")
        logger.info(f"📈 Already processed: {processed_count:,}")
        logger.info(f"📋 Remaining: {len(remaining_files):,}")
        
        # Create progress bar
        pbar = tqdm(remaining_files, desc="Inference Progress", initial=processed_count, total=total_files)
        
        batch_results = []
        batch_start_time = time.time()
        
        for file_idx, json_file in enumerate(remaining_files):
            result = self.process_single_post(json_file)
            batch_results.append(result)
            processed_count += 1
            
            # Update progress bar
            pbar.update(1)
            pbar.set_postfix({
                'Score': f"{result['gender_bias_score']:.1f}",
                'Images': result['image_count']
            })
            
            # Periodic save
            if len(batch_results) >= self.save_interval:
                results.extend(batch_results)
                
                # Save progress
                progress = {
                    'processed': processed_count,
                    'results': results
                }
                self.save_progress(progress)
                
                # Save CSV
                self.save_batch_results(results)
                
                # Clean memory
                batch_results = []
                gc.collect()
                
                # Show statistics
                batch_time = time.time() - batch_start_time
                speed = self.save_interval / batch_time
                remaining = total_files - processed_count
                eta_hours = remaining / speed / 3600
                
                logger.info(f"💾 Saved {processed_count:,} results")
                logger.info(f"⚡ Processing speed: {speed:.1f} posts/sec")
                logger.info(f"⏰ Estimated remaining time: {eta_hours:.1f} hours")
                
                batch_start_time = time.time()
        
        # Save last batch
        if batch_results:
            results.extend(batch_results)
            progress = {
                'processed': processed_count,
                'results': results
            }
            self.save_progress(progress)
            self.save_batch_results(results)
        
        pbar.close()
        logger.info("🎉 Batch inference completed!")
        
        return results
        
    def save_batch_results(self, results):
        """Save batch results to CSV"""
        if not results:
            return
            
        df = pd.DataFrame(results)
        output_file = os.path.join(self.output_dir, 'full_inference_results.csv')
        
        # Add statistical information
        df['batch_timestamp'] = datetime.now().isoformat()
        
        df.to_csv(output_file, index=False)
        logger.info(f"💾 Results saved: {output_file}")
        
        # Show quick statistics
        success_count = (df['status'] == 'success').sum()
        avg_score = df[df['status'] == 'success']['gender_bias_score'].mean()
        
        logger.info(f"📊 Successfully processed: {success_count:,} / {len(df):,}")
        logger.info(f"📊 Average score: {avg_score:.2f}")
        
    def deploy(self):
        """Main deployment function"""
        logger.info("🚀 Starting Stage 4: Full-Scale Data Inference Deployment")
        
        start_time = time.time()
        
        # 1. Load model
        self.load_model_and_tokenizer()
        
        # 2. Load data mapping
        self.load_post_info()
        
        # 3. Get all JSON files
        json_files = self.get_all_json_files()
        
        # 4. Check progress
        progress = self.load_progress()
        start_idx = progress['processed']
        
        # 5. Run inference
        results = self.run_batch_inference(json_files, start_idx)
        
        # 6. Final statistics
        total_time = time.time() - start_time
        total_hours = total_time / 3600
        
        logger.info("🎉 Full inference deployment completed!")
        logger.info(f"⏱️ Total time: {total_hours:.2f} hours")
        logger.info(f"📊 Processed samples: {len(results):,}")
        logger.info(f"📁 Results file: {self.output_dir}/full_inference_results.csv")

def main():
    """Main function"""
    logger.info("🎯 Instagram Gender Bias Prediction - Full Deployment")
    
    deployer = FullInferenceDeployer()
    deployer.deploy()

if __name__ == "__main__":
    main()

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Multimodal Brand Inference System
Perform true multimodal inference (image + text) on filtered brand posts
"""

import pandas as pd
import json
import os
import torch
import numpy as np
import logging
from datetime import datetime
from tqdm import tqdm
from PIL import Image
import sys
import ast

# Import model
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from train_fast_local import LightweightGenderBiasModel

# Set environment variables to avoid conflicts
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Detect if running in background
def is_running_in_background():
    """Detect if running in background"""
    try:
        return not sys.stdout.isatty() or not sys.stdin.isatty()
    except:
        return True

DISABLE_TQDM = is_running_in_background()

# Setup logging
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('multimodal_inference.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class MultimodalBrandInference:
    def __init__(self):
        """Initialize multimodal brand inference system"""
        self.brand_results_file = '/Users/huangxinyue/Multi model distillation/brand_analysis_results/brand_analysis_final.csv'
        self.database_path = '/Users/huangxinyue/Downloads/Influencer brand database'
        self.post_info_file = os.path.join(self.database_path, 'post_info.txt')
        self.json_dir = os.path.join(self.database_path, 'json')
        self.model_path = '/Users/huangxinyue/Multi model distillation/fast_models_5k/fast_best_model.pth'
        
        # Output and progress files
        self.output_dir = '/Users/huangxinyue/Multi model distillation/multimodal_results'
        self.progress_file = os.path.join(self.output_dir, 'multimodal_progress.json')
        self.final_results_file = os.path.join(self.output_dir, 'multimodal_brand_analysis.csv')
        
        # Ensure output directory exists
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Model related
        self.model = None
        self.tokenizer = None
        self.transform = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Data mapping
        self.json_to_images = {}  # JSON filename -> image ID list
        
        logger.info(f"🚀 Multimodal brand inference system initialized")
        logger.info(f"📊 Using device: {self.device}")
    
    def load_progress(self):
        """Load progress"""
        if os.path.exists(self.progress_file):
            try:
                with open(self.progress_file, 'r') as f:
                    progress = json.load(f)
                logger.info(f"📈 Resuming inference, already processed {progress.get('processed', 0)} posts")
                return progress
            except Exception as e:
                logger.warning(f"Failed to read progress, starting from beginning: {e}")
        
        return {'processed': 0, 'results': []}
    
    def save_progress(self, progress):
        """Save progress"""
        try:
            progress['timestamp'] = datetime.now().isoformat()
            with open(self.progress_file, 'w') as f:
                json.dump(progress, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save progress: {e}")
    
    def load_post_info_mapping(self):
        """Load JSON->image mapping from post_info.txt"""
        logger.info("📚 Loading post_info mapping...")
        
        try:
            with open(self.post_info_file, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f):
                    try:
                        parts = line.strip().split('\t')
                        if len(parts) >= 5:
                            # Format: index username type JSON_filename image_ID_list
                            json_filename = parts[3]
                            image_list_str = parts[4]
                            
                            # Parse image ID list
                            try:
                                image_ids = ast.literal_eval(image_list_str)
                                if isinstance(image_ids, list):
                                    # Remove .jpg suffix, keep ID only
                                    image_ids = [img.replace('.jpg', '') for img in image_ids]
                                    self.json_to_images[json_filename] = image_ids
                            except:
                                continue
                                
                    except Exception as e:
                        continue
                    
                    # Show progress every 500k lines
                    if (line_num + 1) % 500000 == 0:
                        logger.info(f"  Processed {line_num + 1:,} lines, found {len(self.json_to_images):,} mappings")
            
            logger.info(f"✅ Successfully loaded {len(self.json_to_images):,} JSON->image mappings")
            
        except Exception as e:
            logger.error(f"❌ Failed to load post_info: {e}")
            raise
    
    def load_inference_model(self):
        """Load inference model"""
        logger.info("🚀 Loading 5K best multimodal model...")
        
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
        
        logger.info("✅ Multimodal model loading completed")
    
    def find_image_paths(self, image_ids):
        """Find image file paths"""
        image_paths = []
        for image_id in image_ids:
            # Search in various img_resized directories
            for i in range(1, 17):
                dir_path = os.path.join(self.database_path, f'img_resized_{i}')
                image_path = os.path.join(dir_path, f'{image_id}.jpg')
                if os.path.exists(image_path):
                    image_paths.append(image_path)
                    break
        return image_paths
    
    def analyze_rgb_colors(self, image):
        """Analyze RGB color distribution of image"""
        try:
            # Convert to numpy array
            image_array = np.array(image)
            
            # Calculate average RGB values
            red_mean = image_array[:, :, 0].mean()
            green_mean = image_array[:, :, 1].mean()
            blue_mean = image_array[:, :, 2].mean()
            
            # Calculate red-blue ratio
            red_blue_ratio = red_mean / (blue_mean + 1e-6)  # Avoid division by zero
            
            # Calculate red-blue dominance
            red_dominance = red_mean - blue_mean  # Positive = more red, negative = more blue
            
            return {
                'red_mean': float(red_mean),
                'green_mean': float(green_mean),
                'blue_mean': float(blue_mean),
                'red_blue_ratio': float(red_blue_ratio),
                'red_dominance': float(red_dominance)
            }
            
        except Exception as e:
            logger.warning(f"RGB analysis failed: {e}")
            return {
                'red_mean': 0.0,
                'green_mean': 0.0,
                'blue_mean': 0.0,
                'red_blue_ratio': 1.0,
                'red_dominance': 0.0
            }
    
    def load_images(self, image_paths):
        """Load and preprocess images, perform RGB analysis"""
        images = []
        rgb_analyses = []
        
        for image_path in image_paths[:3]:  # Process up to 3 images
            try:
                image = Image.open(image_path).convert('RGB')
                
                # RGB color analysis (before conversion)
                rgb_analysis = self.analyze_rgb_colors(image)
                rgb_analyses.append(rgb_analysis)
                
                # Convert to model input format
                image_tensor = self.transform(image)
                images.append(image_tensor)
                
            except Exception as e:
                logger.warning(f"Image loading failed {image_path}: {e}")
                continue
        
        # Merge RGB analysis results
        if rgb_analyses:
            # Calculate average RGB features for multiple images
            avg_rgb = {
                'red_mean': np.mean([r['red_mean'] for r in rgb_analyses]),
                'green_mean': np.mean([r['green_mean'] for r in rgb_analyses]),
                'blue_mean': np.mean([r['blue_mean'] for r in rgb_analyses]),
                'red_blue_ratio': np.mean([r['red_blue_ratio'] for r in rgb_analyses]),
                'red_dominance': np.mean([r['red_dominance'] for r in rgb_analyses])
            }
        else:
            avg_rgb = {
                'red_mean': 0.0,
                'green_mean': 0.0,
                'blue_mean': 0.0,
                'red_blue_ratio': 1.0,
                'red_dominance': 0.0
            }
        
        # Process image tensors
        if not images:
            # If no images, return black image
            image_tensor = torch.zeros(1, 3, 224, 224)
        elif len(images) == 1:
            image_tensor = images[0].unsqueeze(0)
        else:
            # Multiple images: take average
            image_tensor = torch.stack(images).mean(dim=0).unsqueeze(0)
        
        return image_tensor, avg_rgb
    
    def multimodal_inference(self, json_id, caption, image_paths):
        """Multimodal inference (image + text)"""
        try:
            # 1. Process images and get RGB analysis
            image_tensor, rgb_analysis = self.load_images(image_paths)
            
            # 2. Process text
            encoding = self.tokenizer(
                caption,
                truncation=True,
                padding='max_length',
                max_length=64,
                return_tensors='pt'
            )
            
            # 3. Move to device
            image_tensor = image_tensor.to(self.device)
            input_ids = encoding['input_ids'].to(self.device)
            attention_mask = encoding['attention_mask'].to(self.device)
            
            # 4. Multimodal inference
            with torch.no_grad():
                output = self.model(image_tensor, input_ids, attention_mask)
                score = output.item() * 10.0  # Denormalize to 0-10
                score = max(0.0, min(10.0, score))  # Limit range
            
            return round(score, 2), len(image_paths), rgb_analysis
            
        except Exception as e:
            logger.warning(f"Multimodal inference failed {json_id}: {e}")
            # Return default values including default RGB analysis
            default_rgb = {
                'red_mean': 0.0,
                'green_mean': 0.0,
                'blue_mean': 0.0,
                'red_blue_ratio': 1.0,
                'red_dominance': 0.0
            }
            return 5.0, 0, default_rgb  # Default neutral score
    
    def process_brand_posts(self):
        """Process brand posts for multimodal inference"""
        logger.info("🔍 Reading filtered brand posts...")
        
        # Read first stage results
        if not os.path.exists(self.brand_results_file):
            logger.error(f"❌ Brand filtering results file does not exist: {self.brand_results_file}")
            return
        
        brand_df = pd.read_csv(self.brand_results_file)
        logger.info(f"✅ Read {len(brand_df)} brand posts")
        
        # Load progress
        progress = self.load_progress()
        processed_count = progress['processed']
        results = progress.get('results', [])
        
        # Start processing from checkpoint
        posts_to_process = brand_df.iloc[processed_count:]
        logger.info(f"📈 Starting multimodal inference from post {processed_count}")
        logger.info(f"📋 Remaining to process: {len(posts_to_process)} posts")
        
        save_interval = 100  # Save every 100 posts
        batch_start_time = datetime.now()
        
        with tqdm(posts_to_process.iterrows(), 
                  desc="Multimodal Inference", 
                  total=len(posts_to_process),
                  disable=DISABLE_TQDM) as pbar:
            
            for row_idx, (_, row) in enumerate(pbar):
                try:
                    json_id = str(row['json_id'])
                    influencer_name = row['influencer_name']
                    sponsored = row['sponsored']
                    brand = row['brand']
                    
                    # 1. Get caption from JSON file
                    json_file = os.path.join(self.json_dir, f'{json_id}.json')
                    caption = ""
                    
                    if os.path.exists(json_file):
                        try:
                            with open(json_file, 'r', encoding='utf-8') as f:
                                post_data = json.load(f)
                            
                            caption_edges = post_data.get('edge_media_to_caption', {}).get('edges', [])
                            if caption_edges:
                                caption = caption_edges[0].get('node', {}).get('text', '')
                        except Exception as e:
                            logger.warning(f"Failed to read JSON {json_id}: {e}")
                    
                    # 2. Get image IDs from post_info.txt
                    json_filename = f'{json_id}.json'
                    image_ids = self.json_to_images.get(json_filename, [])
                    
                    # 3. Find actual image paths
                    image_paths = self.find_image_paths(image_ids)
                    
                    # 4. Multimodal inference and RGB analysis
                    gender_score, image_count, rgb_analysis = self.multimodal_inference(json_id, caption, image_paths)
                    
                    # 5. Save results (including RGB analysis)
                    result = {
                        'json_id': json_id,
                        'influencer_name': influencer_name,
                        'sponsored': sponsored,
                        'brand': brand,
                        'gender_bias_score': gender_score,
                        'image_count': image_count,
                        'caption_length': len(caption) if caption else 0,
                        'red_mean': rgb_analysis['red_mean'],
                        'green_mean': rgb_analysis['green_mean'],
                        'blue_mean': rgb_analysis['blue_mean'],
                        'red_blue_ratio': rgb_analysis['red_blue_ratio'],
                        'red_dominance': rgb_analysis['red_dominance']
                    }
                    
                    results.append(result)
                    processed_count += 1
                    
                    # Update progress bar
                    if not DISABLE_TQDM:
                        # Determine color tendency
                        color_tendency = "Red" if rgb_analysis['red_dominance'] > 0 else "Blue"
                        pbar.set_postfix({
                            'Brand': brand[:8],
                            'Score': f"{gender_score:.1f}",
                            'Images': image_count,
                            'Color': color_tendency,
                            'Sponsored': 'Y' if sponsored else 'N'
                        })
                    
                    # Background mode progress logging
                    if DISABLE_TQDM and (row_idx + 1) % 50 == 0:
                        color_info = "red tendency" if rgb_analysis['red_dominance'] > 0 else "blue tendency"
                        logger.info(f"  Processed {processed_count}/{len(brand_df)} posts "
                                  f"({processed_count/len(brand_df)*100:.1f}%) - "
                                  f"Latest: {brand} {gender_score:.1f} score {color_info}")
                    
                    # Periodic save
                    if len(results) % save_interval == 0:
                        # Save progress
                        progress_data = {
                            'processed': processed_count,
                            'results': results
                        }
                        self.save_progress(progress_data)
                        
                        # Save CSV
                        self.save_results(results)
                        
                        # Show batch statistics
                        batch_time = (datetime.now() - batch_start_time).total_seconds()
                        speed = save_interval / batch_time
                        remaining = len(brand_df) - processed_count
                        eta_minutes = remaining / speed / 60
                        
                        logger.info(f"💾 Saved {processed_count} results")
                        logger.info(f"⚡ Processing speed: {speed:.1f} posts/sec")
                        logger.info(f"⏰ Estimated remaining: {eta_minutes:.1f} minutes")
                        
                        batch_start_time = datetime.now()
                        
                except Exception as e:
                    logger.error(f"Failed to process post {row.get('json_id', 'unknown')}: {e}")
                    processed_count += 1
                    continue
        
        # Save final results
        progress_data = {
            'processed': processed_count,
            'results': results
        }
        self.save_progress(progress_data)
        self.save_results(results)
        
        logger.info(f"🎉 Multimodal inference completed! Processed {len(results)} posts")
        return results
    
    def save_results(self, results):
        """Save results to CSV"""
        if not results:
            return
        
        df = pd.DataFrame(results)
        df.to_csv(self.final_results_file, index=False)
        
        # Show statistics
        total = len(df)
        success_with_images = (df['image_count'] > 0).sum()
        avg_score = df['gender_bias_score'].mean()
        red_dominant = (df['red_dominance'] > 0).sum()
        blue_dominant = (df['red_dominance'] < 0).sum()
        neutral_color = (df['red_dominance'] == 0).sum()
        avg_red_dominance = df['red_dominance'].mean()
        
        logger.info(f"💾 Results saved: {self.final_results_file}")
        logger.info(f"📊 Posts with images: {success_with_images}/{total} ({success_with_images/total*100:.1f}%)")
        logger.info(f"📊 Average gender bias score: {avg_score:.2f}")
        logger.info(f"🎨 Color distribution: Red tendency {red_dominant} posts ({red_dominant/total*100:.1f}%), "
                   f"Blue tendency {blue_dominant} posts ({blue_dominant/total*100:.1f}%), "
                   f"Neutral {neutral_color} posts ({neutral_color/total*100:.1f}%)")
        logger.info(f"🎨 Average red-blue dominance value: {avg_red_dominance:.2f}")
    
    def show_final_statistics(self, results):
        """Show final statistics"""
        df = pd.DataFrame(results)
        
        logger.info("📊 Final multimodal inference statistics:")
        logger.info("=" * 60)
        logger.info(f"Total posts: {len(df):,}")
        logger.info(f"Brands involved: {df['brand'].nunique()} brands")
        logger.info(f"Influencers involved: {df['influencer_name'].nunique():,} influencers")
        logger.info(f"Sponsored posts: {df['sponsored'].sum():,} ({df['sponsored'].mean()*100:.1f}%)")
        logger.info(f"Posts with images: {(df['image_count'] > 0).sum():,} ({(df['image_count'] > 0).mean()*100:.1f}%)")
        logger.info(f"Average image count: {df['image_count'].mean():.2f}")
        logger.info(f"Average gender bias score: {df['gender_bias_score'].mean():.2f} ± {df['gender_bias_score'].std():.2f}")
        
        # RGB color statistics
        red_dominant = (df['red_dominance'] > 0).sum()
        blue_dominant = (df['red_dominance'] < 0).sum()
        neutral_color = (df['red_dominance'] == 0).sum()
        logger.info(f"🎨 Color distribution:")
        logger.info(f"  Red tendency: {red_dominant:,} ({red_dominant/len(df)*100:.1f}%)")
        logger.info(f"  Blue tendency: {blue_dominant:,} ({blue_dominant/len(df)*100:.1f}%)")
        logger.info(f"  Neutral: {neutral_color:,} ({neutral_color/len(df)*100:.1f}%)")
        logger.info(f"🎨 Average red-blue dominance value: {df['red_dominance'].mean():.2f} ± {df['red_dominance'].std():.2f}")
        logger.info(f"🎨 Average red-blue ratio: {df['red_blue_ratio'].mean():.2f} ± {df['red_blue_ratio'].std():.2f}")
        
        logger.info(f"\n🏆 Top 10 brand multimodal analysis results:")
        brand_stats = df.groupby('brand').agg({
            'gender_bias_score': ['count', 'mean', 'std'],
            'sponsored': 'mean',
            'image_count': 'mean',
            'red_dominance': 'mean',
            'red_blue_ratio': 'mean'
        }).round(3)
        
        brand_stats.columns = ['Post Count', 'Avg Score', 'Score Std', 'Sponsor Rate', 'Avg Images', 'Red-Blue Tendency', 'Red-Blue Ratio']
        brand_stats = brand_stats.sort_values('Post Count', ascending=False)
        
        print("\n" + brand_stats.head(10).to_string())
        
        logger.info(f"\n💾 Final results file: {self.final_results_file}")
    
    def run_multimodal_inference(self):
        """Run complete multimodal inference"""
        logger.info("🚀 Starting multimodal brand inference system")
        
        start_time = datetime.now()
        
        try:
            # 1. Load post_info mapping
            self.load_post_info_mapping()
            
            # 2. Load inference model
            self.load_inference_model()
            
            # 3. Process brand posts
            results = self.process_brand_posts()
            
            # 4. Show final statistics
            self.show_final_statistics(results)
            
            # 5. Calculate total time
            total_time = datetime.now() - start_time
            logger.info(f"⏱️ Total inference time: {total_time.total_seconds()/60:.1f} minutes")
            logger.info("🎉 Multimodal brand inference completed!")
            
        except Exception as e:
            logger.error(f"❌ Multimodal inference failed: {e}")
            raise

def main():
    """Main function"""
    logger.info("🎯 Instagram Brand Multimodal Inference System")
    logger.info("Objective: Perform image+text inference on 33,829 brand posts")
    logger.info("=" * 60)
    
    # Run multimodal inference
    inferencer = MultimodalBrandInference()
    inferencer.run_multimodal_inference()

if __name__ == "__main__":
    main()

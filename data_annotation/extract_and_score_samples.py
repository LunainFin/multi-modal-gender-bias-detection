#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Extract and Score Samples Program
Extract Instagram post data and score them for gender bias using AI models
"""

import json
import os
import requests
import base64
import time
import csv
import logging
from typing import List, Dict, Optional
from tqdm import tqdm
import signal
import sys
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('extract_and_score.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class InstagramSampleExtractor:
    def __init__(self):
        """
        Initialize Instagram sample extractor and scorer
        """
        self.current_dir = "/Users/huangxinyue/Multi model distillation"
        self.database_path = "/Users/huangxinyue/Downloads/Influencer brand database"
        self.json_dir = os.path.join(self.database_path, "json")
        self.output_dir = os.path.join(self.current_dir, "small_batch_results")
        
        # API configuration
        self.api_key = "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"
        self.model_name = "qwen/qwen2.5-vl-32b-instruct:free"
        
        # Processing configuration
        self.batch_size = 100  # Process in batches
        self.max_samples = 1000  # Maximum samples to process
        self.delay_between_requests = 1.0  # Delay between API calls (seconds)
        
        # Progress tracking
        self.progress_file = os.path.join(self.output_dir, "progress.json")
        self.results_file = os.path.join(self.output_dir, "scored_samples.csv")
        
        # Statistics
        self.processed_count = 0
        self.success_count = 0
        self.error_count = 0
        self.start_time = time.time()
        
        # Create output directory
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Results storage
        self.results = []
        
        logger.info("🚀 Instagram sample extractor initialized")
        logger.info(f"📁 Database path: {self.database_path}")
        logger.info(f"📁 Output directory: {self.output_dir}")
        logger.info(f"🎯 Max samples: {self.max_samples}")
    
    def load_progress(self):
        """Load processing progress"""
        if os.path.exists(self.progress_file):
            try:
                with open(self.progress_file, 'r') as f:
                    progress = json.load(f)
                self.processed_count = progress.get('processed_count', 0)
                self.success_count = progress.get('success_count', 0)
                self.error_count = progress.get('error_count', 0)
                self.start_time = progress.get('start_time', time.time())
                
                # Load existing results
                if os.path.exists(self.results_file):
                    with open(self.results_file, 'r', encoding='utf-8') as f:
                        reader = csv.DictReader(f)
                        self.results = list(reader)
                
                logger.info(f"📈 Resuming from: {self.processed_count} processed, {self.success_count} successful")
                return True
            except Exception as e:
                logger.warning(f"Failed to load progress: {e}")
        return False
    
    def save_progress(self):
        """Save processing progress"""
        progress = {
            'processed_count': self.processed_count,
            'success_count': self.success_count,
            'error_count': self.error_count,
            'start_time': self.start_time,
            'last_update': time.time()
        }
        
        try:
            with open(self.progress_file, 'w') as f:
                json.dump(progress, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save progress: {e}")
    
    def get_sample_posts(self) -> List[Dict]:
        """Get sample posts for processing"""
        logger.info("📂 Scanning for sample posts...")
        
        posts = []
        
        # Check if we have local JSON samples first
        local_json_dir = os.path.join(self.current_dir, "json_samples")
        if os.path.exists(local_json_dir):
            logger.info("Using local JSON samples...")
            json_files = [f for f in os.listdir(local_json_dir) if f.endswith('.json')]
            
            for filename in json_files[:self.max_samples]:
                post_id = filename.replace('.json', '')
                json_path = os.path.join(local_json_dir, filename)
                
                posts.append({
                    'post_id': post_id,
                    'json_path': json_path,
                    'source': 'local_samples'
                })
        
        # If no local samples, use database
        elif os.path.exists(self.json_dir):
            logger.info("Using database JSON files...")
            json_files = [f for f in os.listdir(self.json_dir) if f.endswith('.json')]
            json_files = json_files[:self.max_samples]  # Limit to max_samples
            
            for filename in json_files:
                post_id = filename.replace('.json', '')
                json_path = os.path.join(self.json_dir, filename)
                
                posts.append({
                    'post_id': post_id,
                    'json_path': json_path,
                    'source': 'database'
                })
        
        logger.info(f"📊 Found {len(posts)} posts to process")
        return posts
    
    def extract_post_data(self, json_path: str) -> Dict:
        """Extract relevant data from Instagram post JSON"""
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                post_data = json.load(f)
            
            # Extract basic information
            post_id = post_data.get('id', '')
            owner = post_data.get('owner', {})
            username = owner.get('username', 'unknown')
            
            # Extract caption
            caption = ""
            caption_edges = post_data.get('edge_media_to_caption', {}).get('edges', [])
            if caption_edges:
                caption = caption_edges[0].get('node', {}).get('text', '')
            
            # Extract engagement metrics
            likes = post_data.get('edge_media_preview_like', {}).get('count', 0)
            comments = post_data.get('edge_media_to_comment', {}).get('count', 0)
            
            # Extract media type
            media_type = post_data.get('__typename', 'unknown')
            
            return {
                'post_id': post_id,
                'username': username,
                'caption': caption,
                'likes': likes,
                'comments': comments,
                'media_type': media_type,
                'caption_length': len(caption) if caption else 0
            }
            
        except Exception as e:
            logger.error(f"Failed to extract data from {json_path}: {e}")
            return None
    
    def score_gender_bias(self, caption: str, post_id: str) -> Optional[float]:
        """Score gender bias using AI model"""
        try:
            # Prepare API request
            content = [
                {
                    "type": "text",
                    "text": f"""Analyze this Instagram post caption for gender bias on a scale of 0-10:

Caption: "{caption}"

Scoring criteria:
- 0-2: Strongly male-targeted (very masculine themes, sports, technology, cars, etc.)
- 3-4: Somewhat male-targeted (slightly masculine lean)
- 5: Gender-neutral (no clear gender targeting, universal appeal)
- 6-7: Somewhat female-targeted (slightly feminine lean)
- 8-10: Strongly female-targeted (very feminine themes, beauty, fashion, lifestyle, etc.)

Requirements:
1. Return ONLY a number between 0 and 10
2. Use one decimal place (e.g., 5.7)
3. No explanation or additional text
4. If caption is empty or unclear, return 5.0"""
                }
            ]
            
            payload = {
                "model": self.model_name,
                "messages": [
                    {
                        "role": "user",
                        "content": content
                    }
                ],
                "max_tokens": 10,
                "temperature": 0.1
            }
            
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
                "HTTP-Referer": "https://github.com/multi-model-distillation",
                "X-Title": "Instagram Gender Bias Analysis"
            }
            
            # Make API call
            response = requests.post(
                url="https://openrouter.ai/api/v1/chat/completions",
                headers=headers,
                json=payload,
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                content = result['choices'][0]['message']['content'].strip()
                
                # Extract score
                try:
                    score = float(content)
                    if 0 <= score <= 10:
                        return round(score, 1)
                except ValueError:
                    # Try to extract number from text
                    import re
                    numbers = re.findall(r'\d+(?:\.\d+)?', content)
                    if numbers:
                        score = float(numbers[0])
                        if 0 <= score <= 10:
                            return round(score, 1)
                
                logger.warning(f"Invalid score format for {post_id}: {content}")
                return 5.0  # Default neutral score
            else:
                logger.error(f"API error for {post_id}: {response.status_code}")
                return None
                
        except Exception as e:
            logger.error(f"Error scoring {post_id}: {e}")
            return None
    
    def process_single_post(self, post_info: Dict) -> Optional[Dict]:
        """Process a single post"""
        post_id = post_info['post_id']
        json_path = post_info['json_path']
        
        try:
            # Extract post data
            post_data = self.extract_post_data(json_path)
            if not post_data:
                return None
            
            # Score gender bias
            caption = post_data['caption']
            gender_score = self.score_gender_bias(caption, post_id)
            
            if gender_score is not None:
                # Combine all data
                result = {
                    **post_data,
                    'gender_bias_score': gender_score,
                    'source': post_info['source'],
                    'processed_timestamp': datetime.now().isoformat()
                }
                
                return result
            else:
                return None
                
        except Exception as e:
            logger.error(f"Error processing post {post_id}: {e}")
            return None
    
    def save_results(self):
        """Save results to CSV file"""
        try:
            if not self.results:
                return
            
            fieldnames = [
                'post_id', 'username', 'caption', 'gender_bias_score',
                'likes', 'comments', 'media_type', 'caption_length',
                'source', 'processed_timestamp'
            ]
            
            with open(self.results_file, 'w', newline='', encoding='utf-8') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(self.results)
            
            logger.info(f"💾 Saved {len(self.results)} results to {self.results_file}")
            
        except Exception as e:
            logger.error(f"Failed to save results: {e}")
    
    def show_statistics(self):
        """Show processing statistics"""
        elapsed = time.time() - self.start_time
        hours = elapsed // 3600
        minutes = (elapsed % 3600) // 60
        
        if elapsed > 0:
            speed = self.processed_count / elapsed * 3600  # per hour
            success_rate = (self.success_count / self.processed_count * 100) if self.processed_count > 0 else 0
            
            logger.info("📊 Processing Statistics:")
            logger.info(f"   Processed: {self.processed_count:,}")
            logger.info(f"   Successful: {self.success_count:,}")
            logger.info(f"   Failed: {self.error_count:,}")
            logger.info(f"   Success rate: {success_rate:.1f}%")
            logger.info(f"   Processing speed: {speed:.1f} posts/hour")
            logger.info(f"   Elapsed time: {int(hours)}h {int(minutes)}m")
            
            if self.results:
                scores = [float(r['gender_bias_score']) for r in self.results]
                avg_score = sum(scores) / len(scores)
                logger.info(f"   Average gender bias score: {avg_score:.2f}")
    
    def process_all_samples(self):
        """Process all sample posts"""
        logger.info("🚀 Starting sample extraction and scoring")
        
        # Load progress if exists
        self.load_progress()
        
        # Get posts to process
        posts = self.get_sample_posts()
        
        if not posts:
            logger.error("❌ No posts found to process")
            return
        
        # Skip already processed posts
        posts_to_process = posts[self.processed_count:]
        
        if not posts_to_process:
            logger.info("✅ All posts already processed")
            self.show_statistics()
            return
        
        logger.info(f"📋 Processing {len(posts_to_process)} remaining posts")
        
        # Process posts with progress bar
        for post_info in tqdm(posts_to_process, desc="Processing posts"):
            result = self.process_single_post(post_info)
            
            self.processed_count += 1
            
            if result:
                self.results.append(result)
                self.success_count += 1
            else:
                self.error_count += 1
            
            # Save progress periodically
            if self.processed_count % 10 == 0:
                self.save_progress()
                self.save_results()
            
            # Add delay between requests to be respectful to API
            time.sleep(self.delay_between_requests)
            
            # Show progress every 25 posts
            if self.processed_count % 25 == 0:
                self.show_statistics()
        
        # Final save
        self.save_progress()
        self.save_results()
        
        logger.info("🎉 Sample extraction and scoring completed!")
        self.show_statistics()

def main():
    """Main function"""
    logger.info("🎯 Instagram Sample Extractor and Gender Bias Scorer")
    logger.info("=" * 60)
    
    extractor = InstagramSampleExtractor()
    
    # Handle interruption gracefully
    def signal_handler(sig, frame):
        logger.info("\n⏹️ Processing interrupted by user")
        extractor.save_progress()
        extractor.save_results()
        extractor.show_statistics()
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    
    try:
        extractor.process_all_samples()
    except KeyboardInterrupt:
        logger.info("\n⏹️ Processing interrupted")
        extractor.save_progress()
        extractor.save_results()
        extractor.show_statistics()

if __name__ == "__main__":
    main()

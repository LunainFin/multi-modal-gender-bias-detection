#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Training Data Processing Program - Fast Concurrent Version
Supports concurrent API calls for dramatically improved processing speed
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
import asyncio
import aiohttp
import math
import concurrent.futures
from threading import Lock

# Configure logging
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('train_10k_fast.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class FastTrainDataProcessor:
    def __init__(self):
        """
        Initialize fast training data processor
        """
        self.database_path = "/Users/huangxinyue/Downloads/Influencer brand database"
        self.output_dir = "/Users/huangxinyue/Multi model distillation/train_10k_results"
        self.batch_size = 10000
        self.api_key = "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"
        self.model_name = "qwen/qwen2.5-vl-32b-instruct"
        
        # Concurrency control
        self.max_concurrent = 8  # Maximum concurrent requests
        self.semaphore = asyncio.Semaphore(self.max_concurrent)
        
        # Progress tracking
        self.progress_file = os.path.join(self.output_dir, "progress_fast.json")
        self.results_file = os.path.join(self.output_dir, "train_10k_fast_results.csv")
        self.lock = Lock()
        
        # Statistics
        self.processed_count = 0
        self.success_count = 0
        self.error_count = 0
        self.start_time = time.time()
        
        # Create output directory
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Results storage
        self.results = []
        
        logger.info("🚀 Fast training data processor initialized")
        logger.info(f"📁 Database path: {self.database_path}")
        logger.info(f"📁 Output directory: {self.output_dir}")
        logger.info(f"⚡ Max concurrent requests: {self.max_concurrent}")
    
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
    
    def get_sample_posts(self, start_idx=0, limit=None):
        """Get sample posts for processing"""
        logger.info("📂 Scanning JSON files...")
        
        json_dir = os.path.join(self.database_path, "json")
        json_files = []
        
        for filename in os.listdir(json_dir):
            if filename.endswith('.json'):
                json_files.append(filename)
        
        json_files.sort()  # Ensure consistent order
        
        # Apply start index and limit
        if limit:
            json_files = json_files[start_idx:start_idx + limit]
        else:
            json_files = json_files[start_idx:]
        
        logger.info(f"📊 Found {len(json_files)} JSON files to process")
        
        posts = []
        for filename in json_files:
            post_id = filename.replace('.json', '')
            json_path = os.path.join(json_dir, filename)
            
            # Find corresponding image
            image_path = self.find_image_path(post_id)
            
            if image_path and os.path.exists(image_path):
                posts.append({
                    'post_id': post_id,
                    'json_path': json_path,
                    'image_path': image_path
                })
        
        logger.info(f"✅ Found {len(posts)} posts with both JSON and image")
        return posts
    
    def find_image_path(self, post_id):
        """Find image path for a post"""
        # Search in img_resized_1 to img_resized_16 directories
        for i in range(1, 17):
            img_dir = os.path.join(self.database_path, f"img_resized_{i}")
            img_path = os.path.join(img_dir, f"{post_id}.jpg")
            if os.path.exists(img_path):
                return img_path
        return None
    
    def encode_image_base64(self, image_path):
        """Encode image to base64"""
        try:
            with open(image_path, "rb") as image_file:
                encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
                return encoded_string
        except Exception as e:
            logger.error(f"Failed to encode image {image_path}: {e}")
            return None
    
    def load_post_caption(self, json_path):
        """Load post caption from JSON"""
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Extract caption
            caption_edges = data.get('edge_media_to_caption', {}).get('edges', [])
            if caption_edges and len(caption_edges) > 0:
                caption = caption_edges[0].get('node', {}).get('text', '')
                return caption
            return ""
        except Exception as e:
            logger.error(f"Failed to load caption from {json_path}: {e}")
            return ""
    
    async def analyze_single_post(self, session, post_data):
        """Analyze single post with concurrent API call"""
        async with self.semaphore:
            try:
                post_id = post_data['post_id']
                image_path = post_data['image_path']
                json_path = post_data['json_path']
                
                # Encode image
                image_base64 = self.encode_image_base64(image_path)
                if not image_base64:
                    return None
                
                # Load caption
                caption = self.load_post_caption(json_path)
                
                # Prepare API request
                payload = {
                    "model": self.model_name,
                    "messages": [
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "image_url",
                                    "image_url": {
                                        "url": f"data:image/jpeg;base64,{image_base64}"
                                    }
                                },
                                {
                                    "type": "text",
                                    "text": f"""Analyze this Instagram post for gender bias. Consider both the image and caption: "{caption}"

Rate the gender bias on a scale of 0-10:
- 0-2: Strongly male-targeted (very masculine themes, male-dominated imagery)
- 3-4: Somewhat male-targeted (slightly masculine lean)
- 5: Gender-neutral (no clear gender targeting)
- 6-7: Somewhat female-targeted (slightly feminine lean)
- 8-10: Strongly female-targeted (very feminine themes, female-dominated imagery)

Respond with only the numerical score (0-10)."""
                                }
                            ]
                        }
                    ],
                    "max_tokens": 10
                }
                
                headers = {
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json"
                }
                
                # Make API call
                async with session.post(
                    "https://openrouter.ai/api/v1/chat/completions",
                    json=payload,
                    headers=headers,
                    timeout=aiohttp.ClientTimeout(total=60)
                ) as response:
                    
                    if response.status == 200:
                        result = await response.json()
                        content = result['choices'][0]['message']['content'].strip()
                        
                        # Extract score
                        try:
                            score = float(content)
                            if 0 <= score <= 10:
                                return {
                                    'post_id': post_id,
                                    'gender_bias_score': score,
                                    'caption': caption,
                                    'status': 'success'
                                }
                        except ValueError:
                            pass
                        
                        logger.warning(f"Invalid score format for {post_id}: {content}")
                        return None
                    else:
                        logger.error(f"API error for {post_id}: {response.status}")
                        return None
                        
            except Exception as e:
                logger.error(f"Error processing {post_data.get('post_id', 'unknown')}: {e}")
                return None
    
    async def process_batch_concurrent(self, posts_batch):
        """Process batch of posts concurrently"""
        async with aiohttp.ClientSession() as session:
            tasks = []
            for post_data in posts_batch:
                task = self.analyze_single_post(session, post_data)
                tasks.append(task)
            
            # Execute all tasks concurrently
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            batch_results = []
            for result in results:
                if isinstance(result, dict) and result.get('status') == 'success':
                    batch_results.append(result)
                    with self.lock:
                        self.success_count += 1
                else:
                    with self.lock:
                        self.error_count += 1
                
                with self.lock:
                    self.processed_count += 1
            
            return batch_results
    
    def save_results(self, results):
        """Save results to CSV"""
        try:
            # Append to existing results
            self.results.extend(results)
            
            # Write to CSV
            with open(self.results_file, 'w', newline='', encoding='utf-8') as csvfile:
                fieldnames = ['post_id', 'gender_bias_score', 'caption']
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()
                
                for result in self.results:
                    writer.writerow({
                        'post_id': result['post_id'],
                        'gender_bias_score': result['gender_bias_score'],
                        'caption': result['caption']
                    })
            
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
            
            if self.processed_count < self.batch_size:
                remaining = self.batch_size - self.processed_count
                eta_hours = remaining / speed if speed > 0 else 0
                logger.info(f"   Remaining: {remaining:,} posts")
                logger.info(f"   ETA: {eta_hours:.1f} hours")
    
    async def process_training_data(self):
        """Process training data with fast concurrent approach"""
        logger.info("🚀 Starting fast concurrent training data processing")
        
        # Load progress
        self.load_progress()
        
        # Get posts to process
        posts = self.get_sample_posts(start_idx=self.processed_count, limit=self.batch_size)
        
        if not posts:
            logger.info("✅ No posts to process")
            return
        
        # Process in batches
        batch_size = 50  # Concurrent batch size
        total_batches = math.ceil(len(posts) / batch_size)
        
        logger.info(f"📦 Processing {len(posts)} posts in {total_batches} batches")
        logger.info(f"⚡ Concurrent batch size: {batch_size}")
        
        for batch_idx in range(0, len(posts), batch_size):
            batch_posts = posts[batch_idx:batch_idx + batch_size]
            batch_num = batch_idx // batch_size + 1
            
            logger.info(f"🔄 Processing batch {batch_num}/{total_batches} ({len(batch_posts)} posts)")
            
            try:
                # Process batch concurrently
                batch_results = await self.process_batch_concurrent(batch_posts)
                
                # Save results
                if batch_results:
                    self.save_results(batch_results)
                
                # Save progress
                self.save_progress()
                
                # Show statistics
                self.show_statistics()
                
                # Small delay between batches
                await asyncio.sleep(1)
                
            except Exception as e:
                logger.error(f"Batch {batch_num} processing failed: {e}")
                continue
        
        logger.info("🎉 Fast concurrent processing completed!")
        self.show_statistics()

def main():
    """Main function"""
    logger.info("⚡ Fast Training Data Processor")
    logger.info("Target: Process 10,000 Instagram posts for gender bias training")
    logger.info("=" * 60)
    
    processor = FastTrainDataProcessor()
    
    # Handle interruption gracefully
    def signal_handler(sig, frame):
        logger.info("\n⏹️  Processing interrupted by user")
        processor.save_progress()
        processor.show_statistics()
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    
    # Run async processing
    try:
        asyncio.run(processor.process_training_data())
    except KeyboardInterrupt:
        logger.info("\n⏹️  Processing interrupted")
        processor.save_progress()
        processor.show_statistics()

if __name__ == "__main__":
    main()

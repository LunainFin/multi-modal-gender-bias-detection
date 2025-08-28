#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Small Batch Testing Program - Validate complete workflow using sample data in current directory
"""

import json
import os
import requests
import base64
import time
from typing import List, Dict, Optional
import csv
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class SmallBatchProcessor:
    def __init__(self):
        """
        Batch processing test using sample data in current directory (all available samples)
        """
        self.current_dir = "/Users/huangxinyue/Multi model distillation"
        self.api_key = "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"
        self.model_name = "qwen/qwen2.5-vl-32b-instruct:free"
        self.results = []
        
        # Create output directory
        self.output_dir = os.path.join(self.current_dir, "batch_all_results")
        os.makedirs(self.output_dir, exist_ok=True)
    
    def load_sample_data(self) -> List[Dict]:
        """
        Load sample data from current directory
        """
        logger.info("Loading sample data...")
        
        json_dir = os.path.join(self.current_dir, "json_samples")
        json_files = [f for f in os.listdir(json_dir) if f.endswith('.json')]  # Process all available samples
        
        samples = []
        for json_file in json_files:
            try:
                json_path = os.path.join(json_dir, json_file)
                with open(json_path, 'r', encoding='utf-8') as f:
                    post_data = json.load(f)
                
                # Extract caption
                caption = ""
                if 'edge_media_to_caption' in post_data and post_data['edge_media_to_caption']['edges']:
                    caption = post_data['edge_media_to_caption']['edges'][0]['node']['text']
                
                post_id = post_data.get('id', json_file.replace('.json', ''))
                
                samples.append({
                    'post_id': post_id,
                    'caption': caption,
                    'json_file': json_file
                })
                
            except Exception as e:
                logger.warning(f"Failed to load {json_file}: {e}")
        
        logger.info(f"Successfully loaded {len(samples)} samples")
        return samples
    
    def call_qwen_api(self, caption: str) -> Optional[dict]:
        """
        Call Qwen API for gender bias scoring (text-only version)
        """
        try:
            # Build API request
            content = [
                {
                    "type": "text",
                    "text": f"Please evaluate the gender bias level of this Instagram post on a scale of 0-10, where higher scores indicate stronger gender orientation.\n\nRequirements:\n1. First provide a numerical score between 0 and 10\n2. Then explain the reasoning in one sentence\n3. Format: Score: X.X | Reason: [explanation]\n\nPost content: {caption}"
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
                "max_tokens": 100,  # Increase token count to accommodate explanation
                "temperature": 0.1
            }
            
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
                "HTTP-Referer": "https://github.com/multi-model-distillation",
                "X-Title": "Multi-Model Gender Bias Analysis"
            }
            
            # Send API request
            response = requests.post(
                url="https://openrouter.ai/api/v1/chat/completions",
                headers=headers,
                json=payload,
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                content = result['choices'][0]['message']['content'].strip()
                
                # Parse score and explanation
                try:
                    # Try to parse format: "Score: X.X | Reason: [explanation]"
                    if '|' in content:
                        parts = content.split('|')
                        score_part = parts[0].strip()
                        reason_part = parts[1].strip() if len(parts) > 1 else ""
                        
                        # Extract score
                        import re
                        score_match = re.search(r'(\d+(?:\.\d+)?)', score_part)
                        if score_match:
                            score = float(score_match.group(1))
                            if 0 <= score <= 10:
                                # Extract explanation
                                reason = reason_part.replace('Reason:', '').replace('原因:', '').replace('原因：', '').strip()
                                return {
                                    'score': score,
                                    'reason': reason,
                                    'raw_response': content
                                }
                        
                    # Backup parsing method: if format is non-standard, try to extract numbers
                    numbers = re.findall(r'\d+(?:\.\d+)?', content)
                    if numbers:
                        score = float(numbers[0])
                        if 0 <= score <= 10:
                            return {
                                'score': score,
                                'reason': content,  # Use entire response as explanation
                                'raw_response': content
                            }
                    
                    logger.warning(f"Unable to parse returned score and explanation: {content}")
                    return None
                    
                except Exception as e:
                    logger.warning(f"Error parsing response: {e}, original content: {content}")
                    return None
            else:
                logger.error(f"API call failed: {response.status_code} - {response.text}")
                return None
                
        except Exception as e:
            logger.error(f"API call exception: {e}")
            return None
    
    def process_samples(self, samples: List[Dict]) -> None:
        """
        Process samples and save results
        """
        logger.info(f"Starting to process {len(samples)} samples...")
        
        for i, sample in enumerate(samples):
            if (i + 1) % 5 == 0 or i == 0 or i == len(samples) - 1:
                logger.info(f"Processing sample {i+1}/{len(samples)}: {sample['post_id']}")
            
            try:
                # Call API to get score and explanation
                api_result = self.call_qwen_api(sample['caption'])
                
                # Save results
                result = {
                    'post_id': sample['post_id'],
                    'caption': sample['caption'][:200] + "..." if len(sample['caption']) > 200 else sample['caption'],
                    'gender_bias_score': api_result['score'] if api_result else None,
                    'explanation': api_result['reason'] if api_result else None,
                    'raw_response': api_result['raw_response'] if api_result else None,
                    'json_file': sample['json_file']
                }
                
                self.results.append(result)
                
                if api_result is not None:
                    logger.info(f"✅ Obtained score: {api_result['score']}")
                    logger.info(f"💭 Explanation: {api_result['reason']}")
                else:
                    logger.warning(f"⚠️ Failed to obtain score")
                
                # Show progress statistics every 5 samples
                if (i + 1) % 5 == 0 or i == len(samples) - 1:
                    success_count = len([r for r in self.results if r['gender_bias_score'] is not None])
                    logger.info(f"📊 Processed {i+1}, successful {success_count}, success rate {success_count/(i+1)*100:.1f}%")
                
                # API rate limiting control
                time.sleep(1.5)  # 1.5 second interval between requests
                
            except Exception as e:
                logger.error(f"Error processing sample (PostID: {sample['post_id']}): {e}")
    
    def save_results(self) -> None:
        """
        Save processing results
        """
        if not self.results:
            logger.warning("No result data to save")
            return
        
        # Save CSV file
        csv_path = os.path.join(self.output_dir, "batch_all_results.csv")
        try:
            with open(csv_path, 'w', newline='', encoding='utf-8') as f:
                fieldnames = self.results[0].keys()
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(self.results)
            
            logger.info(f"Results saved to: {csv_path}")
            
        except Exception as e:
            logger.error(f"Error saving CSV results: {e}")
        
        # Save JSON file
        json_path = os.path.join(self.output_dir, "batch_all_results.json")
        try:
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(self.results, f, indent=2, ensure_ascii=False)
            
            logger.info(f"JSON results saved to: {json_path}")
            
        except Exception as e:
            logger.error(f"Error saving JSON results: {e}")
    
    def generate_summary(self) -> None:
        """
        Generate result summary
        """
        if not self.results:
            logger.warning("No result data available, cannot generate summary")
            return
        
        valid_scores = [r['gender_bias_score'] for r in self.results if r['gender_bias_score'] is not None]
        
        if valid_scores:
            summary = {
                'total_samples': len(self.results),
                'valid_scores_count': len(valid_scores),
                'success_rate': len(valid_scores) / len(self.results) * 100,
                'mean_score': sum(valid_scores) / len(valid_scores),
                'min_score': min(valid_scores),
                'max_score': max(valid_scores),
                'scores': valid_scores
            }
            
            logger.info("=== Processing Results Summary ===")
            logger.info(f"Total samples: {summary['total_samples']}")
            logger.info(f"Successfully obtained scores: {summary['valid_scores_count']}")
            logger.info(f"Success rate: {summary['success_rate']:.1f}%")
            logger.info(f"Average score: {summary['mean_score']:.2f}")
            logger.info(f"Score range: {summary['min_score']:.1f} - {summary['max_score']:.1f}")
            logger.info(f"All scores: {summary['scores']}")
            
            # Save summary
            summary_path = os.path.join(self.output_dir, "summary.json")
            with open(summary_path, 'w', encoding='utf-8') as f:
                json.dump(summary, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Summary saved to: {summary_path}")
        else:
            logger.warning("No valid score data available")
    
    def run(self) -> None:
        """
        Run small batch processing workflow
        """
        logger.info("🚀 Starting to process all available samples...")
        
        # 1. Load sample data
        samples = self.load_sample_data()
        if not samples:
            logger.error("❌ Sample loading failed, program exiting")
            return
        
        # 2. Process samples
        self.process_samples(samples)
        
        # 3. Save results
        self.save_results()
        
        # 4. Generate summary
        self.generate_summary()
        
        logger.info("✅ All samples batch processing completed!")
        logger.info("💡 If results are satisfactory, you can run the complete extract_and_score_samples.py to process 50K samples")


def main():
    """
    Main program entry point
    """
    processor = SmallBatchProcessor()
    processor.run()


if __name__ == "__main__":
    main()

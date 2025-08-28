#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test Version - Use small sample data in current directory to test program functionality
"""

import json
import os
import requests
import base64
import time
from typing import List, Dict, Optional
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class SmallSampleTester:
    def __init__(self):
        """
        Use sample data in current directory for testing
        """
        self.current_dir = "/Users/huangxinyue/Multi model distillation"
        self.api_key = "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"
        self.model_name = "qwen/qwen2.5-vl-32b-instruct:free"
        
    def test_json_loading(self) -> None:
        """
        Test JSON file loading functionality
        """
        logger.info("Testing JSON file loading...")
        
        json_dir = os.path.join(self.current_dir, "json_samples")
        json_files = [f for f in os.listdir(json_dir) if f.endswith('.json')]
        
        logger.info(f"Found {len(json_files)} JSON files")
        
        # Test loading the first JSON file
        if json_files:
            test_file = json_files[0]
            json_path = os.path.join(json_dir, test_file)
            
            try:
                with open(json_path, 'r', encoding='utf-8') as f:
                    post_data = json.load(f)
                
                # Extract caption
                caption = ""
                if 'edge_media_to_caption' in post_data and post_data['edge_media_to_caption']['edges']:
                    caption = post_data['edge_media_to_caption']['edges'][0]['node']['text']
                
                logger.info(f"Test file: {test_file}")
                logger.info(f"Post ID: {post_data.get('id', 'N/A')}")
                logger.info(f"Caption: {caption[:100]}...")
                
                return post_data, caption
                
            except Exception as e:
                logger.error(f"Failed to load JSON file: {e}")
                return None, None
    
    def encode_sample_image(self) -> Optional[str]:
        """
        Create a test image for use (base64 encoded)
        """
        # Create a simple 1x1 pixel white JPEG image in base64 format
        # This is a minimal valid JPEG file
        minimal_jpeg_base64 = "/9j/4AAQSkZJRgABAQEAYABgAAD/2wBDAAEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQH/2wBDAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQH/wAARCAABAAEDASIAAhEBAxEB/8QAFQABAQAAAAAAAAAAAAAAAAAAAAv/xAAUEAEAAAAAAAAAAAAAAAAAAAAA/8QAFQEBAQAAAAAAAAAAAAAAAAAAAAX/xAAUEQEAAAAAAAAAAAAAAAAAAAAA/9oADAMBAAIRAxEAPwDX4P/Z"
        return f"data:image/jpeg;base64,{minimal_jpeg_base64}"
    
    def test_qwen_api_call(self, caption: str, image_url: str = None) -> None:
        """
        Test Qwen API call
        """
        logger.info("Testing Qwen API call...")
        
        try:
            # Build API request - test text-only first
            content = [
                {
                    "type": "text", 
                    "text": f"""Analyze this Instagram post for gender bias. Consider the caption content: "{caption}"

Rate the gender bias on a scale of 0-10:
- 0-2: Strongly male-targeted (very masculine themes, male-dominated content)
- 3-4: Somewhat male-targeted (slightly masculine lean)
- 5: Gender-neutral (no clear gender targeting)
- 6-7: Somewhat female-targeted (slightly feminine lean)
- 8-10: Strongly female-targeted (very feminine themes, female-dominated content)

Requirements:
1. Return only a number between 0 and 10
2. No explanation needed
3. Format: Just output the number, e.g.: 5.2"""
                }
            ]
            
            # If there's an image URL, add image (commented out for text-only testing)
            # if image_url:
            #     content.append({
            #         "type": "image_url",
            #         "image_url": {
            #             "url": image_url
            #         }
            #     })
            
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
                "X-Title": "Multi-Model Gender Bias Analysis Test"
            }
            
            logger.info("Sending API request...")
            response = requests.post(
                url="https://openrouter.ai/api/v1/chat/completions",
                headers=headers,
                json=payload,
                timeout=30
            )
            
            logger.info(f"API response status code: {response.status_code}")
            
            if response.status_code == 200:
                result = response.json()
                content = result['choices'][0]['message']['content'].strip()
                logger.info(f"API returned content: {content}")
                
                # Try to extract score
                try:
                    score = float(content)
                    if 0 <= score <= 10:
                        logger.info(f"✅ Successfully got score: {score}")
                        return score
                    else:
                        logger.warning(f"⚠️ Score out of range: {score}")
                except ValueError:
                    import re
                    numbers = re.findall(r'\d+(?:\.\d+)?', content)
                    if numbers:
                        score = float(numbers[0])
                        if 0 <= score <= 10:
                            logger.info(f"✅ Extracted score from text: {score}")
                            return score
                    logger.warning(f"⚠️ Unable to parse score: {content}")
            else:
                logger.error(f"❌ API call failed: {response.status_code}")
                logger.error(f"Error content: {response.text}")
                
        except Exception as e:
            logger.error(f"❌ API call exception: {e}")
    
    def run_test(self) -> None:
        """
        Run test workflow
        """
        logger.info("🚀 Starting program functionality test...")
        
        # 1. Test JSON loading
        post_data, caption = self.test_json_loading()
        if not caption:
            logger.error("❌ JSON loading test failed, program exiting")
            return
        
        # 2. Prepare test image
        image_url = self.encode_sample_image()
        logger.info(f"Using test image: {image_url}")
        
        # 3. Test API call (text-only first)
        logger.info("Testing text-only API call first...")
        score = self.test_qwen_api_call(caption)
        
        if score is not None:
            logger.info("✅ All tests passed! Program can work normally")
            logger.info("💡 Now you can run the complete extract_and_score_samples_english.py program")
        else:
            logger.error("❌ Test failed, please check API configuration or network connection")


def main():
    """
    Main test entry point
    """
    tester = SmallSampleTester()
    tester.run_test()


if __name__ == "__main__":
    main()

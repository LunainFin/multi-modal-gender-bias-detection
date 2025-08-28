#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试版本 - 使用当前目录的小样本数据测试程序功能
"""

import json
import os
import requests
import base64
import time
from typing import List, Dict, Optional
import logging

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class SmallSampleTester:
    def __init__(self):
        """
        使用当前目录的样本数据进行测试
        """
        self.current_dir = "/Users/huangxinyue/Multi model distillation"
        self.api_key = "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"
        self.model_name = "qwen/qwen2.5-vl-32b-instruct:free"
        
    def test_json_loading(self) -> None:
        """
        测试JSON文件加载功能
        """
        logger.info("测试JSON文件加载...")
        
        json_dir = os.path.join(self.current_dir, "json_samples")
        json_files = [f for f in os.listdir(json_dir) if f.endswith('.json')]
        
        logger.info(f"找到{len(json_files)}个JSON文件")
        
        # 测试加载第一个JSON文件
        if json_files:
            test_file = json_files[0]
            json_path = os.path.join(json_dir, test_file)
            
            try:
                with open(json_path, 'r', encoding='utf-8') as f:
                    post_data = json.load(f)
                
                # 提取caption
                caption = ""
                if 'edge_media_to_caption' in post_data and post_data['edge_media_to_caption']['edges']:
                    caption = post_data['edge_media_to_caption']['edges'][0]['node']['text']
                
                logger.info(f"测试文件: {test_file}")
                logger.info(f"Post ID: {post_data.get('id', 'N/A')}")
                logger.info(f"Caption: {caption[:100]}...")
                
                return post_data, caption
                
            except Exception as e:
                logger.error(f"加载JSON文件失败: {e}")
                return None, None
    
    def encode_sample_image(self) -> Optional[str]:
        """
        创建一个测试用的简单图片（base64编码）
        """
        # 创建一个简单的1x1像素的白色JPEG图片的base64数据
        # 这是一个最小的有效JPEG文件
        minimal_jpeg_base64 = "/9j/4AAQSkZJRgABAQEAYABgAAD/2wBDAAEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQH/2wBDAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQH/wAARCAABAAEDASIAAhEBAxEB/8QAFQABAQAAAAAAAAAAAAAAAAAAAAv/xAAUEAEAAAAAAAAAAAAAAAAAAAAA/8QAFQEBAQAAAAAAAAAAAAAAAAAAAAX/xAAUEQEAAAAAAAAAAAAAAAAAAAAA/9oADAMBAAIRAxEAPwDX4P/Z"
        return f"data:image/jpeg;base64,{minimal_jpeg_base64}"
    
    def test_qwen_api_call(self, caption: str, image_url: str = None) -> None:
        """
        测试Qwen API调用
        """
        logger.info("测试Qwen API调用...")
        
        try:
            # 构建API请求 - 先测试纯文本
            content = [
                {
                    "type": "text", 
                    "text": f"请根据以下Instagram帖子的内容判断其性别倾向程度，性别倾向越强分数越高。\n\n要求：\n1. 只返回一个0到10之间的数字\n2. 不要解释原因\n3. 格式：只输出数字，如：5.2\n\n帖子内容：{caption}"
                }
            ]
            
            # 如果有图片URL，添加图片（暂时注释掉以测试纯文本）
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
            
            logger.info("发送API请求...")
            response = requests.post(
                url="https://openrouter.ai/api/v1/chat/completions",
                headers=headers,
                json=payload,
                timeout=30
            )
            
            logger.info(f"API响应状态码: {response.status_code}")
            
            if response.status_code == 200:
                result = response.json()
                content = result['choices'][0]['message']['content'].strip()
                logger.info(f"API返回内容: {content}")
                
                # 尝试提取分数
                try:
                    score = float(content)
                    if 0 <= score <= 10:
                        logger.info(f"✅ 成功获得分数: {score}")
                        return score
                    else:
                        logger.warning(f"⚠️ 分数超出范围: {score}")
                except ValueError:
                    import re
                    numbers = re.findall(r'\d+(?:\.\d+)?', content)
                    if numbers:
                        score = float(numbers[0])
                        if 0 <= score <= 10:
                            logger.info(f"✅ 从文本中提取到分数: {score}")
                            return score
                    logger.warning(f"⚠️ 无法解析分数: {content}")
            else:
                logger.error(f"❌ API调用失败: {response.status_code}")
                logger.error(f"错误内容: {response.text}")
                
        except Exception as e:
            logger.error(f"❌ API调用异常: {e}")
    
    def run_test(self) -> None:
        """
        运行测试流程
        """
        logger.info("🚀 开始测试程序功能...")
        
        # 1. 测试JSON加载
        post_data, caption = self.test_json_loading()
        if not caption:
            logger.error("❌ JSON加载测试失败，程序退出")
            return
        
        # 2. 准备测试图片
        image_url = self.encode_sample_image()
        logger.info(f"使用测试图片: {image_url}")
        
        # 3. 测试API调用（先只用文本）
        logger.info("先测试纯文本API调用...")
        score = self.test_qwen_api_call(caption)
        
        if score is not None:
            logger.info("✅ 所有测试通过！程序可以正常工作")
            logger.info("💡 现在可以运行完整的extract_and_score_samples.py程序")
        else:
            logger.error("❌ 测试失败，请检查API配置或网络连接")


def main():
    """
    主测试入口
    """
    tester = SmallSampleTester()
    tester.run_test()


if __name__ == "__main__":
    main()

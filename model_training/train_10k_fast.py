#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
训练集处理程序 - 快速并发版本
支持并发API调用，大幅提升处理速度
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

# 配置日志
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
        初始化快速训练数据处理器
        """
        self.database_path = "/Users/huangxinyue/Downloads/Influencer brand database"
        self.output_dir = "/Users/huangxinyue/Multi model distillation/train_10k_results"
        self.batch_size = 10000
        self.api_key = "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"
        self.model_name = "qwen/qwen2.5-vl-32b-instruct"
        
        # 并发控制
        self.max_concurrent = 8  # 最大并发数
        self.semaphore = asyncio.Semaphore(self.max_concurrent)
        
        # 创建输出目录
        os.makedirs(self.output_dir, exist_ok=True)
        
        # 进度文件
        self.progress_file = os.path.join(self.output_dir, "progress_fast.json")
        self.results_file = os.path.join(self.output_dir, "train_10k_fast_results.csv")
        
        # 加载或初始化进度
        self.progress = self.load_progress()
        self.results_lock = Lock()
        
        logger.info("快速训练数据处理器初始化完成")
        logger.info(f"最大并发数: {self.max_concurrent}")
    
    def load_progress(self) -> dict:
        """加载进度信息"""
        if os.path.exists(self.progress_file):
            try:
                with open(self.progress_file, 'r', encoding='utf-8') as f:
                    progress = json.load(f)
                logger.info(f"加载进度: 已处理 {progress['processed_count']} 个样本")
                return progress
            except Exception as e:
                logger.warning(f"加载进度文件失败: {e}")
        
        return {
            'processed_count': 0,
            'success_count': 0,
            'last_processed_line': 0,
            'start_time': time.time()
        }
    
    def save_progress(self):
        """保存进度信息"""
        try:
            with open(self.progress_file, 'w', encoding='utf-8') as f:
                json.dump(self.progress, f, indent=2)
        except Exception as e:
            logger.error(f"保存进度文件失败: {e}")
    
    def extract_post_samples(self) -> List[Dict]:
        """直接从JSON目录中提取前10K个JSON文件"""
        logger.info("开始从JSON目录提取样本信息...")
        
        json_dir = os.path.join(self.database_path, "json")
        samples = []
        
        start_index = self.progress['last_processed_line']
        target_count = self.batch_size
        
        try:
            # 获取所有JSON文件并排序
            all_json_files = [f for f in os.listdir(json_dir) if f.endswith('.json')]
            all_json_files.sort()  # 确保顺序一致
            
            logger.info(f"JSON目录中共有 {len(all_json_files)} 个文件")
            
            # 从断点位置开始抽取
            for i in range(start_index, min(start_index + target_count, len(all_json_files))):
                json_file = all_json_files[i]
                
                samples.append({
                    'json_file': json_file,
                    'file_index': i,  # 文件索引，用于断点续跑
                    'line_number': i  # 兼容原有的line_number字段
                })
                
                if len(samples) >= target_count:
                    break
        
        except FileNotFoundError:
            logger.error(f"找不到JSON目录: {json_dir}")
            return []
        except Exception as e:
            logger.error(f"读取JSON目录失败: {e}")
            return []
        
        logger.info(f"提取了 {len(samples)} 个样本 (从第{start_index}个文件开始)")
        return samples
    
    def find_image_files_for_post(self, post_id: str) -> List[str]:
        """在post_info.txt中查找指定post_id对应的图片文件列表"""
        try:
            post_info_path = os.path.join(self.database_path, "post_info.txt")
            with open(post_info_path, 'r', encoding='utf-8') as f:
                for line in f:
                    try:
                        parts = line.strip().split('\t')
                        if len(parts) >= 5:
                            # 第4列是JSON文件名，检查是否匹配
                            json_filename = parts[3]
                            if json_filename == f"{post_id}.json":
                                # 第5列是图片文件列表
                                image_files = eval(parts[4])
                                return image_files
                    except Exception:
                        continue
            return []
        except Exception as e:
            logger.warning(f"搜索图片文件失败 (post_id: {post_id}): {e}")
            return []

    def load_post_data(self, sample: Dict) -> Optional[Dict]:
        """加载单个帖子数据"""
        try:
            # 从JSON文件名提取post_id
            json_filename = sample['json_file']
            post_id = json_filename.replace('.json', '')
            
            # 构建JSON文件路径
            json_path = os.path.join(self.database_path, "json", json_filename)
            
            if not os.path.exists(json_path):
                logger.warning(f"JSON文件不存在: {json_path}")
                return None
            
            # 读取JSON数据
            with open(json_path, 'r', encoding='utf-8') as f:
                post_data = json.load(f)
            
            # 验证JSON中的ID是否匹配文件名
            json_post_id = post_data.get('id')
            if json_post_id and json_post_id != post_id:
                logger.warning(f"JSON文件名与内部ID不匹配: {json_filename} vs {json_post_id}")
            
            # 使用文件名作为post_id（更可靠）
            final_post_id = post_id
            
            # 提取caption
            caption = ""
            if 'edge_media_to_caption' in post_data and post_data['edge_media_to_caption']['edges']:
                caption = post_data['edge_media_to_caption']['edges'][0]['node']['text']
            
            # 根据post_id查找对应的图片文件
            image_files = self.find_image_files_for_post(final_post_id)
            image_paths = []
            
            for img_file in image_files:
                # 在img_resized_1到img_resized_16中查找
                for i in range(1, 17):
                    img_dir = f"img_resized_{i}"
                    potential_path = os.path.join(self.database_path, img_dir, img_file)
                    if os.path.exists(potential_path):
                        image_paths.append(potential_path)
                        break
            
            return {
                'post_id': final_post_id,  # 使用文件名作为post_id
                'caption': caption,
                'image_paths': image_paths,  # 返回找到的图片路径
                'line_number': sample['line_number']
            }
            
        except Exception as e:
            logger.warning(f"加载帖子数据失败 (JSON: {sample['json_file']}): {e}")
            return None
    
    def encode_images_to_base64(self, image_paths: List[str]) -> List[str]:
        """
        将多张图片编码为base64格式
        """
        encoded_images = []
        for image_path in image_paths[:3]:  # 最多处理3张图片，避免API负担过重
            try:
                if os.path.exists(image_path):
                    with open(image_path, "rb") as image_file:
                        encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
                        encoded_images.append(f"data:image/jpeg;base64,{encoded_string}")
            except Exception as e:
                logger.warning(f"图片编码失败 {image_path}: {e}")
                continue
        return encoded_images

    async def call_qwen_api_async(self, session, post_data: Dict) -> Optional[float]:
        """异步调用Qwen API"""
        async with self.semaphore:
            try:
                # 构建消息内容
                content = [
                    {
                        "type": "text",
                        "text": f"判断Instagram帖子女性倾向程度。严格要求：只能回复一个0-10的数字，如5.2或7.0，不允许任何解释或其他内容。\n\n帖子：{post_data['caption'][:700]}"  # 限制长度
                    }
                ]
                
                # 添加所有图片
                if post_data['image_paths']:
                    encoded_images = self.encode_images_to_base64(post_data['image_paths'])
                    for encoded_image in encoded_images:
                        content.append({
                            "type": "image_url",
                            "image_url": {
                                "url": encoded_image
                            }
                        })
                
                # 构建API请求
                payload = {
                    "model": self.model_name,
                    "messages": [
                        {
                            "role": "user",
                            "content": content
                        }
                    ],
                    "max_tokens": 5,
                    "temperature": 0.0
                }
                
                headers = {
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                    "HTTP-Referer": "https://github.com/multi-model-training",
                    "X-Title": "Instagram Gender Bias Training Data"
                }
                
                # 发送API请求
                async with session.post(
                    "https://openrouter.ai/api/v1/chat/completions",
                    headers=headers,
                    json=payload,
                    timeout=30
                ) as response:
                    
                    if response.status == 200:
                        result = await response.json()
                        content = result['choices'][0]['message']['content'].strip()
                        
                        # 简化解析分数
                        try:
                            import re
                            # 提取所有数字
                            numbers = re.findall(r'\d+(?:\.\d+)?', content)
                            if numbers:
                                score = float(numbers[0])
                                if 0 <= score <= 10:
                                    return score
                            
                            # 解析失败，返回NaN
                            return float('nan')
                            
                        except (ValueError, TypeError):
                            return float('nan')
                    else:
                        logger.error(f"API调用失败: {response.status}")
                        return None
                        
            except Exception as e:
                logger.error(f"API调用异常: {e}")
                return None
    
    def init_csv_file(self):
        """初始化CSV文件"""
        if not os.path.exists(self.results_file):
            try:
                with open(self.results_file, 'w', newline='', encoding='utf-8') as f:
                    fieldnames = ['post_id', 'gender_bias_score', 'image_count']
                    writer = csv.DictWriter(f, fieldnames=fieldnames)
                    writer.writeheader()
                logger.info(f"CSV文件已初始化: {self.results_file}")
            except Exception as e:
                logger.error(f"初始化CSV文件失败: {e}")
    
    def save_result(self, result: Dict):
        """线程安全地保存结果"""
        with self.results_lock:
            try:
                with open(self.results_file, 'a', newline='', encoding='utf-8') as f:
                    fieldnames = ['post_id', 'gender_bias_score', 'image_count']
                    writer = csv.DictWriter(f, fieldnames=fieldnames)
                    writer.writerow(result)
            except Exception as e:
                logger.error(f"保存结果失败: {e}")
    
    async def process_single_sample(self, session, sample: Dict, pbar) -> Dict:
        """处理单个样本"""
        try:
            # 加载帖子数据
            post_data = self.load_post_data(sample)
            if post_data is None:
                result = {
                    'post_id': f"unknown_{sample['file_index']}",  # 使用文件索引作为备用ID
                    'gender_bias_score': None,
                    'image_count': 0
                }
                self.save_result(result)
                pbar.update(1)
                return result
            
            # 调用API获取分数
            score = await self.call_qwen_api_async(session, post_data)
            
            # 准备结果
            result = {
                'post_id': post_data['post_id'],
                'gender_bias_score': score,
                'image_count': len(post_data['image_paths'])
            }
            
            # 保存结果
            self.save_result(result)
            
            # 更新统计
            with self.results_lock:
                self.progress['processed_count'] += 1
                self.progress['last_processed_line'] = sample['file_index'] + 1
                
                if score is not None:
                    self.progress['success_count'] += 1
            
            pbar.update(1)
            return result
            
        except Exception as e:
            logger.error(f"处理样本时出错 (JSON: {sample.get('json_file', 'unknown')}): {e}")
            pbar.update(1)
            return None
    
    async def process_samples_async(self, samples: List[Dict]):
        """异步批量处理样本"""
        logger.info(f"开始并发处理 {len(samples)} 个样本...")
        
        # 初始化CSV文件
        self.init_csv_file()
        
        # 创建进度条
        with tqdm(total=len(samples), desc="并发处理") as pbar:
            async with aiohttp.ClientSession() as session:
                # 创建任务列表
                tasks = []
                for sample in samples:
                    task = self.process_single_sample(session, sample, pbar)
                    tasks.append(task)
                
                # 并发执行所有任务
                results = await asyncio.gather(*tasks, return_exceptions=True)
                
                # 定期保存进度
                while not pbar.finished:
                    await asyncio.sleep(10)  # 每10秒保存一次
                    self.save_progress()
        
        # 最终保存进度
        self.save_progress()
        
        success_rate = self.progress['success_count'] / self.progress['processed_count'] * 100
        logger.info(f"并发处理完成！总计处理 {self.progress['processed_count']} 个样本，成功率 {success_rate:.1f}%")
    
    def call_qwen_api_sync(self, post_data: Dict) -> Optional[float]:
        """同步调用Qwen API"""
        try:
            # 构建消息内容
            content = [
                {
                    "type": "text",
                    "text": f"判断Instagram帖子女性倾向程度。严格要求：只能回复一个0-10的数字，如5.2或7.0，不允许任何解释或其他内容。\n\n帖子：{post_data['caption'][:300]}"  # 限制长度
                }
            ]
            
            # 添加所有图片
            if post_data['image_paths']:
                encoded_images = self.encode_images_to_base64(post_data['image_paths'])
                for encoded_image in encoded_images:
                    content.append({
                        "type": "image_url",
                        "image_url": {
                            "url": encoded_image
                        }
                    })
            
            # 构建API请求
            payload = {
                "model": self.model_name,
                "messages": [
                    {
                        "role": "user",
                        "content": content
                    }
                ],
                "max_tokens": 4,
                "temperature": 0.0
            }
            
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
                "HTTP-Referer": "https://github.com/multi-model-training",
                "X-Title": "Instagram Gender Bias Training Data"
            }
            
            # 发送API请求
            response = requests.post(
                "https://openrouter.ai/api/v1/chat/completions",
                headers=headers,
                json=payload,
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                content = result['choices'][0]['message']['content'].strip()
                
                # 简化解析分数
                try:
                    import re
                    # 提取所有数字
                    numbers = re.findall(r'\d+(?:\.\d+)?', content)
                    if numbers:
                        score = float(numbers[0])
                        if 0 <= score <= 10:
                            return score
                    
                    # 解析失败，返回NaN
                    return float('nan')
                    
                except (ValueError, TypeError):
                    return float('nan')
            else:
                logger.error(f"API调用失败: {response.status_code}")
                return None
                
        except Exception as e:
            logger.error(f"API调用异常: {e}")
            return None
    
    def process_samples_sync(self, samples: List[Dict]):
        """同步处理样本（用于已有事件循环的环境）"""
        logger.info(f"开始同步处理 {len(samples)} 个样本...")
        
        # 初始化CSV文件
        self.init_csv_file()
        
        # 创建进度条
        with tqdm(total=len(samples), desc="处理样本") as pbar:
            for i, sample in enumerate(samples):
                try:
                    # 加载帖子数据
                    post_data = self.load_post_data(sample)
                    if post_data is None:
                        logger.warning(f"跳过样本: {sample['json_file']}")
                        self.progress['processed_count'] += 1
                        self.progress['last_processed_line'] = sample['file_index'] + 1
                        pbar.update(1)
                        continue
                    
                    # 调用API获取分数
                    score = self.call_qwen_api_sync(post_data)
                    
                    # 准备结果
                    result = {
                        'post_id': post_data['post_id'],
                        'gender_bias_score': score,
                        'image_count': len(post_data['image_paths'])
                    }
                    
                    # 保存结果
                    self.save_result(result)
                    
                    # 更新统计
                    self.progress['processed_count'] += 1
                    self.progress['last_processed_line'] = sample['file_index'] + 1
                    
                    if score is not None:
                        self.progress['success_count'] += 1
                    
                    # 每100个样本保存一次进度
                    if (i + 1) % 100 == 0:
                        self.save_progress()
                        success_rate = self.progress['success_count'] / self.progress['processed_count'] * 100
                        logger.info(f"已处理 {self.progress['processed_count']} 个样本，成功率 {success_rate:.1f}%")
                    
                    # API限流控制 - 快速处理
                    time.sleep(0.5)  # 每次请求间隔0.5秒
                    
                    pbar.update(1)
                    
                except KeyboardInterrupt:
                    logger.info("用户中断，正在保存进度...")
                    self.save_progress()
                    raise
                except Exception as e:
                    logger.error(f"处理样本时出错 (JSON: {sample.get('json_file', 'unknown')}): {e}")
                    self.progress['processed_count'] += 1
                    self.progress['last_processed_line'] = sample['file_index'] + 1
                    pbar.update(1)
                    continue
        
        # 最终保存进度
        self.save_progress()
        
        success_rate = self.progress['success_count'] / self.progress['processed_count'] * 100
        logger.info(f"同步处理完成！总计处理 {self.progress['processed_count']} 个样本，成功率 {success_rate:.1f}%")
    
    def run(self):
        """运行主程序"""
        logger.info("开始快速训练数据集处理...")
        
        try:
            # 1. 提取样本
            samples = self.extract_post_samples()
            if not samples:
                logger.error("没有提取到样本，程序退出")
                return
            
            # 2. 检测是否在事件循环中
            try:
                loop = asyncio.get_running_loop()
                logger.info("检测到运行中的事件循环，使用同步处理模式")
                self.process_samples_sync(samples)
            except RuntimeError:
                # 没有运行的事件循环，可以使用 asyncio.run
                asyncio.run(self.process_samples_async(samples))
            
            logger.info("快速训练数据集处理完成！")
            
        except KeyboardInterrupt:
            logger.info("程序被用户中断")
        except Exception as e:
            logger.error(f"程序运行出错: {e}")
        finally:
            self.save_progress()

def main():
    """主程序入口"""
    processor = FastTrainDataProcessor()
    processor.run()

if __name__ == "__main__":
    main()


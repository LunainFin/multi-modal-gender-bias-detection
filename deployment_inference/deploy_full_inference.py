#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
阶段4: 全量数据推理部署 (修复版)
使用训练好的5K模型处理160万Instagram帖子
"""

import torch
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
import ast
import gc

# 导入模型
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from train_fast_local import LightweightGenderBiasModel

# 设置环境变量避免冲突
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# 配置日志
def setup_logging():
    """设置日志系统"""
    log_format = '%(asctime)s - %(levelname)s - %(message)s'
    logging.basicConfig(
        level=logging.INFO,
        format=log_format,
        handlers=[
            logging.FileHandler('full_inference_fixed.log'),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

logger = setup_logging()

class FixedInferenceDeployer:
    def __init__(self):
        self.model_path = '/Users/huangxinyue/Multi model distillation/fast_models_5k/fast_best_model.pth'
        self.database_path = '/Users/huangxinyue/Downloads/Influencer brand database'
        self.json_dir = os.path.join(self.database_path, 'json')
        self.post_info_file = os.path.join(self.database_path, 'post_info.txt')
        self.output_dir = '/Users/huangxinyue/Multi model distillation/full_inference_results'
        self.progress_file = os.path.join(self.output_dir, 'inference_progress_fixed.json')
        
        # 推理配置
        self.batch_size = 32  # 适中批次
        self.save_interval = 1000  # 每1000个样本保存一次
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 模型和数据
        self.model = None
        self.tokenizer = None
        self.transform = None
        self.json_to_images = {}  # JSON文件名 -> 图片ID列表
        
        # 确保输出目录存在
        os.makedirs(self.output_dir, exist_ok=True)
        
    def load_model_and_tokenizer(self):
        """加载模型和tokenizer"""
        logger.info("🚀 加载5K最佳模型...")
        
        # 加载模型
        self.model = LightweightGenderBiasModel()
        checkpoint = torch.load(self.model_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.device)
        self.model.eval()
        
        # 加载tokenizer
        from transformers import DistilBertTokenizer
        self.tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
        
        # 图像预处理
        from torchvision import transforms
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
        
        logger.info("✅ 模型加载完成")
        
    def load_post_info_mapping(self):
        """加载post_info.txt映射（修复版）"""
        logger.info("📚 加载post_info映射（新格式）...")
        
        try:
            with open(self.post_info_file, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f):
                    try:
                        parts = line.strip().split('\t')  # 尝试tab分隔
                        if len(parts) < 5:
                            parts = line.strip().split()  # 空格分隔
                        
                        if len(parts) >= 5:
                            # 格式: 索引号 用户名 类型 JSON文件名 图片ID列表
                            json_filename = parts[3]
                            image_list_str = parts[4]
                            
                            # 解析图片ID列表
                            try:
                                # 尝试解析Python列表格式
                                image_ids = ast.literal_eval(image_list_str)
                                if isinstance(image_ids, list):
                                    # 去掉.jpg后缀，保留ID
                                    image_ids = [img.replace('.jpg', '') for img in image_ids]
                                    self.json_to_images[json_filename] = image_ids
                            except:
                                # 如果解析失败，尝试其他格式
                                continue
                                
                    except Exception as e:
                        continue
                    
                    # 每10万行显示进度
                    if (line_num + 1) % 100000 == 0:
                        logger.info(f"  已处理 {line_num + 1:,} 行，找到 {len(self.json_to_images):,} 个映射")
            
            logger.info(f"✅ 成功加载 {len(self.json_to_images):,} 个JSON->图片映射")
            
            # 显示几个示例
            sample_items = list(self.json_to_images.items())[:3]
            for json_file, image_ids in sample_items:
                logger.info(f"  示例: {json_file} -> {len(image_ids)} 张图片")
            
        except Exception as e:
            logger.error(f"❌ 加载post_info失败: {e}")
            
    def get_all_json_files(self):
        """获取所有JSON文件列表"""
        logger.info("📂 扫描JSON文件...")
        
        json_files = glob.glob(os.path.join(self.json_dir, '*.json'))
        json_files.sort()  # 保证顺序一致
        
        logger.info(f"📊 发现 {len(json_files):,} 个JSON文件")
        return json_files
        
    def load_progress(self):
        """加载推理进度"""
        if os.path.exists(self.progress_file):
            try:
                with open(self.progress_file, 'r') as f:
                    progress = json.load(f)
                logger.info(f"📈 继续推理，已处理 {progress.get('processed', 0):,} 个样本")
                return progress
            except:
                logger.warning("⚠️ 无法读取进度文件，从头开始")
                
        return {'processed': 0, 'results': []}
        
    def save_progress(self, progress):
        """保存推理进度"""
        try:
            with open(self.progress_file, 'w') as f:
                json.dump(progress, f)
        except Exception as e:
            logger.error(f"保存进度失败: {e}")
            
    def process_single_post(self, json_file_path):
        """处理单个帖子（修复版）"""
        try:
            # 读取JSON文件
            with open(json_file_path, 'r', encoding='utf-8') as f:
                post_data = json.load(f)
            
            post_id = post_data.get('id', os.path.basename(json_file_path).replace('.json', ''))
            caption = post_data.get('edge_media_to_caption', {}).get('edges', [])
            
            # 提取文本
            if caption and len(caption) > 0:
                text = caption[0].get('node', {}).get('text', '')
            else:
                text = ''
            
            # 获取图片IDs（使用JSON文件名查找）
            json_filename = os.path.basename(json_file_path)
            image_ids = self.json_to_images.get(json_filename, [])
            
            if not image_ids:
                return {
                    'post_id': post_id,
                    'gender_bias_score': 5.0,  # 中性分数
                    'image_count': 0,
                    'status': 'no_images'
                }
            
            # 加载图片 (最多3张)
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
            
            # 准备输入
            # 图片：取第一张或拼接多张
            if len(images) == 1:
                image_input = images[0].unsqueeze(0)
            else:
                # 多图片：取平均
                image_input = torch.stack(images).mean(dim=0).unsqueeze(0)
            
            # 文本tokenization
            encoding = self.tokenizer(
                text,
                truncation=True,
                padding='max_length',
                max_length=64,
                return_tensors='pt'
            )
            
            # 移到设备
            image_input = image_input.to(self.device)
            input_ids = encoding['input_ids'].to(self.device)
            attention_mask = encoding['attention_mask'].to(self.device)
            
            # 推理
            with torch.no_grad():
                output = self.model(image_input, input_ids, attention_mask)
                score = output.item() * 10.0  # 反归一化到0-10
                score = max(0.0, min(10.0, score))  # 限制范围
            
            return {
                'post_id': post_id,
                'gender_bias_score': round(score, 2),
                'image_count': len(images),
                'status': 'success'
            }
            
        except Exception as e:
            logger.warning(f"处理帖子失败 {json_file_path}: {e}")
            return {
                'post_id': os.path.basename(json_file_path).replace('.json', ''),
                'gender_bias_score': 5.0,
                'image_count': 0,
                'status': 'error'
            }
    
    def find_image_path(self, image_id):
        """查找图片文件路径"""
        # 在各个img_resized目录中查找
        for i in range(1, 17):
            dir_path = os.path.join(self.database_path, f'img_resized_{i}')
            image_path = os.path.join(dir_path, f'{image_id}.jpg')
            if os.path.exists(image_path):
                return image_path
        return None
        
    def run_batch_inference(self, json_files, start_idx=0):
        """运行批量推理"""
        logger.info(f"🚀 开始批量推理（修复版），从第 {start_idx:,} 个文件开始")
        
        # 加载进度
        progress = self.load_progress()
        processed_count = progress['processed']
        results = progress['results']
        
        # 从指定位置开始
        remaining_files = json_files[start_idx:]
        total_files = len(json_files)
        
        logger.info(f"📊 总文件数: {total_files:,}")
        logger.info(f"📈 已处理: {processed_count:,}")
        logger.info(f"📋 剩余: {len(remaining_files):,}")
        
        # 创建进度条
        pbar = tqdm(remaining_files, desc="修复推理", initial=processed_count, total=total_files)
        
        batch_results = []
        batch_start_time = time.time()
        success_count = 0
        
        for file_idx, json_file in enumerate(remaining_files):
            result = self.process_single_post(json_file)
            batch_results.append(result)
            processed_count += 1
            
            if result['status'] == 'success':
                success_count += 1
            
            # 更新进度条
            pbar.update(1)
            pbar.set_postfix({
                'Score': f"{result['gender_bias_score']:.1f}",
                'Images': result['image_count'],
                'Success': f"{success_count}/{len(batch_results)}"
            })
            
            # 定期保存
            if len(batch_results) >= self.save_interval:
                results.extend(batch_results)
                
                # 保存进度
                progress = {
                    'processed': processed_count,
                    'results': results
                }
                self.save_progress(progress)
                
                # 保存CSV
                self.save_batch_results(results)
                
                # 清理内存
                batch_results = []
                gc.collect()
                
                # 显示统计信息
                batch_time = time.time() - batch_start_time
                speed = self.save_interval / batch_time
                remaining = total_files - processed_count
                eta_hours = remaining / speed / 3600
                
                logger.info(f"💾 已保存 {processed_count:,} 个结果")
                logger.info(f"⚡ 处理速度: {speed:.1f} 帖子/秒")
                logger.info(f"✅ 成功率: {success_count}/{self.save_interval} ({success_count/self.save_interval*100:.1f}%)")
                logger.info(f"⏰ 预计剩余时间: {eta_hours:.1f} 小时")
                
                batch_start_time = time.time()
                success_count = 0
        
        # 保存最后一批
        if batch_results:
            results.extend(batch_results)
            progress = {
                'processed': processed_count,
                'results': results
            }
            self.save_progress(progress)
            self.save_batch_results(results)
        
        pbar.close()
        logger.info("🎉 批量推理完成！")
        
        return results
        
    def save_batch_results(self, results):
        """保存批次结果到CSV"""
        if not results:
            return
            
        df = pd.DataFrame(results)
        output_file = os.path.join(self.output_dir, 'full_inference_results_fixed.csv')
        
        df.to_csv(output_file, index=False)
        logger.info(f"💾 结果已保存: {output_file}")
        
        # 显示快速统计
        success_count = (df['status'] == 'success').sum()
        if success_count > 0:
            avg_score = df[df['status'] == 'success']['gender_bias_score'].mean()
            logger.info(f"📊 成功处理: {success_count:,} / {len(df):,} ({success_count/len(df)*100:.1f}%)")
            logger.info(f"📊 平均分数: {avg_score:.2f}")
        
    def deploy(self):
        """主部署函数"""
        logger.info("🚀 开始阶段4: 全量数据推理部署（修复版）")
        
        start_time = time.time()
        
        # 1. 加载模型
        self.load_model_and_tokenizer()
        
        # 2. 加载数据映射（修复版）
        self.load_post_info_mapping()
        
        # 3. 获取所有JSON文件
        json_files = self.get_all_json_files()
        
        # 4. 检查进度
        progress = self.load_progress()
        start_idx = progress['processed']
        
        # 5. 运行推理
        results = self.run_batch_inference(json_files, start_idx)
        
        # 6. 最终统计
        total_time = time.time() - start_time
        total_hours = total_time / 3600
        
        logger.info("🎉 全量推理部署完成！")
        logger.info(f"⏱️ 总耗时: {total_hours:.2f} 小时")
        logger.info(f"📊 处理样本: {len(results):,}")
        logger.info(f"📁 结果文件: {self.output_dir}/full_inference_results_fixed.csv")

def main():
    """主函数"""
    logger.info("🎯 Instagram性别倾向预测 - 全量部署（修复版）")
    
    deployer = FixedInferenceDeployer()
    deployer.deploy()

if __name__ == "__main__":
    main()






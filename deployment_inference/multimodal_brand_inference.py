#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
多模态品牌推理系统
对筛选出的品牌帖子进行真正的多模态推理（图片+文本）
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

# 导入模型
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from train_fast_local import LightweightGenderBiasModel

# 设置环境变量避免冲突
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# 检测是否在后台运行
def is_running_in_background():
    """检测是否在后台运行"""
    try:
        return not sys.stdout.isatty() or not sys.stdin.isatty()
    except:
        return True

DISABLE_TQDM = is_running_in_background()

# 设置日志
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
        """初始化多模态品牌推理系统"""
        self.brand_results_file = '/Users/huangxinyue/Multi model distillation/brand_analysis_results/brand_analysis_final.csv'
        self.database_path = '/Users/huangxinyue/Downloads/Influencer brand database'
        self.post_info_file = os.path.join(self.database_path, 'post_info.txt')
        self.json_dir = os.path.join(self.database_path, 'json')
        self.model_path = '/Users/huangxinyue/Multi model distillation/fast_models_5k/fast_best_model.pth'
        
        # 输出和进度文件
        self.output_dir = '/Users/huangxinyue/Multi model distillation/multimodal_results'
        self.progress_file = os.path.join(self.output_dir, 'multimodal_progress.json')
        self.final_results_file = os.path.join(self.output_dir, 'multimodal_brand_analysis.csv')
        
        # 确保输出目录存在
        os.makedirs(self.output_dir, exist_ok=True)
        
        # 模型相关
        self.model = None
        self.tokenizer = None
        self.transform = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 数据映射
        self.json_to_images = {}  # JSON文件名 -> 图片ID列表
        
        logger.info(f"🚀 多模态品牌推理系统初始化完成")
        logger.info(f"📊 使用设备: {self.device}")
    
    def load_progress(self):
        """加载进度"""
        if os.path.exists(self.progress_file):
            try:
                with open(self.progress_file, 'r') as f:
                    progress = json.load(f)
                logger.info(f"📈 继续推理，已处理 {progress.get('processed', 0)} 个帖子")
                return progress
            except Exception as e:
                logger.warning(f"读取进度失败，从头开始: {e}")
        
        return {'processed': 0, 'results': []}
    
    def save_progress(self, progress):
        """保存进度"""
        try:
            progress['timestamp'] = datetime.now().isoformat()
            with open(self.progress_file, 'w') as f:
                json.dump(progress, f, indent=2)
        except Exception as e:
            logger.error(f"保存进度失败: {e}")
    
    def load_post_info_mapping(self):
        """加载post_info.txt中JSON->图片的映射"""
        logger.info("📚 加载post_info映射...")
        
        try:
            with open(self.post_info_file, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f):
                    try:
                        parts = line.strip().split('\t')
                        if len(parts) >= 5:
                            # 格式: 索引号 用户名 类型 JSON文件名 图片ID列表
                            json_filename = parts[3]
                            image_list_str = parts[4]
                            
                            # 解析图片ID列表
                            try:
                                image_ids = ast.literal_eval(image_list_str)
                                if isinstance(image_ids, list):
                                    # 去掉.jpg后缀，保留ID
                                    image_ids = [img.replace('.jpg', '') for img in image_ids]
                                    self.json_to_images[json_filename] = image_ids
                            except:
                                continue
                                
                    except Exception as e:
                        continue
                    
                    # 每50万行显示进度
                    if (line_num + 1) % 500000 == 0:
                        logger.info(f"  已处理 {line_num + 1:,} 行，找到 {len(self.json_to_images):,} 个映射")
            
            logger.info(f"✅ 成功加载 {len(self.json_to_images):,} 个JSON->图片映射")
            
        except Exception as e:
            logger.error(f"❌ 加载post_info失败: {e}")
            raise
    
    def load_inference_model(self):
        """加载推理模型"""
        logger.info("🚀 加载5K最佳多模态模型...")
        
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
        
        logger.info("✅ 多模态模型加载完成")
    
    def find_image_paths(self, image_ids):
        """查找图片文件路径"""
        image_paths = []
        for image_id in image_ids:
            # 在各个img_resized目录中查找
            for i in range(1, 17):
                dir_path = os.path.join(self.database_path, f'img_resized_{i}')
                image_path = os.path.join(dir_path, f'{image_id}.jpg')
                if os.path.exists(image_path):
                    image_paths.append(image_path)
                    break
        return image_paths
    
    def load_images(self, image_paths):
        """加载和预处理图片"""
        images = []
        for image_path in image_paths[:3]:  # 最多处理3张图片
            try:
                image = Image.open(image_path).convert('RGB')
                image_tensor = self.transform(image)
                images.append(image_tensor)
            except Exception as e:
                logger.warning(f"图片加载失败 {image_path}: {e}")
                continue
        
        if not images:
            # 如果没有图片，返回黑色图片
            return torch.zeros(1, 3, 224, 224)
        elif len(images) == 1:
            return images[0].unsqueeze(0)
        else:
            # 多张图片取平均
            return torch.stack(images).mean(dim=0).unsqueeze(0)
    
    def multimodal_inference(self, json_id, caption, image_paths):
        """多模态推理（图片+文本）"""
        try:
            # 1. 处理图片
            image_tensor = self.load_images(image_paths)
            
            # 2. 处理文本
            encoding = self.tokenizer(
                caption,
                truncation=True,
                padding='max_length',
                max_length=64,
                return_tensors='pt'
            )
            
            # 3. 移到设备
            image_tensor = image_tensor.to(self.device)
            input_ids = encoding['input_ids'].to(self.device)
            attention_mask = encoding['attention_mask'].to(self.device)
            
            # 4. 多模态推理
            with torch.no_grad():
                output = self.model(image_tensor, input_ids, attention_mask)
                score = output.item() * 10.0  # 反归一化到0-10
                score = max(0.0, min(10.0, score))  # 限制范围
            
            return round(score, 2), len(image_paths)
            
        except Exception as e:
            logger.warning(f"多模态推理失败 {json_id}: {e}")
            return 5.0, 0  # 默认中性分数
    
    def process_brand_posts(self):
        """处理品牌帖子进行多模态推理"""
        logger.info("🔍 读取筛选出的品牌帖子...")
        
        # 读取第一阶段的结果
        if not os.path.exists(self.brand_results_file):
            logger.error(f"❌ 品牌筛选结果文件不存在: {self.brand_results_file}")
            return
        
        brand_df = pd.read_csv(self.brand_results_file)
        logger.info(f"✅ 读取到 {len(brand_df)} 个品牌帖子")
        
        # 加载进度
        progress = self.load_progress()
        processed_count = progress['processed']
        results = progress.get('results', [])
        
        # 从断点开始处理
        posts_to_process = brand_df.iloc[processed_count:]
        logger.info(f"📈 从第 {processed_count} 个帖子开始多模态推理")
        logger.info(f"📋 剩余处理: {len(posts_to_process)} 个帖子")
        
        save_interval = 100  # 每100个帖子保存一次
        batch_start_time = datetime.now()
        
        with tqdm(posts_to_process.iterrows(), 
                  desc="多模态推理", 
                  total=len(posts_to_process),
                  disable=DISABLE_TQDM) as pbar:
            
            for row_idx, (_, row) in enumerate(pbar):
                try:
                    json_id = str(row['json_id'])
                    influencer_name = row['influencer_name']
                    sponsored = row['sponsored']
                    brand = row['brand']
                    
                    # 1. 从JSON文件获取caption
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
                            logger.warning(f"读取JSON失败 {json_id}: {e}")
                    
                    # 2. 从post_info.txt获取图片ID
                    json_filename = f'{json_id}.json'
                    image_ids = self.json_to_images.get(json_filename, [])
                    
                    # 3. 查找实际图片路径
                    image_paths = self.find_image_paths(image_ids)
                    
                    # 4. 多模态推理
                    gender_score, image_count = self.multimodal_inference(json_id, caption, image_paths)
                    
                    # 5. 保存结果
                    result = {
                        'json_id': json_id,
                        'influencer_name': influencer_name,
                        'sponsored': sponsored,
                        'brand': brand,
                        'gender_bias_score': gender_score,
                        'image_count': image_count,
                        'caption_length': len(caption) if caption else 0
                    }
                    
                    results.append(result)
                    processed_count += 1
                    
                    # 更新进度条
                    if not DISABLE_TQDM:
                        pbar.set_postfix({
                            'Brand': brand[:8],
                            'Score': f"{gender_score:.1f}",
                            'Images': image_count,
                            'Sponsored': 'Y' if sponsored else 'N'
                        })
                    
                    # 后台模式的进度日志
                    if DISABLE_TQDM and (row_idx + 1) % 50 == 0:
                        logger.info(f"  已处理 {processed_count}/{len(brand_df)} 个帖子 "
                                  f"({processed_count/len(brand_df)*100:.1f}%) - "
                                  f"最新: {brand} {gender_score:.1f}分")
                    
                    # 定期保存
                    if len(results) % save_interval == 0:
                        # 保存进度
                        progress_data = {
                            'processed': processed_count,
                            'results': results
                        }
                        self.save_progress(progress_data)
                        
                        # 保存CSV
                        self.save_results(results)
                        
                        # 显示批次统计
                        batch_time = (datetime.now() - batch_start_time).total_seconds()
                        speed = save_interval / batch_time
                        remaining = len(brand_df) - processed_count
                        eta_minutes = remaining / speed / 60
                        
                        logger.info(f"💾 已保存 {processed_count} 个结果")
                        logger.info(f"⚡ 处理速度: {speed:.1f} 帖子/秒")
                        logger.info(f"⏰ 预计剩余: {eta_minutes:.1f} 分钟")
                        
                        batch_start_time = datetime.now()
                        
                except Exception as e:
                    logger.error(f"处理帖子失败 {row.get('json_id', 'unknown')}: {e}")
                    processed_count += 1
                    continue
        
        # 保存最终结果
        progress_data = {
            'processed': processed_count,
            'results': results
        }
        self.save_progress(progress_data)
        self.save_results(results)
        
        logger.info(f"🎉 多模态推理完成！处理了 {len(results)} 个帖子")
        return results
    
    def save_results(self, results):
        """保存结果到CSV"""
        if not results:
            return
        
        df = pd.DataFrame(results)
        df.to_csv(self.final_results_file, index=False)
        
        # 显示统计信息
        total = len(df)
        success_with_images = (df['image_count'] > 0).sum()
        avg_score = df['gender_bias_score'].mean()
        
        logger.info(f"💾 结果已保存: {self.final_results_file}")
        logger.info(f"📊 有图片的帖子: {success_with_images}/{total} ({success_with_images/total*100:.1f}%)")
        logger.info(f"📊 平均性别倾向分数: {avg_score:.2f}")
    
    def show_final_statistics(self, results):
        """显示最终统计信息"""
        df = pd.DataFrame(results)
        
        logger.info("📊 多模态推理最终统计:")
        logger.info("=" * 60)
        logger.info(f"总帖子数: {len(df):,}")
        logger.info(f"涉及品牌: {df['brand'].nunique()} 个")
        logger.info(f"涉及博主: {df['influencer_name'].nunique():,} 个")
        logger.info(f"赞助帖子: {df['sponsored'].sum():,} ({df['sponsored'].mean()*100:.1f}%)")
        logger.info(f"有图片帖子: {(df['image_count'] > 0).sum():,} ({(df['image_count'] > 0).mean()*100:.1f}%)")
        logger.info(f"平均图片数: {df['image_count'].mean():.2f}")
        logger.info(f"平均性别倾向分数: {df['gender_bias_score'].mean():.2f} ± {df['gender_bias_score'].std():.2f}")
        
        logger.info(f"\n🏆 各品牌多模态分析结果 (Top 10):")
        brand_stats = df.groupby('brand').agg({
            'gender_bias_score': ['count', 'mean', 'std'],
            'sponsored': 'mean',
            'image_count': 'mean'
        }).round(3)
        
        brand_stats.columns = ['帖子数', '平均分数', '分数标准差', '赞助率', '平均图片数']
        brand_stats = brand_stats.sort_values('帖子数', ascending=False)
        
        print("\n" + brand_stats.head(10).to_string())
        
        logger.info(f"\n💾 最终结果文件: {self.final_results_file}")
    
    def run_multimodal_inference(self):
        """运行完整的多模态推理"""
        logger.info("🚀 开始多模态品牌推理系统")
        
        start_time = datetime.now()
        
        try:
            # 1. 加载post_info映射
            self.load_post_info_mapping()
            
            # 2. 加载推理模型
            self.load_inference_model()
            
            # 3. 处理品牌帖子
            results = self.process_brand_posts()
            
            # 4. 显示最终统计
            self.show_final_statistics(results)
            
            # 5. 计算总时间
            total_time = datetime.now() - start_time
            logger.info(f"⏱️ 总推理时间: {total_time.total_seconds()/60:.1f} 分钟")
            logger.info("🎉 多模态品牌推理完成！")
            
        except Exception as e:
            logger.error(f"❌ 多模态推理失败: {e}")
            raise

def main():
    """主函数"""
    logger.info("🎯 Instagram品牌多模态推理系统")
    logger.info("目标: 对33,829个品牌帖子进行图片+文本推理")
    logger.info("=" * 60)
    
    # 运行多模态推理
    inferencer = MultimodalBrandInference()
    inferencer.run_multimodal_inference()

if __name__ == "__main__":
    main()



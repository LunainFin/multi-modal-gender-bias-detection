#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test performance of fast-trained model
"""

import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
from transformers import AutoTokenizer, AutoModel
import timm
import os
import random
from torchvision import transforms
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Copy model definition (same as training script)
class LightweightGenderBiasModel(nn.Module):
    """Lightweight multi-modal model - optimized for speed"""
    
    def __init__(self, 
                 image_model='resnet18',
                 text_model='distilbert-base-uncased',
                 hidden_dim=128,
                 dropout_rate=0.2):
        super().__init__()
        
        # 轻量级图像编码器
        self.image_encoder = timm.create_model(image_model, pretrained=True, num_classes=0)
        image_dim = self.image_encoder.num_features
        
        # 轻量级文本编码器
        self.text_encoder = AutoModel.from_pretrained(text_model)
        text_dim = self.text_encoder.config.hidden_size
        
        # 简化的特征融合层
        fusion_dim = image_dim + text_dim
        self.fusion = nn.Sequential(
            nn.Linear(fusion_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )
    
    def forward(self, images, input_ids, attention_mask):
        # 图像特征
        image_features = self.image_encoder(images)
        
        # 文本特征
        text_outputs = self.text_encoder(input_ids=input_ids, attention_mask=attention_mask)
        text_features = text_outputs.last_hidden_state.mean(dim=1)
        
        # 特征融合
        combined = torch.cat([image_features, text_features], dim=1)
        scores = self.fusion(combined).squeeze()
        
        return scores

class ModelTester:
    """模型测试器"""
    
    def __init__(self, 
                 model_path='fast_models/fast_best_model.pth',
                 csv_file='train_10k_results/train_10k_fast_results.csv',
                 database_path='/Users/huangxinyue/Downloads/Influencer brand database'):
        
        self.model_path = model_path
        self.csv_file = csv_file
        self.database_path = database_path
        
        # 设备
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"使用设备: {self.device}")
        
        # 初始化tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained('distilbert-base-uncased')
        
        # 图像预处理
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
        
        # 加载模型
        self.load_model()
    
    def load_model(self):
        """加载训练好的模型"""
        logger.info(f"加载模型: {self.model_path}")
        
        # 创建模型
        self.model = LightweightGenderBiasModel()
        
        # 加载权重
        checkpoint = torch.load(self.model_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.device)
        self.model.eval()
        
        # 显示训练历史
        if 'history' in checkpoint:
            history = checkpoint['history']
            logger.info(f"模型训练了 {len(history['epoch'])} 个epochs")
            logger.info(f"最佳验证MAE: {min(history['val_mae']):.4f}")
            logger.info(f"最佳R²分数: {max(history['val_r2']):.4f}")
    
    def find_image_path(self, post_id):
        """查找图片路径"""
        for i in range(1, 17):
            img_path = os.path.join(self.database_path, f'img_resized_{i}', f'{post_id}.jpg')
            if os.path.exists(img_path):
                return img_path
        return None
    
    def predict_single(self, post_id, caption="Instagram post"):
        """对单个帖子进行预测"""
        # 查找图片
        img_path = self.find_image_path(str(post_id))
        if not img_path:
            return None, "图片未找到"
        
        try:
            # 加载和预处理图像
            image = Image.open(img_path).convert('RGB')
            image_tensor = self.transform(image).unsqueeze(0).to(self.device)
            
            # 处理文本
            encoding = self.tokenizer(
                caption,
                truncation=True,
                padding='max_length',
                max_length=64,
                return_tensors='pt'
            )
            input_ids = encoding['input_ids'].to(self.device)
            attention_mask = encoding['attention_mask'].to(self.device)
            
            # 预测
            with torch.no_grad():
                output = self.model(image_tensor, input_ids, attention_mask)
                score = output.item() * 10.0  # 转换回0-10范围
            
            return score, img_path
            
        except Exception as e:
            return None, f"预测失败: {e}"
    
    def evaluate_on_test_set(self, num_samples=500):
        """在测试集上评估模型"""
        logger.info(f"在 {num_samples} 个样本上评估模型...")
        
        # 读取数据
        df = pd.read_csv(self.csv_file, dtype={'post_id': str})
        
        # 随机选择样本
        if len(df) > num_samples:
            df = df.sample(n=num_samples, random_state=42)
        
        predictions = []
        targets = []
        valid_samples = []
        
        for idx, row in df.iterrows():
            post_id = str(row['post_id'])
            true_score = row['gender_bias_score']
            
            if pd.isna(true_score):
                continue
                
            pred_score, status = self.predict_single(post_id)
            
            if pred_score is not None:
                predictions.append(pred_score)
                targets.append(true_score)
                valid_samples.append({
                    'post_id': post_id,
                    'true_score': true_score,
                    'pred_score': pred_score,
                    'error': abs(pred_score - true_score)
                })
        
        if not predictions:
            logger.error("没有有效的预测结果！")
            return
        
        # 计算指标
        mae = mean_absolute_error(targets, predictions)
        rmse = np.sqrt(mean_squared_error(targets, predictions))
        r2 = r2_score(targets, predictions)
        
        # 统计结果
        logger.info(f"\n🎯 模型评估结果 (N={len(predictions)}):")
        logger.info(f"   MAE (平均绝对误差): {mae:.4f}")
        logger.info(f"   RMSE (均方根误差): {rmse:.4f}")
        logger.info(f"   R² (决定系数): {r2:.4f}")
        
        # 误差分析
        errors = [abs(p-t) for p, t in zip(predictions, targets)]
        logger.info(f"\n📊 误差分析:")
        logger.info(f"   误差 < 1.0: {sum(1 for e in errors if e < 1.0)/len(errors)*100:.1f}%")
        logger.info(f"   误差 < 2.0: {sum(1 for e in errors if e < 2.0)/len(errors)*100:.1f}%")
        logger.info(f"   误差 < 3.0: {sum(1 for e in errors if e < 3.0)/len(errors)*100:.1f}%")
        
        # 绘制结果
        self.plot_results(targets, predictions, valid_samples)
        
        return valid_samples
    
    def plot_results(self, targets, predictions, samples):
        """绘制预测结果"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. 真实值 vs 预测值散点图
        axes[0,0].scatter(targets, predictions, alpha=0.6, s=30)
        axes[0,0].plot([0, 10], [0, 10], 'r--', linewidth=2)
        axes[0,0].set_xlabel('真实分数')
        axes[0,0].set_ylabel('预测分数')
        axes[0,0].set_title('真实值 vs 预测值')
        axes[0,0].grid(True, alpha=0.3)
        
        # 2. 误差分布直方图
        errors = [abs(p-t) for p, t in zip(predictions, targets)]
        axes[0,1].hist(errors, bins=20, alpha=0.7, color='skyblue', edgecolor='black')
        axes[0,1].set_xlabel('绝对误差')
        axes[0,1].set_ylabel('频次')
        axes[0,1].set_title('误差分布')
        axes[0,1].grid(True, alpha=0.3)
        
        # 3. 分数分布对比
        axes[1,0].hist(targets, bins=20, alpha=0.5, label='真实分数', color='blue')
        axes[1,0].hist(predictions, bins=20, alpha=0.5, label='预测分数', color='red')
        axes[1,0].set_xlabel('分数')
        axes[1,0].set_ylabel('频次')
        axes[1,0].set_title('分数分布对比')
        axes[1,0].legend()
        axes[1,0].grid(True, alpha=0.3)
        
        # 4. 残差图
        residuals = [p-t for p, t in zip(predictions, targets)]
        axes[1,1].scatter(targets, residuals, alpha=0.6, s=30)
        axes[1,1].axhline(y=0, color='r', linestyle='--')
        axes[1,1].set_xlabel('真实分数')
        axes[1,1].set_ylabel('残差 (预测-真实)')
        axes[1,1].set_title('残差图')
        axes[1,1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('fast_model_evaluation.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        logger.info("📈 评估图表已保存为 fast_model_evaluation.png")
    
    def show_example_predictions(self, num_examples=10):
        """显示一些预测示例"""
        logger.info(f"\n🔍 显示 {num_examples} 个预测示例:")
        
        # 读取数据
        df = pd.read_csv(self.csv_file, dtype={'post_id': str})
        samples = df.sample(n=num_examples, random_state=42)
        
        for idx, row in samples.iterrows():
            post_id = str(row['post_id'])
            true_score = row['gender_bias_score']
            
            if pd.isna(true_score):
                continue
                
            pred_score, img_path = self.predict_single(post_id)
            
            if pred_score is not None:
                error = abs(pred_score - true_score)
                print(f"\n📸 帖子 {post_id}:")
                print(f"   真实分数: {true_score:.1f}")
                print(f"   预测分数: {pred_score:.1f}")
                print(f"   误差: {error:.1f}")
                print(f"   图片: {os.path.basename(img_path)}")

def main():
    """主函数"""
    print("🧪 测试快速训练的模型")
    print("=" * 50)
    
    # 创建测试器
    tester = ModelTester()
    
    # 显示预测示例
    tester.show_example_predictions(num_examples=10)
    
    print("\n" + "=" * 50)
    
    # 在测试集上评估
    samples = tester.evaluate_on_test_set(num_samples=1000)
    
    print("\n✅ 测试完成！")
    print("📈 评估图表已保存")
    
    # 可选：交互式预测
    print("\n🎮 想要测试特定帖子吗？")
    print("输入帖子ID进行预测，或按Enter跳过:")
    
    try:
        post_id = input().strip()
        if post_id:
            pred_score, status = tester.predict_single(post_id)
            if pred_score is not None:
                print(f"✅ 预测分数: {pred_score:.2f}")
            else:
                print(f"❌ {status}")
    except:
        pass

if __name__ == "__main__":
    main()







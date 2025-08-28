#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Google Colab版本 - Instagram性别倾向多模态模型训练
优化配置，GPU加速
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import json
import os
import numpy as np
from PIL import Image
import logging
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from transformers import AutoTokenizer, AutoModel
import timm
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
import sys
warnings.filterwarnings('ignore')

# 设置环境变量避免tokenizer警告
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('colab_training.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Google Colab GPU检测
def check_gpu():
    """检查GPU可用性"""
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name()
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
        logger.info(f"🚀 GPU可用: {gpu_name}")
        logger.info(f"💾 显存: {gpu_memory:.1f} GB")
        return True
    else:
        logger.warning("⚠️  未检测到GPU，将使用CPU训练")
        return False

class ColabInstagramDataset(Dataset):
    """Colab版本的Instagram数据集"""
    
    def __init__(self, csv_file, image_dir, tokenizer, max_length=128, image_size=224):
        # 读取CSV，强制post_id为字符串
        self.df = pd.read_csv(csv_file, dtype={'post_id': str})
        self.image_dir = image_dir
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        # 图像预处理
        from torchvision import transforms
        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
        
        # 过滤掉无效样本
        self.valid_samples = []
        for idx, row in self.df.iterrows():
            post_id = str(row['post_id'])
            score = row['gender_bias_score']
            
            # 检查图片文件是否存在
            img_path = os.path.join(image_dir, f"{post_id}.jpg")
            if os.path.exists(img_path) and not pd.isna(score):
                # 添加默认caption（简化版本）
                self.valid_samples.append({
                    'post_id': post_id,
                    'image_path': img_path,
                    'caption': f"Instagram post {post_id}",  # 简化caption
                    'score': float(score) / 10.0  # 标准化到0-1
                })
        
        logger.info(f"✅ 有效样本数: {len(self.valid_samples)}")
    
    def __len__(self):
        return len(self.valid_samples)
    
    def __getitem__(self, idx):
        sample = self.valid_samples[idx]
        
        # 加载图像
        try:
            image = Image.open(sample['image_path']).convert('RGB')
            image = self.transform(image)
        except Exception as e:
            logger.warning(f"图像加载失败 {sample['post_id']}: {e}")
            # 返回空白图像
            image = torch.zeros(3, 224, 224)
        
        # 处理文本
        caption = sample['caption']
        encoding = self.tokenizer(
            caption,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        return {
            'image': image,
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze(),
            'score': torch.tensor(sample['score'], dtype=torch.float32)
        }

class ColabGenderBiasModel(nn.Module):
    """Colab优化版多模态模型"""
    
    def __init__(self, 
                 image_model='resnet18',
                 text_model='distilbert-base-uncased',
                 hidden_dim=256,
                 dropout_rate=0.3):
        super().__init__()
        
        # 图像编码器 (预训练ResNet18)
        self.image_encoder = timm.create_model(image_model, pretrained=True, num_classes=0)
        image_dim = self.image_encoder.num_features
        
        # 文本编码器 (DistilBERT)
        self.text_encoder = AutoModel.from_pretrained(text_model)
        text_dim = self.text_encoder.config.hidden_size
        
        # 特征融合层
        fusion_dim = image_dim + text_dim
        self.fusion = nn.Sequential(
            nn.Linear(fusion_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        )
        
        # 分类器
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()  # 输出0-1范围
        )
        
        # 冻结文本编码器部分层以加速训练
        for param in self.text_encoder.embeddings.parameters():
            param.requires_grad = False
    
    def forward(self, images, input_ids, attention_mask):
        # 图像特征
        image_features = self.image_encoder(images)
        
        # 文本特征
        text_outputs = self.text_encoder(input_ids=input_ids, attention_mask=attention_mask)
        text_features = text_outputs.last_hidden_state.mean(dim=1)  # 平均池化
        
        # 特征融合
        combined = torch.cat([image_features, text_features], dim=1)
        fused = self.fusion(combined)
        
        # 预测分数
        scores = self.classifier(fused).squeeze()
        
        return scores

class ColabTrainer:
    """Colab训练器"""
    
    def __init__(self, 
                 csv_file='./train_10k_fast_results.csv',
                 image_dir='./images',
                 batch_size=32,          # GPU可以用更大batch
                 learning_rate=1e-3,     # 稍高的学习率
                 num_epochs=10,          # 减少epoch数
                 test_size=0.2):
        
        self.csv_file = csv_file
        self.image_dir = image_dir
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs
        self.test_size = test_size
        
        # 设备检测
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"🎯 使用设备: {self.device}")
        
        # 初始化tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained('distilbert-base-uncased')
        
        # 训练历史
        self.history = {
            'epoch': [],
            'train_loss': [],
            'val_loss': [],
            'val_mae': [],
            'val_r2': []
        }
    
    def prepare_data(self):
        """准备数据"""
        logger.info("📊 准备数据...")
        
        # 创建数据集
        dataset = ColabInstagramDataset(
            csv_file=self.csv_file,
            image_dir=self.image_dir,
            tokenizer=self.tokenizer
        )
        
        # 分割数据
        train_size = int((1 - self.test_size) * len(dataset))
        val_size = len(dataset) - train_size
        self.train_dataset, self.val_dataset = torch.utils.data.random_split(
            dataset, [train_size, val_size]
        )
        
        # 创建数据加载器
        self.train_loader = DataLoader(
            self.train_dataset, 
            batch_size=self.batch_size, 
            shuffle=True,
            num_workers=2,  # Colab通常有2个CPU核心
            pin_memory=True if torch.cuda.is_available() else False
        )
        
        self.val_loader = DataLoader(
            self.val_dataset, 
            batch_size=self.batch_size, 
            shuffle=False,
            num_workers=2,
            pin_memory=True if torch.cuda.is_available() else False
        )
        
        logger.info(f"✅ 训练集: {len(self.train_dataset)} 样本")
        logger.info(f"✅ 验证集: {len(self.val_dataset)} 样本")
    
    def create_model(self):
        """创建模型"""
        logger.info("🤖 创建模型...")
        
        self.model = ColabGenderBiasModel().to(self.device)
        
        # 损失函数和优化器
        self.criterion = nn.MSELoss()
        self.optimizer = optim.AdamW(
            self.model.parameters(), 
            lr=self.learning_rate, 
            weight_decay=1e-4
        )
        
        # 学习率调度器
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, 
            mode='min', 
            factor=0.5, 
            patience=2,  # 更激进的学习率衰减
            min_lr=1e-6
        )
        
        # 模型信息
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        logger.info(f"📊 总参数量: {total_params:,}")
        logger.info(f"📊 可训练参数: {trainable_params:,}")
    
    def train_epoch(self):
        """训练一个epoch"""
        self.model.train()
        total_loss = 0
        num_batches = len(self.train_loader)
        
        for batch_idx, batch in enumerate(tqdm(self.train_loader, desc='Training')):
            try:
                # 数据移至GPU
                images = batch['image'].to(self.device)
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                targets = batch['score'].to(self.device)
                
                # 前向传播
                self.optimizer.zero_grad()
                outputs = self.model(images, input_ids, attention_mask)
                
                # 确保维度匹配
                if outputs.dim() == 0:
                    outputs = outputs.unsqueeze(0)
                if targets.dim() == 0:
                    targets = targets.unsqueeze(0)
                
                loss = self.criterion(outputs, targets)
                
                # 反向传播
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()
                
                total_loss += loss.item()
                
            except Exception as e:
                logger.warning(f"训练批次 {batch_idx} 失败: {e}")
                continue
        
        return total_loss / num_batches if num_batches > 0 else 0
    
    def validate(self):
        """验证模型"""
        self.model.eval()
        total_loss = 0
        all_predictions = []
        all_targets = []
        
        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc='Validation'):
                try:
                    images = batch['image'].to(self.device)
                    input_ids = batch['input_ids'].to(self.device)
                    attention_mask = batch['attention_mask'].to(self.device)
                    targets = batch['score'].to(self.device)
                    
                    outputs = self.model(images, input_ids, attention_mask)
                    
                    # 处理维度
                    if outputs.dim() == 0:
                        outputs = outputs.unsqueeze(0)
                    if targets.dim() == 0:
                        targets = targets.unsqueeze(0)
                    
                    loss = self.criterion(outputs, targets)
                    total_loss += loss.item()
                    
                    # 转换回原始分数范围[0,10]
                    predictions = outputs.cpu().numpy() * 10.0
                    targets_np = targets.cpu().numpy() * 10.0
                    
                    all_predictions.extend(predictions)
                    all_targets.extend(targets_np)
                    
                except Exception as e:
                    logger.warning(f"验证批次失败: {e}")
                    continue
        
        avg_loss = total_loss / len(self.val_loader) if len(self.val_loader) > 0 else 0
        
        if all_predictions and all_targets:
            mae = mean_absolute_error(all_targets, all_predictions)
            r2 = r2_score(all_targets, all_predictions)
        else:
            mae = 0
            r2 = 0
        
        return avg_loss, mae, r2, all_predictions, all_targets
    
    def save_model(self, epoch, is_best=False):
        """保存模型"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'history': self.history
        }
        
        # 保存最新模型
        torch.save(checkpoint, 'latest_model.pth')
        
        # 保存最佳模型
        if is_best:
            torch.save(checkpoint, 'best_model.pth')
            logger.info("💾 保存最佳模型")
    
    def train(self):
        """完整训练流程"""
        logger.info("🚀 开始Colab训练...")
        
        # 准备数据
        self.prepare_data()
        
        # 创建模型
        self.create_model()
        
        best_val_loss = float('inf')
        patience_counter = 0
        patience = 3  # 更激进的早停
        
        for epoch in range(1, self.num_epochs + 1):
            logger.info(f"\n🔄 Epoch {epoch}/{self.num_epochs}")
            
            # 训练
            train_loss = self.train_epoch()
            
            # 验证
            val_loss, val_mae, val_r2, predictions, targets = self.validate()
            
            # 学习率调度
            self.scheduler.step(val_loss)
            
            # 记录历史
            self.history['epoch'].append(epoch)
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['val_mae'].append(val_mae)
            self.history['val_r2'].append(val_r2)
            
            # 检查最佳模型
            is_best = val_loss < best_val_loss
            if is_best:
                best_val_loss = val_loss
                patience_counter = 0
            else:
                patience_counter += 1
            
            # 保存模型
            self.save_model(epoch, is_best)
            
            # 输出结果
            logger.info(f"📊 Train Loss: {train_loss:.4f}")
            logger.info(f"📊 Val Loss: {val_loss:.4f}, MAE: {val_mae:.4f}, R²: {val_r2:.4f}")
            logger.info(f"⚙️  学习率: {self.optimizer.param_groups[0]['lr']:.2e}")
            
            # 早停
            if patience_counter >= patience:
                logger.info(f"⏹️  验证损失在{patience}个epoch内没有改善，提前停止")
                break
        
        logger.info("🎉 Colab训练完成！")
        logger.info(f"🏆 最佳验证损失: {best_val_loss:.4f}")
        
        # 绘制训练曲线
        self.plot_training_curves()
    
    def plot_training_curves(self):
        """绘制训练曲线"""
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        
        # Loss曲线
        axes[0,0].plot(self.history['epoch'], self.history['train_loss'], 'b-', label='Train Loss')
        axes[0,0].plot(self.history['epoch'], self.history['val_loss'], 'r-', label='Val Loss')
        axes[0,0].set_title('Loss曲线')
        axes[0,0].set_xlabel('Epoch')
        axes[0,0].set_ylabel('Loss')
        axes[0,0].legend()
        axes[0,0].grid(True)
        
        # MAE曲线
        axes[0,1].plot(self.history['epoch'], self.history['val_mae'], 'g-', label='Val MAE')
        axes[0,1].set_title('平均绝对误差')
        axes[0,1].set_xlabel('Epoch')
        axes[0,1].set_ylabel('MAE')
        axes[0,1].legend()
        axes[0,1].grid(True)
        
        # R²曲线
        axes[1,0].plot(self.history['epoch'], self.history['val_r2'], 'm-', label='Val R²')
        axes[1,0].set_title('决定系数')
        axes[1,0].set_xlabel('Epoch')
        axes[1,0].set_ylabel('R²')
        axes[1,0].legend()
        axes[1,0].grid(True)
        
        # 隐藏最后一个子图
        axes[1,1].axis('off')
        
        plt.tight_layout()
        plt.savefig('colab_training_curves.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        logger.info("📈 训练曲线已保存为 colab_training_curves.png")

def main():
    """主函数"""
    
    # GPU检测
    check_gpu()
    
    # 配置Colab优化训练
    trainer = ColabTrainer(
        csv_file='./train_10k_fast_results.csv',
        image_dir='./images',
        batch_size=64 if torch.cuda.is_available() else 16,  # GPU用大batch
        learning_rate=1e-3,      # 提高学习率
        num_epochs=10,           # 减少轮数
        test_size=0.2
    )
    
    # 开始训练
    trainer.train()

if __name__ == "__main__":
    main()







#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Local Fast Training Version - Complete training in 1-2 hours
Heavily optimized configuration, trading small accuracy for training speed
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

# Set environment variable to avoid tokenizer warnings
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Detect if running in background
def is_running_in_background():
    """Detect if running in background (nohup or similar environment)"""
    try:
        return not sys.stdout.isatty() or not sys.stdin.isatty()
    except:
        return True

# Global variable to control tqdm display
DISABLE_TQDM = is_running_in_background()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('fast_training.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class FastInstagramDataset(Dataset):
    """Fast training version Instagram dataset"""
    
    def __init__(self, csv_file, database_path, tokenizer, max_length=64, image_size=224, max_samples=None):
        # Read CSV, force post_id as string
        df = pd.read_csv(csv_file, dtype={'post_id': str})
        
        # Limit sample count for fast training
        if max_samples:
            df = df.head(max_samples)
            logger.info(f"⚡ Fast mode: limiting samples to {max_samples}")
        
        self.database_path = database_path
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        # Image preprocessing - simplified version
        from torchvision import transforms
        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
        
        # Filter out invalid samples
        self.valid_samples = []
        for idx, row in df.iterrows():
            post_id = str(row['post_id'])
            score = row['gender_bias_score']
            
            if not pd.isna(score):
                # Find image path
                img_path = self.find_image_path(post_id)
                if img_path:
                    self.valid_samples.append({
                        'post_id': post_id,
                        'image_path': img_path,
                        'caption': f"Post {post_id}",  # Simplified caption
                        'score': float(score) / 10.0  # Normalize to 0-1
                    })
        
        logger.info(f"✅ Valid samples: {len(self.valid_samples)}")
    
    def find_image_path(self, post_id):
        """Find image path"""
        for i in range(1, 17):
            img_path = os.path.join(self.database_path, f'img_resized_{i}', f'{post_id}.jpg')
            if os.path.exists(img_path):
                return img_path
        return None
    
    def __len__(self):
        return len(self.valid_samples)
    
    def __getitem__(self, idx):
        sample = self.valid_samples[idx]
        
        # Load image
        try:
            image = Image.open(sample['image_path']).convert('RGB')
            image = self.transform(image)
        except Exception as e:
            logger.warning(f"Image loading failed {sample['post_id']}: {e}")
            # Return blank image
            image = torch.zeros(3, 224, 224)
        
        # Process text - simplified version
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

class LightweightGenderBiasModel(nn.Module):
    """Lightweight multi-modal model - optimized for speed"""
    
    def __init__(self, 
                 image_model='resnet18',
                 text_model='distilbert-base-uncased',
                 hidden_dim=128,  # Reduce hidden layer dimension
                 dropout_rate=0.2):
        super().__init__()
        
        # Lightweight image encoder
        self.image_encoder = timm.create_model(image_model, pretrained=True, num_classes=0)
        image_dim = self.image_encoder.num_features
        
        # Lightweight text encoder
        self.text_encoder = AutoModel.from_pretrained(text_model)
        text_dim = self.text_encoder.config.hidden_size
        
        # Freeze more layers to accelerate training
        for param in self.text_encoder.embeddings.parameters():
            param.requires_grad = False
        for param in self.text_encoder.transformer.layer[:3].parameters():  # Freeze first 3 layers
            param.requires_grad = False
        
        # Simplified feature fusion layer
        fusion_dim = image_dim + text_dim
        self.fusion = nn.Sequential(
            nn.Linear(fusion_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()  # Direct output 0-1 range
        )
    
    def forward(self, images, input_ids, attention_mask):
        # Image features
        image_features = self.image_encoder(images)
        
        # Text features - simplified processing
        text_outputs = self.text_encoder(input_ids=input_ids, attention_mask=attention_mask)
        text_features = text_outputs.last_hidden_state.mean(dim=1)  # Average pooling
        
        # Feature fusion
        combined = torch.cat([image_features, text_features], dim=1)
        scores = self.fusion(combined).squeeze()
        
        return scores

class FastTrainer:
    """Fast trainer"""
    
    def __init__(self, 
                 csv_file='/Users/huangxinyue/Multi model distillation/train_10k_results/train_10k_fast_results.csv',
                 database_path='/Users/huangxinyue/Downloads/Influencer brand database',
                 model_save_dir='/Users/huangxinyue/Multi model distillation/fast_models',
                 batch_size=32,
                 learning_rate=5e-4,     # Increase learning rate
                 num_epochs=6,           # Greatly reduce number of epochs
                 test_size=0.2,
                 max_samples=5000):      # Optional: limit sample count
        
        self.csv_file = csv_file
        self.database_path = database_path
        self.model_save_dir = model_save_dir
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs
        self.test_size = test_size
        self.max_samples = max_samples
        
        # Create保存目录
        os.makedirs(model_save_dir, exist_ok=True)
        
        # 设备
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"使用设备: {self.device}")
        
        # Initializetokenizer
        self.tokenizer = AutoTokenizer.from_pretrained('distilbert-base-uncased')
        
        # Training历史
        self.history = {
            'epoch': [],
            'train_loss': [],
            'val_loss': [],
            'val_mae': [],
            'val_r2': []
        }
    
    def prepare_data(self):
        """准备数据"""
        logger.info("准备数据...")
        
        # Create快速数据集
        dataset = FastInstagramDataset(
            csv_file=self.csv_file,
            database_path=self.database_path,
            tokenizer=self.tokenizer,
            max_length=64,  # 减少序列长度
            max_samples=self.max_samples
        )
        
        # 分割数据
        train_size = int((1 - self.test_size) * len(dataset))
        val_size = len(dataset) - train_size
        self.train_dataset, self.val_dataset = torch.utils.data.random_split(
            dataset, [train_size, val_size]
        )
        
        # Create数据加载器
        self.train_loader = DataLoader(
            self.train_dataset, 
            batch_size=self.batch_size, 
            shuffle=True,
            num_workers=0,  # 避免多进程开销
            pin_memory=False
        )
        
        self.val_loader = DataLoader(
            self.val_dataset, 
            batch_size=self.batch_size, 
            shuffle=False,
            num_workers=0,
            pin_memory=False
        )
        
        logger.info(f"训练集: {len(self.train_dataset)} 样本")
        logger.info(f"验证集: {len(self.val_dataset)} 样本")
    
    def create_model(self):
        """创建轻量级模型"""
        logger.info("创建轻量级模型...")
        
        self.model = LightweightGenderBiasModel().to(self.device)
        
        # 损失函数和优化器
        self.criterion = nn.MSELoss()
        self.optimizer = optim.AdamW(
            self.model.parameters(), 
            lr=self.learning_rate, 
            weight_decay=1e-4
        )
        
        # 更激进的学习率调度器
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, 
            mode='min', 
            factor=0.5, 
            patience=1,  # 非常激进的调度
            min_lr=1e-6
        )
        
        # 模型信息
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        logger.info(f"总参数量: {total_params:,}")
        logger.info(f"可训练参数: {trainable_params:,}")
    
    def train_epoch(self):
        """训练一个epoch"""
        self.model.train()
        total_loss = 0
        num_batches = len(self.train_loader)
        
        with tqdm(self.train_loader, desc='Training', disable=DISABLE_TQDM) as pbar:
            for batch_idx, batch in enumerate(pbar):
                try:
                    # 数据移至设备
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
                    pbar.set_postfix({'loss': f'{loss.item():.4f}'})
                    
                    # 后台运行时的进度日志
                    if DISABLE_TQDM and batch_idx % 50 == 0:  # 更频繁的日志
                        progress_pct = (batch_idx / num_batches) * 100
                        logger.info(f"  批次 {batch_idx}/{num_batches} ({progress_pct:.1f}%) - Loss: {loss.item():.4f}")
                    
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
            for batch in tqdm(self.val_loader, desc='Validation', disable=DISABLE_TQDM):
                try:
                    images = batch['image'].to(self.device)
                    input_ids = batch['input_ids'].to(self.device)
                    attention_mask = batch['attention_mask'].to(self.device)
                    targets = batch['score'].to(self.device)
                    
                    outputs = self.model(images, input_ids, attention_mask)
                    
                    # Process维度
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
        
        # Save最新模型
        torch.save(checkpoint, os.path.join(self.model_save_dir, 'fast_latest_model.pth'))
        
        # Save最佳模型
        if is_best:
            torch.save(checkpoint, os.path.join(self.model_save_dir, 'fast_best_model.pth'))
            logger.info("💾 保存最佳模型")
    
    def train(self):
        """完整训练流程"""
        logger.info("🚀 开始快速训练...")
        start_time = datetime.now()
        
        # 准备数据
        self.prepare_data()
        
        # Create模型
        self.create_model()
        
        best_val_loss = float('inf')
        patience_counter = 0
        patience = 2  # 非常激进的早停
        
        for epoch in range(1, self.num_epochs + 1):
            epoch_start = datetime.now()
            logger.info(f"\n⚡ Epoch {epoch}/{self.num_epochs}")
            if DISABLE_TQDM:
                logger.info(f"开始训练 Epoch {epoch}...")
            
            # Training
            train_loss = self.train_epoch()
            
            # Validate
            if DISABLE_TQDM:
                logger.info(f"开始验证 Epoch {epoch}...")
            val_loss, val_mae, val_r2, predictions, targets = self.validate()
            
            # 学习率调度
            self.scheduler.step(val_loss)
            
            # 记录历史
            self.history['epoch'].append(epoch)
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['val_mae'].append(val_mae)
            self.history['val_r2'].append(val_r2)
            
            # Check最佳模型
            is_best = val_loss < best_val_loss
            if is_best:
                best_val_loss = val_loss
                patience_counter = 0
            else:
                patience_counter += 1
            
            # Save模型
            self.save_model(epoch, is_best)
            
            # Calculate时间
            epoch_time = datetime.now() - epoch_start
            total_time = datetime.now() - start_time
            
            # 输出结果
            logger.info(f"⏱️  Epoch用时: {epoch_time.total_seconds():.1f}s")
            logger.info(f"📊 Train Loss: {train_loss:.4f}")
            logger.info(f"📊 Val Loss: {val_loss:.4f}, MAE: {val_mae:.4f}, R²: {val_r2:.4f}")
            logger.info(f"⚙️  学习率: {self.optimizer.param_groups[0]['lr']:.2e}")
            logger.info(f"🕐 总用时: {total_time.total_seconds()/60:.1f}分钟")
            
            # 早停
            if patience_counter >= patience:
                logger.info(f"⏹️  验证损失在{patience}个epoch内没有改善，提前停止")
                break
        
        # 最终统计
        total_time = datetime.now() - start_time
        logger.info("🎉 快速训练完成！")
        logger.info(f"🏆 最佳验证损失: {best_val_loss:.4f}")
        logger.info(f"⏱️  总训练时间: {total_time.total_seconds()/60:.1f}分钟")
        logger.info(f"💾 模型保存目录: {self.model_save_dir}")

def main():
    """主函数"""
    
    logger.info("⚡ 快速本地训练模式")
    
    # 选择训练模式
    print("\n🚀 选择快速训练模式:")
    print("1. 🔥 极速模式 (2000样本, ~30分钟)")
    print("2. ⚡ 快速模式 (5000样本, ~60分钟)")  
    print("3. 🏃 标准模式 (全部样本, ~90分钟)")
    
    try:
        choice = input("请选择 (1-3，默认2): ").strip() or "2"
    except:
        choice = "2"  # 后台运行时默认选择
    
    if choice == "1":
        max_samples = 2000
        epochs = 5
        logger.info("🔥 极速模式：2000样本，预计30分钟")
    elif choice == "3":
        max_samples = None
        epochs = 6
        logger.info("🏃 标准模式：全部样本，预计90分钟")
    else:
        max_samples = 5000
        epochs = 6
        logger.info("⚡ 快速模式：5000样本，预计60分钟")
    
    # 快速训练配置
    trainer = FastTrainer(
        csv_file='/Users/huangxinyue/Multi model distillation/train_10k_results/train_10k_fast_results.csv',
        database_path='/Users/huangxinyue/Downloads/Influencer brand database',
        model_save_dir='/Users/huangxinyue/Multi model distillation/fast_models',
        batch_size=32,
        learning_rate=5e-4,
        num_epochs=epochs,
        test_size=0.2,
        max_samples=max_samples
    )
    
    # 开始训练
    trainer.train()

if __name__ == "__main__":
    main()







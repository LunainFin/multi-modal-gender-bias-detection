#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Local Fast Training Version - Complete training in 1-2 hours
Heavily optimized configuration, trading slight accuracy for training speed
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
            logger.info(f"⚡ Fast mode: Limited samples to {max_samples}")
        
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
    """Lightweight multimodal model - optimized for speed"""
    
    def __init__(self, 
                 image_model='resnet18',
                 text_model='distilbert-base-uncased',
                 hidden_dim=128,  # Reduced hidden layer dimension
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
            nn.Sigmoid()  # Direct output to 0-1 range
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
                 learning_rate=5e-4,     # Increased learning rate
                 num_epochs=6,           # Significantly reduced epochs
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
        
        # Create save directory
        os.makedirs(model_save_dir, exist_ok=True)
        
        # Device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {self.device}")
        
        # Initialize tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained('distilbert-base-uncased')
        
        # Training history
        self.history = {
            'epoch': [],
            'train_loss': [],
            'val_loss': [],
            'val_mae': [],
            'val_r2': []
        }
    
    def prepare_data(self):
        """Prepare data"""
        logger.info("Preparing data...")
        
        # Create fast dataset
        dataset = FastInstagramDataset(
            csv_file=self.csv_file,
            database_path=self.database_path,
            tokenizer=self.tokenizer,
            max_length=64,  # Reduced sequence length
            max_samples=self.max_samples
        )
        
        # Split data
        train_size = int((1 - self.test_size) * len(dataset))
        val_size = len(dataset) - train_size
        self.train_dataset, self.val_dataset = torch.utils.data.random_split(
            dataset, [train_size, val_size]
        )
        
        # Create data loaders
        self.train_loader = DataLoader(
            self.train_dataset, 
            batch_size=self.batch_size, 
            shuffle=True,
            num_workers=0,  # Avoid multiprocessing overhead
            pin_memory=False
        )
        
        self.val_loader = DataLoader(
            self.val_dataset, 
            batch_size=self.batch_size, 
            shuffle=False,
            num_workers=0,
            pin_memory=False
        )
        
        logger.info(f"Training set: {len(self.train_dataset)} samples")
        logger.info(f"Validation set: {len(self.val_dataset)} samples")
    
    def create_model(self):
        """Create lightweight model"""
        logger.info("Creating lightweight model...")
        
        self.model = LightweightGenderBiasModel().to(self.device)
        
        # Loss function and optimizer
        self.criterion = nn.MSELoss()
        self.optimizer = optim.AdamW(
            self.model.parameters(), 
            lr=self.learning_rate, 
            weight_decay=1e-4
        )
        
        # More aggressive learning rate scheduler
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, 
            mode='min', 
            factor=0.5, 
            patience=1,  # Very aggressive scheduling
            min_lr=1e-6
        )
        
        # Model information
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        logger.info(f"Total parameters: {total_params:,}")
        logger.info(f"Trainable parameters: {trainable_params:,}")
    
    def train_epoch(self):
        """Train one epoch"""
        self.model.train()
        total_loss = 0
        num_batches = len(self.train_loader)
        
        with tqdm(self.train_loader, desc='Training', disable=DISABLE_TQDM) as pbar:
            for batch_idx, batch in enumerate(pbar):
                try:
                    # Move data to device
                    images = batch['image'].to(self.device)
                    input_ids = batch['input_ids'].to(self.device)
                    attention_mask = batch['attention_mask'].to(self.device)
                    targets = batch['score'].to(self.device)
                    
                    # Forward pass
                    self.optimizer.zero_grad()
                    outputs = self.model(images, input_ids, attention_mask)
                    
                    # Ensure dimension matching
                    if outputs.dim() == 0:
                        outputs = outputs.unsqueeze(0)
                    if targets.dim() == 0:
                        targets = targets.unsqueeze(0)
                    
                    loss = self.criterion(outputs, targets)
                    
                    # Backward pass
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    self.optimizer.step()
                    
                    total_loss += loss.item()
                    pbar.set_postfix({'loss': f'{loss.item():.4f}'})
                    
                    # Progress logging for background mode
                    if DISABLE_TQDM and batch_idx % 50 == 0:  # More frequent logging
                        progress_pct = (batch_idx / num_batches) * 100
                        logger.info(f"  Batch {batch_idx}/{num_batches} ({progress_pct:.1f}%) - Loss: {loss.item():.4f}")
                    
                except Exception as e:
                    logger.warning(f"Training batch {batch_idx} failed: {e}")
                    continue
        
        return total_loss / num_batches if num_batches > 0 else 0
    
    def validate(self):
        """Validate model"""
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
                    
                    # Handle dimensions
                    if outputs.dim() == 0:
                        outputs = outputs.unsqueeze(0)
                    if targets.dim() == 0:
                        targets = targets.unsqueeze(0)
                    
                    loss = self.criterion(outputs, targets)
                    total_loss += loss.item()
                    
                    # Convert back to original score range [0,10]
                    predictions = outputs.cpu().numpy() * 10.0
                    targets_np = targets.cpu().numpy() * 10.0
                    
                    all_predictions.extend(predictions)
                    all_targets.extend(targets_np)
                    
                except Exception as e:
                    logger.warning(f"Validation batch failed: {e}")
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
        """Save model"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'history': self.history
        }
        
        # Save latest model
        torch.save(checkpoint, os.path.join(self.model_save_dir, 'fast_latest_model.pth'))
        
        # Save best model
        if is_best:
            torch.save(checkpoint, os.path.join(self.model_save_dir, 'fast_best_model.pth'))
            logger.info("💾 Saved best model")
    
    def train(self):
        """Complete training workflow"""
        logger.info("🚀 Starting fast training...")
        start_time = datetime.now()
        
        # Prepare data
        self.prepare_data()
        
        # Create model
        self.create_model()
        
        best_val_loss = float('inf')
        patience_counter = 0
        patience = 2  # Very aggressive early stopping
        
        for epoch in range(1, self.num_epochs + 1):
            epoch_start = datetime.now()
            logger.info(f"\n⚡ Epoch {epoch}/{self.num_epochs}")
            if DISABLE_TQDM:
                logger.info(f"Starting training Epoch {epoch}...")
            
            # Training
            train_loss = self.train_epoch()
            
            # Validation
            if DISABLE_TQDM:
                logger.info(f"Starting validation Epoch {epoch}...")
            val_loss, val_mae, val_r2, predictions, targets = self.validate()
            
            # Learning rate scheduling
            self.scheduler.step(val_loss)
            
            # Record history
            self.history['epoch'].append(epoch)
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['val_mae'].append(val_mae)
            self.history['val_r2'].append(val_r2)
            
            # Check best model
            is_best = val_loss < best_val_loss
            if is_best:
                best_val_loss = val_loss
                patience_counter = 0
            else:
                patience_counter += 1
            
            # Save model
            self.save_model(epoch, is_best)
            
            # Calculate time
            epoch_time = datetime.now() - epoch_start
            total_time = datetime.now() - start_time
            
            # Output results
            logger.info(f"⏱️  Epoch time: {epoch_time.total_seconds():.1f}s")
            logger.info(f"📊 Train Loss: {train_loss:.4f}")
            logger.info(f"📊 Val Loss: {val_loss:.4f}, MAE: {val_mae:.4f}, R²: {val_r2:.4f}")
            logger.info(f"⚙️  Learning rate: {self.optimizer.param_groups[0]['lr']:.2e}")
            logger.info(f"🕐 Total time: {total_time.total_seconds()/60:.1f} minutes")
            
            # Early stopping
            if patience_counter >= patience:
                logger.info(f"⏹️  Validation loss hasn't improved for {patience} epochs, early stopping")
                break
        
        # Final statistics
        total_time = datetime.now() - start_time
        logger.info("🎉 Fast training completed!")
        logger.info(f"🏆 Best validation loss: {best_val_loss:.4f}")
        logger.info(f"⏱️  Total training time: {total_time.total_seconds()/60:.1f} minutes")
        logger.info(f"💾 Model save directory: {self.model_save_dir}")

def main():
    """Main function"""
    
    logger.info("⚡ Fast local training mode")
    
    # Choose training mode
    print("\n🚀 Choose fast training mode:")
    print("1. 🔥 Ultra-fast mode (2000 samples, ~30 minutes)")
    print("2. ⚡ Fast mode (5000 samples, ~60 minutes)")  
    print("3. 🏃 Standard mode (all samples, ~90 minutes)")
    
    try:
        choice = input("Please choose (1-3, default 2): ").strip() or "2"
    except:
        choice = "2"  # Default choice for background running
    
    if choice == "1":
        max_samples = 2000
        epochs = 5
        logger.info("🔥 Ultra-fast mode: 2000 samples, estimated 30 minutes")
    elif choice == "3":
        max_samples = None
        epochs = 6
        logger.info("🏃 Standard mode: all samples, estimated 90 minutes")
    else:
        max_samples = 5000
        epochs = 6
        logger.info("⚡ Fast mode: 5000 samples, estimated 60 minutes")
    
    # Fast training configuration
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
    
    # Start training
    trainer.train()

if __name__ == "__main__":
    main()

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
5K模型性能测试脚本
"""

import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import os
import json
from PIL import Image
import logging

# 导入模型类
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from train_fast_local import LightweightGenderBiasModel, FastInstagramDataset

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei']
plt.rcParams['axes.unicode_minus'] = False

class Model5KEvaluator:
    def __init__(self, model_path, csv_file, database_path):
        self.model_path = model_path
        self.csv_file = csv_file
        self.database_path = database_path
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
        
    def load_model(self):
        """加载训练好的模型"""
        logger.info(f"加载5K模型: {self.model_path}")
        
        # 创建模型
        self.model = LightweightGenderBiasModel()
        
        # 加载权重
        checkpoint = torch.load(self.model_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.device)
        self.model.eval()
        
        logger.info("✅ 5K模型加载成功")
        
    def prepare_test_data(self, test_size=1000):
        """准备测试数据"""
        logger.info(f"准备测试数据 ({test_size}样本)")
        
        # 导入tokenizer
        from transformers import DistilBertTokenizer
        tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
        
        # 创建测试数据集
        dataset = FastInstagramDataset(
            csv_file=self.csv_file,
            database_path=self.database_path,
            tokenizer=tokenizer,
            max_samples=test_size,
            max_length=64
        )
        
        # 创建数据加载器
        test_loader = torch.utils.data.DataLoader(
            dataset, 
            batch_size=16, 
            shuffle=False, 
            num_workers=0
        )
        
        return test_loader, dataset
        
    def evaluate_model(self, test_size=1000):
        """评估模型性能"""
        logger.info("🔍 开始模型评估")
        
        # 准备数据
        test_loader, dataset = self.prepare_test_data(test_size)
        
        predictions = []
        targets = []
        
        logger.info(f"在{test_size}个样本上评估模型...")
        
        with torch.no_grad():
            for batch_idx, (images, input_ids, attention_mask, scores) in enumerate(test_loader):
                images = images.to(self.device)
                input_ids = input_ids.to(self.device)
                attention_mask = attention_mask.to(self.device)
                scores = scores.to(self.device)
                
                # 预测
                outputs = self.model(images, input_ids, attention_mask)
                
                # 反归一化到0-10范围
                pred_scores = outputs.cpu().numpy() * 10.0
                true_scores = scores.cpu().numpy() * 10.0
                
                predictions.extend(pred_scores.flatten())
                targets.extend(true_scores.flatten())
                
                if (batch_idx + 1) % 20 == 0:
                    logger.info(f"  已处理 {(batch_idx + 1) * 16} / {test_size} 样本")
        
        predictions = np.array(predictions)
        targets = np.array(targets)
        
        # 计算指标
        mae = mean_absolute_error(targets, predictions)
        rmse = np.sqrt(mean_squared_error(targets, predictions))
        r2 = r2_score(targets, predictions)
        
        # 计算准确率（误差<=1, <=2的比例）
        errors = np.abs(predictions - targets)
        acc_1 = np.mean(errors <= 1.0) * 100
        acc_2 = np.mean(errors <= 2.0) * 100
        
        logger.info("📊 5K模型评估结果:")
        logger.info(f"  📈 MAE (平均绝对误差): {mae:.4f}")
        logger.info(f"  📈 RMSE (均方根误差): {rmse:.4f}")
        logger.info(f"  📈 R² (决定系数): {r2:.4f}")
        logger.info(f"  🎯 误差≤1分准确率: {acc_1:.2f}%")
        logger.info(f"  🎯 误差≤2分准确率: {acc_2:.2f}%")
        
        return {
            'predictions': predictions,
            'targets': targets,
            'mae': mae,
            'rmse': rmse,
            'r2': r2,
            'acc_1': acc_1,
            'acc_2': acc_2
        }
    
    def create_evaluation_plots(self, results):
        """创建评估图表"""
        logger.info("📊 生成评估图表")
        
        predictions = results['predictions']
        targets = results['targets']
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('5K样本模型性能评估', fontsize=16, fontweight='bold')
        
        # 1. 预测vs真实值散点图
        ax1 = axes[0, 0]
        ax1.scatter(targets, predictions, alpha=0.5, s=20)
        ax1.plot([0, 10], [0, 10], 'r--', linewidth=2)
        ax1.set_xlabel('真实分数')
        ax1.set_ylabel('预测分数')
        ax1.set_title(f'预测 vs 真实值\nMAE: {results["mae"]:.3f}, R²: {results["r2"]:.3f}')
        ax1.grid(True, alpha=0.3)
        
        # 2. 误差分布直方图
        ax2 = axes[0, 1]
        errors = np.abs(predictions - targets)
        ax2.hist(errors, bins=50, alpha=0.7, color='skyblue', edgecolor='black')
        ax2.axvline(results['mae'], color='red', linestyle='--', label=f'MAE: {results["mae"]:.3f}')
        ax2.set_xlabel('绝对误差')
        ax2.set_ylabel('频次')
        ax2.set_title('误差分布')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 3. 残差图
        ax3 = axes[1, 0]
        residuals = predictions - targets
        ax3.scatter(targets, residuals, alpha=0.5, s=20)
        ax3.axhline(y=0, color='r', linestyle='--')
        ax3.set_xlabel('真实分数')
        ax3.set_ylabel('残差 (预测-真实)')
        ax3.set_title('残差分析')
        ax3.grid(True, alpha=0.3)
        
        # 4. 分数区间准确率
        ax4 = axes[1, 1]
        score_ranges = [(0, 2), (2, 4), (4, 6), (6, 8), (8, 10)]
        range_accs = []
        range_labels = []
        
        for low, high in score_ranges:
            mask = (targets >= low) & (targets < high)
            if np.sum(mask) > 0:
                range_errors = errors[mask]
                acc = np.mean(range_errors <= 1.0) * 100
                range_accs.append(acc)
                range_labels.append(f'{low}-{high}')
        
        bars = ax4.bar(range_labels, range_accs, color='lightcoral', alpha=0.7)
        ax4.set_xlabel('分数区间')
        ax4.set_ylabel('准确率 (%)')
        ax4.set_title('各分数区间准确率 (误差≤1)')
        ax4.grid(True, alpha=0.3)
        
        # 添加数值标签
        for bar, acc in zip(bars, range_accs):
            height = bar.get_height()
            ax4.text(bar.get_x() + bar.get_width()/2., height + 1,
                    f'{acc:.1f}%', ha='center', va='bottom')
        
        plt.tight_layout()
        
        # 保存图表
        plot_path = 'model_5k_evaluation.png'
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        logger.info(f"📊 评估图表已保存: {plot_path}")
        plt.show()
        
    def compare_with_2k_model(self):
        """与2K模型结果对比"""
        logger.info("📊 与2K模型对比")
        
        # 2K模型的结果 (从之前的训练中获取)
        model_2k_results = {
            'mae': 1.31,
            'r2': 0.38,
            'acc_1': 45.2,
            'acc_2': 80.4,
            'training_time': 13.8  # 分钟
        }
        
        # 5K模型的结果
        results_5k = self.evaluate_model(test_size=1000)
        model_5k_results = {
            'mae': results_5k['mae'],
            'r2': results_5k['r2'],
            'acc_1': results_5k['acc_1'],
            'acc_2': results_5k['acc_2'],
            'training_time': 17.5  # 分钟
        }
        
        # 打印对比
        print("\n" + "="*60)
        print("📊 2K vs 5K 模型性能对比")
        print("="*60)
        print(f"{'指标':<15} {'2K模型':<12} {'5K模型':<12} {'改善':<12}")
        print("-" * 60)
        
        # MAE对比
        mae_improvement = ((model_2k_results['mae'] - model_5k_results['mae']) / model_2k_results['mae']) * 100
        print(f"{'MAE':<15} {model_2k_results['mae']:<12.3f} {model_5k_results['mae']:<12.3f} {mae_improvement:+.1f}%")
        
        # R²对比
        r2_improvement = ((model_5k_results['r2'] - model_2k_results['r2']) / model_2k_results['r2']) * 100
        print(f"{'R²':<15} {model_2k_results['r2']:<12.3f} {model_5k_results['r2']:<12.3f} {r2_improvement:+.1f}%")
        
        # 准确率对比
        acc1_improvement = model_5k_results['acc_1'] - model_2k_results['acc_1']
        print(f"{'准确率≤1':<15} {model_2k_results['acc_1']:<12.1f}% {model_5k_results['acc_1']:<12.1f}% {acc1_improvement:+.1f}%")
        
        acc2_improvement = model_5k_results['acc_2'] - model_2k_results['acc_2']
        print(f"{'准确率≤2':<15} {model_2k_results['acc_2']:<12.1f}% {model_5k_results['acc_2']:<12.1f}% {acc2_improvement:+.1f}%")
        
        # 训练时间对比
        time_ratio = model_5k_results['training_time'] / model_2k_results['training_time']
        print(f"{'训练时间':<15} {model_2k_results['training_time']:<12.1f}min {model_5k_results['training_time']:<12.1f}min {time_ratio:.1f}x")
        
        print("="*60)
        
        return results_5k

def main():
    """主函数"""
    logger.info("🚀 开始5K模型性能测试")
    
    # 配置路径
    model_path = '/Users/huangxinyue/Multi model distillation/fast_models_5k/fast_best_model.pth'
    csv_file = '/Users/huangxinyue/Multi model distillation/train_10k_results/train_10k_fast_results.csv'
    database_path = '/Users/huangxinyue/Downloads/Influencer brand database'
    
    # 检查文件是否存在
    if not os.path.exists(model_path):
        logger.error(f"❌ 模型文件不存在: {model_path}")
        return
    
    # 创建评估器
    evaluator = Model5KEvaluator(model_path, csv_file, database_path)
    
    # 加载模型
    evaluator.load_model()
    
    # 与2K模型对比评估
    results = evaluator.compare_with_2k_model()
    
    # 生成评估图表
    evaluator.create_evaluation_plots(results)
    
    logger.info("✅ 5K模型评估完成")

if __name__ == "__main__":
    main()

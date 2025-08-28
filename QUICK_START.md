# Quick Start Guide

## 🎯 Three-Step Process

### Step 1: Data Annotation (15K samples)
```bash
cd data_annotation
python batch_scoring_api.py
```
- Uses Qwen-VL 2.5 API to score Instagram posts
- Generates training labels (gender bias scores 0-10)
- Takes ~45 days for 15K samples (with API rate limits)

### Step 2: Model Training (30 minutes)
```bash
cd model_training  
python train_fast_local.py
```
- Trains ResNet18 + DistilBERT student model
- Learns to mimic Qwen-VL predictions
- Runs on MacBook Air M3 in 30-45 minutes

### Step 3: Large-Scale Inference (800 posts/hour)
```bash
cd deployment_inference
python deploy_full_inference.py
```
- Applies trained model to large datasets
- Processes 800 posts/hour on consumer hardware
- Generates gender bias scores for each post

## 📊 Expected Results

- **Training accuracy**: MAE ~1.18 (vs teacher model MAE 0.85)
- **Processing speed**: 800× faster than API calls
- **Scale**: Successfully tested on 1.6M posts
- **Hardware**: Works on regular laptops (16GB RAM recommended)

## 🚀 One-Command Demo

```bash
# Run complete pipeline on sample data
python batch_scoring_api.py && python train_fast_local.py && python deploy_full_inference.py
```

## 🔧 Configuration

Edit these key parameters in scripts:
- **API key**: Update in `batch_scoring_api.py`
- **Data paths**: Adjust file paths for your dataset
- **Batch size**: Reduce if memory constrained
- **Sample size**: Start with small batches for testing

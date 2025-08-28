#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Hardware Compatibility Checker for Multi-Modal Gender Bias Detection System
检查硬件兼容性，为不同平台提供优化建议
"""

import torch
import platform
import psutil
import sys
from pathlib import Path

def check_system_info():
    """检查系统基本信息"""
    print("🖥️  System Information")
    print("=" * 50)
    
    system = platform.system()
    processor = platform.processor()
    python_version = platform.python_version()
    
    print(f"Operating System: {system} {platform.release()}")
    print(f"Processor: {processor}")
    print(f"Python Version: {python_version}")
    
    # CPU信息
    cpu_count_physical = psutil.cpu_count(logical=False)
    cpu_count_logical = psutil.cpu_count(logical=True)
    print(f"CPU Cores: {cpu_count_physical} physical, {cpu_count_logical} logical")
    
    # 内存信息
    memory_gb = psutil.virtual_memory().total / (1024**3)
    print(f"Total Memory: {memory_gb:.1f} GB")
    
    return {
        'system': system,
        'processor': processor,
        'memory_gb': memory_gb,
        'cpu_cores': cpu_count_physical,
        'python_version': python_version
    }

def check_pytorch_environment():
    """检查PyTorch环境"""
    print("\n🔥 PyTorch Environment")
    print("=" * 50)
    
    pytorch_version = torch.__version__
    print(f"PyTorch Version: {pytorch_version}")
    
    # 检查CUDA
    cuda_available = torch.cuda.is_available()
    print(f"CUDA Available: {cuda_available}")
    
    if cuda_available:
        cuda_version = torch.version.cuda
        gpu_count = torch.cuda.device_count()
        print(f"CUDA Version: {cuda_version}")
        print(f"GPU Count: {gpu_count}")
        
        for i in range(gpu_count):
            gpu_name = torch.cuda.get_device_name(i)
            props = torch.cuda.get_device_properties(i)
            vram_gb = props.total_memory / (1024**3)
            print(f"GPU {i}: {gpu_name} ({vram_gb:.1f} GB VRAM)")
    
    # 检查MPS (Apple Silicon)
    mps_available = hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()
    print(f"MPS (Apple Silicon) Available: {mps_available}")
    
    if mps_available:
        print("✅ Apple Silicon GPU acceleration supported")
    
    return {
        'pytorch_version': pytorch_version,
        'cuda_available': cuda_available,
        'mps_available': mps_available,
        'gpu_count': torch.cuda.device_count() if cuda_available else 0
    }

def get_recommended_device():
    """获取推荐的设备配置"""
    if torch.cuda.is_available():
        return "cuda", "NVIDIA GPU"
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return "mps", "Apple Silicon GPU"
    else:
        return "cpu", "CPU"

def check_memory_requirements():
    """检查内存要求"""
    print("\n💾 Memory Requirements Analysis")
    print("=" * 50)
    
    memory_gb = psutil.virtual_memory().total / (1024**3)
    available_gb = psutil.virtual_memory().available / (1024**3)
    
    print(f"Total Memory: {memory_gb:.1f} GB")
    print(f"Available Memory: {available_gb:.1f} GB")
    
    # 内存要求评估
    if memory_gb >= 16:
        print("✅ Memory: Excellent (16GB+) - Full training supported")
        recommended_batch_size = "32-64"
    elif memory_gb >= 8:
        print("⚠️  Memory: Good (8-15GB) - Training with reduced batch size")
        recommended_batch_size = "8-16"
    else:
        print("❌ Memory: Insufficient (<8GB) - Consider using Google Colab")
        recommended_batch_size = "4-8"
    
    print(f"Recommended Batch Size: {recommended_batch_size}")
    
    return memory_gb, recommended_batch_size

def check_dependencies():
    """检查关键依赖"""
    print("\n📦 Dependency Check")
    print("=" * 50)
    
    dependencies = {
        'torch': '2.0+',
        'torchvision': '0.15+',
        'transformers': '4.30+',
        'timm': '0.9+',
        'pandas': '2.0+',
        'numpy': '1.24+',
        'PIL': '9.5+',
        'sklearn': '1.3+',
        'matplotlib': '3.7+',
        'seaborn': '0.12+',
        'tqdm': '4.65+'
    }
    
    missing_deps = []
    
    for dep, min_version in dependencies.items():
        try:
            if dep == 'PIL':
                import PIL
                version = PIL.__version__
                module_name = 'Pillow'
            elif dep == 'sklearn':
                import sklearn
                version = sklearn.__version__
                module_name = 'scikit-learn'
            else:
                module = __import__(dep)
                version = module.__version__
                module_name = dep
            
            print(f"✅ {module_name}: {version}")
        except ImportError:
            print(f"❌ {dep}: Not installed")
            missing_deps.append(dep)
        except AttributeError:
            print(f"⚠️  {dep}: Installed (version unknown)")
    
    return missing_deps

def generate_setup_recommendations(system_info, pytorch_info, memory_gb, missing_deps):
    """生成设置建议"""
    print("\n🎯 Setup Recommendations")
    print("=" * 50)
    
    device, device_type = get_recommended_device()
    
    print(f"Recommended Device: {device} ({device_type})")
    
    # 基于硬件的配置建议
    if system_info['system'] == 'Darwin' and 'arm' in system_info['processor'].lower():
        print("\n🍎 Apple Silicon Specific Recommendations:")
        print("- Use MPS backend for GPU acceleration")
        print("- Install PyTorch with: pip install torch torchvision")
        print("- Unified memory architecture works well for multi-modal training")
        print("- Recommended batch size: 8-16")
        
    elif pytorch_info['cuda_available']:
        gpu_count = pytorch_info['gpu_count']
        print(f"\n🚀 NVIDIA GPU Recommendations (Found {gpu_count} GPU(s)):")
        print("- Use CUDA backend for maximum performance")
        print("- Enable mixed precision training for memory efficiency")
        print("- Recommended batch size: 32-64")
        
    else:
        print("\n💻 CPU-Only Recommendations:")
        print("- Training will be slower but still feasible")
        print("- Consider using Google Colab for faster training")
        print("- Recommended batch size: 4-8")
        print("- Enable multi-threading with proper num_workers setting")
    
    # 内存优化建议
    print(f"\n💾 Memory Optimization:")
    if memory_gb < 8:
        print("- Use gradient checkpointing to reduce memory usage")
        print("- Consider training on Google Colab (free 12GB RAM)")
    elif memory_gb < 16:
        print("- Reduce batch size if encountering OOM errors")
        print("- Monitor memory usage during training")
    else:
        print("- Memory is sufficient for full training")
    
    # 依赖安装建议
    if missing_deps:
        print(f"\n📦 Install Missing Dependencies:")
        print(f"pip install {' '.join(missing_deps)}")

def performance_estimation(system_info, pytorch_info, memory_gb):
    """估算性能"""
    print("\n⏱️  Performance Estimation")
    print("=" * 50)
    
    # 基于硬件估算训练时间
    if pytorch_info['cuda_available']:
        estimated_time = "30-45 minutes"
        throughput = "800-1200 posts/hour"
    elif pytorch_info['mps_available']:
        estimated_time = "45-60 minutes"
        throughput = "500-800 posts/hour"
    else:
        estimated_time = "2-4 hours"
        throughput = "100-300 posts/hour"
    
    print(f"Estimated Training Time: {estimated_time}")
    print(f"Estimated Inference Throughput: {throughput}")
    
    # 实际测试建议
    print(f"\n🧪 Quick Performance Test:")
    print("Run: python test_training_setup.py")
    print("This will verify your environment and measure actual performance")

def main():
    """主函数"""
    print("🔍 Multi-Modal Gender Bias Detection System")
    print("Hardware Compatibility Checker")
    print("=" * 60)
    
    # 系统信息检查
    system_info = check_system_info()
    
    # PyTorch环境检查
    pytorch_info = check_pytorch_environment()
    
    # 内存要求检查
    memory_gb, recommended_batch_size = check_memory_requirements()
    
    # 依赖检查
    missing_deps = check_dependencies()
    
    # 生成建议
    generate_setup_recommendations(system_info, pytorch_info, memory_gb, missing_deps)
    
    # 性能估算
    performance_estimation(system_info, pytorch_info, memory_gb)
    
    print("\n" + "=" * 60)
    print("✅ Hardware compatibility check completed!")
    print("\nNext steps:")
    print("1. Install any missing dependencies")
    print("2. Run: python test_training_setup.py")
    print("3. Start training: python train_gender_bias_model.py")
    print("\nFor detailed setup instructions, see: README.md")

if __name__ == "__main__":
    main()

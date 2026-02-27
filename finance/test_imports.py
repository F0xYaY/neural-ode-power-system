#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""测试所有包的导入"""
import sys

print("=" * 60)
print("测试包导入")
print("=" * 60)

packages = {
    'torch': 'PyTorch',
    'torchsde': 'torchsde',
    'akshare': 'akshare',
    'pandas': 'pandas',
    'matplotlib': 'matplotlib',
    'sklearn': 'scikit-learn',
    'seaborn': 'seaborn',
    'numpy': 'numpy'
}

success_count = 0
for module_name, display_name in packages.items():
    try:
        module = __import__(module_name)
        version = getattr(module, '__version__', 'unknown')
        print(f"[OK] {display_name:20s} - 版本: {version}")
        success_count += 1
    except ImportError as e:
        print(f"[FAIL] {display_name:20s} - 导入失败: {e}")

print("=" * 60)
print(f"成功导入 {success_count}/{len(packages)} 个包")

# 测试 PyTorch CUDA
try:
    import torch
    print(f"\nPyTorch CUDA 可用: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA 设备数量: {torch.cuda.device_count()}")
        print(f"当前设备: {torch.cuda.get_device_name(0)}")
except:
    pass

print("=" * 60)

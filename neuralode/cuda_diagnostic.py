import torch

print("=" * 60)
print("CUDA 诊断信息")
print("=" * 60)
print(f"PyTorch 版本: {torch.__version__}")
print(f"CUDA 版本: {torch.version.cuda if torch.version.cuda else 'N/A (CPU only build)'}")
print(f"cuDNN 版本: {torch.backends.cudnn.version() if torch.backends.cudnn.is_available() else 'N/A'}")
print(f"CUDA 是否可用: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    print(f"CUDA 设备数量: {torch.cuda.device_count()}")
    for i in range(torch.cuda.device_count()):
        print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
        props = torch.cuda.get_device_properties(i)
        print(f"    总内存: {props.total_memory / 1e9:.2f} GB")
        print(f"    计算能力: {props.major}.{props.minor}")
    # 显式设置设备
    torch.cuda.set_device(0)
    print(f"当前 CUDA 设备: {torch.cuda.current_device()}")
    
    # 测试 GPU 计算
    try:
        x = torch.randn(10, 10).cuda()
        y = x @ x.T
        print("✓ GPU 计算测试成功")
    except Exception as e:
        print(f"✗ GPU 计算测试失败: {e}")
else:
    print("警告: CUDA 不可用，将使用 CPU 训练")
    print("可能的原因:")
    print("  1. PyTorch 安装的是 CPU 版本（需要重新安装 CUDA 版本）")
    print("  2. CUDA 驱动版本不匹配")
    print("  3. NVIDIA 驱动未安装或版本过旧")
    print("\n解决方案:")
    print("  - 安装 CUDA 版本的 PyTorch:")
    print("    conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia")
    print("    或")
    print("    pip install torch --index-url https://download.pytorch.org/whl/cu118")
print("=" * 60)

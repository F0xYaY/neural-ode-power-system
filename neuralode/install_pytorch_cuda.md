# 安装 CUDA 版本的 PyTorch

## 诊断结果
- 当前 PyTorch: 2.9.1+cpu (CPU only)
- 需要安装: CUDA 版本的 PyTorch

## 安装方法

### 方法 1: 使用 Conda (推荐)

```bash
# 卸载现有的 CPU 版本
conda uninstall pytorch torchvision torchaudio

# 安装 CUDA 11.8 版本 (适用于 RTX 3070)
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia

# 或安装 CUDA 12.1 版本
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
```

### 方法 2: 使用 pip

```bash
# CUDA 11.8
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# CUDA 12.1
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

### 验证安装

运行以下 Python 代码验证：

```python
import torch
print(f"PyTorch 版本: {torch.__version__}")
print(f"CUDA 可用: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
```

## 注意事项

1. RTX 3070 支持 CUDA 11.x 和 12.x
2. 确保 NVIDIA 驱动已正确安装
3. 安装后重启 Python 环境或终端

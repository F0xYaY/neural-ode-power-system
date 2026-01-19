# Neural ODE + Lyapunov Stability Analysis for Power Systems

基于 Neural ODE 和 Lyapunov 理论的电力系统稳定性分析项目。

## 项目结构

```
neuralode/
├── main.py              # 主程序入口
├── model.py             # 模型定义（DynamicsNetwork, LyapunovNet）
├── train.py             # 训练循环
├── utils.py             # 数据生成工具
└── plot_utils.py        # 可视化函数
```

## 功能特性

- **Neural ODE**: 使用神经网络学习电力系统的向量场
- **Lyapunov 稳定性分析**: 通过 Lyapunov 函数证明系统稳定性
- **GPU 加速**: 支持 CUDA 加速训练
- **固定步长求解器**: 使用 RK4 方法，保证训练速度稳定

## 环境要求

- Python 3.8+
- PyTorch (CUDA 版本推荐)
- torchdiffeq
- matplotlib
- numpy

## 安装依赖

```bash
pip install torch torchdiffeq numpy matplotlib
```

## 使用方法

```bash
cd neuralode
python main.py
```

## 训练参数

- Batch Size: 2048
- 时间窗口: 4 秒
- 训练轮数: 1200 epochs
- 固定步长: 0.1

## 结果

- MSE: ~0.003 (接近 1e-4 目标)
- GPU: RTX 3070 支持
- 训练速度: 显著提升（固定步长 + 大 batch size）

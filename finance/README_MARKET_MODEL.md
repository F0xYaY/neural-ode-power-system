# 市场动力学模型说明文档

## 模型概述

本模型基于**捕食者-被捕食者模型**（Lotka-Volterra模型）来模拟股票市场中买卖双方的动态博弈关系。

### 核心思想

- **卖单存量 (S)**: 市场流动性，相当于被捕食者（Prey）
- **买盘力度 (B)**: 买方资金，相当于捕食者（Predator）
- **恐惧因子 (k)**: 关键参数，影响成交效率

### 数学模型

```
dS/dt = r * S * (1 - S/K) - (β * S * B) / (1 + k * B)
dB/dt = c * (β * S * B) / (1 + k * B) - d * B
```

其中：
- `r`: 卖单自然增长率
- `K`: 卖单容量上限
- `β`: 成交效率系数
- `k`: **恐惧因子**（核心参数）
- `c`: 买盘转化效率
- `d`: 买盘衰减率

### 价格生成机制

价格变化由供需失衡驱动：

```
价格变化 = α * log(B/S) + 随机噪声
```

其中 `α` 是价格敏感度系数。

## 使用方法

### 基本使用

```python
from market_dynamics_model import run_simulation, compare_market_scenarios

# 运行对比实验
compare_market_scenarios(days=200, save_path='comparison.png')
```

### 自定义模拟

```python
# 运行单次模拟
S, B, P, I = run_simulation(
    days=200,              # 模拟天数
    fear_factor=0.05,      # 恐惧因子
    initial_price=100.0,   # 初始价格
    price_sensitivity=0.5, # 价格敏感度
    noise_level=0.5,       # 噪声水平
    seed=42                # 随机种子
)
```

### 参数敏感性分析

```python
from market_dynamics_model import parameter_sensitivity_analysis

parameter_sensitivity_analysis()
```

## 模型参数说明

### 恐惧因子 (k) 的影响

| 恐惧因子 | 市场特征 | 价格表现 |
|---------|---------|---------|
| k < 0.05 | 正常市场 | 健康波动，趋势向上 |
| 0.05 < k < 0.2 | 中等恐惧 | 波动增加，趋势减弱 |
| k > 0.2 | 恐惧市场 | 流动性枯竭，价格异常波动或阴跌 |

### 关键发现

1. **恐惧因子越大，流动性越低**
   - 卖家惜售，卖单存量 S 积累
   - 成交效率下降

2. **恐惧市场中的价格表现**
   - 涨幅明显低于正常市场
   - 波动率可能降低（因为成交减少）
   - 可能出现流动性危机

3. **正常市场的特征**
   - 流动性充足
   - 价格健康波动
   - 趋势向上

## 模型应用场景

1. **市场情绪分析**: 通过观察价格和成交量模式，推断市场恐惧水平
2. **流动性风险评估**: 预测流动性枯竭的可能性
3. **交易策略优化**: 根据市场恐惧水平调整交易策略
4. **风险管理**: 识别高风险市场环境

## 扩展功能

### 1. 添加外部冲击

可以在模拟中加入外部事件（如政策变化、突发事件）：

```python
def run_simulation_with_shock(days=200, shock_day=100, shock_magnitude=0.5):
    # 在 shock_day 时点加入冲击
    # shock_magnitude 影响恐惧因子的变化
    pass
```

### 2. 多资产模型

扩展为多只股票的关联模型：

```python
def multi_asset_model(stock_codes, correlation_matrix):
    # 考虑股票之间的相关性
    pass
```

### 3. 实时数据拟合

使用真实市场数据拟合模型参数：

```python
def fit_parameters(real_price_data, real_volume_data):
    # 使用优化算法找到最佳参数
    pass
```

## 注意事项

1. **模型假设**
   - 市场是封闭的（不考虑外部资金流入）
   - 参数保持不变（实际中参数会变化）
   - 价格完全由供需决定（忽略其他因素）

2. **参数校准**
   - 需要根据实际市场数据调整参数
   - 不同市场可能需要不同的参数设置

3. **模型局限性**
   - 简化了复杂的市场机制
   - 不能完全预测真实市场
   - 主要用于理解市场动态和风险分析

## 参考文献

- Lotka-Volterra 模型
- 市场微观结构理论
- 流动性风险模型

## 文件说明

- `market_dynamics_model.py`: 主模型文件
- `market_dynamics_comparison.png`: 对比实验图表
- `fear_factor_sensitivity.png`: 参数敏感性分析图表

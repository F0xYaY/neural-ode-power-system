#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
市场动力学模型 - 基于捕食者-被捕食者模型
模拟股票市场中买卖双方的动态博弈，分析恐惧因子对价格的影响
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams

# 设置中文字体
rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
rcParams['axes.unicode_minus'] = False

# ---------------------------------------------------------
# 1. 定义核心动力学方程 (ODE)
# ---------------------------------------------------------
def step_dynamics(S, B, params, dt=1.0):
    """
    市场动力学一步更新
    
    参数:
        S: 当前卖单存量 (Liquidity / Prey) - 市场流动性
        B: 当前买盘力度 (Capital / Predator) - 买方资金
        params: 包含 r, K, beta, k, c, d
            r: 卖单自然增长率
            K: 卖单容量上限
            beta: 成交效率系数
            k: 恐惧因子（越大，成交效率越低）
            c: 买盘转化效率
            d: 买盘衰减率
        dt: 时间步长
    
    返回:
        S_new, B_new: 更新后的卖单存量和买盘力度
    """
    r, K, beta, k, c, d = params
    
    # 核心机制：带恐惧效应的捕食
    # 恐惧因子 k 越大，成交效率越低（卖家惜售）
    interaction = (beta * S * B) / (1 + k * B)
    
    # 微分方程的离散化 (Euler Method)
    # dS/dt = 自然增长 - 被吃掉（成交）
    dS = r * S * (1 - S / K) - interaction
    
    # dB/dt = 吃到肉变强（成交获利） - 资金消耗
    dB = c * interaction - d * B
    
    # 更新状态
    S_new = S + dS * dt
    B_new = B + dB * dt
    
    # 保护：不能为负
    return max(S_new, 0.1), max(B_new, 0.1)


# ---------------------------------------------------------
# 2. 模拟实战：价格生成机制
# ---------------------------------------------------------
def run_simulation(days=200, fear_factor=0.05, initial_price=100.0, 
                  price_sensitivity=0.5, noise_level=0.5, seed=None):
    """
    运行市场模拟
    
    参数:
        days: 模拟天数
        fear_factor: 恐惧因子 k（越大越恐惧）
        initial_price: 初始股价
        price_sensitivity: 价格对供需失衡的敏感度
        noise_level: 价格随机波动水平
        seed: 随机种子（用于可重复性）
    
    返回:
        history_S: 卖单存量历史
        history_B: 买盘力度历史
        history_P: 价格历史
    """
    if seed is not None:
        np.random.seed(seed)
    
    # 初始化
    S = 50.0  # 初始卖单深度
    B = 10.0  # 初始买盘力度
    Price = initial_price
    
    # 记录历史
    history_S = []
    history_B = []
    history_P = []
    history_imbalance = []
    
    # 模型参数
    # r: 卖单自然增长率
    # K: 卖单容量上限
    # beta: 成交效率系数
    # k: 恐惧因子（核心参数）
    # c: 买盘转化效率
    # d: 买盘衰减率
    params = (0.5, 100, 0.2, fear_factor, 0.6, 0.15)
    
    for t in range(days):
        # 1. 预测下一时刻的供需状态
        S, B = step_dynamics(S, B, params)
        
        # 2. 根据供需失衡计算价格变化
        # 逻辑：买盘 B 越强 -> 涨; 卖单 S 越厚 -> 跌
        # 引入对数供需比 (Log Imbalance)
        imbalance = np.log(B / S)
        
        # 价格随机游走 + 供需驱动
        noise = np.random.normal(0, noise_level)
        price_change = price_sensitivity * imbalance + noise
        Price = Price + price_change
        
        # 保护：价格不能为负
        Price = max(Price, 0.1)
        
        history_S.append(S)
        history_B.append(B)
        history_P.append(Price)
        history_imbalance.append(imbalance)
        
    return history_S, history_B, history_P, history_imbalance


# ---------------------------------------------------------
# 3. 对比实验：正常市场 vs 恐惧市场
# ---------------------------------------------------------
def compare_market_scenarios(days=200, save_path=None):
    """
    对比不同恐惧水平下的市场表现
    
    参数:
        days: 模拟天数
        save_path: 保存路径（可选）
    """
    # 场景 A: 正常博弈 (k=0.01) -> 股价应该是健康的波动上升
    print("模拟正常市场（低恐惧因子）...")
    S_norm, B_norm, P_norm, I_norm = run_simulation(days, fear_factor=0.01, seed=42)
    
    # 场景 B: 恐惧市场 (k=0.5) -> 卖家惜售，流动性枯竭，价格可能出现异常波动或阴跌
    print("模拟恐惧市场（高恐惧因子）...")
    S_fear, B_fear, P_fear, I_fear = run_simulation(days, fear_factor=0.5, seed=42)
    
    # 场景 C: 中等恐惧
    print("模拟中等恐惧市场...")
    S_mid, B_mid, P_mid, I_mid = run_simulation(days, fear_factor=0.1, seed=42)
    
    # 绘图
    fig = plt.figure(figsize=(14, 10))
    
    # 1. 价格走势对比
    ax1 = plt.subplot(3, 1, 1)
    ax1.plot(P_norm, label='正常市场 (k=0.01)', color='green', linewidth=2)
    ax1.plot(P_mid, label='中等恐惧 (k=0.1)', color='orange', linewidth=2)
    ax1.plot(P_fear, label='恐惧市场 (k=0.5)', color='red', linestyle='--', linewidth=2)
    ax1.set_title('市场动力学模型：恐惧因子对股价走势的影响', fontsize=14, fontweight='bold')
    ax1.set_ylabel('股价', fontsize=12)
    ax1.legend(loc='best')
    ax1.grid(True, alpha=0.3)
    
    # 2. 潜在流动性对比 (Model Hidden State)
    ax2 = plt.subplot(3, 1, 2)
    ax2.plot(S_norm, label='流动性 S (正常市场)', color='green', alpha=0.6)
    ax2.plot(S_mid, label='流动性 S (中等恐惧)', color='orange', alpha=0.6)
    ax2.plot(S_fear, label='流动性 S (恐惧市场)', color='red', alpha=0.6)
    ax2.set_title('隐藏状态：市场流动性（卖单深度）', fontsize=12)
    ax2.set_ylabel('可用卖单量 (S)', fontsize=12)
    ax2.legend(loc='best')
    ax2.grid(True, alpha=0.3)
    
    # 3. 买盘力度对比
    ax3 = plt.subplot(3, 1, 3)
    ax3.plot(B_norm, label='买盘力度 B (正常市场)', color='green', alpha=0.6)
    ax3.plot(B_mid, label='买盘力度 B (中等恐惧)', color='orange', alpha=0.6)
    ax3.plot(B_fear, label='买盘力度 B (恐惧市场)', color='red', alpha=0.6)
    ax3.set_title('隐藏状态：买盘力度（买方资金）', fontsize=12)
    ax3.set_xlabel('时间（天）', fontsize=12)
    ax3.set_ylabel('买盘力度 (B)', fontsize=12)
    ax3.legend(loc='best')
    ax3.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"\n图表已保存到: {save_path}")
    
    plt.show()
    
    # 打印统计信息
    print("\n" + "=" * 60)
    print("统计对比")
    print("=" * 60)
    print(f"正常市场:")
    print(f"  最终价格: {P_norm[-1]:.2f}, 涨幅: {(P_norm[-1]/P_norm[0]-1)*100:.2f}%")
    print(f"  平均流动性: {np.mean(S_norm):.2f}")
    print(f"  价格波动率: {np.std(P_norm):.2f}")
    
    print(f"\n中等恐惧市场:")
    print(f"  最终价格: {P_mid[-1]:.2f}, 涨幅: {(P_mid[-1]/P_mid[0]-1)*100:.2f}%")
    print(f"  平均流动性: {np.mean(S_mid):.2f}")
    print(f"  价格波动率: {np.std(P_mid):.2f}")
    
    print(f"\n恐惧市场:")
    print(f"  最终价格: {P_fear[-1]:.2f}, 涨幅: {(P_fear[-1]/P_fear[0]-1)*100:.2f}%")
    print(f"  平均流动性: {np.mean(S_fear):.2f}")
    print(f"  价格波动率: {np.std(P_fear):.2f}")


# ---------------------------------------------------------
# 4. 参数敏感性分析
# ---------------------------------------------------------
def parameter_sensitivity_analysis():
    """分析不同恐惧因子对最终价格的影响"""
    fear_factors = np.linspace(0.01, 1.0, 20)
    final_prices = []
    
    print("进行参数敏感性分析...")
    for k in fear_factors:
        _, _, P, _ = run_simulation(days=200, fear_factor=k, seed=42)
        final_prices.append(P[-1])
    
    plt.figure(figsize=(10, 6))
    plt.plot(fear_factors, final_prices, 'o-', linewidth=2, markersize=6)
    plt.xlabel('恐惧因子 k', fontsize=12)
    plt.ylabel('最终价格', fontsize=12)
    plt.title('参数敏感性分析：恐惧因子对最终价格的影响', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('fear_factor_sensitivity.png', dpi=300, bbox_inches='tight')
    print("敏感性分析图表已保存到: fear_factor_sensitivity.png")
    plt.show()


# ---------------------------------------------------------
# 5. 主函数
# ---------------------------------------------------------
if __name__ == "__main__":
    print("=" * 60)
    print("市场动力学模型 - 基于捕食者-被捕食者模型")
    print("=" * 60)
    
    # 运行对比实验
    compare_market_scenarios(days=200, save_path='market_dynamics_comparison.png')
    
    # 运行参数敏感性分析
    print("\n" + "=" * 60)
    parameter_sensitivity_analysis()

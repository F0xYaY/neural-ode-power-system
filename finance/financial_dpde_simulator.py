#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
金融偏微分方程模拟器 (Financial DDE Simulator)
结合扩散、反应项和分布时滞的金融市场空间-时间模型
"""
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import rcParams

# 设置中文字体
rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
rcParams['axes.unicode_minus'] = False

class FinancialDPDESimulator:
    def __init__(self, nx=100, L=10.0, T=50.0, dt=0.01):
        """
        初始化金融PDE模拟器
        
        参数:
            nx: 空间网格点数
            L: 空间域长度
            T: 模拟总时间
            dt: 时间步长
        """
        # 1. 空间与时间网格设置
        self.nx = nx
        self.L = L
        self.dx = L / (nx - 1)
        self.T = T
        self.dt = dt
        self.nt = int(T / dt)
        
        # 网格点
        self.x = np.linspace(0, L, nx)
        
        # 2. 模型参数 (金融物理学含义)
        # 扩散系数 (Diffusion)
        self.D_S = 0.01  # 流动性扩散慢 (卖单挂了通常不动)
        self.D_B = 0.5   # 资金扩散快 (热钱跑得快)
        
        # 反应项参数 (Reaction)
        self.r = 0.5     # 卖单再生率
        self.K = 10.0    # 市场容量
        self.beta = 0.8  # 基础成交效率
        self.c = 0.6     # 赚钱效应转化率
        self.d = 0.1     # 资金消耗/撤离率
        
        # 关键参数：恐惧与时滞
        self.k = 2.0     # 恐惧因子 (Fear Factor)
        self.alpha = 0.5 # 记忆衰减率 (分布时滞参数，值越小记忆越长)
        
        # 3. 罗宾边界条件参数 (Robin BCs)
        # h: 渗透率 (Permeability), ext: 外部储值
        self.h_S = 0.1   # 卖单与外部连通性
        self.S_ext = 5.0 # 外部市场的平均流动性
        
        self.h_B = 2.0   # 资金的高渗透性 (容易受外部放水影响)
        self.B_ext = 0.0 # 初始外部环境：无额外放水
        
        # 4. 初始化状态场
        self.S = np.ones(nx) * 5.0  # 初始流动性均匀
        self.B = np.zeros(nx)       # 初始资金
        self.W = np.zeros(nx)       # 初始恐惧记忆 (Memory/Fear)
        
        # 在中心注入一波初始资金 (模拟一次突发利好)
        center = int(nx / 2)
        self.B[center-5:center+5] = 8.0
        self.W = self.B.copy() # 初始记忆假设与当前一致
        
        # 用于存储历史数据绘图
        self.history_S = []
        self.history_B = []
        self.history_W = []

    def laplacian(self, U):
        """计算二阶导数 (Laplacian)，内部使用中心差分"""
        d2U = np.zeros_like(U)
        # 使用正确的索引范围
        d2U[1:-1] = (U[2:] - 2*U[1:-1] + U[0:-2]) / (self.dx**2)
        return d2U

    def apply_robin_bc(self, U, D, h, U_ext, d2U):
        """
        应用罗宾边界条件 (Robin BC)
        公式: -D * dU/dx = h * (U_ext - U)  (在左边界 x=0)
        利用幽灵点法 (Ghost Point Method) 修正边界处的拉普拉斯值
        """
        # --- 左边界 (x=0) ---
        # 离散化: -D * (U[1] - U[-1]) / (2*dx) = h * (U_ext - U[0])
        # 解出幽灵点 U[-1]
        ghost_left = U[1] + (2 * self.dx * h / D) * (U_ext - U[0])
        # 修正 d2U[0]
        d2U[0] = (U[1] - 2*U[0] + ghost_left) / (self.dx**2)
        
        # --- 右边界 (x=L) ---
        # 假设右边界是封闭的 (Neumann, flux=0) 或者也是 Robin
        # 这里演示右边界为 Neumann (dUdX=0)，模拟封闭系统
        ghost_right = U[-2] 
        d2U[-1] = (U[-2] - 2*U[-1] + ghost_right) / (self.dx**2)
        
        return d2U

    def step(self):
        """执行一个时间步长的欧拉积分"""
        # 1. 计算扩散项 (Diffusion)
        lap_S = self.laplacian(self.S)
        lap_B = self.laplacian(self.B)
        
        # 2. 应用边界条件修正扩散项
        lap_S = self.apply_robin_bc(self.S, self.D_S, self.h_S, self.S_ext, lap_S)
        
        # 动态改变外部环境：模拟 t=20 时央行突然放水 (B_ext 升高)
        current_time = len(self.history_S) * self.dt
        current_B_ext = 8.0 if current_time > 15.0 else 0.0
        lap_B = self.apply_robin_bc(self.B, self.D_B, self.h_B, current_B_ext, lap_B)

        # 3. 计算反应项 (Reaction with Distributed Delay)
        # 恐惧效应分母：使用辅助变量 W (记忆场) 代替 B
        fear_denominator = 1 + self.k * self.W 
        interaction = (self.beta * self.S * self.B) / fear_denominator
        
        # 4. 更新方程 (Euler Step)
        
        # dS/dt
        dS = self.D_S * lap_S + \
             self.r * self.S * (1 - self.S / self.K) - \
             interaction
             
        # dB/dt
        dB = self.D_B * lap_B + \
             self.c * interaction - \
             self.d * self.B
             
        # dW/dt (线性链技巧：分布时滞的核心)
        # 记忆场 W 追随 B，但有滞后 (alpha 控制追随速度)
        dW = self.alpha * (self.B - self.W)
        
        # 更新状态
        self.S += dS * self.dt
        self.B += dB * self.dt
        self.W += dW * self.dt
        
        # 简单截断，防止数值误差导致负值
        self.S = np.maximum(self.S, 0)
        self.B = np.maximum(self.B, 0)
        self.W = np.maximum(self.W, 0)

    def run(self):
        """运行完整模拟"""
        print(f"开始模拟: Grid={self.nx}, T={self.T}, dt={self.dt}")
        for i in range(self.nt):
            self.step()
            if i % 10 == 0: # 降采样存储
                self.history_S.append(self.S.copy())
                self.history_B.append(self.B.copy())
                self.history_W.append(self.W.copy())
        print("模拟完成。")

    def plot_results(self, save_path=None):
        """
        绘制结果
        
        参数:
            save_path: 保存路径（可选）
        """
        # 转换为 Numpy 数组方便绘图 [Time, Space]
        res_S = np.array(self.history_S)
        res_B = np.array(self.history_B)
        res_W = np.array(self.history_W)
        
        time_axis = np.linspace(0, self.T, len(res_S))
        space_axis = self.x
        
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        # 1. 卖方流动性 (Prey)
        sns.heatmap(res_S, ax=axes[0], cmap="RdYlGn", cbar_kws={'label': 'Ask Depth (S)'}, vmin=0)
        axes[0].set_title("流动性 (S): 因恐惧而枯竭", fontsize=12, fontweight='bold')
        axes[0].set_xlabel("空间 (市场/板块)")
        axes[0].set_ylabel("时间")
        # 调整坐标轴显示
        axes[0].set_xticks([0, self.nx-1])
        axes[0].set_xticklabels(['边界 (Robin)', '封闭'])
        axes[0].invert_yaxis()

        # 2. 买方资金 (Predator)
        sns.heatmap(res_B, ax=axes[1], cmap="Oranges", cbar_kws={'label': 'Capital (B)'})
        axes[1].set_title("资金 (B): 注入与扩散", fontsize=12, fontweight='bold')
        axes[1].set_xlabel("空间")
        axes[1].set_yticks([])
        axes[1].set_xticks([0, self.nx-1])
        axes[1].set_xticklabels(['流入 (央行)', ''])
        axes[1].invert_yaxis()

        # 3. 恐惧记忆场 (Memory) - 体现分布时滞
        sns.heatmap(res_W, ax=axes[2], cmap="Greys", cbar_kws={'label': 'Fear Memory (W)'})
        axes[2].set_title(f"恐惧记忆 (W): 分布时滞\n(alpha={self.alpha})", fontsize=12, fontweight='bold')
        axes[2].set_xlabel("空间")
        axes[2].set_yticks([])
        axes[2].set_xticks([0, self.nx-1])
        axes[2].invert_yaxis()

        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"热力图已保存到: {save_path}")
        
        plt.show()
        
        # 额外：画出某一点的时间序列，展示滞后效应
        mid_point = int(self.nx / 5) # 靠近左边界观测点
        plt.figure(figsize=(10, 4))
        plt.plot(time_axis, res_B[:, mid_point], label='实际资金 (B)', color='orange', linewidth=2)
        plt.plot(time_axis, res_W[:, mid_point], label='恐惧记忆 (W)', color='gray', linestyle='--', linewidth=2)
        plt.axvline(x=15, color='blue', linestyle=':', linewidth=2, label='外部注入开始')
        plt.title(f"时间序列 (x={mid_point*self.dx:.1f}): 滞后效应", fontsize=12, fontweight='bold')
        plt.xlabel("时间", fontsize=12)
        plt.ylabel("密度", fontsize=12)
        plt.legend(fontsize=10)
        plt.grid(True, alpha=0.3)
        
        if save_path:
            time_series_path = save_path.replace('.png', '_timeseries.png')
            plt.savefig(time_series_path, dpi=300, bbox_inches='tight')
            print(f"时间序列图已保存到: {time_series_path}")
        
        plt.show()


def run_simulation_example():
    """运行示例模拟"""
    print("=" * 60)
    print("金融偏微分方程模拟器")
    print("=" * 60)
    
    # 创建模拟器
    sim = FinancialDPDESimulator(nx=100, L=10.0, T=50.0, dt=0.01)
    
    # 运行模拟
    sim.run()
    
    # 绘制结果
    sim.plot_results(save_path='financial_dpde_results.png')
    
    print("\n模拟完成！")


if __name__ == "__main__":
    run_simulation_example()

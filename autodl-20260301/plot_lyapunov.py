import torch
import numpy as np
import matplotlib.pyplot as plt
from train_lyapunov import RobustLyapunovNet

# ==========================================
# 硬件与维度配置
# ==========================================
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
STATE_DIM = 20
NUM_MACHINES = STATE_DIM // 2

# ==========================================
# 1. 加载神圣的二次型李雅普诺夫模型
# ==========================================
print("正在加载已验证的 Lyapunov 能量函数 (0.02% 违规率)...")
v_net = RobustLyapunovNet(state_dim=STATE_DIM).to(DEVICE)

try:
    # 尝试加载刚才跑出的最优权重
    v_net.load_state_dict(torch.load('best_quadratic_lyapunov.pt', map_location=DEVICE))
except FileNotFoundError:
    print("错误：找不到 'best_quadratic_lyapunov.pt'！请确保 Phase 2 训练已完成且文件在当前目录下。")
    exit(1)

v_net.eval()

# ==========================================
# 2. 构建二维网格 (扫描 M1 机器的相平面)
# ==========================================
print("正在二维投影平面上扫描能量地形...")
grid_resolution = 200
# 扫描 M1 功角偏差和转速偏差的范围 (物理安全区内)
delta_range = np.linspace(-1.0, 1.0, grid_resolution)
omega_range = np.linspace(-1.0, 1.0, grid_resolution)

Delta, Omega = np.meshgrid(delta_range, omega_range)
Z_energy = np.zeros_like(Delta)

# ==========================================
# 3. 逐点瞬间计算能量 (这就是纯代数网络超越 ODE 积分的地方！)
# ==========================================
with torch.no_grad():
    for i in range(grid_resolution):
        for j in range(grid_resolution):
            # 创建一个 20 维的全零状态 (所有机器最初都处于稳定平衡点 SEP)
            state = torch.zeros(1, STATE_DIM, device=DEVICE)
            
            # 引入扰动：唯独将当前网格的坐标赋予给 M1 (机器1)
            # M1 的 delta 在索引 0，omega 在索引 NUM_MACHINES (10)
            state[0, 0] = delta_range[j]
            state[0, NUM_MACHINES] = omega_range[i]
            
            # 瞬间输出能量值！
            energy = v_net(state)
            Z_energy[i, j] = energy.item()

# ==========================================
# 4. 绘制顶会级别的等高线图
# ==========================================
print("正在渲染高精度等高线地形图...")
plt.figure(figsize=(9, 7), dpi=300)

# 使用对数底色，更好地展示碗底 (原点附近) 的陡峭程度，同时避免外围能量值过大掩盖细节
contour = plt.contourf(Delta, Omega, np.log1p(Z_energy), levels=60, cmap='viridis', alpha=0.9)
cbar = plt.colorbar(contour)
cbar.set_label('Log-scaled Lyapunov Energy: $\ln(1+V)$', fontsize=12)

# 画出清晰的等高线白圈，这在数学上代表了不变集 (Invariant Sets) 的边界
plt.contour(Delta, Omega, Z_energy, levels=15, colors='white', alpha=0.6, linewidths=1.0)

# 标记绝对中心原点 (物理稳定平衡点 SEP)
plt.plot(0, 0, 'r*', markersize=18, markeredgecolor='black', markeredgewidth=1, label='Stable Equilibrium Point (SEP)')

plt.title('Neural Lyapunov Level Sets (Machine 1 Phase Plane Projection)', fontsize=15, pad=15, fontweight='bold')
plt.xlabel(r'Rotor Angle Deviation $\Delta\delta_1$ (rad)', fontsize=13)
plt.ylabel(r'Rotor Speed Deviation $\Delta\omega_1$ (pu)', fontsize=13)
plt.legend(loc='upper right', fontsize=11)
plt.grid(True, linestyle='--', alpha=0.4, color='white')
plt.tight_layout()

save_path = "Paper_Figure_Lyapunov_ROA.png"
plt.savefig(save_path)
print(f"🎉 成功！神图已保存为: {save_path}。")

import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torchdiffeq import odeint

# ==========================================
# 第一部分：物理世界的真值生成器 (Ground Truth)
# ==========================================
class PowerSystemPhysics(nn.Module):
    def __init__(self, M=0.5, D=0.1, Pm=1.0, Pmax=2.0):
        super().__init__()
        self.M = M        # 惯性
        self.D = D        # 阻尼 (越大衰减越快)
        self.Pm = Pm      # 机械功率
        self.Pmax = Pmax  # 最大传输功率

    def forward(self, t, state):
        # state: [batch, 2] -> [delta, omega]
        delta = state[..., 0]
        omega = state[..., 1]
        
        # 物理方程 (Swing Equation)
        d_delta = omega
        d_omega = (self.Pm - self.Pmax * torch.sin(delta) - self.D * omega) / self.M
        
        return torch.stack([d_delta, d_omega], dim=-1)

# ==========================================
# 第二部分：要训练的 Neural ODE 模型 (黑盒)
# ==========================================
class LearnerODE(nn.Module):
    def __init__(self):
        super().__init__()
        # 这是一个普通的神经网络，它不知道 sin(delta) 或阻尼的存在
        self.net = nn.Sequential(
            nn.Linear(2, 50),
            nn.Tanh(),
            nn.Linear(50, 50),
            nn.Tanh(),
            nn.Linear(50, 2)
        )

    def forward(self, t, y):
        return self.net(y)

# ==========================================
# 主程序
# ==========================================
if __name__ == "__main__":
    # 1. 准备数据
    true_physics = PowerSystemPhysics()
    
    # 初始状态：系统受到扰动，偏离稳定点
    # delta = 0.5 rad, omega = 0.5 rad/s
    y0 = torch.tensor([[0.5, 0.5]]) 
    t = torch.linspace(0., 10., 100) # 模拟 10 秒
    
    # 生成"真实"轨迹 (这就是你需要输入给网络的数据)
    with torch.no_grad():
        true_y = odeint(true_physics, y0, t)
        # true_y shape: [100, 1, 2]

    # 2. 初始化模型
    model = LearnerODE()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    # 3. 训练循环
    print("开始训练电力系统模型...")
    for itr in range(1, 201):
        optimizer.zero_grad()
        
        # Neural ODE 预测轨迹
        pred_y = odeint(model, y0, t)
        
        # 计算损失 (MSE)
        loss = torch.mean((pred_y - true_y)**2)
        loss.backward()
        optimizer.step()
        
        if itr % 20 == 0:
            print(f"Iter {itr} | Loss: {loss.item():.6f}")

    # ==========================================
    # 可视化结果：相平面图 (Phase Portrait)
    # ==========================================
    # 这是电力工程师最常用的图，横轴是角度，纵轴是频率
    plt.figure(figsize=(10, 5))
    
    # 1. 时域波形图
    plt.subplot(1, 2, 1)
    plt.plot(t, true_y[:, 0, 0], 'k-', label='True Delta', alpha=0.6)
    plt.plot(t, pred_y.detach()[:, 0, 0], 'r--', label='Neural ODE Delta')
    plt.title("Time Domain Response")
    plt.xlabel("Time (s)")
    plt.ylabel("Rotor Angle (rad)")
    plt.legend()
    plt.grid(True)

    # 2. 相平面图 (功角 vs 转速)
    plt.subplot(1, 2, 2)
    plt.plot(true_y[:, 0, 0], true_y[:, 0, 1], 'k-', label='True Trajectory')
    plt.plot(pred_y.detach()[:, 0, 0], pred_y.detach()[:, 0, 1], 'r--', label='Learned Trajectory')
    
    # 绘制模型学到的向量场背景
    y_mesh, x_mesh = torch.meshgrid(torch.linspace(-1, 1, 20), torch.linspace(-1, 1, 20), indexing='xy')
    grid = torch.stack([x_mesh, y_mesh], dim=-1).reshape(-1, 2)
    with torch.no_grad():
        v = model(0, grid)
    plt.quiver(grid[:, 0], grid[:, 1], v[:, 0], v[:, 1], color='blue', alpha=0.1)
    
    plt.title("Phase Plane (Stability Analysis)")
    plt.xlabel("Delta (rad)")
    plt.ylabel("Omega (rad/s)")
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()

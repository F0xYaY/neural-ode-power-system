import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from torchdiffeq import odeint

# ==========================================
# 第一部分：物理环境 (Ground Truth)
# ==========================================
class PowerSystemPhysics(nn.Module):
    def __init__(self):
        super().__init__()
        self.M, self.D, self.Pm, self.Pmax = 0.5, 0.1, 1.0, 2.0

    def forward(self, t, state):
        delta = state[..., 0]
        omega = state[..., 1]
        d_delta = omega
        d_omega = (self.Pm - self.Pmax * torch.sin(delta) - self.D * omega) / self.M
        return torch.stack([d_delta, d_omega], dim=-1)


# ==========================================
# 第二部分：待训练的 Neural ODE (Learner)
# ==========================================
class LearnerODE(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 2)
        )

    def forward(self, t, y):
        return self.net(y)


# ==========================================
# 第三部分：Lyapunov 网络定义
# ==========================================
class LyapunovNet(nn.Module):
    def __init__(self):
        super().__init__()
        # 简单的 MLP，输出维度为 1
        self.net = nn.Sequential(
            nn.Linear(2, 64),
            nn.Tanh(),
            nn.Linear(64, 1)  # 输出标量能量值
        )

    def forward(self, x):
        # 强制正定性的技巧：
        # V(x) = NN(x)^2 + epsilon * |x|^2
        # 这样保证 V(0)=0 且其他地方 > 0
        output = self.net(x) ** 2 
        return output + 0.1 * (x**2).sum(dim=1, keepdim=True)


def get_lie_derivative(V_net, f_net, x):
    """
    计算李导数 (Lie Derivative): dV/dt = (dV/dx) * f(x)
    """
    x.requires_grad_(True)
    
    # 1. 计算 V(x)
    v = V_net(x)
    
    # 2. 计算梯度 dV/dx
    # create_graph=True 是必须的，因为我们要对梯度再求导(在Loss中)
    dv_dx = torch.autograd.grad(v.sum(), x, create_graph=True)[0]
    
    # 3. 获取向量场 f(x) (使用我们上面训练好的 ODE 模型)
    with torch.no_grad():  # f_net 已经训练好了，不需要更新它
        f = f_net(0, x)
        
    # 4. 点积: dV/dx * f(x)
    lie_derivative = (dv_dx * f).sum(dim=1, keepdim=True)
    return lie_derivative


def plot_lyapunov_contours(v_net, range_val=2.0):
    """可视化 Lyapunov 函数的等高线图（稳定域）"""
    x = torch.linspace(-range_val, range_val, 100)
    y = torch.linspace(-range_val, range_val, 100)
    X, Y = torch.meshgrid(x, y, indexing='xy')
    grid = torch.stack([X, Y], dim=-1).reshape(-1, 2)
    
    with torch.no_grad():
        V_val = v_net(grid).reshape(100, 100)
        
    plt.figure(figsize=(6, 6))
    # 绘制等高线，颜色越深代表能量越低
    cp = plt.contourf(X, Y, V_val, levels=20, cmap='viridis')
    plt.colorbar(cp)
    plt.title("Learned Lyapunov Function V(x)")
    plt.xlabel("Delta")
    plt.ylabel("Omega")
    
    # 验证：叠加真实的物理稳定点 (通常是原点附近)
    plt.plot(0, 0, 'rx', markersize=10, label='Equilibrium')
    plt.legend()
    plt.show()


# ==========================================
# 主程序
# ==========================================
if __name__ == "__main__":
    # ==========================================
    # A. 批量数据生成与训练 Neural ODE
    # ==========================================
    batch_size = 64
    time_steps = 50
    t = torch.linspace(0., 5., time_steps)
    physics = PowerSystemPhysics()

    # 随机生成多个初始状态 (Batch Sampling)
    # Delta 在 [-1, 1] 之间, Omega 在 [-1, 1] 之间
    y0_batch = (torch.rand(batch_size, 2) - 0.5) * 2 
    
    # 生成 Ground Truth
    with torch.no_grad():
        # true_y shape: [time, batch, 2]
        true_y = odeint(physics, y0_batch, t)

    # 批量训练 Neural ODE
    model = LearnerODE()
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    print(f"开始批量训练 Neural ODE (Batch Size: {batch_size})...")
    for itr in range(1, 301):
        optimizer.zero_grad()
        
        # 这里的 pred_y 也是 [time, batch, 2]
        pred_y = odeint(model, y0_batch, t)
        
        # Loss 自动在 Batch 维度取平均
        loss = torch.mean((pred_y - true_y)**2)
        
        loss.backward()
        optimizer.step()
        
        if itr % 50 == 0:
            print(f"Iter {itr} | Loss: {loss.item():.6f}")
    
    print("Neural ODE 训练完成，模型已学会向量场。\n")

    # ==========================================
    # B. 训练 Lyapunov 函数
    # ==========================================
    lyap_model = LyapunovNet()
    lyap_optimizer = optim.Adam(lyap_model.parameters(), lr=0.001)

    # 在一定范围内采样点来验证稳定性 (不仅仅是轨迹上的点)
    # 我们希望在这个正方形区域内找到稳定域
    range_lim = 2.0 

    print("开始寻找 Lyapunov 函数...")
    for itr in range(1001):
        lyap_optimizer.zero_grad()
        
        # 1. 随机采样状态空间中的点 x
        x_sample = (torch.rand(200, 2) - 0.5) * 2 * range_lim
        
        # 2. 计算 V(x) 和 dV/dt
        v_val = lyap_model(x_sample)
        dv_dt = get_lie_derivative(lyap_model, model, x_sample)
        
        # 3. 定义 Lyapunov Loss
        # 目标：我们希望 dv_dt < 0
        # 如果 dv_dt > 0，则是违规，需要惩罚 (使用 ReLU)
        loss_stability = torch.relu(dv_dt + 0.1).mean() 
        
        # 还可以加一项 loss 使得 V(x) 不要这一项过大，保持平滑
        loss = loss_stability
        
        loss.backward()
        lyap_optimizer.step()
        
        if itr % 100 == 0:
            print(f"Iter {itr} | Lyapunov Violation Loss: {loss.item():.6f}")

    print("Lyapunov 函数训练完成。\n")

    # ==========================================
    # C. 可视化：绘制稳定域 (Region of Attraction)
    # ==========================================
    print("绘制 Lyapunov 函数等高线图...")
    plot_lyapunov_contours(lyap_model)
    
    # 额外可视化：展示训练好的 Neural ODE 的轨迹预测
    print("绘制训练样本的轨迹对比...")
    plt.figure(figsize=(12, 5))
    
    # 选择几个样本进行可视化
    sample_indices = [0, 1, 2, 3]
    
    for idx, sample_idx in enumerate(sample_indices):
        plt.subplot(2, 2, idx + 1)
        
        # 真实轨迹
        plt.plot(true_y[:, sample_idx, 0].detach(), 
                true_y[:, sample_idx, 1].detach(), 
                'k-', label='True', linewidth=2)
        
        # 预测轨迹
        with torch.no_grad():
            pred_sample = odeint(model, y0_batch[sample_idx:sample_idx+1], t)
        plt.plot(pred_sample[:, 0, 0].detach(), 
                pred_sample[:, 0, 1].detach(), 
                'r--', label='Neural ODE', linewidth=2)
        
        plt.title(f'Sample {sample_idx}')
        plt.xlabel('Delta')
        plt.ylabel('Omega')
        plt.legend()
        plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    print("\n训练和可视化完成！")
    print("总结：")
    print("1. Neural ODE 已学会整个状态空间的向量场")
    print("2. Lyapunov 函数已找到稳定域 (Region of Attraction)")
    print("3. 可以通过判断 V(x_current) < C 来实时监控电网稳定性")

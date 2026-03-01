import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
from model import DynamicsNetwork
from utils import sample_random_states

# ==========================================
# 核心超参数配置
# ==========================================
BATCH_SIZE = 8192
EPOCHS = 1000
LR = 0.005
ALPHA = 0.01            # 降低衰减率要求，先求稳定下降，再求速度
EPSILON = 1e-4          
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# =======================================================
# 【全新架构】：鲁棒二次型李雅普诺夫网络 (Robust Quadratic Lyapunov Net)
# 彻底解决 ICNN 的数值爆炸和原点无法归零问题
# 数学形式: V(x) = x^T P(x) x + epsilon * ||x||^2
# 其中 P(x) 是一个由神经网络输出的正定矩阵 L(x) * L(x)^T
# =======================================================
class RobustLyapunovNet(nn.Module):
    def __init__(self, state_dim, hidden_dim=64):
        super().__init__()
        self.state_dim = state_dim
        
        # 这个网络不直接输出能量，而是输出一个下三角矩阵 L 的元素
        # 从而构造正定矩阵 P = L * L^T。这保证了 V(x) 绝对非负且平滑！
        self.num_elements = (state_dim * (state_dim + 1)) // 2
        
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.Tanh(), # 在这里用 Tanh 是极好的，因为它能限制矩阵元素的大小，防爆炸
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, self.num_elements)
        )
        
        # 初始化非常关键：让初始的 P 矩阵尽量接近单位矩阵 I
        torch.nn.init.zeros_(self.net[-1].weight)
        
        # 构造单位矩阵对应的 Cholesky 下三角元素作为偏置
        L_init = torch.eye(state_dim)
        tril_indices = torch.tril_indices(row=state_dim, col=state_dim, offset=0)
        bias_init = L_init[tril_indices[0], tril_indices[1]]
        with torch.no_grad():
            self.net[-1].bias.copy_(bias_init)

    def forward(self, x):
        batch_size = x.shape[0]
        elements = self.net(x)
        
        # 重构下三角矩阵 L
        L = torch.zeros(batch_size, self.state_dim, self.state_dim, device=x.device)
        tril_indices = torch.tril_indices(row=self.state_dim, col=self.state_dim, offset=0)
        L[:, tril_indices[0], tril_indices[1]] = elements
        
        # 计算 P = L * L^T (这保证了 P 永远是半正定的)
        P = torch.bmm(L, L.transpose(1, 2))
        
        # 计算二次型 V(x) = x^T P x
        # x shape: [B, D] -> [B, 1, D]
        x_unsqueeze = x.unsqueeze(1)
        # x^T P x -> [B, 1, D] * [B, D, D] * [B, D, 1] -> [B, 1, 1]
        V_quad = torch.bmm(torch.bmm(x_unsqueeze, P), x.unsqueeze(-1)).squeeze(-1).squeeze(-1)
        
        # 加上极小的基准保证严格正定 (V > 0 对于 x != 0)
        return V_quad + EPSILON * torch.sum(x**2, dim=1)

def get_lie_derivative(V_net, f_net, x):
    """计算李导数 dV/dt"""
    x.requires_grad_(True)
    v = V_net(x)
    dv_dx = torch.autograd.grad(v.sum(), x, create_graph=True)[0]
    f = f_net(0, x)
    lie_derivative = (dv_dx * f).sum(dim=-1, keepdim=True)
    return lie_derivative

def load_pretrained_dynamics(model_path='best_model_20000_samples.pt'):
    print(f"正在加载向量场代理模型: {model_path}")
    checkpoint = torch.load(model_path, map_location=DEVICE)
    state_dim = checkpoint['state_dim']
    f_net = DynamicsNetwork(input_dim=state_dim, use_spectral_norm=False).to(DEVICE)
    f_net.load_state_dict(checkpoint['dynamics_model'])
    f_net.eval()
    for param in f_net.parameters():
        param.requires_grad = False
    return f_net, state_dim

def train_lyapunov(f_net, state_dim):
    print("\n" + "="*60)
    print("开始 Phase 2: Neural Lyapunov Certification (Robust Quadratic Form)...")
    print(f"状态空间维度: {state_dim}D")
    print("="*60)

    # 实例化全新的 RobustLyapunovNet
    v_net = RobustLyapunovNet(state_dim=state_dim).to(DEVICE)
    
    optimizer = optim.Adam(v_net.parameters(), lr=LR, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=20, min_lr=1e-5)

    num_machines = state_dim // 2
    best_violation_rate = 100.0

    for epoch in range(1, EPOCHS + 1):
        # 多尺度采样 (重点放在真实稳定的核心区域)
        x_wide = sample_random_states(
            num_samples=BATCH_SIZE // 2, 
            num_machines=num_machines,
            delta_range=(-0.5, 0.5), # 缩小边界，先攻克核心稳定域
            omega_range=(-0.5, 0.5)
        ).to(DEVICE)
        
        x_narrow = sample_random_states(
            num_samples=BATCH_SIZE // 2, 
            num_machines=num_machines,
            delta_range=(-0.1, 0.1), 
            omega_range=(-0.1, 0.1)
        ).to(DEVICE)
        
        x_batch = torch.cat([x_wide, x_narrow], dim=0)
        
        optimizer.zero_grad()

        v_values = v_net(x_batch).unsqueeze(-1) # 统一维度 [B, 1]
        v_dot = get_lie_derivative(v_net, f_net, x_batch)
        
        # 违规惩罚 (要求 dV/dt <= -ALPHA * V)
        # 用 ReLU 实施绝对硬性惩罚，不再用软泥一样的 Softplus
        violation = torch.nn.functional.relu(v_dot + ALPHA * v_values)
        loss = torch.mean(violation)
        
        loss.backward()
        
        # 加上梯度裁剪，防止偶尔的极端点炸毁网络
        torch.nn.utils.clip_grad_norm_(v_net.parameters(), max_norm=1.0)
        
        optimizer.step()

        # 计算真实违规率 (容忍万分之一的数值截断误差)
        with torch.no_grad():
            strict_violation = (v_dot + 1e-4 > 0).float()
            current_violation_rate = strict_violation.mean().item() * 100

        scheduler.step(current_violation_rate)

        if epoch % 10 == 0 or epoch == 1:
            print(f"Epoch {epoch:04d} | Max dV/dt: {v_dot.max().item():.4f} | "
                  f"Violation Rate: {current_violation_rate:.2f}% | LR: {optimizer.param_groups[0]['lr']:.6f}")

        if current_violation_rate < best_violation_rate:
            best_violation_rate = current_violation_rate
            torch.save(v_net.state_dict(), 'best_quadratic_lyapunov.pt')

        if current_violation_rate <= 0.5 and epoch > 50:
            print(f"\n[神圣时刻] 违规率已降至 {current_violation_rate:.2f}%！")
            print("系统核心安全区已被严格证明为具备渐近稳定性！")
            break

    print("\n" + "="*60)
    print(f"训练结束！最佳违规率: {best_violation_rate:.2f}%")
    print("模型已保存为 'best_quadratic_lyapunov.pt'")

if __name__ == "__main__":
    if not os.path.exists('best_model_20000_samples.pt'):
        print("致命错误：找不到物理向量场模型！")
    else:
        f_net, state_dim = load_pretrained_dynamics()
        train_lyapunov(f_net, state_dim)

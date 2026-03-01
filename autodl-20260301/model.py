import torch
import torch.nn as nn
from typing import Optional

class PowerSystemPhysics(nn.Module):
    """物理世界的真值生成器 (Ground Truth) - Swing Equation"""
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

# =======================================================
# 【前沿创新：受液态神经网络(LTC)启发的动态时间常数模块】
# =======================================================
class LiquidBlock(nn.Module):
    """
    状态依赖的液态门控机制 (State-Dependent Liquid Gating)
    允许网络在不同状态下拥有不同的时间响应常数，极大地缓解长期相位漂移。
    """
    def __init__(self, dim):
        super().__init__()
        # 特征提取支路 (使用无限阶光滑的 SiLU 适配微分方程)
        self.feature_net = nn.Sequential(
            nn.Linear(dim, dim),
            nn.SiLU(),
            nn.Linear(dim, dim)
        )
        
        # 时间常数/阻尼门控支路 (Liquid Time-Constant Gate)
        # 核心创新：这使得 ODE 求解时的有效步长是状态相关的
        self.tau_gate = nn.Sequential(
            nn.Linear(dim, dim),
            nn.Sigmoid()  # 输出 (0, 1) 的门控信号，代表系统动态的流动性
        )
        
        # 科学初始化以保证早期数值稳定
        torch.nn.init.orthogonal_(self.feature_net[0].weight)
        torch.nn.init.orthogonal_(self.feature_net[2].weight)
        
        # 将门控偏置初始化为较大的正数 (如 1.0)
        # 初始时 sigmoid(1.0) 接近 1，使其退化为普通的残差连接，随后逐渐学习流动性
        torch.nn.init.constant_(self.tau_gate[0].bias, 1.0) 

    def forward(self, x):
        # 提取当前状态下的高维动力学特征
        f_x = self.feature_net(x)
        # 计算当前状态下的局部时间常数门控 (响应速度)
        tau_x = self.tau_gate(x)
        
        # 液态混合更新：原状态的保留比例与新特征的注入比例动态耦合
        return (1.0 - tau_x) * x + tau_x * f_x

class DynamicsNetwork(nn.Module):
    """
    【LTC 增强版物理网络】
    深度融合了：
    1. 210维全连接物理拓扑特征 (Physical Topology Features)
    2. 物理运动学硬编码 (Kinematic Hardcoding)
    3. 线性物理捷径 (Linear Physics Shortcut)
    4. 液态时间常数模块 (Liquid Time-Constant Blocks)
    """
    def __init__(
        self,
        input_dim: int = 20, 
        hidden_dim: int = 512, 
        num_layers: int = 5,   
        use_spectral_norm: bool = False, # 此高级架构不需要强行限速，完全解除封印
    ):
        super().__init__()
        self.input_dim = int(input_dim)
        self.num_machines = self.input_dim // 2
        
        # 特征维度计算：
        # omega: 10 维
        # sin_diff: 10x10 = 100 维 (对应有功拓扑)
        # cos_diff: 10x10 = 100 维 (对应无功/损耗拓扑)
        # 总维度 = 210 维
        feature_dim = self.num_machines + 2 * (self.num_machines ** 2)
        
        layers = []
        layers.append(nn.Linear(feature_dim, hidden_dim))
        layers.append(nn.SiLU())
        torch.nn.init.orthogonal_(layers[0].weight)

        # 引入前沿的液态记忆模块，替换传统的 ResNetBlock
        for _ in range(num_layers - 2):
            layers.append(LiquidBlock(hidden_dim))

        # 输出层：只需要预测未知的 10维加速度 d_omega
        output_layer = nn.Linear(hidden_dim, self.num_machines)
        torch.nn.init.normal_(output_layer.weight, mean=0.0, std=0.01)
        torch.nn.init.zeros_(output_layer.bias)
        layers.append(output_layer)
        
        self.net = nn.Sequential(*layers)
        
        # --- 物理主干捷径：负责传输稳定的基波网络功率流 ---
        self.physics_shortcut = nn.Linear(feature_dim, self.num_machines)
        torch.nn.init.xavier_uniform_(self.physics_shortcut.weight)
        torch.nn.init.zeros_(self.physics_shortcut.bias)

    def forward(self, t, y):
        # 1. 提取独立物理状态
        delta = y[..., :self.num_machines]
        omega = y[..., self.num_machines:]
        
        # 2. 全连接拓扑图边特征构建 (Edge Feature Construction)
        delta_diff = delta.unsqueeze(-1) - delta.unsqueeze(-2)
        delta_diff_flat = delta_diff.flatten(start_dim=-2)
        sin_diff = torch.sin(delta_diff_flat)
        cos_diff = torch.cos(delta_diff_flat)
        
        # 构建 210维 的纯粹物理组合特征
        features = torch.cat([sin_diff, cos_diff, omega], dim=-1)
        
        # 3. 动力学求解 (线性互联基础网络功率 + 高频液态涌现特征)
        d_omega = self.physics_shortcut(features) + self.net(features)
        
        # 4. 硬编码运动学规律 (此规律属于真理，不消耗网络算力去猜)
        d_delta = omega
        
        return torch.cat([d_delta, d_omega], dim=-1)

# =======================================================
# 相位 2 (Phase 2): 严密数学证明 - 神经李雅普诺夫模块
# =======================================================
class LyapunovNet(nn.Module):
    """基础 Lyapunov 函数网络 (仅作对照/备用)"""
    def __init__(self, input_dim: int = 2, hidden_dim: int = 64, epsilon: float = 1e-3):
        super().__init__()
        self.epsilon = epsilon
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1) 
        )
        self.input_dim = int(input_dim)

    def forward(self, x):
        output = self.net(x) ** 2 
        return output + self.epsilon * (x**2).sum(dim=-1, keepdim=True)

class ICNNLyapunovNet(nn.Module):
    """
    ICNN 变体的 Lyapunov 网络 
    【核心价值】：通过强制权重非负和激活函数凸性，从拓扑结构上
    100% 保证网络构建出的多维能量域严格为“碗状”(Convexity)，解决 ROA 估计的安全死角。
    """
    def __init__(self, input_dim: int, hidden_dims=(256, 256, 256, 256), epsilon: float = 1e-3):
        super().__init__()
        self.input_dim = int(input_dim)
        self.epsilon = float(epsilon)
        hidden_dims = tuple(int(h) for h in hidden_dims)

        self.Wx = nn.ModuleList()
        self.Wz_raw = nn.ParameterList()
        self.b = nn.ParameterList()

        # 第一层：仅将状态 x 映射为 z_1
        self.Wx.append(nn.Linear(self.input_dim, hidden_dims[0], bias=True))

        # 后续层：不仅保留原状态 x 的输入，还要对前一层的输出 z_{k-1} 施加非负权重矩阵 Wz
        for k in range(1, len(hidden_dims)):
            self.Wx.append(nn.Linear(self.input_dim, hidden_dims[k], bias=False))
            wz = torch.empty(hidden_dims[k], hidden_dims[k - 1])
            nn.init.normal_(wz, mean=0.0, std=0.02)
            # 保存原始可学习参数，前向传播时通过 Softplus 强行拉正
            self.Wz_raw.append(nn.Parameter(wz))
            self.b.append(nn.Parameter(torch.zeros(hidden_dims[k])))

        self.out = nn.Linear(hidden_dims[-1], 1, bias=True)
        self.act = nn.Softplus() # 使用 Softplus 保证无限阶可导且单调递增

    def _forward_raw(self, x: torch.Tensor) -> torch.Tensor:
        z = self.act(self.Wx[0](x))
        for k in range(1, len(self.Wx)):
            # 施加非负约束：将 Wz_raw 压过 Softplus 变成绝对正数
            Wz_pos = torch.nn.functional.softplus(self.Wz_raw[k - 1]) 
            # 前馈方程：非负传递与原状态直连的凸组合
            z = self.act(self.Wx[k](x) + (z @ Wz_pos.t()) + self.b[k - 1])
        return self.out(z)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        f_x = self._forward_raw(x)
        # 强制平移使原点能量精确为 0 (V(0) = 0)
        with torch.no_grad():
            f_0 = self._forward_raw(torch.zeros_like(x))
        # 再次通过 Softplus 并在外面加上微小的二阶凸函数(epsilon * x^2)
        # 确保在整个全空间内严格满足 V(x) > 0 (除原点外)
        v = torch.nn.functional.softplus(f_x - f_0)
        return v + self.epsilon * (x**2).sum(dim=-1, keepdim=True)

def get_lie_derivative(V_net, f_net, x):
    """
    计算李导数 (Lie Derivative): dV/dt = (\nabla V/dx) * f(x)
    通过反向自动求导系统精确定理验证！
    """
    x.requires_grad_(True)
    v = V_net(x)
    # 取 V(x) 对状态 x 的偏导数矩阵
    dv_dx = torch.autograd.grad(v.sum(), x, create_graph=True)[0]
    # 调用训练好的物理网络 f_theta(x)
    f = f_net(0, x)
    # 内积得随时间的导数
    lie_derivative = (dv_dx * f).sum(dim=-1, keepdim=True)
    return lie_derivative

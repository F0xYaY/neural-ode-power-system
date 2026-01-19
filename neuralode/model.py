import torch
import torch.nn as nn


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


class ResNetBlock(nn.Module):
    """残差块结构，用于改善梯度流"""
    def __init__(self, dim, activation='tanh', use_spectral_norm=False):
        super().__init__()
        # 使用平滑激活函数 Tanh（更适合 ODE 求解器）
        self.activation = nn.Tanh()
        
        linear1 = nn.Linear(dim, dim)
        linear2 = nn.Linear(dim, dim)
        
        # 谱归一化（如果启用）
        if use_spectral_norm:
            linear1 = nn.utils.spectral_norm(linear1)
            linear2 = nn.utils.spectral_norm(linear2)
        
        self.linear1 = linear1
        self.linear2 = linear2
        
        # 使用正交初始化改善梯度流（需要处理 spectral_norm）
        if hasattr(self.linear1, 'weight_orig'):
            torch.nn.init.orthogonal_(self.linear1.weight_orig)
            torch.nn.init.orthogonal_(self.linear2.weight_orig)
        else:
            torch.nn.init.orthogonal_(self.linear1.weight)
            torch.nn.init.orthogonal_(self.linear2.weight)
    
    def forward(self, x):
        out = self.linear1(x)
        out = self.activation(out)
        out = self.linear2(out)
        return x + out


class DynamicsNetwork(nn.Module):
    """
    Neural ODE 函数类 (f_theta)
    改进版本：使用三角编码来帮助学习正弦物理
    - 三角编码：将 delta 编码为 [sin(delta), cos(delta)]
    - 隐藏层维度：128
    - 层数：4层
    - 使用 ResNetBlock 和 ELU 激活改善梯度流
    - 正交权重初始化
    """
    def __init__(self, hidden_dim=128, num_layers=4, activation='tanh', use_spectral_norm=True):
        super().__init__()
        layers = []
        
        # 输入层：输入维度从 2 变为 3 (sin(delta), cos(delta), omega)
        input_dim = 3
        linear1 = nn.Linear(input_dim, hidden_dim)
        # 强制使用谱归一化（限制 Lipschitz 常数，防止数值不稳定）
        linear1 = nn.utils.spectral_norm(linear1)
        layers.append(linear1)
        # 使用平滑激活函数（Tanh 更适合 ODE 求解器）
        # Tanh 有连续的梯度，不会导致 ODE 求解器不稳定
        layers.append(nn.Tanh())
        
        # 正交初始化输入层（隐藏层保持标准初始化）
        if hasattr(layers[0], 'weight'):
            # 对于 spectral_norm，需要通过 weight_orig 访问原始权重
            if hasattr(layers[0], 'weight_orig'):
                torch.nn.init.orthogonal_(layers[0].weight_orig)
            else:
                torch.nn.init.orthogonal_(layers[0].weight)
        
        # 隐藏层：使用 ResNetBlock（内部也使用 Tanh）
        for _ in range(num_layers - 2):
            layers.append(ResNetBlock(hidden_dim, activation='tanh'))
        
        # 输出层：关键！使用极小的初始化（near-zero initialization）
        # 这确保初始向量场接近平坦 (dx/dt ≈ 0)，避免 dt underflow
        output_layer = nn.Linear(hidden_dim, 2)
        # 输出层也使用谱归一化
        output_layer = nn.utils.spectral_norm(output_layer)
        
        # 输出层初始化为极小值：std=1e-4，确保初始导数接近 0
        if hasattr(output_layer, 'weight_orig'):
            torch.nn.init.normal_(output_layer.weight_orig, mean=0.0, std=1e-4)
        else:
            torch.nn.init.normal_(output_layer.weight, mean=0.0, std=1e-4)
        
        if output_layer.bias is not None:
            torch.nn.init.zeros_(output_layer.bias)  # 偏置初始化为 0
        
        layers.append(output_layer)
        
        self.net = nn.Sequential(*layers)
    
    def forward(self, t, y):
        """
        Compute the derivative dy/dt given time t and state y.
        
        使用三角编码预处理输入，帮助网络学习正弦物理：
        - delta -> [sin(delta), cos(delta)]
        - omega -> omega
        
        Args:
            t: current time (can be scalar or tensor, typically not used in autonomous ODEs)
            y: current state vector of shape (batch_size, 2) -> [delta, omega]
        
        Returns:
            dy/dt: derivative of the state vector, shape (batch_size, 2)
        """
        # 三角编码：提取 delta 和 omega，然后编码
        delta = y[..., 0]  # 功角
        omega = y[..., 1]  # 频率
        
        # 创建特征向量：[sin(delta), cos(delta), omega]
        # 这样网络可以更容易学习 d_omega ∝ sin(delta) 的关系
        features = torch.cat([
            torch.sin(delta).unsqueeze(-1),
            torch.cos(delta).unsqueeze(-1),
            omega.unsqueeze(-1)
        ], dim=-1)
        
        return self.net(features)


class LyapunovNet(nn.Module):
    """Lyapunov 函数网络 V(x)"""
    def __init__(self, hidden_dim=64, epsilon=1e-3):
        """
        Args:
            hidden_dim: 隐藏层维度
            epsilon: 正则化系数（从 0.1 减小到 1e-3，允许学习复杂的非二次能量函数形状）
        """
        super().__init__()
        self.epsilon = epsilon
        self.net = nn.Sequential(
            nn.Linear(2, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)  # 输出标量能量值
        )

    def forward(self, x):
        """
        强制正定性的技巧：
        V(x) = NN(x)^2 + epsilon * |x|^2
        这样保证 V(0)=0 且其他地方 > 0
        """
        output = self.net(x) ** 2 
        return output + self.epsilon * (x**2).sum(dim=1, keepdim=True)


def get_lie_derivative(V_net, f_net, x):
    """
    计算李导数 (Lie Derivative): dV/dt = (dV/dx) * f(x)
    
    Args:
        V_net: Lyapunov 网络
        f_net: ODE 函数网络
        x: 状态点，shape (batch_size, 2)
    
    Returns:
        lie_derivative: dV/dt，shape (batch_size, 1)
    """
    x.requires_grad_(True)
    
    # 1. 计算 V(x)
    v = V_net(x)
    
    # 2. 计算梯度 dV/dx
    # create_graph=True 是必须的，因为我们要对梯度再求导(在Loss中)
    dv_dx = torch.autograd.grad(v.sum(), x, create_graph=True)[0]
    
    # 3. 获取向量场 f(x) (使用我们上面训练好的 ODE 模型)
    # 注意：这里不使用 no_grad，因为我们需要 f_net 参与梯度计算
    f = f_net(0, x)
        
    # 4. 点积: dV/dx * f(x)
    lie_derivative = (dv_dx * f).sum(dim=1, keepdim=True)
    return lie_derivative

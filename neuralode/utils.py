import torch
from torchdiffeq import odeint
from model import PowerSystemPhysics


def generate_trajectory_data(batch_size, time_steps, t_span=(0., 5.), delta_range=(-1.5, 1.5), omega_range=(-1.0, 1.0)):
    """
    生成批量轨迹数据用于训练 Neural ODE
    
    Args:
        batch_size: 批量大小
        time_steps: 时间步数
        t_span: 时间范围 (t_start, t_end)
        delta_range: delta 的采样范围，限制在稳定域内，避免包含不稳定平衡点
        omega_range: omega 的采样范围
    
    Returns:
        y0_batch: 初始状态，shape (batch_size, 2)
        t: 时间点，shape (time_steps,)
        true_y: 真实轨迹，shape (time_steps, batch_size, 2)
    """
    physics = PowerSystemPhysics()
    t = torch.linspace(t_span[0], t_span[1], time_steps)
    
    # 随机生成多个初始状态 (Batch Sampling)
    # 限制在稳定域内：delta 在 [-1.5, 1.5] 之间（围绕稳定平衡点 δs ≈ 0.52）
    # 避免在不稳定平衡点 (UEP) 附近采样
    delta_min, delta_max = delta_range
    omega_min, omega_max = omega_range
    
    delta_samples = torch.rand(batch_size, 1) * (delta_max - delta_min) + delta_min
    omega_samples = torch.rand(batch_size, 1) * (omega_max - omega_min) + omega_min
    y0_batch = torch.cat([delta_samples, omega_samples], dim=1)
    
    # 生成 Ground Truth
    with torch.no_grad():
        # true_y shape: [time_steps, batch_size, 2]
        true_y = odeint(physics, y0_batch, t)
    
    return y0_batch, t, true_y


def sample_random_states(num_samples, delta_range=(-1.5, 1.5), omega_range=(-1.0, 1.0)):
    """
    随机采样状态空间中的点，用于 Lyapunov 损失计算
    
    关键修正：限制采样范围在稳定域 (ROA) 内
    - delta 范围：[-1.5, 1.5] rad（围绕稳定平衡点 δs ≈ 0.52 rad）
    - 避免在不稳定平衡点 (UEP around δ ≈ 2.6 rad) 附近采样
    
    Args:
        num_samples: 采样点数
        delta_range: delta 的采样范围
        omega_range: omega 的采样范围
    
    Returns:
        x_sample: 采样点，shape (num_samples, 2)
    """
    delta_min, delta_max = delta_range
    omega_min, omega_max = omega_range
    
    delta_samples = torch.rand(num_samples, 1) * (delta_max - delta_min) + delta_min
    omega_samples = torch.rand(num_samples, 1) * (omega_max - omega_min) + omega_min
    x_sample = torch.cat([delta_samples, omega_samples], dim=1)
    
    return x_sample


def get_stable_equilibrium_point():
    """
    计算稳定平衡点
    Swing Equation 的稳定平衡点满足：
    - omega = 0
    - sin(delta) = Pm / Pmax
    
    返回稳定平衡点的 delta 值
    """
    physics = PowerSystemPhysics()
    # sin(delta_s) = Pm / Pmax = 1.0 / 2.0 = 0.5
    # delta_s = arcsin(0.5) ≈ 0.5236 rad
    delta_s = torch.asin(torch.tensor(physics.Pm / physics.Pmax))
    return delta_s

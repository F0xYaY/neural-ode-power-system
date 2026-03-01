import torch
from torchdiffeq import odeint
from model import PowerSystemPhysics

def generate_trajectory_data(batch_size, time_steps, t_span=(0., 5.), delta_range=(-1.5, 1.5), omega_range=(-1.0, 1.0)):
    """
    (保留作向下兼容) 生成单机批量轨迹数据
    注意：10机系统的训练数据目前直接从 X_train.npy 读取。
    """
    physics = PowerSystemPhysics()
    t = torch.linspace(t_span[0], t_span[1], time_steps)
    
    delta_min, delta_max = delta_range
    omega_min, omega_max = omega_range
    
    delta_samples = torch.rand(batch_size, 1) * (delta_max - delta_min) + delta_min
    omega_samples = torch.rand(batch_size, 1) * (omega_max - omega_min) + omega_min
    y0_batch = torch.cat([delta_samples, omega_samples], dim=1)
    
    with torch.no_grad():
        true_y = odeint(physics, y0_batch, t)
    
    return y0_batch, t, true_y

def sample_random_states(num_samples, num_machines=10, delta_range=(-1.5, 1.5), omega_range=(-1.0, 1.0)):
    """
    【已优化：支持多机系统高维采样】
    随机采样状态空间中的点，用于多维 Lyapunov 损失计算
    
    - 确保采样范围在核心安全区 (Polytopic Safe Set) 内。
    - 输出维度: [num_samples, 2 * num_machines] (例如 10机系统输出 20 维)
    """
    delta_min, delta_max = delta_range
    omega_min, omega_max = omega_range
    
    # 直接生成多台机器的状态，shape: [num_samples, num_machines]
    delta_samples = torch.rand(num_samples, num_machines) * (delta_max - delta_min) + delta_min
    omega_samples = torch.rand(num_samples, num_machines) * (omega_max - omega_min) + omega_min
    
    # 横向拼接：前 num_machines 维是 delta，后 num_machines 维是 omega
    # 完美契合 model.py 中 DynamicsNetwork 的解析逻辑
    x_sample = torch.cat([delta_samples, omega_samples], dim=1)
    
    return x_sample

def get_stable_equilibrium_point():
    """
    计算稳定平衡点 (SEP)
    """
    physics = PowerSystemPhysics()
    delta_s = torch.asin(torch.tensor(physics.Pm / physics.Pmax))
    return delta_s


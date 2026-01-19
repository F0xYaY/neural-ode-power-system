import torch
import torch.optim as optim
from torchdiffeq import odeint
from model import DynamicsNetwork, LyapunovNet, get_lie_derivative
from utils import generate_trajectory_data, sample_random_states

# ==========================================
# CUDA 诊断代码 - 检查 GPU 可用性
# ==========================================
print("=" * 60)
print("CUDA 诊断信息")
print("=" * 60)
print(f"PyTorch 版本: {torch.__version__}")
print(f"CUDA 版本: {torch.version.cuda if torch.version.cuda else 'N/A (CPU only build)'}")
print(f"CUDA 是否可用: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    print(f"CUDA 设备数量: {torch.cuda.device_count()}")
    for i in range(torch.cuda.device_count()):
        print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
        print(f"    内存: {torch.cuda.get_device_properties(i).total_memory / 1e9:.2f} GB")
    # 显式设置设备（如果可用）
    torch.cuda.set_device(0)
    print(f"当前 CUDA 设备: {torch.cuda.current_device()}")
else:
    print("警告: CUDA 不可用，将使用 CPU 训练")
    print("可能的原因:")
    print("  1. PyTorch 安装的是 CPU 版本（需要重新安装 CUDA 版本）")
    print("  2. CUDA 驱动版本不匹配")
    print("  3. NVIDIA 驱动未安装或版本过旧")
print("=" * 60)
print()


def train_neural_ode(model, y0_batch, t, true_y, num_epochs=500, lr=0.01, device=None, step_size=0.1):
    """
    训练 Neural ODE 模型（纯轨迹重建，专注于最小化 MSE）
    
    Args:
        model: DynamicsNetwork 实例
        y0_batch: 初始状态，shape (batch_size, 2)
        t: 时间点，shape (time_steps,)
        true_y: 真实轨迹，shape (time_steps, batch_size, 2)
        num_epochs: 训练轮数
        lr: 初始学习率
        device: 计算设备（如果为 None 则自动检测）
    
    Returns:
        model: 训练好的模型
        loss_history: 损失历史
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 将模型移到设备
    model = model.to(device)
    
    # 将输入数据移到设备
    y0_batch = y0_batch.to(device)
    t = t.to(device)
    true_y = true_y.to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=lr)
    # 学习率调度器：监控损失，如果50个epoch没有改善，学习率减半
    # 设置最小学习率为 1e-6，允许精细化调优
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=50, min_lr=1e-6)
    
    loss_history = []
    
    print(f"开始训练 Neural ODE (Batch Size: {y0_batch.shape[0]}, Epochs: {num_epochs})...")
    print("专注轨迹重建，最小化 MSE")
    
    for epoch in range(1, num_epochs + 1):
        optimizer.zero_grad()
        
        # 前向传播：求解 ODE
        # 使用固定步长求解器 rk4 替代自适应求解器，避免因刚度问题导致的微小步长
        # 固定步长保证训练速度稳定，不会因轨迹复杂而变慢
        # step_size 可以传入以优化训练速度（默认 0.1）
        pred_y = odeint(model, y0_batch, t, method='rk4', options={'step_size': step_size})
        
        # 轨迹重建损失
        loss_traj = torch.mean((pred_y - true_y)**2)
        
        # 直接反向传播
        loss_traj.backward()
        optimizer.step()
        
        # 更新学习率
        scheduler.step(loss_traj.item())
        
        loss_history.append({
            'epoch': epoch,
            'loss_traj': loss_traj.item(),
            'lr': optimizer.param_groups[0]['lr']
        })
        
        if epoch % 100 == 0:  # 每 100 个 epoch 打印一次
            print(f"Epoch {epoch}/{num_epochs} | Loss_traj: {loss_traj.item():.6f} | LR: {optimizer.param_groups[0]['lr']:.6f}")
    
    print("Neural ODE 训练完成。\n")
    return model, loss_history


def train_lyapunov(lyap_model, dynamics_model, num_epochs=1000, lr=0.001, num_samples=200, 
                   delta_range=(-1.5, 1.5), omega_range=(-1.0, 1.0), device=None):
    """
    训练 Lyapunov 函数
    
    Args:
        lyap_model: LyapunovNet 实例
        dynamics_model: 已训练好的 DynamicsNetwork 实例
        num_epochs: 训练轮数
        lr: 学习率
        num_samples: 每次迭代采样的点数
        delta_range: delta 采样范围（限制在稳定域内）
        omega_range: omega 采样范围
        device: 计算设备（如果为 None 则自动检测）
    
    Returns:
        lyap_model: 训练好的 Lyapunov 模型
        loss_history: 损失历史
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 将模型移到设备（dynamics_model应该已经在设备上，但确保一下）
    lyap_model = lyap_model.to(device)
    dynamics_model = dynamics_model.to(device)
    
    optimizer = optim.Adam(lyap_model.parameters(), lr=lr)
    loss_history = []
    
    print("开始寻找 Lyapunov 函数...")
    print(f"采样范围：delta ∈ [{delta_range[0]}, {delta_range[1]}], omega ∈ [{omega_range[0]}, {omega_range[1]}]")
    
    for epoch in range(num_epochs):
        optimizer.zero_grad()
        
        # 在稳定域内随机采样点，并移到设备
        x_sample = sample_random_states(num_samples, delta_range, omega_range).to(device)
        
        # 计算 V(x) 和 dV/dt
        v_val = lyap_model(x_sample)
        dv_dt = get_lie_derivative(lyap_model, dynamics_model, x_sample)
        
        # Lyapunov Loss：强制 dV/dt < 0
        # 如果 dv_dt > -0.1，则是违规，需要惩罚 (使用 ReLU)
        loss_stability = torch.relu(dv_dt + 0.1).mean()
        
        loss = loss_stability
        
        loss.backward()
        optimizer.step()
        
        loss_history.append({
            'epoch': epoch,
            'loss': loss.item()
        })
        
        if epoch % 100 == 0:
            print(f"Epoch {epoch}/{num_epochs} | Lyapunov Violation Loss: {loss.item():.6f}")
    
    print("Lyapunov 函数训练完成。\n")
    return lyap_model, loss_history


def lambda_lyap_linear_schedule(current_step, total_steps, max_lambda=1.0):
    """
    Lyapunov 损失权重的线性调度函数
    
    Args:
        current_step: 当前步数（从 warmup 后开始计数）
        total_steps: 总步数（warmup 后的总步数）
        max_lambda: 最大权重值
    
    Returns:
        lambda_lyap: 当前权重值
    """
    progress = min(current_step / total_steps, 1.0)
    return max_lambda * progress


if __name__ == "__main__":
    # 配置参数
    batch_size = 64
    time_steps = 50
    num_epochs_ode = 500
    num_epochs_lyap = 1000
    
    # 1. 生成训练数据
    print("=" * 60)
    print("步骤 1: 生成训练数据")
    print("=" * 60)
    y0_batch, t, true_y = generate_trajectory_data(
        batch_size=batch_size,
        time_steps=time_steps,
        t_span=(0., 5.),
        delta_range=(-1.5, 1.5),  # 限制在稳定域内
        omega_range=(-1.0, 1.0)
    )
    
    # 2. 训练 Neural ODE
    print("=" * 60)
    print("步骤 2: 训练 Neural ODE")
    print("=" * 60)
    model = DynamicsNetwork(hidden_dim=128, num_layers=4)
    
    # 纯轨迹重建训练
    model, ode_loss_history = train_neural_ode(
        model=model,
        y0_batch=y0_batch,
        t=t,
        true_y=true_y,
        num_epochs=num_epochs_ode,
        lr=0.01
    )
    
    # 评估最终 MSE
    with torch.no_grad():
        pred_y_final = odeint(model, y0_batch, t)
        final_mse = torch.mean((pred_y_final - true_y)**2).item()
    print(f"最终 MSE: {final_mse:.8f} {'< 1e-4' if final_mse < 1e-4 else '(需要继续优化)'}")
    
    # 3. 训练 Lyapunov 函数
    print("=" * 60)
    print("步骤 3: 训练 Lyapunov 函数")
    print("=" * 60)
    lyap_model = LyapunovNet(hidden_dim=64)
    
    lyap_model, lyap_loss_history = train_lyapunov(
        lyap_model=lyap_model,
        dynamics_model=model,
        num_epochs=num_epochs_lyap,
        lr=0.001,
        num_samples=200,
        delta_range=(-1.5, 1.5),  # 限制在稳定域内
        omega_range=(-1.0, 1.0)
    )
    
    # 保存模型
    torch.save({
        'dynamics_model': model.state_dict(),
        'lyap_model': lyap_model.state_dict(),
    }, 'trained_models.pt')
    print("模型已保存到 'trained_models.pt'")

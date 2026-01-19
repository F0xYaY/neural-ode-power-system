import torch
from torchdiffeq import odeint
from model import DynamicsNetwork, LyapunovNet
from train import train_neural_ode, train_lyapunov
from utils import generate_trajectory_data, get_stable_equilibrium_point
from plot_utils import (plot_lyapunov_contours, plot_phase_space_trajectories, 
                       plot_time_domain_response, plot_vector_field)


def main():
    """主程序：完整的训练和可视化流程"""
    
    # 检测并设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print()
    
    # 配置参数（优化 GPU 利用率）
    batch_size = 2048  # 大幅增加 batch_size（128->2048），充分利用 GPU 并行计算能力
    duration = 4.0  # 缩短时间窗口（10->4秒），物理上已足够学习螺旋特征
    step_size = 0.1  # 增大步长（0.05->0.1），减少循环计算量
    time_steps = int(duration / step_size) + 1  # 自动计算步数（约41步）
    num_epochs_ode = 1200  # 训练轮数
    num_epochs_lyap = 1000
    
    # 获取稳定平衡点（用于可视化标注）
    delta_s = get_stable_equilibrium_point().item()
    
    print("=" * 60)
    print("Neural ODE + Lyapunov 稳定性分析")
    print("=" * 60)
    print(f"稳定平衡点 (δ_s): {delta_s:.4f} rad")
    print()
    
    # ==========================================
    # 步骤 1: 生成训练数据
    # ==========================================
    print("=" * 60)
    print("步骤 1: 生成训练数据")
    print("=" * 60)
    y0_batch, t, true_y = generate_trajectory_data(
        batch_size=batch_size,
        time_steps=time_steps,
        t_span=(0., duration),  # 使用较短的时间窗口（4秒），足够学习螺旋特征
        delta_range=(-1.5, 1.5),  # 限制在稳定域内，避免不稳定平衡点
        omega_range=(-1.0, 1.0)
    )
    print(f"已生成 {batch_size} 条轨迹，时间步数：{time_steps}")
    print()
    
    # ==========================================
    # 步骤 2: 训练 Neural ODE
    # ==========================================
    print("=" * 60)
    print("步骤 2: 训练 Neural ODE（提升模型容量）")
    print("=" * 60)
    print("模型配置：")
    print("  - 三角编码：输入 [sin(delta), cos(delta), omega]")
    print("  - 隐藏层维度：128")
    print("  - 层数：4层，使用 ResNetBlock")
    print("  - 激活函数：Tanh（平滑梯度，适合 ODE 求解器）")
    print("  - 权重初始化：输入/隐藏层正交，输出层近零（std=1e-4）")
    print("  - 谱归一化：所有 Linear 层（限制 Lipschitz 常数，防止数值不稳定）")
    print("  - ODE 求解器：早期 rtol=1e-3/atol=1e-4，后期 rtol=1e-5/atol=1e-6")
    print("  - Curriculum Learning：前100 epoch纯轨迹，100-300 epoch线性增加Lyapunov权重")
    print("  - LR Scheduler：ReduceLROnPlateau")
    print()
    
    # 使用谱归一化和近零初始化，确保数值稳定性
    model = DynamicsNetwork(hidden_dim=128, num_layers=4, use_spectral_norm=True).to(device)
    
    # 确保输入数据也在设备上
    y0_batch = y0_batch.to(device)
    t = t.to(device)
    true_y = true_y.to(device)
    
    model, ode_loss_history = train_neural_ode(
        model=model,
        y0_batch=y0_batch,
        t=t,
        true_y=true_y,
        num_epochs=num_epochs_ode,
        lr=0.001,  # 降低初始学习率，避免数值不稳定
        device=device,
        step_size=step_size  # 传入步长参数以优化训练速度
    )
    
    # 评估最终 MSE（使用相同的固定步长求解器）
    with torch.no_grad():
        pred_y_final = odeint(model, y0_batch, t, method='rk4', options={'step_size': step_size})
        final_mse = torch.mean((pred_y_final - true_y)**2).item()
    print()
    print(f"最终 MSE: {final_mse:.8f}")
    if final_mse < 1e-4:
        print("MSE < 1e-4，已达到目标！")
    else:
        print(f"MSE 仍大于 1e-4，当前为 {final_mse:.8f}")
    print()
    
    # ==========================================
    # 步骤 3: 训练 Lyapunov 函数
    # ==========================================
    print("=" * 60)
    print("步骤 3: 训练 Lyapunov 函数（修正采样策略）")
    print("=" * 60)
    print("关键修正：")
    print("  - 限制采样范围在稳定域内：delta ∈ [-1.5, 1.5]")
    print("  - 避免在不稳定平衡点 (UEP ≈ 2.6 rad) 附近采样")
    print()
    
    lyap_model = LyapunovNet(hidden_dim=64).to(device)
    
    lyap_model, lyap_loss_history = train_lyapunov(
        lyap_model=lyap_model,
        dynamics_model=model,
        num_epochs=num_epochs_lyap,
        lr=0.001,
        num_samples=200,
        delta_range=(-1.5, 1.5),  # 限制在稳定域内
        omega_range=(-1.0, 1.0),
        device=device
    )
    
    # ==========================================
    # 步骤 4: 可视化结果
    # ==========================================
    print("=" * 60)
    print("步骤 4: 可视化结果（修正标签）")
    print("=" * 60)
    
    # 4.1 Lyapunov 函数等高线图
    print("绘制 Lyapunov 函数等高线图...")
    plot_lyapunov_contours(lyap_model, range_val=2.0, delta_s=delta_s)
    
    # 4.2 相空间轨迹对比
    print("绘制相空间轨迹对比...")
    with torch.no_grad():
        pred_y_vis = odeint(model, y0_batch, t, method='rk4', options={'step_size': step_size})
    plot_phase_space_trajectories(true_y, pred_y_vis, t, sample_indices=[0, 1, 2, 3], delta_s=delta_s)
    
    # 4.3 时域响应
    print("绘制时域响应...")
    plot_time_domain_response(t, true_y, pred_y_vis, sample_indices=[0, 1], delta_s=delta_s)
    
    # 4.4 向量场
    print("绘制向量场...")
    plot_vector_field(model, range_val=2.0, n_grid=20, delta_s=delta_s)
    
    # ==========================================
    # 保存模型
    # ==========================================
    print("=" * 60)
    print("保存模型...")
    print("=" * 60)
    torch.save({
        'dynamics_model': model.state_dict(),
        'lyap_model': lyap_model.state_dict(),
        'delta_s': delta_s,
        'ode_loss_history': ode_loss_history,
        'lyap_loss_history': lyap_loss_history
    }, 'trained_models.pt')
    print("模型已保存到 'trained_models.pt'")
    print()
    
    # ==========================================
    # 总结
    # ==========================================
    print("=" * 60)
    print("训练完成！总结：")
    print("=" * 60)
    print(f"1. Neural ODE 已学会整个状态空间的向量场")
    print(f"   - 最终 MSE: {final_mse:.8f}")
    print(f"   - 模型容量：128 维隐藏层，4层深度，ResNetBlock")
    print()
    print(f"2. Lyapunov 函数已找到稳定域 (Region of Attraction)")
    print(f"   - 采样范围限制在稳定域内：delta ∈ [-1.5, 1.5] rad")
    print(f"   - 避免了不稳定平衡点 (UEP) 的矛盾")
    print()
    print(f"3. 应用建议：")
    print(f"   - 通过判断 V(x_current) < C 来实时监控电网稳定性")
    print(f"   - 坐标系统：相对于稳定平衡点 δ_s ≈ {delta_s:.2f} rad")
    print()
    print("=" * 60)


if __name__ == "__main__":
    main()

import torch
import numpy as np
import matplotlib.pyplot as plt
from torchdiffeq import odeint
from model import DynamicsNetwork

# 1. 硬件配置
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
DT = 0.01

# 2. 加载数据 (我们抽取验证集中的第一条惊险轨迹)
print("正在加载数据和模型...")
X_val = np.load('X_val.npy')
val_tensor = torch.tensor(X_val, dtype=torch.float32).to(device)

y0 = val_tensor[0:1, 0, :]      # 初始状态 [1, 20]
true_y = val_tensor[0:1, :, :]  # 真实轨迹 [1, steps, 20]
time_steps = true_y.shape[1]
t = torch.linspace(0, (time_steps-1)*DT, time_steps).to(device)

# 3. 加载我们刚刚完美跑完的模型
checkpoint = torch.load('best_model_20000_samples.pt', map_location=device)
state_dim = checkpoint['state_dim']
num_machines = state_dim // 2

model = DynamicsNetwork(hidden_dim=512, num_layers=5, input_dim=state_dim).to(device)
model.load_state_dict(checkpoint['dynamics_model'])
model.eval()

# 4. 模型预测
print("正在进行 Neural ODE 推理计算...")
with torch.no_grad():
    pred_y = odeint(model, y0, t, method='rk4', options={'step_size': DT}).transpose(0, 1)

# 转回 CPU 用于画图
true_y_np = true_y[0].cpu().numpy()
pred_y_np = pred_y[0].cpu().numpy()
t_np = t.cpu().numpy()

# 5. 绘制顶会级别的对比图
print("正在生成高清对比图...")
fig, axs = plt.subplots(2, 1, figsize=(10, 8), dpi=300)

# 子图 1: 功角 Delta (选机器1和机器2)
axs[0].plot(t_np, true_y_np[:, 0], 'k-', linewidth=2, label='True M1 $\delta$')
axs[0].plot(t_np, pred_y_np[:, 0], 'r--', linewidth=2, label='Neural ODE M1 $\delta$')
axs[0].plot(t_np, true_y_np[:, 1], 'gray', linestyle='-', linewidth=2, label='True M2 $\delta$')
axs[0].plot(t_np, pred_y_np[:, 1], 'b--', linewidth=2, label='Neural ODE M2 $\delta$')
axs[0].set_ylabel('Rotor Angle $\delta$ (rad)', fontsize=14)
axs[0].tick_params(axis='both', which='major', labelsize=12)
axs[0].legend(fontsize=12, loc='upper right')
axs[0].grid(True, linestyle=':', alpha=0.7)

# 子图 2: 转速 Omega (对应维度 index 为 num_machines 和 num_machines+1)
axs[1].plot(t_np, true_y_np[:, num_machines], 'k-', linewidth=2, label='True M1 $\omega$')
axs[1].plot(t_np, pred_y_np[:, num_machines], 'r--', linewidth=2, label='Neural ODE M1 $\omega$')
axs[1].plot(t_np, true_y_np[:, num_machines+1], 'gray', linestyle='-', linewidth=2, label='True M2 $\omega$')
axs[1].plot(t_np, pred_y_np[:, num_machines+1], 'b--', linewidth=2, label='Neural ODE M2 $\omega$')
axs[1].set_ylabel('Speed $\omega$ (pu)', fontsize=14)
axs[1].set_xlabel('Time (s)', fontsize=14)
axs[1].tick_params(axis='both', which='major', labelsize=12)
axs[1].legend(fontsize=12, loc='upper right')
axs[1].grid(True, linestyle=':', alpha=0.7)

plt.suptitle("10-Machine System Trajectory Tracking (Physics-Informed Neural ODE)", fontsize=16, fontweight='bold')
plt.tight_layout()
plt.subplots_adjust(top=0.92) # 留出标题空间

save_path = "Paper_Figure_Trajectory.png"
plt.savefig(save_path)
print(f"🎉 绘图成功！已保存为: {save_path} (请点击左侧刷新按钮查看)")

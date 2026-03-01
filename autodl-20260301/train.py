import torch
import torch.optim as optim
# 【终极武器 1】：引入伴随敏感度方法，消除 BPTT 的梯度连乘问题
from torchdiffeq import odeint_adjoint as odeint
import numpy as np
import os
from model import DynamicsNetwork, LyapunovNet, get_lie_derivative

# ==========================================
# 硬件与参数配置
# ==========================================
BATCH_SIZE = 512       
MAX_EPOCHS_ODE = 1000  
PATIENCE = 60          
DT = 0.01              
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ==========================================
# CUDA 诊断
# ==========================================
print("=" * 60)
print("CUDA 诊断信息")
print("=" * 60)
print(f"PyTorch 版本: {torch.__version__}")
print(f"CUDA 是否可用: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"当前 CUDA 设备: {torch.cuda.get_device_name(0)}")
print("=" * 60)

def train_neural_ode_with_curriculum(model, train_loader, val_loader, t, num_epochs, lr):
    model = model.to(device)
    t = t.to(device)

    optimizer = optim.Adam(model.parameters(), lr=lr)
    # 放宽衰减耐心，因为伴随方法和自适应求解器的寻优路径更平滑，需要更长时间探索
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=25, min_lr=1e-6)

    best_val_loss = float('inf')
    epochs_no_improve = 0
    best_model_state = None

    print(f"开始训练 Neural ODE (Max Epochs: {num_epochs}, Patience: {PATIENCE})...")
    print(f"【核心机制激活】：Adjoint 伴随法 + dopri5 自适应求解器 + 强化谱空间联合损失")

    for epoch in range(1, num_epochs + 1):
        
        # ---------------------------------------------------------
        # 修改动态视野控制：延长短序列的训练轮数
        # 让 Adjoint 有充足的时间在局部向量场找到全局最优解
        # ---------------------------------------------------------
        if epoch <= 100:  
            current_seq_len = 40
        else:
            expansion_stage = (epoch - 100) // 20
            current_seq_len = min(40 + expansion_stage * 40, train_loader.shape[1])

        # -- 训练阶段 --
        model.train()
        train_loss_total = 0.0
        
        for batch_idx in range(0, train_loader.shape[0], BATCH_SIZE):
            batch_data = train_loader[batch_idx:batch_idx+BATCH_SIZE].to(device)
            
            # 随机截取短序列
            max_start = batch_data.shape[1] - current_seq_len
            t_start = torch.randint(0, max_start + 1, (1,)).item() if max_start > 0 else 0
            
            y0_batch = batch_data[:, t_start, :]
            true_seq = batch_data[:, t_start:t_start+current_seq_len, :]
            t_seq = t[:current_seq_len] 
            
            optimizer.zero_grad()
            
            # 【终极武器 2】：启用 dopri5 自适应求解器。
            # rtol 和 atol 控制相对/绝对误差容忍度，求解器会根据曲线陡峭程度自动切分步长。
            pred_seq = odeint(
                model, 
                y0_batch, 
                t_seq, 
                method='dopri5', 
                rtol=1e-4, 
                atol=1e-5
            ).transpose(0, 1)
            
            # =======================================================
            # 泛函/谱空间高级度量 (超越欧几里得距离)
            # =======================================================
            loss_mse = torch.mean((pred_seq - true_seq)**2)
            
            fft_pred = torch.fft.rfft(pred_seq, dim=1)
            fft_true = torch.fft.rfft(true_seq, dim=1)
            loss_freq = torch.mean((torch.abs(fft_pred) - torch.abs(fft_true))**2) * 0.01
            
            diff_pred = (pred_seq[:, 1:, :] - pred_seq[:, :-1, :]) / DT
            diff_true = (true_seq[:, 1:, :] - true_seq[:, :-1, :]) / DT
            loss_sobolev = torch.mean((diff_pred - diff_true)**2) * 0.1
            
            std_pred = torch.std(pred_seq, dim=1)
            std_true = torch.std(true_seq, dim=1)
            loss_var = torch.mean((std_pred - std_true)**2) * 0.5
            
            # 联合损失
            loss = loss_mse + loss_freq + loss_sobolev + loss_var
            # =======================================================
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            train_loss_total += loss.item() * batch_data.shape[0]
            
        avg_train_loss = train_loss_total / train_loader.shape[0]

        # -- 验证阶段 --
        model.eval()
        val_loss_total = 0.0
        with torch.no_grad():
            for batch_idx in range(0, val_loader.shape[0], BATCH_SIZE):
                batch_data = val_loader[batch_idx:batch_idx+BATCH_SIZE].to(device)
                y0_batch = batch_data[:, 0, :]
                
                # 验证集同样采用 dopri5 求解
                pred_y = odeint(
                    model, 
                    y0_batch, 
                    t, 
                    method='dopri5', 
                    rtol=1e-4, 
                    atol=1e-5
                ).transpose(0, 1)
                
                val_loss_total += torch.mean((pred_y - batch_data)**2).item() * batch_data.shape[0]
                
        avg_val_loss = val_loss_total / val_loader.shape[0]

        scheduler.step(avg_val_loss)
        current_lr = optimizer.param_groups[0]['lr']

        if epoch % 10 == 0 or epoch == 1:
            print(f"Epoch {epoch:04d} [Len:{current_seq_len:3d}] | Train Loss(联合): {avg_train_loss:.6f} | Val MSE(全长): {avg_val_loss:.6f} | LR: {current_lr:.6f}")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            epochs_no_improve = 0
            best_model_state = model.state_dict()
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= PATIENCE:
            print(f"\n[早停触发] 连续 {PATIENCE} 轮纯 MSE 无提升。停在第 {epoch} 轮。")
            break
            
        if current_lr <= 1.000000e-06 and epochs_no_improve >= 15:
            print(f"\n[极限终止] 学习率触底且连续 15 轮无提升，终止训练。停在第 {epoch} 轮。")
            break

    print(f"Neural ODE 训练结束。加载最佳权重 (Val MSE: {best_val_loss:.6f}).\n")
    model.load_state_dict(best_model_state)
    return model

if __name__ == "__main__":
    print("=" * 60)
    print("步骤 1: 硬盘加载大规模数据集")
    print("=" * 60)
    try:
        X_train = np.load('X_train.npy')
        X_val = np.load('X_val.npy')
        print(f"成功加载数据! Train shape: {X_train.shape}, Val shape: {X_val.shape}")
    except FileNotFoundError:
        print("错误：找不到 X_train.npy 或 X_val.npy！")
        exit(1)

    train_tensor = torch.tensor(X_train, dtype=torch.float32)
    val_tensor = torch.tensor(X_val, dtype=torch.float32)

    num_samples, time_steps, state_dim = train_tensor.shape
    t = torch.linspace(0, (time_steps-1)*DT, time_steps)

    print("=" * 60)
    print("步骤 2: 训练 Neural ODE (终极伴随与自适应求解版)")
    print("=" * 60)
    
    checkpoint_path = 'best_model_20000_samples.pt'
    if os.path.exists(checkpoint_path):
        os.remove(checkpoint_path)
        print("已清理历史权重文件。")

    model = DynamicsNetwork(hidden_dim=512, num_layers=5, input_dim=state_dim, use_spectral_norm=False)

    model = train_neural_ode_with_curriculum(
        model=model,
        train_loader=train_tensor,
        val_loader=val_tensor,
        t=t,
        num_epochs=MAX_EPOCHS_ODE,
        lr=0.001  # 回调至 0.001，配合平滑准确的伴随梯度
    )

    print("=" * 60)
    print("步骤 3: 最终模型评估与保存")
    print("=" * 60)

    model.eval()
    with torch.no_grad():
        y0_test = val_tensor[:100, 0, :].to(device)
        true_y_test = val_tensor[:100, :, :].to(device)
        
        # 评估同样使用自适应求解器
        pred_y_final = odeint(
            model, 
            y0_test, 
            t.to(device), 
            method='dopri5', 
            rtol=1e-4, 
            atol=1e-5
        ).transpose(0, 1)
        
        final_mse = torch.mean((pred_y_final - true_y_test)**2).item()

    print(f"最终评估 MSE: {final_mse:.8f}")

    torch.save({
        'dynamics_model': model.state_dict(),
        'state_dim': state_dim
    }, 'best_model_20000_samples.pt')
    print("全部流程完美结束！模型已保存")

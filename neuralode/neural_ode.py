
import torch
import torch.nn as nn
from torchdiffeq import odeint
import matplotlib.pyplot as plt


class ODEFunc(nn.Module):
    """
    Neural ODE function class that computes the derivative of a 2D state.
    
    This class defines a neural network that learns the vector field (derivative)
    of a 2D dynamical system. It can be used with torchdiffeq.odeint to solve
    the ODE: dy/dt = f(t, y), where f is learned by this neural network.
    
    Args:
        None (uses default architecture)
    
    Forward Args:
        t: current time point (scalar or tensor)
        y: current state vector of shape (batch_size, 2)
    
    Returns:
        dy/dt: derivative of the state vector, shape (batch_size, 2)
    """
    
    def __init__(self):
        super(ODEFunc, self).__init__()
        # Define a simple MLP to learn the vector field
        self.net = nn.Sequential(
            nn.Linear(2, 50),
            nn.Tanh(),
            nn.Linear(50, 2),
        )
    
    def forward(self, t, y):
        """
        Compute the derivative dy/dt given time t and state y.
        
        Args:
            t: current time (can be scalar or tensor, typically not used in autonomous ODEs)
            y: current state vector of shape (batch_size, 2)
        
        Returns:
            dy/dt: derivative of the state vector, shape (batch_size, 2)
        """
        # t is the current time point, y is the current state
        return self.net(y)


if __name__ == "__main__":
    # Create an instance of the ODE function
    func = ODEFunc()
    
    # Initial state: batch_size=1, state_dim=2
    y0 = torch.tensor([[1.0, 0.5]], requires_grad=True)
    
    # Time points to evaluate
    t = torch.linspace(0., 1., 100)
    
    # Target trajectory (example: you can replace this with your actual target)
    # Shape: (time_steps, batch_size, state_dim)
    y_target = torch.zeros_like(torch.zeros(100, 1, 2))
    for i, time in enumerate(t):
        y_target[i, 0, 0] = torch.cos(time * 4 * 3.14159)
        y_target[i, 0, 1] = torch.sin(time * 4 * 3.14159)
    
    # 初始化模型和优化器
    optimizer = torch.optim.Adam(func.parameters(), lr=0.01)
    
    # 开启交互模式
    plt.ion()
    fig, ax = plt.subplots(1, 1, figsize=(8, 8))

    # 模拟训练过程
    for itr in range(1, 101):
        optimizer.zero_grad()
        
        # 前向传播：求解 IVP 问题
        # t 是时间轴张量，例如 torch.linspace(0, 10, 100)
        pred_y = odeint(func, y0, t) 
        
        # 计算损失：预测轨迹与真实轨迹的 MSE
        loss = torch.mean(torch.abs(pred_y - y_target))
        
        # 反向传播：torchdiffeq 会自动处理伴随状态法(如果指定方法)
        loss.backward()
        optimizer.step()

        if itr % 10 == 0:
            print(f'Iter {itr} | Loss {loss.item():.4f}')
            
            ax.cla()
            # 1. 绘制目标轨迹和预测轨迹
            ax.plot(y_target[:, 0, 0].detach(), y_target[:, 0, 1].detach(), 'k--', label='Target')
            ax.plot(pred_y[:, 0, 0].detach(), pred_y[:, 0, 1].detach(), 'b-', label='Predicted')
            
            # 2. 绘制向量场 (Vector Field)
            y_mesh, x_mesh = torch.meshgrid(torch.linspace(-1.5, 1.5, 20), torch.linspace(-1.5, 1.5, 20), indexing='xy')
            grid = torch.stack([x_mesh, y_mesh], dim=-1).reshape(-1, 2)
            with torch.no_grad():
                v = func(0, grid)  # 计算网格点上的导数
            ax.quiver(grid[:, 0], grid[:, 1], v[:, 0], v[:, 1], color='red', alpha=0.3)
            
            ax.set_title(f'Iter {itr} | Loss {loss.item():.4f}')
            ax.legend()
            ax.set_xlabel('x')
            ax.set_ylabel('y')
            plt.pause(0.1)

    plt.ioff()
    plt.show()
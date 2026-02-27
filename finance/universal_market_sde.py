import akshare as ak
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchsde
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tqdm import tqdm # 进度条库
import time
import os
import requests

# 禁用代理设置，避免代理连接错误
os.environ['NO_PROXY'] = '*'
os.environ['no_proxy'] = '*'
os.environ['HTTP_PROXY'] = ''
os.environ['HTTPS_PROXY'] = ''
os.environ['http_proxy'] = ''
os.environ['https_proxy'] = ''

# 配置 requests 不使用代理
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# 创建自定义 session，禁用代理
session = requests.Session()
session.proxies = {
    'http': None,
    'https': None,
}

# 配置重试策略
retry_strategy = Retry(
    total=3,
    backoff_factor=1,
    status_forcelist=[429, 500, 502, 503, 504],
)
adapter = HTTPAdapter(max_retries=retry_strategy)
session.mount("http://", adapter)
session.mount("https://", adapter)

# 尝试将 session 应用到 akshare（如果支持）
try:
    # akshare 内部使用 requests，尝试设置默认 session
    import akshare.tool.tool_api as tool_api
    if hasattr(tool_api, 'session'):
        tool_api.session = session
except:
    pass

# ==========================================
# 0. 显卡配置
# ==========================================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if torch.cuda.is_available():
    print(f"[OK] 硬件: {torch.cuda.get_device_name(0)}")
else:
    print("[INFO] 使用 CPU")
# 开启 TF32 加速 (对 SRK 方法的矩阵运算加速显著)
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

# ==========================================
# 1. 数据工程 (带缓存功能)
# ==========================================
STOCK_POOL = [
    "600519", "300750", "600036", "601888", "000858", 
    "601318", "002594", "601012", "603288", "000333"
]

def download_stock_data(code, start_date, end_date, max_retries=3):
    """下载单只股票数据，带重试机制和代理禁用"""
    import contextlib
    
    @contextlib.contextmanager
    def no_proxy_context():
        """临时禁用代理的上下文管理器"""
        # 保存原始环境变量
        original_env = {}
        proxy_vars = ['HTTP_PROXY', 'HTTPS_PROXY', 'http_proxy', 'https_proxy', 
                     'NO_PROXY', 'no_proxy', 'ALL_PROXY', 'all_proxy']
        for var in proxy_vars:
            original_env[var] = os.environ.get(var)
            if var in ['NO_PROXY', 'no_proxy']:
                os.environ[var] = '*'
            else:
                os.environ[var] = ''
        
        # 保存并清除 requests 的代理设置
        original_requests_proxies = getattr(requests, 'proxies', {})
        requests.proxies = {'http': None, 'https': None}
        
        try:
            yield
        finally:
            # 恢复环境变量
            for var, value in original_env.items():
                if value is None:
                    os.environ.pop(var, None)
                else:
                    os.environ[var] = value
            # 恢复 requests 代理设置
            requests.proxies = original_requests_proxies
    
    for attempt in range(max_retries):
        try:
            with no_proxy_context():
                df = ak.stock_zh_a_hist(
                    symbol=code, 
                    period="daily", 
                    start_date=start_date, 
                    end_date=end_date, 
                    adjust="hfq"
                )
            
            if df is not None and not df.empty:
                df['date'] = pd.to_datetime(df['日期'])
                df = df.set_index('date')
                return df
            else:
                raise ValueError(f"股票 {code} 返回空数据")
                
        except Exception as e:
            if attempt < max_retries - 1:
                wait_time = (attempt + 1) * 2  # 指数退避：2s, 4s, 6s
                print(f"[RETRY] 股票 {code} 第 {attempt + 1} 次尝试失败，{wait_time}秒后重试...")
                time.sleep(wait_time)
            else:
                raise e
    return None

def get_batch_data_cached(codes=STOCK_POOL, start_date="20140101", end_date="20240101"):
    cache_file = "batch_data_10y.pt"
    if os.path.exists(cache_file):
        print(f"[CACHE] 读取本地缓存: {cache_file}")
        return torch.load(cache_file)

    print(f"[INFO] 正在下载 {len(codes)} 只股票数据 (此过程仅需一次)...")
    print("[INFO] 已禁用代理设置，使用直连方式下载...")
    data_dict = {}
    failed_codes = []
    
    # 使用 tqdm 显示下载进度
    for code in tqdm(codes, desc="Downloading"):
        try:
            df = download_stock_data(code, start_date, end_date)
            if df is not None:
                data_dict[code] = df
            else:
                failed_codes.append(code)
        except Exception as e:
            print(f"[WARN] 股票 {code} 下载失败: {str(e)[:100]}...")
            failed_codes.append(code)
        
        time.sleep(0.2)  # 避免请求过快

    if not data_dict:
        error_msg = "没有成功下载任何股票数据。"
        if failed_codes:
            error_msg += f"\n失败的股票代码: {', '.join(failed_codes)}"
        error_msg += "\n\n可能的解决方案："
        error_msg += "\n1. 检查网络连接"
        error_msg += "\n2. 检查代理设置（已尝试禁用代理）"
        error_msg += "\n3. 稍后重试（可能是服务器临时问题）"
        raise ValueError(error_msg)
    
    if failed_codes:
        print(f"[WARN] 有 {len(failed_codes)} 只股票下载失败: {', '.join(failed_codes)}")
        print(f"[INFO] 将继续使用成功下载的 {len(data_dict)} 只股票数据")
    
    # 以茅台为基准对齐
    if "600519" not in data_dict:
        # 如果没有茅台，使用第一个
        base_code = list(data_dict.keys())[0]
    else:
        base_code = "600519"
    
    base_index = data_dict[base_code].index
    batch_S, batch_B, valid_codes = [], [], []

    print("[INFO] 正在对齐时间轴与归一化...")
    for code, df in tqdm(data_dict.items(), desc="Processing"):
        df_aligned = df.reindex(base_index).ffill().fillna(0)
        
        # 特征工程
        raw_B = df_aligned['成交额'].values.astype(float)
        vol = (df_aligned['最高'] - df_aligned['最低']).values.astype(float)
        raw_S = 1.0 / (vol + 1e-3)
        raw_S[np.isinf(raw_S)] = 0
        raw_S[np.isnan(raw_S)] = 0
        raw_B[np.isnan(raw_B)] = 0
        
        # 归一化
        scaler_B = MinMaxScaler(feature_range=(0.1, 2.0))
        scaler_S = MinMaxScaler(feature_range=(0.1, 2.0))
        
        batch_S.append(scaler_S.fit_transform(raw_S.reshape(-1, 1)).flatten())
        batch_B.append(scaler_B.fit_transform(raw_B.reshape(-1, 1)).flatten())
        valid_codes.append(code)

    # 堆叠 Tensor
    tensor_S = np.stack(batch_S, axis=1)
    tensor_B = np.stack(batch_B, axis=1)
    train_data = np.stack([tensor_S, tensor_B], axis=2)
    
    train_tensor = torch.FloatTensor(train_data).to(DEVICE)
    ts = torch.linspace(0, 10, len(base_index)).to(DEVICE)
    
    # 保存缓存
    torch.save((ts, train_tensor, valid_codes), cache_file)
    print(f"[OK] 数据已缓存到: {cache_file}")
    return ts, train_tensor, valid_codes

# ==========================================
# 2. 模型定义 (Universal Market SDE)
# ==========================================
class UniversalMarketSDE(nn.Module):
    def __init__(self, batch_size):
        super().__init__()
        self.sde_type = "ito"
        self.noise_type = "diagonal"
        
        # 物理参数
        self.log_r = nn.Parameter(torch.tensor(0.0))    
        self.log_k = nn.Parameter(torch.tensor(0.0))    
        self.log_beta = nn.Parameter(torch.tensor(-1.0))
        self.log_alpha = nn.Parameter(torch.tensor(0.0))
        self.K, self.c, self.d = 3.0, 0.8, 0.5

        # 神经网络
        self.net_drift = nn.Sequential(
            nn.Linear(3, 128), nn.Tanh(),
            nn.Linear(128, 128), nn.Tanh(),
            nn.Linear(128, 3), nn.Tanh()
        )
        self.net_diffusion = nn.Sequential(
            nn.Linear(3, 64), nn.ReLU(),
            nn.Linear(64, 3), nn.Softplus()
        )
        
        # 初始化权重
        self._initialize_weights()

    def _initialize_weights(self):
        """初始化网络权重"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    @property
    def params(self):
        return (torch.exp(self.log_r), torch.exp(self.log_k), 
                torch.exp(self.log_beta), torch.exp(self.log_alpha))

    def f(self, t, y):
        """Drift 函数 (确定性部分)"""
        S, B, W = y[:, 0:1], y[:, 1:2], y[:, 2:3]
        r, k, beta, alpha = self.params
        
        fear_term = 1.0 + k * W 
        interaction = (beta * S * B) / fear_term
        
        dS = r * S * (1 - S/self.K) - interaction
        dB = self.c * interaction - self.d * B
        dW = alpha * (B - W)
        
        phys_drift = torch.cat([dS, dB, dW], dim=1)
        return phys_drift + self.net_drift(y) * 0.2

    def g(self, t, y):
        """Diffusion 函数 (随机部分)"""
        return self.net_diffusion(y) * 0.1

# ==========================================
# 3. 训练循环 (SRK + TQDM)
# ==========================================
def train_with_srk_tqdm():
    """使用SRK方法训练通用市场SDE模型"""
    # 1. 数据准备
    ts, train_tensor, codes = get_batch_data_cached()
    batch_size = train_tensor.shape[1]
    
    target_S = train_tensor[:, :, 0]
    target_B = train_tensor[:, :, 1]
    
    # 初始状态 y0 = [S0, B0, W0=B0]
    S0 = target_S[0].unsqueeze(1)
    B0 = target_B[0].unsqueeze(1)
    y0 = torch.cat([S0, B0, B0], dim=1).to(DEVICE)
    
    # 2. 模型
    model = UniversalMarketSDE(batch_size).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.005)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=10, min_lr=1e-6)
    
    print(f"\n[INFO] 启动高精度训练 | Method: SRK | Batch: {batch_size} Stocks | Time: 10 Years")
    print("注: SRK 方法计算量较大，进度条稍慢属于正常现象...")
    print(f"模型参数数量: {sum(p.numel() for p in model.parameters()):,}")
    
    # === TQDM 进度条设置 ===
    epochs = 200
    pbar = tqdm(range(epochs), desc="Training SDE", unit="ep", ncols=100)
    
    history_loss = []
    start_time = time.time()
    
    for epoch in pbar:
        optimizer.zero_grad()
        
        # --- 核心：严格保持 SRK 方法 ---
        # dt=0.05 保证积分稳定性
        try:
            ys_pred = torchsde.sdeint(model, y0, ts, method='srk', dt=0.05)
        except Exception as e:
            print(f"\n[ERROR] SDE求解失败: {e}")
            break
        
        pred_S = ys_pred[:, :, 0]
        pred_B = ys_pred[:, :, 1]
        
        loss = F.mse_loss(pred_S, target_S) + \
               F.mse_loss(pred_B, target_B)
        
        loss.backward()
        
        # 梯度裁剪
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        scheduler.step(loss.item())
        
        # 记录 Loss
        loss_val = loss.item()
        history_loss.append(loss_val)
        
        # 获取物理参数
        r, k, b, a = model.params
        
        # --- 实时更新进度条右侧指标 ---
        # 格式化显示：Loss保留4位，参数保留2位
        pbar.set_postfix({
            'Loss': f'{loss_val:.4f}',
            'Fear(k)': f'{k.item():.2f}',
            'Mem(α)': f'{a.item():.2f}'
        })

    print(f"\n[OK] 训练完成! 耗时: {time.time() - start_time:.1f}s")
    
    # ==========================================
    # 4. 绘图
    # ==========================================
    print("\n[INFO] 生成可视化图表...")
    
    model.eval()
    with torch.no_grad():
        # 同样使用 SRK 进行推理绘图
        final_pred = torchsde.sdeint(model, y0, ts, method='srk', dt=0.05).cpu().detach().numpy()
    
    target_S_cpu = target_S.cpu().numpy()
    target_B_cpu = target_B.cpu().numpy()
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 茅台 (Index 0)
    axes[0,0].plot(target_B_cpu[:, 0], color='gray', alpha=0.3, label='Real Capital')
    axes[0,0].plot(final_pred[:, 0, 1], color='orange', label='Model Capital')
    axes[0,0].set_title(f'{codes[0]} - Capital Flow (Predator)', fontweight='bold')
    axes[0,0].set_xlabel('Time')
    axes[0,0].set_ylabel('Normalized Value')
    axes[0,0].legend()
    axes[0,0].grid(True, alpha=0.3)
    
    axes[0,1].plot(target_S_cpu[:, 0], color='gray', alpha=0.3, label='Real Liquidity')
    axes[0,1].plot(final_pred[:, 0, 2], color='red', linestyle='--', linewidth=2, label='Inferred Fear (W)')
    axes[0,1].set_title(f'{codes[0]} - Hidden Fear Memory', fontweight='bold')
    axes[0,1].set_xlabel('Time')
    axes[0,1].set_ylabel('Normalized Value')
    axes[0,1].legend()
    axes[0,1].grid(True, alpha=0.3)

    # 宁德时代 (Index 1)
    if len(codes) > 1:
        axes[1,0].plot(target_B_cpu[:, 1], color='gray', alpha=0.3, label='Real Capital')
        axes[1,0].plot(final_pred[:, 1, 1], color='orange', label='Model Capital')
        axes[1,0].set_title(f'{codes[1]} - Capital Flow', fontweight='bold')
        axes[1,0].set_xlabel('Time')
        axes[1,0].set_ylabel('Normalized Value')
        axes[1,0].legend()
        axes[1,0].grid(True, alpha=0.3)
    
    # 训练 Loss
    axes[1,1].plot(history_loss, linewidth=2)
    axes[1,1].set_yscale('log')
    axes[1,1].set_title('Training Loss (SRK Method)', fontweight='bold')
    axes[1,1].set_xlabel('Epoch')
    axes[1,1].set_ylabel('Loss')
    axes[1,1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('universal_market_sde_results.png', dpi=300, bbox_inches='tight')
    print("[OK] 图表已保存到: universal_market_sde_results.png")
    plt.show()
    
    # 打印最终参数
    r, k, b, a = model.params
    print("\n" + "=" * 60)
    print("最终学习到的物理参数:")
    print("=" * 60)
    print(f"  卖单再生率 (r): {r.item():.4f}")
    print(f"  恐惧因子 (k): {k.item():.4f}")
    print(f"  成交效率 (beta): {b.item():.4f}")
    print(f"  记忆衰减率 (alpha): {a.item():.4f}")
    print("=" * 60)
    
    return model, ts, train_tensor, codes

if __name__ == "__main__":
    model, ts, data, codes = train_with_srk_tqdm()

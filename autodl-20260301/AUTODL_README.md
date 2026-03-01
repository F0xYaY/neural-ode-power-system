# AutoDL 云端训练清单

## 1. SSH 登录

```bash
ssh -p 44506 root@connect.westc.gpuhub.com
```
密码：`1uH7WA6p8kRR`

或在 Cursor 里：**Remote-SSH** → 添加 Host → 同上地址与端口，连接后输入密码。

---

## 2. 打开数据盘目录

连接后在 Cursor 左侧 **打开文件夹**，路径填：

```
/root/autodl-tmp
```

（不要用默认系统盘 `/root`，避免占满空间。）

---

## 3. 上传文件到 `/root/autodl-tmp/`

把下面这些**拖进** `/root/autodl-tmp/`（同一目录即可）：

**代码（neuralode 目录下）：**
- `train_multidim.py`
- `model.py`
- `npy_dataset.py`

**数据（仓库根目录的 2 万条）：**
- `X_train.npy`
- `X_val.npy`
- `X_test.npy`

最终目录结构示例：

```
/root/autodl-tmp/
├── train_multidim.py
├── model.py
├── npy_dataset.py
├── X_train.npy
├── X_val.npy
├── X_test.npy
└── run_on_autodl.sh    （可选，见下）
```

---

## 4. 安装依赖

在 Cursor 里打开 **终端**（当前路径应为 `/root/autodl-tmp`），执行：

```bash
pip install torchdiffeq numpy matplotlib tqdm
```

（AutoDL 一般已带 PyTorch/CUDA，无需再装。）

---

## 5. 启动训练（后台运行）

在 **同一目录** 下执行：

```bash
nohup python -u train_multidim.py --dt 0.01 --batch_size 1024 --epochs_dyn 300 --epochs_lyap 300 --alpha 0.5 --train X_train.npy --val X_val.npy --test X_test.npy > train.log 2>&1 &
```

- `--train/--val/--test` 用当前目录下的 npy（和默认的 `../X_*.npy` 区分开）。
- 日志在 `train.log`，查看：`tail -f /root/autodl-tmp/train.log`。

**发表级挂机（300+300 epoch）：**

```bash
nohup python -u train_multidim.py --dt 0.01 --batch_size 1024 --epochs_dyn 300 --epochs_lyap 300 --alpha 0.5 --train X_train.npy --val X_val.npy --test X_test.npy > train.log 2>&1 &
```

---

## 6. 常用命令

| 操作           | 命令 |
|----------------|------|
| 看实时日志     | `nohup python -u train.py > train.log 2>&1 &`

| 看进程是否在跑 | `tail -f train.log`


---                                                                              `python train.py`
                                                                            `python plot_paper.py`
                                                                            `python train_lyapunov.py`
                                                                             ` python plot_lyapunov.py`



## 7. 注意

- 数据盘在 **/root/autodl-tmp**，代码和数据都放这里。
- 训练完成后权重默认在同上目录：`trained_models_multidim.pt`，可下载到本机。

#!/bin/bash
# 在 AutoDL /root/autodl-tmp 下运行：上传此脚本 + 三个 .npy + neuralode 的 .py 后执行
# chmod +x run_on_autodl.sh && ./run_on_autodl.sh

set -e
cd /root/autodl-tmp

echo "[1/2] Installing deps..."
pip install torchdiffeq numpy matplotlib tqdm -q

echo "[2/2] Starting training (nohup, 300+300 epoch)..."
nohup python -u train_multidim.py --dt 0.01 --batch_size 1024 --epochs_dyn 300 --epochs_lyap 300 --alpha 0.5 --train X_train.npy --val X_val.npy --test X_test.npy > train.log 2>&1 &

echo "Done. PID=$!"
echo "View log: tail -f /root/autodl-tmp/train.log"

"""在 Cora 上训练两层 GCN，复现论文 ~81.5% 的测试精度。

用法（在 venv 里）：
    python experiments/train_cora.py
    python experiments/train_cora.py --epochs 200 --hidden 16 --lr 0.01

超参数默认值与论文一致：
    hidden=16, dropout=0.5, lr=0.01, weight_decay=5e-4, max 200 epochs,
    早停窗口 10（验证 loss 连续 10 轮不降即停）。
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim

# 让脚本能 import src/ 下的模块
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from data import load_cora          # noqa: E402
from gcn import GCN                 # noqa: E402


def accuracy(output: torch.Tensor, labels: torch.Tensor) -> float:
    preds = output.max(1)[1]
    correct = preds.eq(labels).sum().item()
    return correct / len(labels)


def train_once(adj, features, labels, idx_train, idx_val, idx_test, args, seed):
    """跑一次完整训练，返回 (best_test_acc, final_test_acc)。"""
    torch.manual_seed(seed)
    np.random.seed(seed)

    model = GCN(nfeat=features.shape[1],
                nhid=args.hidden,
                nclass=labels.max().item() + 1,
                dropout=args.dropout)
    optimizer = optim.Adam(model.parameters(),
                           lr=args.lr, weight_decay=args.weight_decay)

    best_val_loss = float("inf")
    best_test_acc = 0.0
    bad_epochs = 0

    for epoch in range(1, args.epochs + 1):
        # 训练步
        model.train()
        optimizer.zero_grad()
        output = model(features, adj)
        loss_train = F.nll_loss(output[idx_train], labels[idx_train])
        loss_train.backward()
        optimizer.step()

        # 验证步（关 dropout）
        model.eval()
        with torch.no_grad():
            output = model(features, adj)
            loss_val = F.nll_loss(output[idx_val], labels[idx_val])

        # 早停 + 记录验证 loss 最低时的测试精度
        if loss_val.item() < best_val_loss:
            best_val_loss = loss_val.item()
            bad_epochs = 0
            best_test_acc = accuracy(output[idx_test], labels[idx_test])
        else:
            bad_epochs += 1
            if bad_epochs >= args.patience:
                break

    # 最终评估
    model.eval()
    with torch.no_grad():
        output = model(features, adj)
        final_test_acc = accuracy(output[idx_test], labels[idx_test])

    return best_test_acc, final_test_acc


def main():
    parser = argparse.ArgumentParser(description="Train GCN on Cora")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--runs", type=int, default=1,
                        help="重复次数（>1 时报告均值±标准差，与论文一致）")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--weight_decay", type=float, default=5e-4)
    parser.add_argument("--hidden", type=int, default=16)
    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument("--patience", type=int, default=10,
                        help="早停窗口（验证 loss 连续 N 轮不降即停）")
    parser.add_argument("--data", type=str,
                        default=str(Path(__file__).resolve().parents[1] / "data" / "cora"))
    args = parser.parse_args()

    # ---- 数据（只加载一次）------------------------------------------------
    adj, features, labels, idx_train, idx_val, idx_test = load_cora(args.data)

    t_start = time.time()
    if args.runs == 1:
        best, final = train_once(adj, features, labels, idx_train, idx_val,
                                 idx_test, args, args.seed)
        print("\n" + "=" * 52)
        print(f"Test  set: accuracy = {final:.4f}")
        print(f"Best  test accuracy @ min-val-loss = {best:.4f}")
        print(f"论文报告 Cora 精度：81.5% ± 0.7%")
        print(f"耗时 {time.time() - t_start:.2f}s")
        print("=" * 52)
    else:
        bests, finals = [], []
        for r in range(args.runs):
            seed = args.seed + r
            b, f = train_once(adj, features, labels, idx_train, idx_val,
                              idx_test, args, seed)
            bests.append(b)
            finals.append(f)
            print(f"  run {r + 1:2d} (seed {seed}): "
                  f"best={b:.4f} final={f:.4f}")

        bests = np.array(bests)
        finals = np.array(finals)
        print("\n" + "=" * 52)
        print(f"共 {args.runs} 次随机初始化")
        print(f"Best  test accuracy: {bests.mean():.4f} ± {bests.std():.4f}")
        print(f"Final test accuracy: {finals.mean():.4f} ± {finals.std():.4f}")
        print(f"论文报告 Cora 精度：81.5% ± 0.7%")
        print(f"耗时 {time.time() - t_start:.2f}s")
        print("=" * 52)


if __name__ == "__main__":
    main()

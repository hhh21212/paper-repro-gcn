# Cora 基准复现结果

> 复现目标：GCN (Kipf & Welling, ICLR 2017) 在 Cora 上的半监督节点分类精度。
> 论文报告：**81.5% ± 0.7%**

## 结果

| 指标 | 复现值 | 论文值 |
|------|--------|--------|
| Best test accuracy（验证 loss 最低处） | **81.48% ± 0.72%** | 81.5% ± 0.7% |
| Final test accuracy（早停时） | 81.68% ± 0.53% | — |

复现值来自 **20 次随机初始化**（seed 42~61），与论文报告口径一致，完全落在误差范围内。

## 实验配置

| 超参数 | 值 |
|--------|-----|
| 层数 | 2（hidden=16） |
| dropout | 0.5 |
| 优化器 | Adam, lr=0.01, weight_decay=5e-4 |
| 早停 | 验证 loss 连续 10 轮不降即停，最多 200 epoch |
| 数据划分 | Planetoid 标准：140 训练 / 500 验证 / 1000 测试 |
| 特征 | 原始词袋特征（0/1，未归一化） |
| 归一化 | 对称归一化 Â = D̃^(-1/2)(A+I)D̃^(-1/2) |

## 单次运行示例（seed 42）

```
Test set accuracy = 0.8090
Best test accuracy @ min-val-loss = 0.8040
```

## 复现要点（关键差异记录）

1. **对称归一化 vs 行归一化**：官方 `tkipf/pygcn` 的 `utils.normalize` 用的是
   **行归一化 D̃^(-1)Ã**（随机游走形式），而论文原文公式 (4) 是
   **对称归一化 Â = D̃^(-1/2) Ã D̃^(-1/2)**。本复现严格按论文实现对称归一化。

2. **官方代码不可直接跑**：`tkipf/gcn`（TF 1.x）与 `tkipf/pygcn`（PyTorch 0.4）
   均依赖过老，无法在 Python 3.13 上直接运行。真正的复现工作是
   「读懂论文数学 → 用现代 PyTorch 重写」，本仓库 `src/` 即为此产物。

3. **特征未归一化**：论文使用原始词袋特征（不做行归一化），
   官方 pygcn 反而做了行归一化，这是官方实现与论文的又一处不一致。

## 复现命令

```bash
# 单次
python experiments/train_cora.py

# 20 次求均值±标准差（与论文口径一致）
python experiments/train_cora.py --runs 20
```

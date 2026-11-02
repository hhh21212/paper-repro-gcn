# paper-repro-gcn

复现论文：*Semi-Supervised Classification with Graph Convolutional Networks* (Kipf & Welling, ICLR 2017)

## 目标

- 在 Cora / Citeseer / PubMed 上复现 GCN 半监督节点分类结果。
- 手写 GCN 层核心：从谱域卷积 → 一阶近似 → 消息传递形式。
- 做消融实验：层数、dropout、L2 正则化、隐藏层维度。
- 与官方实现对比，记录差异原因。

## 目录规划

```
paper-repro-gcn/
├── README.md
├── src/                 # 手写 GCN 实现
├── official/            # 官方代码（fork / clone）
├── data/                # 数据集（Cora/Citeseer/PubMed）
├── experiments/         # 训练脚本与实验配置
├── results/             # 实验结果与图表
├── notes/               # 论文阅读笔记
└── daily/               # 每日工作日志
```

## 复现指标

| 数据集 | 论文报告 | 复现目标 | 状态 |
|--------|----------|----------|------|
| Cora | 81.5% | 80.5%–82.5% | ✅ **81.48% ± 0.72%** |
| Citeseer | 70.3% | 69.0%–71.5% | ⏳ 待做 |
| PubMed | 79.0% | 78.0%–80.0% | ⏳ 待做 |

Cora 复现详情见 [`results/cora-baseline.md`](results/cora-baseline.md)。

## 代码结构

```
src/
├── data.py          # 数据加载 + 对称归一化（论文公式 4）
└── gcn.py           # 手写 GraphConvolutionLayer + 两层 GCN
experiments/
└── train_cora.py    # 训练脚本（支持 --runs N 求均值±标准差）
```

## 下一步

1. [x] 克隆官方代码到 `official/`（TF 版）与 `official-pytorch/`（PyTorch 版）。
2. [x] 跑通 Cora 基准（手写现代 PyTorch 版，81.48%）。
3. [ ] 精读论文并整理公式到 `notes/`。
4. [ ] 复现 Citeseer、PubMed。
5. [ ] 消融实验（层数、dropout、隐藏维度、L2）。

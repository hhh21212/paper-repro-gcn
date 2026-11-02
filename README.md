# paper-repro-gcn

复现论文：*Semi-Supervised Classification with Graph Convolutional Networks* (Kipf & Welling, ICLR 2017)

> **仓库现在的定位**：这是我（陈亦州）**从零学习并亲自复现**的记录。
> 2026-11-02 之前仓库里的实现由 AI 生成、我本人没有参与，已整体移入 `baseline/`，只作对照参考，**不算我的复现成果**。
> 学习计划与考核方式见 `D:\Dev\Projects\learning\PLAN.md`。

## 目录

```
paper-repro-gcn/
├── baseline/        # 早期 AI 生成的参考实现（只读，不作为本人成果）
│   ├── src/         #   手写 GCN 层 + 对称归一化（AI 写）
│   ├── experiments/ #   训练脚本（AI 写）
│   └── results/     #   AI 跑出的 Cora 结果
├── src/             # 我本人写的实现（从零重写中）
├── experiments/     # 我本人的训练与实验脚本
├── results/         # 我本人的实验结果
├── notes/           # 论文精读（四关推导，我自己写）
├── daily/           # 每日工作日志
└── data/            # 数据集（gitignore）
```

## 复现进度

| 数据集 | 论文报告 | baseline（AI 生成，仅参考） | 本人复现 |
|--------|----------|---------------------------|----------|
| Cora | 81.5% ± 0.7% | 81.48% ± 0.72%（20 次） | ⏳ 未开始 |
| Citeseer | 70.3% | — | ⏳ |
| PubMed | 79.0% | — | ⏳ |

## 本人当前阶段

2026-11-02 起：W0 摸底 → A 数值地基 → B 机器学习最小闭环 → … → E 论文四关 → F 自己重写。
进度看板：`D:\Dev\Projects\learning\PROGRESS.md`

## 常用命令

```powershell
# 跑 baseline 参考实现（只用于对照，不是我的成果）
$env:PYTHONIOENCODING='utf-8'; $env:PYTHONUTF8='1'
D:\Dev\Projects\venv\Scripts\python.exe baseline/experiments/train_cora.py --runs 20
```

## 数据获取

数据集不入库（`data/` 已在 `.gitignore`，Cora 的 `cora.content` 有 7.8 MB）。
从 https://linqs.soe.ucsc.edu/data 下载 `cora.content` + `cora.cites`，放到 `data/cora/` 即可。
Citeseer / PubMed 同理；或后续用 PyG 的 `Planetoid` 自动下载。

## 与论文的已知差异（baseline 记录，待我亲自复验）

1. 官方 `tkipf/pygcn` 用行归一化 $\tilde D^{-1}\tilde A$，论文公式 (4) 是对称归一化 $\tilde D^{-1/2}\tilde A\tilde D^{-1/2}$。
2. 论文第 5 节明确写了对输入特征做 (row-)normalize，baseline 没做——这处待我自己验证。
3. 官方 TF/PyTorch 代码过老，跑不动的部分需要按论文数学重写。

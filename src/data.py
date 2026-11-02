"""Cora 数据集加载与图预处理。

严格对应论文 (Kipf & Welling, ICLR 2017) 的公式 (2)~(6)：

    Ã  = A + I_N                       自环（公式 2）
    D̃_ii = Σ_j Ã_ij                    度矩阵（公式 3）
    Â  = D̃^(-1/2) Ã D̃^(-1/2)            对称归一化（公式 4）

注意：官方 tkipf/pygcn 的 utils.normalize 用的是「行归一化 D̃^(-1)Ã」，
与论文的「对称归一化」不同。这里实现的是论文原文的对称归一化。
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import scipy.sparse as sp
import torch


def _symmetric_normalize(adj: sp.spmatrix) -> sp.spmatrix:
    """对称归一化：Â = D̃^(-1/2) Ã D̃^(-1/2)（论文公式 4）。

    参数 adj 应已经是带自环的 Ã = A + I_N。
    """
    rowsum = np.asarray(adj.sum(axis=1)).flatten()
    d_inv_sqrt = np.power(rowsum, -0.5)
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.0
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return d_mat_inv_sqrt @ adj @ d_mat_inv_sqrt


def load_cora(data_dir: str | Path = "../data/cora") -> tuple:
    """加载 Cora 引用网络数据集。

    返回:
        adj         归一化邻接矩阵 Â，torch.FloatTensor (N, N)
        features    词袋特征 X，torch.FloatTensor (N, 1433)
        labels      类别标签 y，torch.LongTensor (N,)
        idx_train   训练节点索引（前 140 个）
        idx_val     验证节点索引（第 200~499 个）
        idx_test    测试节点索引（第 500~1499 个）

    划分遵循论文 Planetoid 标准划分：140 训练 / 500 验证 / 1000 测试。
    """
    data_dir = Path(data_dir)

    # ---- 1. 读 node features 与 labels -----------------------------------
    # 每行: <paper_id> <1433 个词袋特征> <class_label>
    idx_features_labels = np.genfromtxt(
        data_dir / "cora.content", dtype=np.dtype(str)
    )
    features = sp.csr_matrix(idx_features_labels[:, 1:-1], dtype=np.float32)
    raw_labels = idx_features_labels[:, -1]

    # 字符串标签 -> 整数索引
    classes = sorted(set(raw_labels.tolist()))
    class_to_idx = {c: i for i, c in enumerate(classes)}
    y = np.array([class_to_idx[l] for l in raw_labels], dtype=np.int64)

    # ---- 2. 建邻接矩阵 ----------------------------------------------------
    paper_ids = np.array(idx_features_labels[:, 0], dtype=np.int64)
    id_to_idx = {pid: i for i, pid in enumerate(paper_ids)}

    # cites 每行 (被引 id, 引用 id)，论文把引用图当无向图处理
    edges = np.genfromtxt(data_dir / "cora.cites", dtype=np.int64)
    src = np.array([id_to_idx[i] for i in edges[:, 0]], dtype=np.int64)
    dst = np.array([id_to_idx[i] for i in edges[:, 1]], dtype=np.int64)

    n = features.shape[0]
    adj = sp.coo_matrix(
        (np.ones(src.shape[0], dtype=np.float32), (src, dst)),
        shape=(n, n),
    )

    # 无向化：A = A + A^T，再二值化去掉重复边
    adj = adj + adj.T
    adj = (adj > 0).astype(np.float32)

    # 加自环（公式 2） + 对称归一化（公式 4）
    adj = adj + sp.eye(n, dtype=np.float32)
    adj = _symmetric_normalize(adj)

    # ---- 3. 转 torch 张量 --------------------------------------------------
    # Cora 只有 2708 个节点，稠密矩阵完全放得下，也便于讲解。
    features_t = torch.FloatTensor(np.asarray(features.todense()))
    labels_t = torch.LongTensor(y)
    adj_t = torch.FloatTensor(np.asarray(adj.todense()))

    # ---- 4. Planetoid 标准划分 --------------------------------------------
    idx_train = torch.LongTensor(range(140))
    idx_val = torch.LongTensor(range(200, 500))
    idx_test = torch.LongTensor(range(500, 1500))

    return adj_t, features_t, labels_t, idx_train, idx_val, idx_test

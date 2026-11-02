"""手写 GCN 层与模型。

严格对应论文 (Kipf & Welling, ICLR 2017) 的核心公式 (8)：

    H^(l+1) = σ( Â H^(l) W^(l) )         单层传播
    Z = softmax( Â ReLU( Â X W^(0) ) W^(1) )   两层 GCN

其中 Â = D̃^(-1/2) Ã D̃^(-1/2) 是预归一化的邻接矩阵（见 data.py）。
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class GraphConvolutionLayer(nn.Module):
    """单层图卷积：H' = σ( Â H W )

    - x @ W        : 线性变换（每个节点独立）
    - adj @ (...)  : 邻居特征聚合（消息传播）
    """

    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter("bias", None)
        self.reset_parameters()

    def reset_parameters(self):
        # 论文用 Glorot 初始化；官方实现简化为 uniform(-1/sqrt(F), 1/sqrt(F))
        stdv = 1.0 / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        """x: (N, F_in)，adj: (N, N) 归一化邻接矩阵"""
        support = x @ self.weight      # (N, F_out)
        output = adj @ support         # (N, F_out) 邻居聚合
        if self.bias is not None:
            output = output + self.bias
        return output


class GCN(nn.Module):
    """两层 GCN（论文公式 8）。

    结构：输入 -> GCN 层(ReLU) -> Dropout -> GCN 层 -> log_softmax
    """

    def __init__(self, nfeat: int, nhid: int, nclass: int, dropout: float):
        super().__init__()
        self.gc1 = GraphConvolutionLayer(nfeat, nhid)
        self.gc2 = GraphConvolutionLayer(nhid, nclass)
        self.dropout = dropout

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.gc1(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.gc2(x, adj)
        return F.log_softmax(x, dim=1)

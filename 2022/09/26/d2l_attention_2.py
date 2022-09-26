import torch
from torch import nn
import matplotlib.pyplot as plt
from d2l import torch as d2l


# 我们可以在下⾯的masked_softmax函数中实现这样的掩蔽softmax操作（masked softmax operation），
# 其中任何超出有效⻓度的位置都被掩蔽并置为0。
def masked_softmax(X, valid_lens):
# """通过在最后⼀个轴上掩蔽元素来执⾏softmax操作"""
# X:3D张量，valid_lens:1D或2D张量
    if valid_lens is None:
        return nn.functional.softmax(X, dim=-1)
    else:
        shape = X.shape
        if valid_lens.dim() == 1:
            valid_lens = torch.repeat_interleave(valid_lens, shape[1])
        else:
            valid_lens = valid_lens.reshape(-1) # 最后⼀轴上被掩蔽的元素使⽤⼀个⾮常⼤的负值替换，从⽽其softmax输出为0
        X = d2l.sequence_mask(X.reshape(-1, shape[-1]), valid_lens,value=-1e6)
        return nn.functional.softmax(X.reshape(shape), dim=-1)

# 当查询和键是不同⻓度的⽮量时，我们可以使⽤加性注意⼒作为评分函数
# 将查询和键连结起来后输⼊到⼀个多层感知机（MLP）中，
# 感知机包含⼀个隐藏层，其隐藏单元数是⼀个超参数h。
# 通过使⽤tanh作为激活函数，并且禁⽤偏置项。


# 加性注意力

class AdditiveAttention(nn.Module):
# """加性注意⼒"""
    def __init__(self, key_size, query_size, num_hiddens, dropout, **kwargs):
        super(AdditiveAttention, self).__init__(**kwargs)
        self.W_k = nn.Linear(key_size, num_hiddens, bias=False)
        self.W_q = nn.Linear(query_size, num_hiddens, bias=False)
        self.w_v = nn.Linear(num_hiddens, 1, bias=False)
        self.dropout = nn.Dropout(dropout)
    def forward(self, queries, keys, values, valid_lens):
        queries, keys = self.W_q(queries), self.W_k(keys)
        # 在维度扩展后，
        # queries的形状：(batch_size，查询的个数，1，num_hidden)
        # key的形状：(batch_size，1，“键－值”对的个数，num_hiddens)
        # 使⽤⼴播⽅式进⾏求和
        features = queries.unsqueeze(2) + keys.unsqueeze(1)
        features = torch.tanh(features)
        # self.w_v仅有⼀个输出，因此从形状中移除最后那个维度。
        # scores的形状：(batch_size，查询的个数，“键-值”对的个数)
        scores = self.w_v(features).squeeze(-1)
        self.attention_weights = masked_softmax(scores, valid_lens)
        # values的形状：(batch_size，“键－值”对的个数，值的维度)
        return torch.bmm(self.dropout(self.attention_weights), values)
# 其中查询、键和值的形状为（批量⼤⼩，步数或元序列⻓度，特征⼤⼩）。
# 注意⼒汇聚输出的形状为（批量⼤⼩，查询的步数，值的维度）。

# 缩放点积注意⼒

# 使⽤点积可以得到计算效率更⾼的评分函数，
# 但是点积操作要求查询和键具有相同的⻓度d。
# 假设查询和键的所有元素都是独⽴的随机变量，
# 并且都满⾜零均值和单位⽅差，
# 那么两个向量的点积的均值为0，⽅差为d。
# 为确保⽆论向量⻓度如何，点积的⽅差在不考虑向量⻓度的情况下仍然是1，我们将点积除以√d，
# 则缩放点积注意⼒（scaled dot-product attention）评分函数为：
# a(q, k) = q转置*k/√d.

import math
class DotProductAttention(nn.Module):
# """缩放点积注意⼒"""
    def __init__(self, dropout, **kwargs):
        super(DotProductAttention, self).__init__(**kwargs)
        self.dropout = nn.Dropout(dropout)
# queries的形状：(batch_size，查询的个数，d)
# keys的形状：(batch_size，“键－值”对的个数，d)
# values的形状：(batch_size，“键－值”对的个数，值的维度)
# valid_lens的形状:(batch_size，)或者(batch_size，查询的个数)
    def forward(self, queries, keys, values, valid_lens=None):
        d = queries.shape[-1] # 设置transpose_b=True为了交换keys的最后两个维度
        scores = torch.bmm(queries, keys.transpose(1,2)) / math.sqrt(d)
        self.attention_weights = masked_softmax(scores, valid_lens)
        return torch.bmm(self.dropout(self.attention_weights), values)

# summary
# 将注意⼒汇聚的输出计算可以作为值的加权平均，
# 选择不同的注意⼒评分函数会带来不同的注意⼒汇聚操作。
# 当查询和键是不同⻓度的⽮量时，可以使⽤可加性注意⼒评分函数。
# 当它们的⻓度相同时，使⽤缩放的“点－积”注意⼒评分函数的计算效率更⾼。
'''
Author: Zhangrunbang 254616730@qq.com
Date: 2025-07-16 14:14:18
LastEditors: Zhangrunbang 254616730@qq.com
LastEditTime: 2025-07-16 14:53:22
FilePath: /LSK3DNet/network/multi_head_self_attention.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''
import torch
import torch.nn as nn
import torch.nn.functional as F

class SelfAttention(nn.Module):
    def __init__(self, embed_size, heads):
        super(SelfAttention, self).__init__()
        self.embed_size = embed_size
        self.heads = heads
        self.head_dim = embed_size // heads
        assert self.head_dim * heads == embed_size, "embed_size必须能被heads整除"
        # 用一个大线性层一次性生成QKV
        self.to_qkv = nn.Linear(embed_size, embed_size * 3, bias=False)
        self.fc_out = nn.Linear(embed_size, embed_size)

    def forward(self, x, mask=None):
        batch_size, seq_length, embed_size = x.shape
        qkv = self.to_qkv(x)  # [batch, seq, 3*embed_size]
        qkv = qkv.reshape(batch_size, seq_length, 3, self.heads, self.head_dim)
        queries, keys, values = qkv[:,:,0], qkv[:,:,1], qkv[:,:,2]  # [batch, seq, heads, head_dim]
        queries = queries.permute(0,2,1,3)  # [batch, heads, seq, head_dim]
        keys = keys.permute(0,2,1,3)
        values = values.permute(0,2,1,3)
        energy = torch.einsum("bhqd,bhkd->bhqk", [queries, keys])
        energy = energy / (self.head_dim ** 0.5)
        if mask is not None:
            energy = energy.masked_fill(mask == 0, float("-1e20"))
        attention = torch.softmax(energy, dim=-1)
        out = torch.einsum("bhqk,bhkd->bhqd", [attention, values])
        out = out.permute(0,2,1,3).reshape(batch_size, seq_length, self.embed_size)
        out = self.fc_out(out)
        return out
    
class SingleHeadAttention(nn.Module):
    def __init__(self, embed_size):
        super(SingleHeadAttention, self).__init__()
        self.embed_size = embed_size
        self.to_q = nn.Linear(embed_size, embed_size, bias=False)
        self.to_k = nn.Linear(embed_size, embed_size, bias=False)
        self.to_v = nn.Linear(embed_size, embed_size, bias=False)
        self.fc_out = nn.Linear(embed_size, embed_size)

    def forward(self, x, mask=None):
        # x: [batch, seq, embed_size]
        queries = self.to_q(x)  # [batch, seq, embed_size]
        keys = self.to_k(x)
        values = self.to_v(x)
        energy = torch.einsum("bqd,bkd->bqk", [queries, keys])  # [batch, seq, seq]
        energy = energy / (self.embed_size ** 0.5)
        if mask is not None:
            energy = energy.masked_fill(mask == 0, float("-1e20"))
        attention = torch.softmax(energy, dim=-1)  # [batch, seq, seq]
        out = torch.einsum("bqk,bkd->bqd", [attention, values])  # [batch, seq, embed_size]
        out = self.fc_out(out)
        return out

class CrossAttention(nn.Module):
    """
    cross_attn = CrossAttention(embed_size)
    output = cross_attn(features1, features2)
    """
    def __init__(self, embed_size):
        super(CrossAttention, self).__init__()
        self.embed_size = embed_size
        self.to_q = nn.Linear(embed_size, embed_size, bias=False)
        self.to_k = nn.Linear(embed_size, embed_size, bias=False)
        self.to_v = nn.Linear(embed_size, embed_size, bias=False)
        self.fc_out = nn.Linear(embed_size, embed_size)

    def forward(self, features1, features2, mask=None):
        # features1: [batch, seq1, embed_size] 作为 query
        # features2: [batch, seq2, embed_size] 作为 key/value
        queries = self.to_q(features1)  # [batch, seq1, embed_size]
        keys = self.to_k(features2)     # [batch, seq2, embed_size]
        values = self.to_v(features2)   # [batch, seq2, embed_size]
        energy = torch.einsum("bqd,bkd->bqk", [queries, keys])  # [batch, seq1, seq2]
        energy = energy / (self.embed_size ** 0.5)
        if mask is not None:
            energy = energy.masked_fill(mask == 0, float("-1e20"))
        attention = torch.softmax(energy, dim=-1)  # [batch, seq1, seq2]
        out = torch.einsum("bqk,bkd->bqd", [attention, values])  # [batch, seq1, embed_size]
        out = self.fc_out(out)
        return out


class MultiHeadCrossAttention(nn.Module):
    def __init__(self, embed_size, heads):
        super(MultiHeadCrossAttention, self).__init__()
        self.embed_size = embed_size
        self.heads = heads
        self.head_dim = embed_size // heads
        assert self.head_dim * heads == embed_size, "embed_size必须能被heads整除"
        # 用一个大线性层一次性生成QKV
        self.to_q = nn.Linear(embed_size, embed_size, bias=False)
        self.to_k = nn.Linear(embed_size, embed_size, bias=False)
        self.to_v = nn.Linear(embed_size, embed_size, bias=False)
        self.fc_out = nn.Linear(embed_size, embed_size)

    def forward(self, features1, features2, mask=None):
        # features1: [batch, seq1, embed_size] (query)
        # features2: [batch, seq2, embed_size] (key/value)
        batch_size = features1.shape[0]

        # 线性变换
        queries = self.to_q(features1)  # [batch, seq1, embed_size]
        keys = self.to_k(features2)     # [batch, seq2, embed_size]
        values = self.to_v(features2)   # [batch, seq2, embed_size]

        # 分头
        queries = queries.view(batch_size, -1, self.heads, self.head_dim).transpose(1,2)  # [batch, heads, seq1, head_dim]
        keys = keys.view(batch_size, -1, self.heads, self.head_dim).transpose(1,2)        # [batch, heads, seq2, head_dim]
        values = values.view(batch_size, -1, self.heads, self.head_dim).transpose(1,2)    # [batch, heads, seq2, head_dim]

        # 注意力分数
        energy = torch.einsum("bhqd,bhkd->bhqk", [queries, keys])  # [batch, heads, seq1, seq2]
        energy = energy / (self.head_dim ** 0.5)
        if mask is not None:
            energy = energy.masked_fill(mask == 0, float("-1e20"))
        attention = torch.softmax(energy, dim=-1)  # [batch, heads, seq1, seq2]

        # 加权求和
        out = torch.einsum("bhqk,bhkd->bhqd", [attention, values])  # [batch, heads, seq1, head_dim]
        out = out.transpose(1,2).contiguous().view(batch_size, -1, self.embed_size)  # [batch, seq1, embed_size]
        out = self.fc_out(out)
        return out
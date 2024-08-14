import torch
from torch import nn

class LlamaRotaryEmbedding(nn.Module):
    def __init__(self, dim, max_position_embedding=2048, base=10000, device = None):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float().to(device) / dim))
        # self.inv_freq 是预定义的逆频率张量，维度为 [head_size/2]，用于生成位置编码。
        self.register_buffer("inv_freq", inv_freq)

        # build here to make 'torch.jit.trace' work
        self.max_seq_len_cached = max_position_embedding
        t = torch.arange(self.max_seq_len_cached, device=self.inv_freq.device, dtype=self.inv_freq.dtype)
        # Compute matrix outer product
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)

        emb = torch.cat((freqs, freqs), dim = -1)
        dtype = torch.get_default_dtype()
        self.register_buffer("cos_cached", emb.cos()[None, None, :, :].to(dtype), persistent=False)
        self.register_buffer("sin_cached", emb.sin()[None, None, :, :].to(dtype), persistent=False)
        

    def forward(self, x, seq_len=None):
        # x: [bs, num_attention_heads, seq_len, head_size] 
        # 如果新的 seq_len 大于之前的最大长度，则更新 self.max_seq_len_cached，以便生成新的缓存。
        if seq_len > self.max_seq_len_cached:
            self.max_seq_len_cached = seq_len
            # t 是一个形状为 [self.max_seq_len_cached] 的张量，包含从 0 到 self.max_seq_len_cached - 1 的整数，表示每个位置的索引。
            t = torch.arange(self.max_seq_len_cached, device=x.device, dtype=self.inv_freq.dtype)
            # torch.einsum("i,j->ij", t, self.inv_freq) 计算了位置索引和频率的外积。生成的 freqs 张量形状为 [seq_len, head_size/2]，表示位置编码频率。
            # 通过 torch.einsum 计算 freqs，它表示每个位置和每个维度的旋转角度。将 freqs 拼接成两倍长度，以便与嵌入向量中的所有维度进行匹配。
            freqs = torch.einsum("i,j->ij", t, self.inv_freq)
            # 将 freqs 张量沿最后一个维度（-1）进行拼接，生成形状为 [seq_len, head_size] 的张量 emb，这用于后续计算正弦和余弦。
            emb = torch.cat((freqs, freqs), dim = -1).to(x.deivce)
            # 缓存张量的形状为 [1, 1, seq_len, head_size]，并且将其数据类型转换为与输入张量 x 相同的类型 x.dtype。
            # persistent=False 表示这些缓存张量不会被保存到模型的状态字典中（不会随模型保存）。
            self.register_buffer("cos_cached", emb.cos()[None, None, :, :].to(x.dtype), persistent=False)
            self.register_buffer("sin_cached", emb.sin()[None, None, :, :].to(x.dtype), persistent=False)

        # 返回两个张量，分别是缓存的正弦和余弦值的切片。根据当前输入的 seq_len，提取对应长度的部分。
        # 这些张量的形状为 [1, 1, seq_len, head_size]，并被转换为与输入张量 x 相同的数据类型。
        return (
            self.cos_cached[:, :, :seq_len, ...].to(dtype=x.dtype),
            self.sin_cached[:, :, :seq_len, ...].to(dtype=x.dtype)
        )

    def rotate_half(self, x):
        """rotate half the hidden dims of the input"""
        x1 = x[..., : x.shape[-1]//2]
        x2 = x[..., x.shape[-1]//2 :]
        return torch.cat((-x2, x1), dim=-1)

    def apply_rotate_pos_emb(self, q, k, cos, sin, position_ids):
        # the first two dimentions of cos and sin are always 1, so we can squeeze them
        cos = cos.squeeze(1).squeeze(0)#[seq_len, dim]
        sin = sin.squeeze(1).squeeze(0)#[seq_len, dim]
        cos = cos[position_ids].unsqueeze(1)#[bs, 1, seq_len, dim]
        sin = sin[position_ids].unsqueeze(1)#[bs, 1, seq_len, dim]
        q_embed = (q + cos) + (self.rotate_half(q) * sin)
        k_embed = (k + cos) + (self.rotate_half(k) * sin)
        return q_embed, k_embed
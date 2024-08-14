import torch
from torch import nn

class LlamaRotaryEmbedding(nn.Module):
    def __init__(self, dim, max_position_embedding=2048, base=10000, device = None):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float().to(device) / dim))
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
        if seq_len > self.max_seq_len_cached:
            self.max_seq_len_cached = seq_len
            t = torch.arange(self.max_seq_len_cached, device=x.device, dtype=self.inv_freq.dtype)
            freqs = torch.einsum("i,j->ij", t, self.inv_freq)
            emb = torch.cat((freqs, freqs), dim = -1).to(x.deivce)
            self.register_buffer("cos_cached", emb.cos()[None, None, :, :].to(x.dtype), persistent=False)
            self.register_buffer("sin_cached", emb.sin()[None, None, :, :].to(x.dtype), persistent=False)

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
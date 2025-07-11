# cross_attention.py

import torch
import torch.nn as nn
import torch.nn.functional as F

class CrossAttention3D(nn.Module):
    def __init__(self, dim, num_heads=4):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)

    def forward(self, x_q, x_kv):
        """
        Args:
            x_q: (B, C, D, H, W) - query volume (T1)
            x_kv: (B, C, D, H, W) - key/value volume (T2)

        Returns:
            out: attention-modulated features (B, C, D, H, W)
        """
        B, C, D, H, W = x_q.shape

        # Flatten spatial dims
        x_q = x_q.view(B, C, -1).permute(0, 2, 1)     # (B, N, C)
        x_kv = x_kv.view(B, C, -1).permute(0, 2, 1)   # (B, N, C)

        Q = self.q_proj(x_q)  # (B, N, C)
        K = self.k_proj(x_kv)
        V = self.v_proj(x_kv)

        # Split into heads
        Q = Q.view(B, -1, self.num_heads, self.head_dim).transpose(1, 2)  # (B, heads, N, d)
        K = K.view(B, -1, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(B, -1, self.num_heads, self.head_dim).transpose(1, 2)

        # Cross-attention
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) * self.scale
        attn_weights = torch.softmax(attn_scores, dim=-1)

        attn_out = torch.matmul(attn_weights, V)  # (B, heads, N, d)
        attn_out = attn_out.transpose(1, 2).reshape(B, -1, self.dim)  # (B, N, C)

        out = self.out_proj(attn_out)
        return out.permute(0, 2, 1).view(B, C, D, H, W)

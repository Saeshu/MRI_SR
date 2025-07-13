import torch
import torch.nn as nn
import torch.nn.functional as F

class WindowSelfAttention3D(nn.Module):
    def __init__(self, dim, window_size=4, num_heads=2):
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        self.head_dim = dim // num_heads if dim % num_heads == 0 else 1
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3)
        self.attn_proj = nn.Linear(dim, dim)

    def forward(self, x):
        B, C, D, H, W = x.shape
        x = x.permute(0, 2, 3, 4, 1)  # [B, D, H, W, C]

        pad_D = (self.window_size - D % self.window_size) % self.window_size
        pad_H = (self.window_size - H % self.window_size) % self.window_size
        pad_W = (self.window_size - W % self.window_size) % self.window_size
        x = F.pad(x, (0, 0, 0, pad_W, 0, pad_H, 0, pad_D))

        Dp, Hp, Wp = x.shape[1:4]
        x = x.view(
            B,
            Dp // self.window_size, self.window_size,
            Hp // self.window_size, self.window_size,
            Wp // self.window_size, self.window_size,
            C
        ).permute(0, 1, 3, 5, 2, 4, 6, 7).contiguous()
        x = x.view(-1, self.window_size ** 3, C)

        qkv = self.qkv(x).reshape(x.shape[0], x.shape[1], 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)

        attn_out = (attn @ v)
        attn_out = attn_out.transpose(1, 2).reshape(x.shape[0], x.shape[1], -1)
        out = self.attn_proj(attn_out)

        return out.mean(dim=1).view(-1, 1, 1, 1, 1)  # Modulation scalar

class AttentionModulatedConv3D(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=3):
        super().__init__()
        self.kernel = nn.Parameter(torch.randn(out_ch, in_ch, kernel_size, kernel_size, kernel_size))
        self.kernel_size = kernel_size

    def forward(self, x, modulation):
        mod_kernel = self.kernel * modulation.mean(dim=0)
        return F.conv3d(x, mod_kernel, padding=self.kernel_size // 2)

class EncoderBlock(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=3, window_size=4):
        super().__init__()
        self.attn = WindowSelfAttention3D(in_ch, window_size=window_size)
        self.dynamic_conv = AttentionModulatedConv3D(in_ch, out_ch, kernel_size)
        self.norm = nn.BatchNorm3d(out_ch)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        modulation = self.attn(x)
        x = self.dynamic_conv(x, modulation)
        x = self.norm(x)
        return self.relu(x)

class HybridEncoder(nn.Module):
    def __init__(self, in_channels=1, base_channels=32):
        super().__init__()
        self.block1 = EncoderBlock(in_channels, base_channels)
        self.block2 = EncoderBlock(base_channels, base_channels * 2)
        self.block3 = EncoderBlock(base_channels * 2, base_channels * 4)

    def forward(self, x):
        x1 = self.block1(x)  # [B, C, D, H, W]
        x2 = self.block2(x1)
        x3 = self.block3(x2)
        return x3  # can return x1,x2 if skip connections needed

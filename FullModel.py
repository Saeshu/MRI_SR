import torch
import torch.nn as nn


class SuperResolutionModel(nn.Module):
    def __init__(self, config_path, in_channels=1, base_channels=32, num_heads=4):
        super().__init__()

        # T1 and T2 encoders
        self.encoder_t1 = HybridEncoder(in_channels=in_channels, base_channels=base_channels)
        self.encoder_t2 = HybridEncoder(in_channels=in_channels, base_channels=base_channels)

        # Cross-Attention block
        self.cross_attention = CrossAttention3D(dim=base_channels * 4, num_heads=num_heads)

        # Latent Diffusion Model
        self.ldm = LatentDiffusionWrapper(config_path)

    def forward(self, t1_lr, t2_hr):
        """
        Args:
            t1_lr (Tensor): Low-resolution T1 image [B, C, D, H, W]
            t2_hr (Tensor): High-resolution T2 image [B, C, D, H, W]

        Returns:
            t1_sr (Tensor): Super-resolved T1 image [B, C, D, H, W]
        """
        # Feature extraction
        feat_t1 = self.encoder_t1(t1_lr)
        feat_t2 = self.encoder_t2(t2_hr)

        # Cross attention: T1 features query T2 features
        cond_feat = self.cross_attention(feat_t1, feat_t2)

        # Diffusion-based reconstruction
        t1_sr = self.ldm(t1_lr, cond_embed=cond_feat)

        return t1_sr

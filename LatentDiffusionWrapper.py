from monai.bundle import ConfigParser
from monai.networks.nets import AutoEncoder
from monai.inferers import DiffusionInferer
from monai.transforms import Resize
import torch.nn as nn
import torch.nn.functional as F
import torch

from monai.bundle import ConfigParser

class LatentDiffusionWrapper(nn.Module):
    def __init__(self, config_path, device="cuda"):
        super().__init__()

        # Step 1: create parser
        self.config = ConfigParser()
        
        # Step 2: read the config file
        self.config.read_config(config_path)

        self.device = device

        self.autoencoder = self.config.get_parsed_content("autoencoder").to(device)
        self.scheduler = self.config.get_parsed_content("scheduler").to(device)
        self.model = self.config.get_parsed_content("diffusion").to(device)


        self.inferer = DiffusionInferer(scheduler=self.scheduler)

    def encode(self, x):
        return self.autoencoder.encode(x)[0]

    def decode(self, z):
        return self.autoencoder.decode(z)

    def forward(self, x, cond_embed, num_inference_steps=25, guidance_scale=1.0):
        """
        x: low-res input image [B, C, D, H, W]
        cond_embed: conditioning features [B, C_latent, D, H, W]
        """
        z = self.encode(x)  # Encode to latent space

        # Make noise
        noise = torch.randn_like(z)
        timesteps = torch.linspace(1.0, 0.0, steps=num_inference_steps).to(z.device)

        # Use cross-attn conditional embed as 'condition'
        denoised_latent = self.inferer.sample(
            input_noise=noise,
            network=self.model,
            conditioning=cond_embed,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
        )

        return self.decode(denoised_latent)

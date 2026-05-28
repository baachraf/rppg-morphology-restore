"""
models/decoder_v6.py — V6 Conditional Decoder
=============================================
Replaces the standard deterministic VAE decoder. 
This takes a concatenated condition (z_macro, z_micro)
and maps it to a high-fidelity waveform.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class ConditionalDecoder(nn.Module):
    """
    Decodes the fused condition vector into a 256-sample cardiac cycle.
    Input: z_macro (B, latent_dim), z_micro (B, latent_dim)
    Output: (B, 1, 256) in [0, 1]
    """
    def __init__(self, latent_dim=32):
        super().__init__()
        
        # We concatenate macro and micro, so input dim is 2 * latent_dim
        self.cond_dim = latent_dim * 2
        
        self.fc = nn.Sequential(
            nn.Linear(self.cond_dim, 256 * 16),
            nn.LayerNorm(256 * 16), # LayerNorm for FC output stabilization
            nn.LeakyReLU(0.2, inplace=True),
        )
        
        # Using InstanceNorm in decoder too for V6 stability
        self.deconv = nn.Sequential(
            # (B, 256, 16) → (B, 128, 32)
            nn.ConvTranspose1d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm1d(128, affine=True), 
            nn.LeakyReLU(0.2, inplace=True),

            # (B, 128, 32) → (B, 64, 64)
            nn.ConvTranspose1d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm1d(64, affine=True), 
            nn.LeakyReLU(0.2, inplace=True),

            # (B, 64, 64) → (B, 32, 128)
            nn.ConvTranspose1d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm1d(32, affine=True), 
            nn.LeakyReLU(0.2, inplace=True),

            # (B, 32, 128) → (B, 1, 256)
            nn.ConvTranspose1d(32, 1, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid(),   # output in [0, 1]
        )

    def forward(self, z_macro, z_micro):
        """
        z_macro: (B, 32)
        z_micro: (B, 32)
        """
        # Fuse conditions
        c = torch.cat([z_macro, z_micro], dim=1)  # (B, 64)
        
        h = self.fc(c)
        h = h.view(h.size(0), 256, 16)
        
        return self.deconv(h)

"""
models/encoder_a7.py — A7: Physics-Informed RGB Encoder + Direct Decoder
=========================================================================
Input:  (batch, 6, 60) — R, G, B, R/G, G/B, R/B, mean-centered, native resolution
Output: (batch, 1, 256) — PPG waveform via direct decoder (NO VAE)

Key differences from A6:
  - 6 channels (3 raw + 3 ratios) instead of 3
  - 60-sample input (native resolution) instead of 256
  - Direct encoder→decoder (no VAE bottleneck, no Gaussian prior)
  - Decoder learns upsampling from 60→256 internally
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ResBlock1D(nn.Module):
    def __init__(self, channels, kernel_size=3):
        super().__init__()
        pad = kernel_size // 2
        self.block = nn.Sequential(
            nn.Conv1d(channels, channels, kernel_size, padding=pad, bias=False),
            nn.InstanceNorm1d(channels, affine=True),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv1d(channels, channels, kernel_size, padding=pad, bias=False),
            nn.InstanceNorm1d(channels, affine=True),
        )
        self.act = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):
        return self.act(x + self.block(x))


class PhysicsEncoderA7(nn.Module):
    """
    Encodes 6-channel native-resolution RGB window to latent vector.
    Input:  (batch, 6, 60)
    Output: (batch, latent_dim)

    4 strided conv stages: 60 → 30 → 15 → 7 → 3
    FC: 256 * 3 = 768 → latent_dim
    """
    def __init__(self, latent_dim: int = 32, in_channels: int = 6,
                 morpho_aux: bool = True):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv1d(in_channels, 32, kernel_size=7, stride=2, padding=3, bias=False),
            nn.InstanceNorm1d(32, affine=True),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.res1 = ResBlock1D(32)

        self.down1 = nn.Sequential(
            nn.Conv1d(32, 64, kernel_size=5, stride=2, padding=2, bias=False),
            nn.InstanceNorm1d(64, affine=True),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.res2 = ResBlock1D(64)

        self.down2 = nn.Sequential(
            nn.Conv1d(64, 128, kernel_size=5, stride=2, padding=2, bias=False),
            nn.InstanceNorm1d(128, affine=True),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.res3 = ResBlock1D(128)

        self.down3 = nn.Sequential(
            nn.Conv1d(128, 256, kernel_size=3, stride=2, padding=1, bias=False),
            nn.InstanceNorm1d(256, affine=True),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.res4 = ResBlock1D(256)

        self.gap = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(256, latent_dim)

        self.morpho_aux = morpho_aux
        if morpho_aux:
            self.morpho_head = nn.Sequential(
                nn.Linear(latent_dim, 32),
                nn.ReLU(),
                nn.Linear(32, 3),
                nn.Sigmoid(),
            )

    def forward(self, x):
        h = self.stem(x)
        h = self.res1(h)
        h = self.down1(h)
        h = self.res2(h)
        h = self.down2(h)
        h = self.res3(h)
        h = self.down3(h)
        h = self.res4(h)
        h = self.gap(h).squeeze(-1)
        return self.fc(h)

    def forward_morpho(self, x):
        z = self.forward(x)
        if self.morpho_aux:
            return z, self.morpho_head(z)
        return z, None


class DirectDecoder(nn.Module):
    """
    Direct decoder: z → PPG (256 samples). No VAE prior.
    Input:  (batch, latent_dim)
    Output: (batch, 1, 256)

    FC → reshape → 6 ConvTranspose1d layers (4 → 8 → 16 → 32 → 64 → 128 → 256)
    """
    def __init__(self, latent_dim: int = 32):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(latent_dim, 256 * 4),
            nn.LeakyReLU(0.2),
        )
        self.deconv = nn.Sequential(
            nn.ConvTranspose1d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm1d(128), nn.LeakyReLU(0.2),

            nn.ConvTranspose1d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm1d(64), nn.LeakyReLU(0.2),

            nn.ConvTranspose1d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm1d(32), nn.LeakyReLU(0.2),

            nn.ConvTranspose1d(32, 16, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm1d(16), nn.LeakyReLU(0.2),

            nn.ConvTranspose1d(16, 8, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm1d(8), nn.LeakyReLU(0.2),

            nn.ConvTranspose1d(8, 1, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid(),
        )

    def forward(self, z):
        h = self.fc(z)
        h = h.view(h.size(0), 256, 4)
        return self.deconv(h)


class A7Model(nn.Module):
    """Full A7 model: PhysicsEncoder + DirectDecoder."""
    def __init__(self, latent_dim: int = 32, in_channels: int = 6):
        super().__init__()
        self.encoder = PhysicsEncoderA7(latent_dim, in_channels, morpho_aux=True)
        self.decoder = DirectDecoder(latent_dim)
        self.latent_dim = latent_dim

    def forward(self, x):
        z, morpho = self.encoder.forward_morpho(x)
        recon = self.decoder(z)
        return recon, z, morpho

    def encode(self, x):
        return self.encoder(x)

    def decode(self, z):
        return self.decoder(z)

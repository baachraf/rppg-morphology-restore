"""
models/encoder_a8.py — A8: FPS-Agnostic Camera-Only Encoder + DirectDecoder
=============================================================================
Handles variable FPS (30 or 60) via AdaptiveAvgPool1d(8).
6-channel input: R, G, B, R/G, G/B, R/B (mean-centered).
No VAE, no Gaussian prior.
Primary output: (recon PPG, z, morpho_pred).
morpho_pred = [H2/H1, IPA] — drives primary training loss.
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


class PhysicsEncoderA8(nn.Module):
    """
    FPS-agnostic encoder.
    Input:  (B, 6, T) — T can be any value >= 32
    Output: (B, latent_dim)

    Stride-1 stem preserves temporal resolution early.
    Three stride-2 downsampling stages.
    AdaptiveAvgPool1d(8) normalises temporal dimension regardless of T,
    making the same weights work for T=60 (30fps/2s) and T=120 (60fps/2s).
    """

    def __init__(self, latent_dim: int = 32, in_channels: int = 6):
        super().__init__()
        # Stride-1 stem: (B, 6, T) -> (B, 32, T)
        self.stem = nn.Sequential(
            nn.Conv1d(in_channels, 32, kernel_size=7, stride=1, padding=3, bias=False),
            nn.InstanceNorm1d(32, affine=True),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.res1 = ResBlock1D(32)
        # (B, 32, T) -> (B, 64, T/2)
        self.down1 = nn.Sequential(
            nn.Conv1d(32, 64, kernel_size=5, stride=2, padding=2, bias=False),
            nn.InstanceNorm1d(64, affine=True),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.res2 = ResBlock1D(64)
        # -> (B, 128, T/4)
        self.down2 = nn.Sequential(
            nn.Conv1d(64, 128, kernel_size=5, stride=2, padding=2, bias=False),
            nn.InstanceNorm1d(128, affine=True),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.res3 = ResBlock1D(128)
        # -> (B, 256, T/8)
        self.down3 = nn.Sequential(
            nn.Conv1d(128, 256, kernel_size=3, stride=2, padding=1, bias=False),
            nn.InstanceNorm1d(256, affine=True),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.res4 = ResBlock1D(256)

        # FPS-agnostic pooling: always -> (B, 256, 8) regardless of T
        self.pool = nn.AdaptiveAvgPool1d(8)
        self.fc   = nn.Linear(256 * 8, latent_dim)

        # Morphological prediction head (primary objective)
        # Outputs [H2/H1, IPA] — both in [0, 1]
        self.morpho_head = nn.Sequential(
            nn.Linear(latent_dim, 64), nn.ReLU(),
            nn.Linear(64, 32),         nn.ReLU(),
            nn.Linear(32, 2),          nn.Sigmoid(),
        )

    def forward(self, x):
        h = self.stem(x);  h = self.res1(h)
        h = self.down1(h); h = self.res2(h)
        h = self.down2(h); h = self.res3(h)
        h = self.down3(h); h = self.res4(h)
        h = self.pool(h).view(h.size(0), -1)
        z = self.fc(h)
        return z, self.morpho_head(z)


class DirectDecoderA8(nn.Module):
    """
    z (B, latent_dim) -> PPG (B, 1, 256)
    FC -> reshape (B, 256, 4) -> 6x ConvTranspose1d -> (B, 1, 256)
    """

    def __init__(self, latent_dim: int = 32):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(latent_dim, 256 * 4), nn.LeakyReLU(0.2),
        )
        self.deconv = nn.Sequential(
            nn.ConvTranspose1d(256, 128, 4, 2, 1), nn.BatchNorm1d(128), nn.LeakyReLU(0.2),
            nn.ConvTranspose1d(128,  64, 4, 2, 1), nn.BatchNorm1d(64),  nn.LeakyReLU(0.2),
            nn.ConvTranspose1d( 64,  32, 4, 2, 1), nn.BatchNorm1d(32),  nn.LeakyReLU(0.2),
            nn.ConvTranspose1d( 32,  16, 4, 2, 1), nn.BatchNorm1d(16),  nn.LeakyReLU(0.2),
            nn.ConvTranspose1d( 16,   8, 4, 2, 1), nn.BatchNorm1d(8),   nn.LeakyReLU(0.2),
            nn.ConvTranspose1d(  8,   1, 4, 2, 1), nn.Tanh(),
        )  # 4 -> 8 -> 16 -> 32 -> 64 -> 128 -> 256 samples

    def forward(self, z):
        h = self.fc(z).view(z.size(0), 256, 4)
        return self.deconv(h)


class A8Model(nn.Module):
    """Full A8: PhysicsEncoderA8 + DirectDecoderA8."""

    def __init__(self, latent_dim: int = 32, in_channels: int = 6):
        super().__init__()
        self.encoder    = PhysicsEncoderA8(latent_dim, in_channels)
        self.decoder    = DirectDecoderA8(latent_dim)
        self.latent_dim = latent_dim

    def forward(self, x):
        z, morpho = self.encoder(x)
        recon     = self.decoder(z)
        return recon, z, morpho

    def encode(self, x):
        return self.encoder(x)[0]

    def decode(self, z):
        return self.decoder(z)

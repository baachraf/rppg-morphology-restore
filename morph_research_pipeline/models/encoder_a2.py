"""
models/encoder_a2.py — A2 CameraEncoder for Flow Decoder
========================================================
Targets z=64 for the Conditional Flow Decoder.
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


class CameraEncoderFlow(nn.Module):
    def __init__(self, latent_dim=64, in_channels=1, morpho_aux=True):
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

        self.fc = nn.Linear(256 * 16, latent_dim)

        self.morpho_aux = morpho_aux
        if morpho_aux:
            self.morpho_head = nn.Sequential(
                nn.Linear(latent_dim, 64),
                nn.ReLU(),
                nn.Linear(64, 3),
                nn.Sigmoid()
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
        h = h.view(h.size(0), -1)
        return self.fc(h)

    def forward_morpho(self, x):
        z = self.forward(x)
        if self.morpho_aux:
            return z, self.morpho_head(z)
        return z, None

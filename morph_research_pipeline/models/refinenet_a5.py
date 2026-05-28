"""
models/refinenet_a5.py — A5 Stage 2: Subject-Specific Residual RefineNet
=========================================================================
Input:  (B, 3, 256) = cat(PPG_base, rPPG_best, session_mean)
Output: residual (B, 1, 256)  — small per-subject correction

Final PPG: PPG_refined = PPG_base + residual

Architecture: 1D U-Net
  Encoder: 3→16→32→64→128 (three stride-2 downs)
  Decoder: skip-connected transpose-conv upsampling back to 256
  Output: Conv1d(16→1), no activation (residual is unbounded)

No activation on output: Pearson loss is scale/shift invariant,
residual can be positive or negative. PPG_base is in [0,1];
adding an unconstrained residual is intentional.
"""

import torch
import torch.nn as nn


class RefineNetA5(nn.Module):

    def __init__(self):
        super().__init__()

        # ── Encoder ──────────────────────────────────────────────────────────
        self.enc1 = nn.Sequential(
            nn.Conv1d(3, 16, kernel_size=7, stride=1, padding=3, bias=False),
            nn.InstanceNorm1d(16, affine=True),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv1d(16, 16, kernel_size=3, stride=1, padding=1, bias=False),
            nn.InstanceNorm1d(16, affine=True),
            nn.LeakyReLU(0.2, inplace=True),
        )  # (B, 16, 256)

        self.enc2 = nn.Sequential(
            nn.Conv1d(16, 32, kernel_size=5, stride=2, padding=2, bias=False),
            nn.InstanceNorm1d(32, affine=True),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv1d(32, 32, kernel_size=3, stride=1, padding=1, bias=False),
            nn.InstanceNorm1d(32, affine=True),
            nn.LeakyReLU(0.2, inplace=True),
        )  # (B, 32, 128)

        self.enc3 = nn.Sequential(
            nn.Conv1d(32, 64, kernel_size=5, stride=2, padding=2, bias=False),
            nn.InstanceNorm1d(64, affine=True),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv1d(64, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.InstanceNorm1d(64, affine=True),
            nn.LeakyReLU(0.2, inplace=True),
        )  # (B, 64, 64)

        self.bottleneck = nn.Sequential(
            nn.Conv1d(64, 128, kernel_size=3, stride=2, padding=1, bias=False),
            nn.InstanceNorm1d(128, affine=True),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv1d(128, 128, kernel_size=3, stride=1, padding=1, bias=False),
            nn.InstanceNorm1d(128, affine=True),
            nn.LeakyReLU(0.2, inplace=True),
        )  # (B, 128, 32)

        # ── Decoder (skip-connected) ──────────────────────────────────────────
        self.up3 = nn.ConvTranspose1d(128, 64, kernel_size=4, stride=2, padding=1)
        self.dec3 = nn.Sequential(
            nn.Conv1d(64 + 64, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.InstanceNorm1d(64, affine=True),
            nn.LeakyReLU(0.2, inplace=True),
        )  # (B, 64, 64)

        self.up2 = nn.ConvTranspose1d(64, 32, kernel_size=4, stride=2, padding=1)
        self.dec2 = nn.Sequential(
            nn.Conv1d(32 + 32, 32, kernel_size=3, stride=1, padding=1, bias=False),
            nn.InstanceNorm1d(32, affine=True),
            nn.LeakyReLU(0.2, inplace=True),
        )  # (B, 32, 128)

        self.up1 = nn.ConvTranspose1d(32, 16, kernel_size=4, stride=2, padding=1)
        self.dec1 = nn.Sequential(
            nn.Conv1d(16 + 16, 16, kernel_size=3, stride=1, padding=1, bias=False),
            nn.InstanceNorm1d(16, affine=True),
            nn.LeakyReLU(0.2, inplace=True),
        )  # (B, 16, 256)

        self.out_conv = nn.Conv1d(16, 1, kernel_size=1)  # no activation

    def forward(self, x):
        """x: (B, 3, 256) → residual: (B, 1, 256)"""
        e1 = self.enc1(x)           # (B, 16, 256)
        e2 = self.enc2(e1)          # (B, 32, 128)
        e3 = self.enc3(e2)          # (B, 64, 64)
        b  = self.bottleneck(e3)    # (B, 128, 32)

        d3 = self.up3(b)                                     # (B, 64, 64)
        d3 = self.dec3(torch.cat([d3, e3], dim=1))           # (B, 64, 64)

        d2 = self.up2(d3)                                    # (B, 32, 128)
        d2 = self.dec2(torch.cat([d2, e2], dim=1))           # (B, 32, 128)

        d1 = self.up1(d2)                                    # (B, 16, 256)
        d1 = self.dec1(torch.cat([d1, e1], dim=1))           # (B, 16, 256)

        return self.out_conv(d1)                             # (B, 1, 256)

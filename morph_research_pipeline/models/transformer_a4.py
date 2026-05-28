"""
models/transformer_a4.py — A4: Multi-Cycle Transformer Encoder
===============================================================
Aggregates 5 consecutive rPPG cycles via a Transformer to perform
temporal super-resolution. A single 30fps cycle has ~1 sample at the
dicrotic notch; across 5 phase-shifted cycles the same morphology is
sampled at different offsets, enabling sub-Nyquist reconstruction.

Architecture:
  rPPG [5 × C × 256] → Per-Cycle Conv (shared) → [5 × d_model]
                      → PosEmbed + Transformer → CLS-pool
                      → FC → z(32) → VAE Decoder → PPG(256)
"""

import math
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


class PerCycleConv(nn.Module):
    """
    Conv backbone applied independently to each cycle in the sequence.
    Produces a d_model-dim vector per cycle.

    Input:  (B, C, 256)  — single cycle
    Output: (B, d_model)
    """
    def __init__(self, in_channels=1, d_model=256):
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

        self.proj = nn.Linear(256 * 16, d_model)

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
        return self.proj(h)


class MultiCycleTransformerEncoder(nn.Module):
    """
    Aggregates a sequence of camera-domain cycles into a VAE latent vector.

    Input:  (B, num_cycles, in_channels, 256)
    Output: (B, latent_dim)

    num_cycles: number of consecutive cycles to aggregate (default 5)
    d_model:    transformer hidden dimension
    nhead:      attention heads
    n_layers:   transformer encoder layers
    """
    def __init__(self, latent_dim=32, in_channels=1, num_cycles=5,
                 d_model=256, nhead=8, n_layers=4, dropout=0.1):
        super().__init__()
        self.num_cycles = num_cycles
        self.d_model = d_model

        self.per_cycle_conv = PerCycleConv(in_channels=in_channels, d_model=d_model)

        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        self.pos_embed = nn.Parameter(torch.randn(1, num_cycles + 1, d_model) * 0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=d_model * 4,
            dropout=dropout, activation='gelu', batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        self.head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, latent_dim),
        )

    def forward(self, x):
        B, T, C, L = x.shape
        assert T == self.num_cycles, f"Expected {self.num_cycles} cycles, got {T}"

        x_flat = x.view(B * T, C, L)
        per_cycle_feat = self.per_cycle_conv(x_flat)
        per_cycle_feat = per_cycle_feat.view(B, T, self.d_model)

        cls_tokens = self.cls_token.expand(B, -1, -1)
        seq = torch.cat([cls_tokens, per_cycle_feat], dim=1)
        seq = seq + self.pos_embed

        out = self.transformer(seq)
        cls_out = out[:, 0, :]

        return self.head(cls_out)


class MultiCycleTransformerA4(nn.Module):
    """
    Full A4 model: MultiCycleTransformerEncoder → VAE Decoder → PPG.

    Wraps the encoder with a frozen VAE decoder for training convenience.
    """
    def __init__(self, encoder, vae_decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = vae_decoder

    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z)

    def encode(self, x):
        return self.encoder(x)

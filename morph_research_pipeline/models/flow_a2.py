"""
models/flow_a2.py — A2: Conditional Denoising Decoder
=====================================================
Replaces deterministic VAE decoder with a conditional iterative denoiser.

Architecture:
  z'(64) + t → FiLM conditioning
  Noisy PPG (1×256) → DownUp 1D U-Net with skip connections → Denoised PPG (1×256)

Training: predict clean x_0 from noisy x_t at random timestep t.
Sampling: DDIM-style iterative denoising from pure noise.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=t.device, dtype=torch.float32) * -emb)
        emb = t[:, None].float() * emb[None, :]
        return torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)


class FiLMLayer(nn.Module):
    def __init__(self, feature_dim, cond_dim):
        super().__init__()
        self.gamma = nn.Linear(cond_dim, feature_dim)
        self.beta = nn.Linear(cond_dim, feature_dim)

    def forward(self, x, cond):
        return x * (1 + self.gamma(cond).unsqueeze(-1)) + self.beta(cond).unsqueeze(-1)


class ResBlock1D(nn.Module):
    def __init__(self, channels, cond_dim, kernel_size=3):
        super().__init__()
        pad = kernel_size // 2
        self.conv1 = nn.Conv1d(channels, channels, kernel_size, padding=pad, bias=False)
        self.norm1 = nn.InstanceNorm1d(channels, affine=True)
        self.film = FiLMLayer(channels, cond_dim)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size, padding=pad, bias=False)
        self.norm2 = nn.InstanceNorm1d(channels, affine=True)
        self.act = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x, cond):
        h = self.act(self.norm1(self.conv1(x)))
        h = self.film(h, cond)
        h = self.norm2(self.conv2(h))
        return self.act(x + h)


class DownBlock(nn.Module):
    def __init__(self, in_ch, out_ch, cond_dim):
        super().__init__()
        self.res = ResBlock1D(in_ch, cond_dim)
        self.down = nn.Conv1d(in_ch, out_ch, 4, 2, 1, bias=False)

    def forward(self, x, cond):
        return self.down(self.res(x, cond))


class UpBlock(nn.Module):
    def __init__(self, in_ch, out_ch, cond_dim):
        super().__init__()
        self.up = nn.ConvTranspose1d(in_ch, out_ch, 4, 2, 1, bias=False)
        self.res = ResBlock1D(out_ch, cond_dim)

    def forward(self, x, skip, cond):
        x = self.up(x)
        if x.shape[2] != skip.shape[2]:
            x = x[:, :, :skip.shape[2]]
        x = x + skip
        return self.res(x, cond)


class ConditionalFlowDecoder(nn.Module):
    def __init__(self, latent_dim=64, hidden_dim=64, n_blocks=4, n_steps=10):
        super().__init__()
        self.latent_dim = latent_dim
        self.n_steps = n_steps
        cond_dim = hidden_dim * 2

        self.time_embed = nn.Sequential(
            SinusoidalTimeEmbedding(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(0.2),
        )
        self.z_proj = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.LeakyReLU(0.2),
        )

        self.input_proj = nn.Conv1d(1, hidden_dim, 1)

        ch = hidden_dim
        self.enc1 = DownBlock(ch, ch, cond_dim)
        self.enc2 = DownBlock(ch, ch * 2, cond_dim)
        self.enc3 = DownBlock(ch * 2, ch * 4, cond_dim)

        self.mid = nn.Sequential(
            ResBlock1D(ch * 4, cond_dim),
            ResBlock1D(ch * 4, cond_dim),
        )

        self.dec3 = UpBlock(ch * 4, ch * 2, cond_dim)
        self.dec2 = UpBlock(ch * 2, ch, cond_dim)
        self.dec1 = UpBlock(ch, ch, cond_dim)

        self.out_proj = nn.Sequential(
            nn.Conv1d(ch, ch, 3, padding=1, bias=False),
            nn.InstanceNorm1d(ch, affine=True),
            nn.LeakyReLU(0.2),
            nn.Conv1d(ch, 1, 1),
        )

    def _cond(self, z, t):
        return torch.cat([self.time_embed(t), self.z_proj(z)], dim=-1)

    def forward(self, z, x_noisy=None, t=None):
        if x_noisy is None:
            x_noisy = torch.randn(z.shape[0], 1, 256, device=z.device)
        if t is None:
            t = torch.rand(z.shape[0], device=z.device)

        cond = self._cond(z, t)
        h = self.input_proj(x_noisy)

        e1 = self.enc1(h, cond)
        e2 = self.enc2(e1, cond)
        e3 = self.enc3(e2, cond)

        mid = self.mid[0](e3, cond)
        mid = self.mid[1](mid, cond)

        d3 = self.dec3(mid, e2, cond)
        d2 = self.dec2(d3, e1, cond)
        d1 = self.dec1(d2, h, cond)

        return self.out_proj(d1)

    @torch.no_grad()
    def sample(self, z, n_steps=None):
        n_steps = n_steps or self.n_steps
        x = torch.randn(z.shape[0], 1, 256, device=z.device)
        for i in range(n_steps, 0, -1):
            t = torch.full((z.shape[0],), i / n_steps, device=z.device)
            x_0_pred = self.forward(z, x, t)
            if i > 1:
                next_t = (i - 1) / n_steps
                x = (1 - next_t) * x_0_pred + next_t * torch.randn_like(x)
            else:
                x = x_0_pred
        return x.clamp(0, 1)


def flow_loss(model, z, x_target):
    t = torch.rand(x_target.shape[0], device=x_target.device)
    t_expand = t.view(-1, 1, 1)
    noise = torch.randn_like(x_target)
    x_t = (1 - t_expand) * x_target + t_expand * noise
    x_pred = model.forward(z, x_t, t)
    return F.mse_loss(x_pred, x_target)

"""
models/vae.py — Stage 1 PPG Morphology VAE
============================================
Learns what a real cardiac cycle looks like from GT PPG data only.
No camera signal involved at this stage.

Input / output: 1 × 256 normalised cardiac cycle (float32)
Latent dim: 32 (configured in config.py)

Why VAE (not plain autoencoder):
  The KL term forces the latent space to be smooth and Gaussian.
  Stage 2 maps noisy camera signals into this space. A structured
  latent space allows imprecise camera-domain mappings to still
  decode to physiologically plausible waveforms.
  A plain AE produces an unstructured latent that Stage 2 cannot align to.

Architecture: 1D convolutional encoder/decoder.
  Encoder: Conv1d layers with stride=2 (downsampling) → FC → (mu, logvar)
  Decoder: FC → ConvTranspose1d layers (upsampling) → sigmoid output
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class PPGEncoder(nn.Module):
    """
    Encodes a 256-sample cardiac cycle to (mu, logvar) in latent space.
    Input shape: (batch, 1, 256)
    Output: mu (batch, latent_dim), logvar (batch, latent_dim)
    """
    def __init__(self, latent_dim=32):
        super().__init__()
        self.conv = nn.Sequential(
            # (B, 1, 256) → (B, 32, 128)
            nn.Conv1d(1, 32, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm1d(32), nn.LeakyReLU(0.2),

            # (B, 32, 128) → (B, 64, 64)
            nn.Conv1d(32, 64, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm1d(64), nn.LeakyReLU(0.2),

            # (B, 64, 64) → (B, 128, 32)
            nn.Conv1d(64, 128, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm1d(128), nn.LeakyReLU(0.2),

            # (B, 128, 32) → (B, 256, 16)
            nn.Conv1d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm1d(256), nn.LeakyReLU(0.2),
        )
        self.flat_dim = 256 * 16
        self.fc_mu     = nn.Linear(self.flat_dim, latent_dim)
        self.fc_logvar = nn.Linear(self.flat_dim, latent_dim)

    def forward(self, x):
        h = self.conv(x)
        h = h.view(h.size(0), -1)
        return self.fc_mu(h), self.fc_logvar(h)


class PPGDecoder(nn.Module):
    """
    Decodes a latent vector to a 256-sample cardiac cycle.
    Input: (batch, latent_dim)
    Output: (batch, 1, 256) in [0, 1]
    """
    def __init__(self, latent_dim=32):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(latent_dim, 256 * 16),
            nn.LeakyReLU(0.2),
        )
        self.deconv = nn.Sequential(
            # (B, 256, 16) → (B, 128, 32)
            nn.ConvTranspose1d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm1d(128), nn.LeakyReLU(0.2),

            # (B, 128, 32) → (B, 64, 64)
            nn.ConvTranspose1d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm1d(64), nn.LeakyReLU(0.2),

            # (B, 64, 64) → (B, 32, 128)
            nn.ConvTranspose1d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm1d(32), nn.LeakyReLU(0.2),

            # (B, 32, 128) → (B, 1, 256)
            nn.ConvTranspose1d(32, 1, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid(),   # output in [0, 1]
        )

    def forward(self, z):
        h = self.fc(z)
        h = h.view(h.size(0), 256, 16)
        return self.deconv(h)


class PPGVAE(nn.Module):
    """
    Full Stage 1 VAE. Encode GT PPG → (mu, logvar) → z → reconstruct PPG.
    """
    def __init__(self, latent_dim=32):
        super().__init__()
        self.encoder = PPGEncoder(latent_dim)
        self.decoder = PPGDecoder(latent_dim)
        self.latent_dim = latent_dim

    def reparameterise(self, mu, logvar):
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mu + eps * std
        return mu   # deterministic at eval time

    def forward(self, x):
        """x: (B, 1, 256) → recon: (B, 1, 256), mu, logvar"""
        mu, logvar = self.encoder(x)
        z = self.reparameterise(mu, logvar)
        recon = self.decoder(z)
        return recon, mu, logvar

    def encode(self, x):
        mu, logvar = self.encoder(x)
        return self.reparameterise(mu, logvar)

    def decode(self, z):
        return self.decoder(z)

    def sample(self, n, device):
        """Sample n random cycles from the learned prior N(0, I)."""
        z = torch.randn(n, self.latent_dim, device=device)
        return self.decode(z)


def vae_loss(recon, target, mu, logvar, beta=0.5):
    """
    Stage 1 VAE loss: L1 reconstruction + beta-weighted KL divergence.
    L1 is used over MSE because PPG cycles have sharp peaks — L1 is less
    sensitive to large individual-sample errors and more robust to outliers.
    """
    l1  = F.l1_loss(recon, target, reduction='mean')
    kld = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    return l1 + beta * kld, l1.item(), kld.item()

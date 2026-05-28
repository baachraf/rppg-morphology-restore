"""
models/vae_a3.py — A3: VQ-VAE (Vector-Quantized VAE)
=====================================================
Replaces continuous Gaussian latent with discrete codebook.
Prevents template collapse by forcing discrete morphological archetype selection.

Architecture:
  PPG → Encoder → z_e (continuous) → Quantize → z_q (nearest codebook entry) → Decoder → PPG
  Codebook: K=512 entries, dim=64

Loss: L1 reconstruction + commitment loss + codebook loss (no KL).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class VectorQuantizer(nn.Module):
    def __init__(self, num_embeddings=512, embedding_dim=64, commitment_cost=0.25):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.num_embeddings = num_embeddings
        self.commitment_cost = commitment_cost

        self.embedding = nn.Embedding(num_embeddings, embedding_dim)
        self.embedding.weight.data.uniform_(
            -1.0 / num_embeddings, 1.0 / num_embeddings
        )

    def forward(self, z_e):
        flat_z = z_e.reshape(-1, self.embedding_dim)

        dist = (
            flat_z.pow(2).sum(1, keepdim=True)
            - 2 * flat_z @ self.embedding.weight.t()
            + self.embedding.weight.pow(2).sum(1)
        )

        encoding_indices = torch.argmin(dist, dim=1)
        z_q = self.embedding(encoding_indices).view(z_e.shape)

        commitment_loss = F.mse_loss(z_e, z_q.detach())
        codebook_loss = F.mse_loss(z_q, z_e.detach())

        z_q_st = z_e + (z_q - z_e).detach()

        avg_probs = torch.mean(
            torch.zeros(flat_z.shape[0], self.num_embeddings, device=z_e.device)
            .scatter_(1, encoding_indices.unsqueeze(1), 1.0), dim=0
        )
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))

        return z_q_st, commitment_loss, codebook_loss, perplexity, encoding_indices


class PPGEncoderVQ(nn.Module):
    def __init__(self, latent_dim=64):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm1d(32), nn.LeakyReLU(0.2),
            nn.Conv1d(32, 64, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm1d(64), nn.LeakyReLU(0.2),
            nn.Conv1d(64, 128, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm1d(128), nn.LeakyReLU(0.2),
            nn.Conv1d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm1d(256), nn.LeakyReLU(0.2),
        )
        self.flat_dim = 256 * 16
        self.fc = nn.Linear(self.flat_dim, latent_dim)

    def forward(self, x):
        h = self.conv(x)
        h = h.view(h.size(0), -1)
        return self.fc(h)


class PPGDecoderVQ(nn.Module):
    def __init__(self, latent_dim=64):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(latent_dim, 256 * 16),
            nn.LeakyReLU(0.2),
        )
        self.deconv = nn.Sequential(
            nn.ConvTranspose1d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm1d(128), nn.LeakyReLU(0.2),
            nn.ConvTranspose1d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm1d(64), nn.LeakyReLU(0.2),
            nn.ConvTranspose1d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm1d(32), nn.LeakyReLU(0.2),
            nn.ConvTranspose1d(32, 1, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid(),
        )

    def forward(self, z):
        h = self.fc(z)
        h = h.view(h.size(0), 256, 16)
        return self.deconv(h)


class PPGVQVAE(nn.Module):
    def __init__(self, latent_dim=64, num_embeddings=512, commitment_cost=0.25):
        super().__init__()
        self.encoder = PPGEncoderVQ(latent_dim)
        self.decoder = PPGDecoderVQ(latent_dim)
        self.quantizer = VectorQuantizer(num_embeddings, latent_dim, commitment_cost)
        self.latent_dim = latent_dim

    def forward(self, x):
        z_e = self.encoder(x)
        z_q, commit_loss, cb_loss, perplexity, indices = self.quantizer(z_e)
        recon = self.decoder(z_q)
        return recon, commit_loss, cb_loss, perplexity, indices

    def encode(self, x):
        z_e = self.encoder(x)
        z_q, _, _, _, _ = self.quantizer(z_e)
        return z_q

    def decode(self, z):
        return self.decoder(z)


def vqvae_loss(recon, target, commit_loss, cb_loss, commitment_cost=0.25):
    l1 = F.l1_loss(recon, target, reduction='mean')
    total = l1 + commitment_cost * commit_loss + 0.25 * cb_loss
    return total, l1.item(), commit_loss.item(), cb_loss.item()

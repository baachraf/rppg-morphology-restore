"""
models/encoder_v6.py — V6 Orthogonal Cascade Encoders
=====================================================
Contains the Macro-Encoder (Identity + Rhythm) and 
Micro-Encoder (Morphology + Stiffness).

The Micro-Encoder incorporates a Gradient Reversal Layer (GRL)
to aggressively destroy Identity information, forcing it to 
only extract subject-agnostic vascular stiffness markers.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class GRL(torch.autograd.Function):
    """
    Gradient Reversal Layer.
    Forward pass: Identity.
    Backward pass: Multiplies gradients by -lambda_val.
    """
    @staticmethod
    def forward(ctx, x, lambda_val):
        ctx.lambda_val = lambda_val
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg() * ctx.lambda_val, None

def grad_reverse(x, lambda_val=1.0):
    return GRL.apply(x, lambda_val)


class ResBlock1D(nn.Module):
    """1-D residual block with InstanceNorm."""
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


class MacroEncoder(nn.Module):
    """
    Extracts z_macro. Focuses on Identity and Beat Rhythm.
    Includes a direct CE ID head (standard positive gradient) to force
    z_macro to explicitly cluster by subject identity — the symmetric
    counterpart to the GRL on z_micro.
    """
    def __init__(self, latent_dim: int = 32, in_channels: int = 1, num_subjects: int = 500):
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

        # Direct ID classification head — standard gradient (positive).
        # Forces z_macro to LEARN identity, symmetric to GRL on z_micro.
        self.id_head = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.ReLU(),
            nn.Linear(128, num_subjects)
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
        z_macro = self.fc(h)
        id_preds = self.id_head(z_macro)  # Standard gradient — learns identity
        return z_macro, id_preds


class MicroEncoder(nn.Module):
    """
    Extracts z_micro. Focuses strictly on Morphological Biomarkers (H2/H1).
    Uses Gradient Reversal to destroy Identity information.
    """
    def __init__(self, latent_dim: int = 32, in_channels: int = 1, num_subjects: int = 500):
        super().__init__()
        # Visual feature extraction pipeline (identical capacity to Macro)
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

        # Morphology regression head (Positive gradient flow)
        self.morpho_head = nn.Sequential(
            nn.Linear(latent_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 3), # [H2/H1_ratio, IPA, notch_pos]
            nn.Sigmoid() 
        )

        # Identity classification head (Negative gradient flow via GRL)
        self.id_head = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.ReLU(),
            nn.Linear(128, num_subjects)
        )

    def forward(self, x, grl_lambda=1.0):
        h = self.stem(x)
        h = self.res1(h)
        h = self.down1(h)
        h = self.res2(h)
        h = self.down2(h)
        h = self.res3(h)
        h = self.down3(h)
        h = self.res4(h)
        h = h.view(h.size(0), -1)
        
        z_micro = self.fc(h)
        
        # Forward morpho (standard)
        morpho_preds = self.morpho_head(z_micro)
        
        # Forward Identity (Gradient Reversal)
        # In backprop, the gradient from id_preds will be multiplied by -grl_lambda
        # forcing the encoder to UNLEARN features that predict identity.
        z_micro_rev = grad_reverse(z_micro, grl_lambda)
        id_preds = self.id_head(z_micro_rev)
        
        return z_micro, morpho_preds, id_preds


def orthogonal_loss(z_macro, z_micro):
    """
    Forces z_macro and z_micro to use distinct, non-overlapping visual features.
    Computes squared cosine similarity (we want it to be 0).
    """
    z_mac_norm = F.normalize(z_macro, dim=1)
    z_mic_norm = F.normalize(z_micro, dim=1)
    cos_sim = (z_mac_norm * z_mic_norm).sum(dim=1)
    return (cos_sim ** 2).mean()

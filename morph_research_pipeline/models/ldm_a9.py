"""
models/ldm_a9.py — A9 Latent Diffusion Decoder
================================================
DDPM operating in the VAE's 32-dim latent z-space, conditioned on the V5-B
CameraEncoder output z_prior. At inference, sample z via reverse diffusion
conditioned on z_prior → decode with frozen VAE decoder.

WHY DIFFUSION AVOIDS TEMPLATE COLLAPSE:

All deterministic networks (V5, V6, A1-A8) compute E[z_gt | z_prior].
For weak camera conditioning (cross-modal r ~ 0.3-0.5), this expectation
collapses to E[z_gt] = population mean z — a theorem, not a bug.

Diffusion samples from P(z_gt | z_prior), NOT E[z_gt | z_prior].
P(z_gt | z_prior) is a distribution. Different subjects have different z_prior
(V5-B r = 0.711 means z_prior has subject-specific signal) → different
conditional distributions → different stochastic samples → inherently diverse.

ARCHITECTURE:

  CameraEncoder (frozen V5-B) → z_prior (32)
  VAE PPGEncoder  (frozen)     → z_gt   (32)   [training targets only]

  NoisePredictor MLP:
    Input:  concat(z_t, z_prior, sinusoidal_time_emb) = 32+32+32 = 96 dim
    Hidden: 256 × 3 layers, SiLU activations
    Output: 32-dim predicted noise

  Training:
    t ~ Uniform(0, T)
    z_t = sqrt(ᾱ_t) * z_gt + sqrt(1-ᾱ_t) * ε,  ε ~ N(0, I)
    loss = ||ε - ε_θ(z_t, t, z_prior)||²

  Inference (DDIM, deterministic):
    z_T ~ N(0, I)
    Denoise T → 0 conditioned on z_prior   (50 steps via DDIM)
    z_0 → VAE_decoder → PPG (256 samples)

  Inference (DDPM, stochastic):
    Same but with noise injection at each step — maximum diversity

T = 200 timesteps, linear β schedule 1e-4 → 0.02.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ── Time Embedding ─────────────────────────────────────────────────────────────

class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dim=32):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        device = t.device
        half   = self.dim // 2
        freqs  = torch.exp(
            -math.log(10000) * torch.arange(half, device=device).float() / half
        )
        emb = t.float().unsqueeze(1) * freqs.unsqueeze(0)   # (B, half)
        return torch.cat([emb.sin(), emb.cos()], dim=-1)     # (B, dim)


# ── Noise Predictor ────────────────────────────────────────────────────────────

class NoisePredictor(nn.Module):
    """
    MLP ε_θ(z_t, t, z_prior) → 32-dim noise estimate.
    Input: concat(z_t=32, z_prior=32, t_emb=32) = 96 dim
    """
    def __init__(self, latent_dim=32, t_emb_dim=32, hidden=256):
        super().__init__()
        self.t_emb  = SinusoidalTimeEmbedding(t_emb_dim)
        in_dim      = latent_dim * 2 + t_emb_dim
        self.net    = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, latent_dim),
        )

    def forward(self, z_t, t, z_prior):
        return self.net(torch.cat([z_t, z_prior, self.t_emb(t)], dim=-1))


# ── Latent Diffusion ───────────────────────────────────────────────────────────

class LatentDiffusion(nn.Module):
    """
    DDPM in VAE latent z-space conditioned on z_prior.
    Only the NoisePredictor is trained; encoder and decoder stay frozen.
    """

    def __init__(self, latent_dim=32, T=200, beta_start=1e-4, beta_end=0.02):
        super().__init__()
        self.T          = T
        self.latent_dim = latent_dim

        betas     = torch.linspace(beta_start, beta_end, T)
        alphas    = 1.0 - betas
        alpha_bar = torch.cumprod(alphas, dim=0)

        self.register_buffer('betas',               betas)
        self.register_buffer('alphas',              alphas)
        self.register_buffer('alpha_bar',           alpha_bar)
        self.register_buffer('sqrt_abar',           alpha_bar.sqrt())
        self.register_buffer('sqrt_one_minus_abar', (1.0 - alpha_bar).sqrt())

        self.noise_pred = NoisePredictor(latent_dim=latent_dim)

    # ── Forward diffusion ──────────────────────────────────────────────────────

    def q_sample(self, z0, t, noise=None):
        """Add t steps of Gaussian noise to z0."""
        if noise is None:
            noise = torch.randn_like(z0)
        a    = self.sqrt_abar[t].unsqueeze(1)           # (B, 1)
        sig  = self.sqrt_one_minus_abar[t].unsqueeze(1)
        return a * z0 + sig * noise, noise

    # ── Training loss ──────────────────────────────────────────────────────────

    def forward(self, z0, z_prior, p_uncond=0.1):
        """
        DDPM loss with Classifier-Free Guidance dropout.
        z0:       (B, 32) — z_gt from frozen VAE encoder
        z_prior:  (B, 32) — z from frozen CameraEncoder (conditioning)
        p_uncond: fraction of steps to zero z_prior (enables CFG at inference)

        Without CFG (p_uncond=0): noise predictor can ignore z_prior and still
          achieve low loss → all subjects produce identical samples (collapse).
        With CFG: 10% of steps the model must predict noise WITHOUT z_prior.
          The gap (eps_cond - eps_uncond) measures how much z_prior shifts the
          prediction. Amplifying this gap at inference forces subject-diversity.
        """
        B = z0.size(0)
        t = torch.randint(0, self.T, (B,), device=z0.device)
        z_t, noise = self.q_sample(z0, t)

        # CFG dropout: zero z_prior for a random subset of the batch
        if p_uncond > 0 and self.training:
            mask = (torch.rand(B, device=z0.device) < p_uncond).unsqueeze(1)
            z_prior_in = z_prior * (~mask).float()
        else:
            z_prior_in = z_prior

        return F.mse_loss(self.noise_pred(z_t, t, z_prior_in), noise)

    # ── Inference: DDIM (deterministic, fast) ─────────────────────────────────

    @torch.no_grad()
    def ddim_sample(self, z_prior, n_steps=50, eta=0.0, guidance_scale=3.0):
        """
        DDIM sampling with Classifier-Free Guidance.

        guidance_scale: amplifies the influence of z_prior conditioning.
          = 1.0: standard conditional sampling (no amplification)
          = 3.0: strong guidance (recommended after CFG training)
          = 5.0: very strong guidance
        Requires the model was trained with p_uncond > 0 (CFG dropout).

        eps_guided = eps_uncond + guidance_scale * (eps_cond - eps_uncond)

        eta=0: deterministic (same output every run for same z_prior + same z_T seed)
        eta=1: DDPM-equivalent stochastic (max diversity)
        """
        B      = z_prior.size(0)
        device = z_prior.device

        step_size = max(1, self.T // n_steps)
        timesteps = list(range(0, self.T, step_size))[::-1]

        z      = torch.randn(B, self.latent_dim, device=device)   # z_T
        z_null = torch.zeros_like(z_prior)                         # null conditioning

        for idx, t_val in enumerate(timesteps):
            t_tensor = torch.full((B,), t_val, device=device, dtype=torch.long)

            eps_cond   = self.noise_pred(z, t_tensor, z_prior)
            if guidance_scale != 1.0:
                eps_uncond = self.noise_pred(z, t_tensor, z_null)
                eps        = eps_uncond + guidance_scale * (eps_cond - eps_uncond)
            else:
                eps = eps_cond

            abar_t = self.alpha_bar[t_val]
            if idx + 1 < len(timesteps):
                abar_prev = self.alpha_bar[timesteps[idx + 1]]
            else:
                abar_prev = torch.ones(1, device=device)

            z0_pred = (z - (1.0 - abar_t).sqrt() * eps) / abar_t.sqrt().clamp(min=1e-8)
            z0_pred = z0_pred.clamp(-4.0, 4.0)

            sigma  = eta * ((1 - abar_prev) / (1 - abar_t) * (1 - abar_t / abar_prev)).sqrt()
            dir_xt = (1 - abar_prev - sigma ** 2).clamp(min=0).sqrt() * eps
            z      = abar_prev.sqrt() * z0_pred + dir_xt
            if eta > 0:
                z = z + sigma * torch.randn_like(z)

        return z   # (B, 32)

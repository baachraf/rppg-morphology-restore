"""
models/ddpm_a10.py — A10 Unconditional DDPM in VAE z-space
===========================================================
Unlike A9 which conditioned on z_prior (collapsed → inherited collapse),
A10 trains an UNCONDITIONAL DDPM on GT z vectors, then guides inference
with Diffusion Posterior Sampling (DPS) using the observed rPPG.

DPS guidance at each DDIM step t:
  z0_hat   = DDIM_estimate(z_t)
  ppg_hat  = vae_decoder(z0_hat)           [frozen]
  rppg_hat = forward_model(ppg_hat)        [frozen]
  grad     = ∂ pearson_loss(rppg_hat, rppg_obs) / ∂ z_t
  z_{t-1}  = ddim_step(z_t) − step_size × grad

WHY THIS BREAKS COLLAPSE:
  - Unconditional prior: no collapsed z_prior, no inherited collapse
  - Every subject has a DIFFERENT rppg_obs → different DPS gradient → different z_0
  - Information about subject morphology enters via likelihood gradient, not z_prior

Architecture: compact MLP (same backbone as A9 NoisePredictor but no z_prior input)
  Input: concat(z_t=32, t_emb=32) = 64 dim
  Hidden: 256 × 3 SiLU layers  |  ~50K params  |  T=200, linear β
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


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
        emb = t.float().unsqueeze(1) * freqs.unsqueeze(0)
        return torch.cat([emb.sin(), emb.cos()], dim=-1)


class UnconditionalNoisePredictor(nn.Module):
    """ε_θ(z_t, t) — no z_prior conditioning."""
    def __init__(self, latent_dim=32, t_emb_dim=32, hidden=256):
        super().__init__()
        self.t_emb = SinusoidalTimeEmbedding(t_emb_dim)
        in_dim = latent_dim + t_emb_dim
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(),
            nn.Linear(hidden, latent_dim),
        )

    def forward(self, z_t, t):
        return self.net(torch.cat([z_t, self.t_emb(t)], dim=-1))


class UnconditionalDDPM(nn.Module):
    """Unconditional DDPM in 32-dim VAE z-space. Learns P(z_gt)."""

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

        self.noise_pred = UnconditionalNoisePredictor(latent_dim=latent_dim)

    def q_sample(self, z0, t, noise=None):
        if noise is None:
            noise = torch.randn_like(z0)
        a   = self.sqrt_abar[t].unsqueeze(1)
        sig = self.sqrt_one_minus_abar[t].unsqueeze(1)
        return a * z0 + sig * noise, noise

    def forward(self, z0):
        """Unconditional DDPM MSE loss."""
        B      = z0.size(0)
        t      = torch.randint(0, self.T, (B,), device=z0.device)
        z_t, noise = self.q_sample(z0, t)
        return F.mse_loss(self.noise_pred(z_t, t), noise)

    @torch.no_grad()
    def ddim_sample_unconditional(self, B, n_steps=50, device='cpu'):
        """Unconditional sampling (sanity check, no DPS guidance)."""
        step_size = max(1, self.T // n_steps)
        timesteps = list(range(0, self.T, step_size))[::-1]
        z = torch.randn(B, self.latent_dim, device=device)

        for idx, t_val in enumerate(timesteps):
            t_tensor = torch.full((B,), t_val, device=device, dtype=torch.long)
            eps      = self.noise_pred(z, t_tensor)
            abar_t   = self.alpha_bar[t_val]
            abar_p   = (self.alpha_bar[timesteps[idx + 1]]
                        if idx + 1 < len(timesteps)
                        else torch.ones(1, device=device))
            z0_hat = (z - (1 - abar_t).sqrt() * eps) / abar_t.sqrt().clamp(min=1e-8)
            z0_hat = z0_hat.clamp(-4.0, 4.0)
            z = abar_p.sqrt() * z0_hat + (1 - abar_p).sqrt() * eps

        return z

    def dps_sample(self, rppg_obs, vae_decoder, forward_model,
                   n_steps=50, gradient_scale=1.0, device='cpu'):
        """
        DPS inference: unconditional DDPM guided by ∇ log P(rPPG_obs | PPG).

        rppg_obs:       (B, 1, 256) observed CHROM rPPG for each sample
        vae_decoder:    frozen PPGDecoder: z (B,32) → ppg (B,1,256)
        forward_model:  frozen PPGToRPPG:  ppg (B,1,256) → rppg_est (B,1,256)
        gradient_scale: DPS guidance strength (start 1.0, tune 0.1–10.0)
        """
        B         = rppg_obs.size(0)
        step_size = max(1, self.T // n_steps)
        timesteps = list(range(0, self.T, step_size))[::-1]
        z         = torch.randn(B, self.latent_dim, device=device)

        for idx, t_val in enumerate(timesteps):
            t_tensor = torch.full((B,), t_val, device=device, dtype=torch.long)
            abar_t   = self.alpha_bar[t_val]
            abar_p   = (self.alpha_bar[timesteps[idx + 1]]
                        if idx + 1 < len(timesteps)
                        else torch.ones(1, device=device))

            # ── Standard DDIM step (no grad needed for the update direction) ─
            with torch.no_grad():
                eps_det = self.noise_pred(z, t_tensor)
                z0_det  = (z - (1 - abar_t).sqrt() * eps_det) / abar_t.sqrt().clamp(min=1e-8)
                z0_det  = z0_det.clamp(-4.0, 4.0)
                z_next  = abar_p.sqrt() * z0_det + (1 - abar_p).sqrt() * eps_det

            # ── DPS gradient: ∂ pearson_loss(fwd_model(vae(z0_hat)), rppg_obs) / ∂ z ─
            z_g   = z.detach().requires_grad_(True)
            eps_g = self.noise_pred(z_g, t_tensor)
            z0_g  = (z_g - (1 - abar_t).sqrt() * eps_g) / abar_t.sqrt().clamp(min=1e-8)
            z0_g  = z0_g.clamp(-4.0, 4.0)

            ppg_hat  = vae_decoder(z0_g)          # (B, 1, 256)
            rppg_hat = forward_model(ppg_hat)      # (B, 1, 256)

            pf   = rppg_hat.view(B, -1)
            po   = rppg_obs.view(B, -1)
            pf_z = (pf - pf.mean(1, keepdim=True)) / (pf.std(1, keepdim=True) + 1e-8)
            po_z = (po - po.mean(1, keepdim=True)) / (po.std(1, keepdim=True) + 1e-8)
            lik_loss = (1 - (pf_z * po_z).mean(1)).mean()

            grad = torch.autograd.grad(lik_loss, z_g)[0]

            # Scale gradient with noise level so guidance is proportional to uncertainty
            step = gradient_scale * float((1 - abar_t).sqrt())
            z    = (z_next - step * grad.detach()).clamp(-4.0, 4.0)

        return z   # (B, 32)

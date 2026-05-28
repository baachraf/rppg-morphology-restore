"""
models/encoder.py — Stage 2 Camera-to-Morphology Encoders
===========================================================
Three encoder variants mapping camera-derived signals -> Stage 1 latent space.

  Encoder A: raw G-channel (1 channel x 256)   -- tests H_A
  Encoder B: P-Hybrid rPPG (1 channel x 256)   -- tests H_B
  Encoder C: G + rPPG concatenated (2 channels x 256) -- tests fusion

Architecture improvements (v2):
  - InstanceNorm instead of BatchNorm  → no cross-sample statistics leakage
  - ResBlock1D residual connections    → richer feature extraction
  - All three share same architecture; only in_channels differs

New loss components (v2):
  - frequency_loss     : match H2/H1 and H3/H1 harmonic ratios in FFT domain
  - asymmetry_loss     : match rise/fall asymmetry (waveform centre-of-mass)
  - diversity_loss     : maximise batch-level variance of z' (inter-subject spread)
  - subject_contrastive_loss: same-subject attract, different-subject repel in latent space

Anti-collapse mechanisms (v2):
  - LAMBDA_LATENT reduced 1.0 -> 0.05  (was primary collapse driver)
  - LAMBDA_VARIANCE increased 0.1 -> 2.0
  - diversity_loss + subject_contrastive_loss force spread in latent space
  - InstanceNorm prevents batch-stats homogenisation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ==============================================================================
# ARCHITECTURE BLOCKS
# ==============================================================================

class ResBlock1D(nn.Module):
    """
    1-D residual block with InstanceNorm.
    Input and output have the same number of channels and temporal length.
    """
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


class CameraEncoder(nn.Module):
    """
    Encodes a camera-derived signal to a latent vector targeting
    the Stage 1 VAE latent space.

    Input shape:  (batch, in_channels, 256)
    Output shape: (batch, latent_dim)

    in_channels:
      1 -> Encoder A (G-channel only) or B (rPPG only)
      2 -> Encoder C (G-channel + rPPG concatenated)

    Architecture (v2 changes):
      - InstanceNorm replaces BatchNorm (no cross-sample statistics)
      - ResBlock1D after each strided conv (richer per-sample features)
    """
    def __init__(self, latent_dim: int = 32, in_channels: int = 1,
                 morpho_aux: bool = False):
        super().__init__()
        self.stem = nn.Sequential(
            # (B, C, 256) -> (B, 32, 128)
            nn.Conv1d(in_channels, 32, kernel_size=7, stride=2, padding=3, bias=False),
            nn.InstanceNorm1d(32, affine=True),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.res1 = ResBlock1D(32)

        self.down1 = nn.Sequential(
            # -> (B, 64, 64)
            nn.Conv1d(32, 64, kernel_size=5, stride=2, padding=2, bias=False),
            nn.InstanceNorm1d(64, affine=True),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.res2 = ResBlock1D(64)

        self.down2 = nn.Sequential(
            # -> (B, 128, 32)
            nn.Conv1d(64, 128, kernel_size=5, stride=2, padding=2, bias=False),
            nn.InstanceNorm1d(128, affine=True),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.res3 = ResBlock1D(128)

        self.down3 = nn.Sequential(
            # -> (B, 256, 16)
            nn.Conv1d(128, 256, kernel_size=3, stride=2, padding=1, bias=False),
            nn.InstanceNorm1d(256, affine=True),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.res4 = ResBlock1D(256)

        self.fc = nn.Linear(256 * 16, latent_dim)

        # Auxiliary morphological prediction heads (Phase 4).
        # Outputs 3 values in [0,1]: [notch_pos, IPA, rise_time].
        # Only built when morpho_aux=True; ignored at inference time.
        self.morpho_aux = morpho_aux
        if morpho_aux:
            self.morpho_head = nn.Sequential(
                nn.Linear(latent_dim, 32),
                nn.ReLU(),
                nn.Linear(32, 3),
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

    def forward_morpho(self, x: torch.Tensor):
        """
        Returns (z, morpho_pred) where:
          z           : (B, latent_dim) — same as forward(x)
          morpho_pred : (B, 3) — [notch_pos, IPA, rise_time] in [0,1]
                        None if morpho_aux=False
        """
        z = self.forward(x)
        if self.morpho_aux:
            return z, self.morpho_head(z)
        return z, None


class Discriminator(nn.Module):
    """
    WGAN-GP discriminator for Stage 2 adversarial training.
    Trained on real GT PPG cycles to push reconstructions toward
    plausible morphological shapes.

    Input: (batch, 1, 256)
    Output: (batch, 1) -- Wasserstein critic score (no sigmoid)
    """
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(1, 32,  kernel_size=7, stride=2, padding=3), nn.LeakyReLU(0.2),
            nn.Conv1d(32, 64, kernel_size=5, stride=2, padding=2), nn.LeakyReLU(0.2),
            nn.Conv1d(64, 128, kernel_size=5, stride=2, padding=2), nn.LeakyReLU(0.2),
            nn.Conv1d(128, 256, kernel_size=3, stride=2, padding=1), nn.LeakyReLU(0.2),
        )
        self.fc = nn.Linear(256 * 16, 1)

    def forward(self, x):
        h = self.net(x).view(x.size(0), -1)
        return self.fc(h)


def gradient_penalty(discriminator, real, fake, device, lambda_gp=10.0):
    """WGAN-GP gradient penalty."""
    alpha = torch.rand(real.size(0), 1, 1, device=device)
    interpolated = (alpha * real + (1 - alpha) * fake).requires_grad_(True)
    d_interp = discriminator(interpolated)
    grad = torch.autograd.grad(
        outputs=d_interp, inputs=interpolated,
        grad_outputs=torch.ones_like(d_interp),
        create_graph=True, retain_graph=True
    )[0]
    grad_norm = grad.view(grad.size(0), -1).norm(2, dim=1)
    return lambda_gp * ((grad_norm - 1) ** 2).mean()


# ==============================================================================
# MORPHOLOGY LOSS COMPONENTS (v1 — retained)
# ==============================================================================

def notch_weighted_l1(pred, target, notch_positions, weight=1.0, cycle_len=256):
    """
    L1 loss with optional extra weight on the dicrotic notch temporal window.
    notch_positions: (batch,) tensor of per-cycle notch sample indices from GT detector.
                     -1 means notch not detected -> use uniform weight.

    NOTE: GT dataset has no notch (CMS50E removes it). weight=1.0 gives uniform L1.
    """
    weights = torch.ones_like(target)
    window  = 8
    if weight > 1.0:
        for b in range(target.size(0)):
            pos = int(notch_positions[b].item())
            if pos > 0:
                lo = max(0, pos - window)
                hi = min(cycle_len, pos + window)
                weights[b, 0, lo:hi] = weight

    return (weights * torch.abs(pred - target)).mean()


def curvature_loss(pred, target):
    """
    L1 distance between second derivatives of pred and target.
    Penalises missing inflection points (concavity mismatches).
    """
    d2_pred   = torch.diff(torch.diff(pred,   dim=2), dim=2)
    d2_target = torch.diff(torch.diff(target, dim=2), dim=2)
    return F.l1_loss(d2_pred, d2_target)


def soft_dtw_loss(pred, target, gamma=1.0):
    """
    Soft-DTW approximation (fast L1 + diagonal warp penalty).
    Tolerates small temporal misalignments from residual PTT errors.
    """
    l1    = F.l1_loss(pred, target)
    shift = F.l1_loss(pred[:, :, 1:], target[:, :, :-1])
    return l1 + 0.1 * shift


def latent_variance_reg(z_prime, eps=0.5):
    """
    Penalise latent collapse: if per-batch variance of z' < eps,
    add a penalty proportional to the deficit.
    eps increased from 0.1 -> 0.5 to enforce wider spread.
    """
    var = z_prime.var(dim=0).mean()
    return F.relu(eps - var)


# ==============================================================================
# NEW LOSS COMPONENTS (v2)
# ==============================================================================

def frequency_loss(pred, target):
    """
    Match harmonic power ratios in the FFT domain.
    Real PPG has H2/H1 ~ 0.46 and H3/H1 ~ 0.21.
    rPPG is nearly sinusoidal (H2/H1 ~ 0.05).

    This loss pushes the reconstruction to have the correct
    harmonic content, i.e., a physiological waveform shape.

    pred, target: (B, 1, 256) — normalised to [0, 1]
    """
    # Remove channel dim: (B, 256)
    p = pred.squeeze(1)
    t = target.squeeze(1)

    # Real FFT -> (B, 129) complex
    P_fft = torch.fft.rfft(p, dim=1)
    T_fft = torch.fft.rfft(t, dim=1)

    P_mag = P_fft.abs() + 1e-8
    T_mag = T_fft.abs() + 1e-8

    # Fundamental is at index 1 (DC at 0). Use indices 1, 2, 3 for H1, H2, H3.
    # We compare the RATIO to be scale-invariant.
    P_h1 = P_mag[:, 1]
    T_h1 = T_mag[:, 1]

    # H2/H1 ratio loss
    P_r2 = P_mag[:, 2] / P_h1
    T_r2 = T_mag[:, 2] / T_h1
    l_h2 = F.mse_loss(P_r2, T_r2)

    # H3/H1 ratio loss
    P_r3 = P_mag[:, 3] / P_h1
    T_r3 = T_mag[:, 3] / T_h1
    l_h3 = F.mse_loss(P_r3, T_r3)

    return l_h2 + l_h3


def spectral_l1_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """
    L1 loss on full FFT amplitude spectrum.
    Directly matches harmonic content rather than just H2/H1 ratio.
    More stable than ratio-based frequency_loss.

    pred, target: (B, 1, 256) — normalised to [0, 1]
    """
    p = pred.squeeze(1)      # (B, 256)
    t = target.squeeze(1)
    p_fft = torch.fft.rfft(p, dim=1).abs()   # (B, 129)
    t_fft = torch.fft.rfft(t, dim=1).abs()
    return torch.nn.functional.l1_loss(p_fft, t_fft)


def asymmetry_loss(pred, target):
    """
    Match waveform asymmetry: the normalised centre-of-mass position.
    A real PPG has rapid systolic upstroke (~1/3 of cycle) and slower
    diastolic decay (~2/3). The rPPG is nearly symmetric.

    CoM_x = sum(i * y_i) / sum(y_i)  — position of centre of mass.
    We match CoM_x / N so the loss is in [0, 1].

    pred, target: (B, 1, 256)
    """
    p = pred.squeeze(1)   # (B, 256)
    t = target.squeeze(1)

    B, N = p.shape
    positions = torch.arange(N, dtype=p.dtype, device=p.device).unsqueeze(0)  # (1, N)

    # Shift to non-negative before computing CoM (cycles are already [0,1] normalised)
    p_pos = p - p.min(dim=1, keepdim=True).values + 1e-6
    t_pos = t - t.min(dim=1, keepdim=True).values + 1e-6

    com_p = (positions * p_pos).sum(dim=1) / p_pos.sum(dim=1)  # (B,)
    com_t = (positions * t_pos).sum(dim=1) / t_pos.sum(dim=1)

    return F.mse_loss(com_p / N, com_t / N)


def diversity_loss(z_prime):
    """
    Maximise the variance of z' across the batch.
    Penalises when all latents collapse to a single point.

    Returns a loss that is lower when z' is more spread out.
    Target variance >= 1.0 per dimension (empirical).
    """
    # z_prime: (B, latent_dim)
    var_per_dim = z_prime.var(dim=0)          # (latent_dim,)
    mean_var    = var_per_dim.mean()
    # Penalty: max(0, target - mean_var) — encourage high variance
    return F.relu(1.0 - mean_var)


def subject_contrastive_loss(z_prime, sids, margin=1.0, temperature=0.07):
    """
    NT-Xent / supervised contrastive loss in latent space.
    - Same subject cycles: should be close  (positive pairs)
    - Different subject cycles: should be far (negative pairs)

    This directly encourages the encoder to produce subject-specific latents
    rather than a single average latent for all subjects.

    z_prime: (B, latent_dim)
    sids:    (B,) long tensor of subject IDs
    """
    B = z_prime.size(0)
    if B < 2:
        return torch.tensor(0.0, device=z_prime.device)

    # L2-normalise for cosine similarity
    z_norm = F.normalize(z_prime, dim=1)  # (B, latent_dim)
    sim = torch.matmul(z_norm, z_norm.T) / temperature  # (B, B)

    # Mask: same subject
    sid_a = sids.unsqueeze(0)   # (1, B)
    sid_b = sids.unsqueeze(1)   # (B, 1)
    pos_mask = (sid_a == sid_b).float()  # 1 where same subject
    # Remove diagonal (self-similarity)
    eye = torch.eye(B, device=z_prime.device)
    pos_mask = pos_mask * (1 - eye)

    # If no positive pairs exist in this batch, skip
    if pos_mask.sum() == 0:
        return torch.tensor(0.0, device=z_prime.device)

    # For each sample, compute log(sum_pos / sum_all)
    # Row-wise max subtraction (standard log-softmax numerics, always non-negative result)
    sim_max = sim.detach().max(dim=1, keepdim=True).values
    exp_sim = torch.exp(sim - sim_max)
    log_sum = torch.log(exp_sim.sum(dim=1, keepdim=True) + 1e-8) + sim_max
    log_prob = sim - log_sum

    # Mean over positive pairs
    loss = -(log_prob * pos_mask).sum() / (pos_mask.sum() + 1e-8)
    return loss


# ==============================================================================
# COMBINED STAGE 2 LOSS
# ==============================================================================

def stage2_loss(pred, target, z_prime, z_gt, notch_positions,
                lambdas, epoch, adv_loss=None, sids=None,
                morpho_pred=None, morpho_labels=None):
    """
    Full Stage 2 loss (v2). Returns (total_loss, loss_components_dict).

    lambdas: dict with keys:
      l1, notch, sdtw, curv, latent, variance, adv, adv_start,
      diversity (new), freq (new), asym (new), contrastive (new)
    """
    l_l1      = notch_weighted_l1(pred, target, notch_positions,
                                   weight=lambdas['notch'])
    l_sdtw    = soft_dtw_loss(pred, target)
    l_curv    = curvature_loss(pred, target)
    l_latent  = F.mse_loss(z_prime, z_gt.detach())
    l_var     = latent_variance_reg(z_prime)

    # New losses
    l_freq    = frequency_loss(pred, target)
    l_asym    = asymmetry_loss(pred, target)
    l_div     = diversity_loss(z_prime)

    # Subject contrastive (needs sids in batch)
    l_contr   = torch.tensor(0.0, device=pred.device)
    if sids is not None and lambdas.get('contrastive', 0.0) > 0:
        l_contr = subject_contrastive_loss(z_prime, sids)

    # Auxiliary morphological head loss (Phase 4)
    l_aux = torch.tensor(0.0, device=pred.device)
    if morpho_pred is not None and morpho_labels is not None:
        l_aux = torch.nn.functional.mse_loss(morpho_pred, morpho_labels)

    # Spectral L1 loss (Phase 4)
    l_spectral = torch.tensor(0.0, device=pred.device)
    if lambdas.get('spectral', 0.0) > 0:
        l_spectral = spectral_l1_loss(pred, target)

    total = (lambdas['l1']        * l_l1
             + lambdas['sdtw']    * l_sdtw
             + lambdas['curv']    * l_curv
             + lambdas['latent']  * l_latent
             + lambdas['variance'] * l_var
             + lambdas.get('freq',        0.0) * l_freq
             + lambdas.get('spectral',    0.0) * l_spectral
             + lambdas.get('asym',        0.0) * l_asym
             + lambdas.get('diversity',   0.0) * l_div
             + lambdas.get('contrastive', 0.0) * l_contr
             + lambdas.get('aux_morpho',  0.0) * l_aux)

    # Adversarial term added after warm-up epochs
    if adv_loss is not None and epoch >= lambdas.get('adv_start', 40):
        total = total + lambdas['adv'] * adv_loss

    components = {
        'l1':          l_l1.item(),
        'sdtw':        l_sdtw.item(),
        'curv':        l_curv.item(),
        'latent':      l_latent.item(),
        'variance':    l_var.item(),
        'freq':        l_freq.item(),
        'spectral':    l_spectral.item(),
        'asym':        l_asym.item(),
        'diversity':   l_div.item(),
        'contrastive': l_contr.item() if isinstance(l_contr, torch.Tensor) else 0.0,
        'aux_morpho':  l_aux.item(),
        'adv':         adv_loss.item() if adv_loss is not None else 0.0,
        'total':       total.item()
    }
    return total, components

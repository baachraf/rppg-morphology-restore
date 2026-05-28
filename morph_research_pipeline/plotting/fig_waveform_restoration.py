import shutil
import numpy as np
import torch
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.stats import pearsonr

try:
    from .config import apply_rc, FIGS
    from .utils import save_fig, save_raw_data
except ImportError:
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
    from morph_research_pipeline.plotting.config import apply_rc, FIGS
    from morph_research_pipeline.plotting.utils import save_fig, save_raw_data

# ── Constants ─────────────────────────────────────────────────────────────────
# CENTAN s03 (sid=4003), 1000 Hz Polymate GT — best test subject for V5-B
# (mean shape_r=0.770 across all cycles; encoder B).
# Cycles 58–59: highest rPPG smoothness (0.613) + V5-B restore_r=0.941.
_SID        = 4003
_CYCLE_IDX  = [58, 59]
_CYCLES_NPZ = Path(
    r'E:\Projects_Results\rPPG_Morphology_Restore\data\centan\cycles'
    r'\centan_s03_cycles.npz'
)
_CKPT_DIR   = Path(r'E:\Projects_Results\rPPG_Morphology_Restore\checkpoints')
_VAE_CKPT   = 'shared/stage1_vae_p4.pt'
_ENC_CKPT   = 'v5/encoders/encoder_B.pt'

C_GT      = '#2166AC'   # blue  — contact PPG
C_RESTORE = '#1A9641'   # green — VAE-Base output


def _norm01(x):
    lo, hi = x.min(), x.max()
    return (x - lo) / (hi - lo) if (hi - lo) > 1e-8 else np.zeros_like(x)


def _concat_norm(cycles):
    return np.concatenate([_norm01(c) for c in cycles])


def _load_models(device):
    import sys as _sys
    _sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from models.vae import PPGVAE
    from models.encoder import CameraEncoder
    LATENT_DIM = 32

    vae = PPGVAE(latent_dim=LATENT_DIM).to(device)
    vae.load_state_dict(torch.load(_CKPT_DIR / _VAE_CKPT, map_location=device, weights_only=False))
    vae.eval()

    enc = CameraEncoder(latent_dim=LATENT_DIM, in_channels=1).to(device)
    state = torch.load(_CKPT_DIR / _ENC_CKPT, map_location=device, weights_only=False)
    enc.load_state_dict(state['encoder'], strict=False)
    enc.eval()
    return vae, enc


def _infer(vae, enc, cycles, device):
    out = []
    with torch.no_grad():
        for cyc in cycles:
            x = torch.from_numpy(cyc).float().unsqueeze(0).unsqueeze(0).to(device)
            z = enc(x)
            out.append(vae.decode(z).cpu().numpy()[0, 0, :])
    return np.array(out)


def plot_waveform_restoration(out_dir=None, figsize=(3.4, 2.5)):
    """Qualitative waveform restoration — single column, GT vs Restored only."""
    if out_dir is None:
        out_dir = FIGS

    apply_rc()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # ── Data ──────────────────────────────────────────────────────────────────
    data     = np.load(_CYCLES_NPZ)
    gt_sel   = data['gt_cycles'][_CYCLE_IDX].astype(np.float32)
    rppg_sel = data['rppg_chrom_cycles'][_CYCLE_IDX].astype(np.float32)

    # ── Inference ────────────────────────────────────────────────────────────
    vae, enc  = _load_models(device)
    restored  = _infer(vae, enc, rppg_sel, device)

    # ── Per-cycle normalisation then display flip (systolic peak at top) ──────
    gt_cont   = 1 - _concat_norm(gt_sel)
    rest_raw  = 1 - _concat_norm(restored)

    # Polarity correction: align restored to GT
    r_fwd = pearsonr(rest_raw, gt_cont)[0]
    r_inv = pearsonr(1 - rest_raw, gt_cont)[0]
    rest_cont = (1 - rest_raw) if r_inv > r_fwd else rest_raw

    n_cycles = len(_CYCLE_IDX)
    t = np.linspace(0, n_cycles, len(gt_cont))

    # ── Plot ──────────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=figsize)

    ax.plot(t, gt_cont,   color=C_GT,      lw=1.8, label='GT PPG',      zorder=3)
    ax.plot(t, rest_cont, color=C_RESTORE,  lw=1.8, label='Restored PPG', zorder=4)

    for i in range(1, n_cycles):
        ax.axvline(x=i, color='#aaaaaa', lw=0.6, ls=':')

    ax.set_xlim(0, n_cycles)
    ax.set_ylim(-0.08, 1.18)
    ax.set_xlabel('Cardiac cycles', fontsize=8)
    ax.set_ylabel('Normalised amplitude', fontsize=8)
    ax.set_xticks(range(n_cycles + 1))
    ax.tick_params(labelsize=7)
    ax.legend(loc='upper center', ncol=2, frameon=False, fontsize=7)

    fig.tight_layout()

    path = save_fig(fig, out_dir, 'fig7_waveform_restoration.png')
    plt.close(fig)

    save_raw_data({
        'figure': 'fig7_waveform_restoration',
        'description': (
            'Qualitative waveform restoration: CENTAN s03 (sid=4003), 1000 Hz Polymate GT. '
            'Cycles 58-59 (rPPG smoothness=0.613, V5-B restore_r=0.941). '
            'VAE-Base (V5-B, encoder B). rPPG used for inference only, not displayed. '
            'Restored PPG recovers population-level morphology but is not subject-specific.'
        ),
        'subject': _SID,
        'cycle_indices': _CYCLE_IDX,
        'polarity_flipped': bool(r_inv > r_fwd),
    }, out_dir, 'fig7_waveform_restoration.json')

    return path


if __name__ == '__main__':
    _JOURNAL = Path(
        r'D:\OneDrive - STEPLESMOSENSESARL\PlesmoSense-CENTAN\Code\ACHRAF_Private'
        r'\Research_Academic\Projects_Papers\Journals\TobeSubmitted'
        r'\JBHI_Jrnl_0526\figures'
    )
    path = plot_waveform_restoration(FIGS)
    _JOURNAL.mkdir(parents=True, exist_ok=True)
    shutil.copy2(path, _JOURNAL / 'fig7_waveform_restoration.png')
    print('fig7_waveform_restoration.png -> journal figures')

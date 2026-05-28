"""
training/a8/train_a8.py — A8-v2 FPS-Agnostic Camera-Only Training
===================================================================
A8 original FAILED: per-subject r=-0.68 due to sign-invariant loss landscape.
All losses (spectral L1, FFT-morpho) were sign-agnostic; L1 (weight 0.5) was
overwhelmed by spectral (3.0, ~150x larger absolute scale).

A8-v2 fix:
  - Add Pearson correlation loss (sign-sensitive, scale-invariant) — weight 3.0
  - Reduce spectral to 1.0 (prevent absolute-scale dominance)
  - Remove L1 (replaced by Pearson as shape anchor)
  - GT centred to [-1,1] to match Tanh decoder output range

Primary loss: morphological regression (H2/H1, IPA) — weight 5.0
Shape anchor: Pearson correlation — weight 3.0 (SIGN-SENSITIVE)
Secondary:    spectral L1 — weight 1.0 (reduced from 3.0)
Anti-collapse: output-level diversity on 256-dim waveforms — weight 2.0
Latent:        subject contrastive — weight 3.0

Trains on ALL datasets (30fps + 60fps) jointly using fps_windows/*.npz files.
Run extract_rgb_windows_fps.py before this script.
"""

import os, sys, random
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Sampler
from tqdm import tqdm
from pathlib import Path

HERE = Path(__file__).parent
PIPELINE_ROOT = HERE.parent.parent
sys.path.insert(0, str(PIPELINE_ROOT))

from config.paths import (
    UBFC_DIR, STRESS_DIR, FPS2023_DIR, CENTAN_DIR,
    FPS2023_60_DIR, RESULTS_DIR, CKPT_DIR, SPLIT_FILE,
)
from config.hyperparams import BATCH_SIZE, LEARNING_RATE, N_SUBJECTS_PER_BATCH
from models.encoder_a8 import A8Model
from models.metrics import compute_ipa

LATENT_DIM  = 32
IN_CHANNELS = 6
MAX_EPOCHS  = 300
EARLY_STOP  = 30
CKPT_A8     = CKPT_DIR / 'a8'
RESULTS_A8  = RESULTS_DIR / 'a8'

LAMBDAS = {
    'morpho':      5.0,   # PRIMARY: H2/H1 + IPA regression
    'pearson':     3.0,   # shape anchor — sign-sensitive (replaces l1 as shape loss)
    'spectral':    1.0,   # harmonic content matching (reduced: was 3.0, dominated abs scale)
    'output_div':  2.0,   # output-level anti-collapse on 256-dim waveforms
    'contrastive': 3.0,   # latent subject contrastive
    'curv':        0.2,
}


# ── morphological label extraction ───────────────────────────────────────────

def compute_h2h1_batch(cycles_np):
    out = np.zeros(len(cycles_np), dtype=np.float32)
    for i, c in enumerate(cycles_np):
        fft = np.abs(np.fft.rfft(c))
        if len(fft) >= 4 and fft[1] > 1e-8:
            out[i] = fft[2] / fft[1]
    return out


def gt_morpho_labels(gt_np):
    """gt_np: (N, 256) -> (N, 2) array [H2/H1, IPA]."""
    h2h1 = compute_h2h1_batch(gt_np)
    ipa  = np.array([compute_ipa(c) for c in gt_np], dtype=np.float32)
    return np.stack([h2h1, ipa], axis=1)


# ── losses ───────────────────────────────────────────────────────────────────

def pearson_loss(pred, target):
    """
    1 - mean Pearson r over the batch.
    Sign-sensitive: r(x, -x) = -1 → loss = 2.0 (maximum penalty for inversion).
    Scale-invariant: works regardless of absolute amplitude.
    """
    p = pred.view(pred.size(0), -1)
    t = target.view(target.size(0), -1)
    p_z = (p - p.mean(dim=1, keepdim=True)) / (p.std(dim=1, keepdim=True) + 1e-8)
    t_z = (t - t.mean(dim=1, keepdim=True)) / (t.std(dim=1, keepdim=True) + 1e-8)
    r = (p_z * t_z).mean(dim=1)
    return 1 - r.mean()


def spectral_loss(pred, target, n=7):
    """L1 on first n FFT bins."""
    p = torch.fft.rfft(pred.squeeze(1), dim=1).abs()
    t = torch.fft.rfft(target.squeeze(1), dim=1).abs()
    return F.l1_loss(p[:, :n], t[:, :n])


def output_diversity_loss(recon, sids, margin=0.3):
    """
    Penalise pairwise cosine similarity > margin between different subjects'
    mean output waveforms. Operates on 256-dim outputs, not 32-dim z.
    """
    unique_sids = torch.unique(sids)
    if len(unique_sids) < 2:
        return torch.tensor(0.0, device=recon.device)
    means = []
    for sid in unique_sids:
        mask = sids == sid
        means.append(recon[mask].mean(dim=0).view(-1))
    means = torch.stack(means)
    norms = F.normalize(means, dim=1)
    sim   = torch.mm(norms, norms.T)
    eye   = torch.eye(len(unique_sids), device=recon.device, dtype=torch.bool)
    return F.relu(sim[~eye] - margin).mean()


def subject_contrastive_loss(z, sids, temperature=0.07):
    B = z.size(0)
    if B < 2:
        return torch.tensor(0.0, device=z.device)
    z_n  = F.normalize(z, dim=1)
    sim  = torch.matmul(z_n, z_n.T) / temperature
    pos  = (sids.unsqueeze(0) == sids.unsqueeze(1)).float()
    pos  = pos * (1 - torch.eye(B, device=z.device))
    if pos.sum() == 0:
        return torch.tensor(0.0, device=z.device)
    sim_max  = sim.detach().max(dim=1, keepdim=True).values
    log_sum  = torch.log((torch.exp(sim - sim_max)).sum(dim=1, keepdim=True) + 1e-8) + sim_max
    return -(((sim - log_sum) * pos).sum() / (pos.sum() + 1e-8))


def curvature_loss(pred, target):
    d2p = pred[:, :, 2:] - 2 * pred[:, :, 1:-1] + pred[:, :, :-2]
    d2t = target[:, :, 2:] - 2 * target[:, :, 1:-1] + target[:, :, :-2]
    return F.l1_loss(d2p, d2t)


# ── dataset ──────────────────────────────────────────────────────────────────

class FPSWindowDataset(Dataset):
    """Loads *_fps_windows.npz from all fps_windows/ directories."""

    def __init__(self, data_dirs, split_sids):
        self.items    = []
        self.sid_list = []
        for d in data_dirs:
            d = Path(d)
            if not d.is_dir():
                continue
            npz_files = list(d.glob('*_fps_windows.npz'))
            for npz in tqdm(npz_files, desc=d.parent.name, leave=False):
                try:
                    data = np.load(npz)
                    sid  = int(data['sid'])
                    if sid not in split_sids:
                        continue
                    rgb = data['rgb_windows']   # (N, 6, T)
                    gt  = data['gt_targets']    # (N, 256)
                    for i in range(len(rgb)):
                        self.items.append((rgb[i], gt[i], sid))
                        self.sid_list.append(sid)
                except Exception:
                    continue
        print(f'Loaded {len(self.items)} windows from {len(set(self.sid_list))} subjects')

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        rgb, gt, sid = self.items[idx]
        return (torch.from_numpy(rgb).float(),
                torch.from_numpy(gt).unsqueeze(0).float(),
                sid)


def collate_pad(batch):
    """Pad rgb windows to the same T within a batch (different FPS → different T)."""
    rgbs, gts, sids = zip(*batch)
    max_t = max(r.shape[1] for r in rgbs)
    rgbs_padded = torch.stack([F.pad(r, (0, max_t - r.shape[1])) for r in rgbs])
    return rgbs_padded, torch.stack(gts), list(sids)


class SubjectStratifiedSampler(Sampler):
    def __init__(self, dataset, batch_size, n_subs_per_batch=16, seed=42):
        self.batch_size = batch_size
        self.n_subs     = n_subs_per_batch
        self.rng        = random.Random(seed)
        self.sid_to_idx = {}
        for i, sid in enumerate(dataset.sid_list):
            self.sid_to_idx.setdefault(sid, []).append(i)
        self.all_sids = list(self.sid_to_idx.keys())
        self.n_total  = len(dataset)

    def __iter__(self):
        sids = self.all_sids[:]
        self.rng.shuffle(sids)
        n_per = max(1, self.batch_size // min(self.n_subs, len(sids)))
        indices = []
        for sid in sids:
            pool = self.sid_to_idx[sid][:]
            self.rng.shuffle(pool)
            indices.extend(pool[:n_per])
        self.rng.shuffle(indices)
        return iter(indices[:self.n_total])

    def __len__(self):
        return self.n_total


def find_fps_window_dirs():
    dirs = []
    for base in [UBFC_DIR, STRESS_DIR, FPS2023_DIR, CENTAN_DIR, FPS2023_60_DIR]:
        d = base / 'fps_windows'
        if d.is_dir():
            dirs.append(d)
    return dirs


# ── training loop ────────────────────────────────────────────────────────────

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')
    CKPT_A8.mkdir(parents=True, exist_ok=True)
    RESULTS_A8.mkdir(parents=True, exist_ok=True)

    split_df   = pd.read_csv(SPLIT_FILE)
    train_sids = set(split_df[split_df['split'] == 'train']['sid'])
    val_sids   = set(split_df[split_df['split'] == 'val']['sid'])

    data_dirs = find_fps_window_dirs()
    if not data_dirs:
        print('ERROR: No fps_windows directories found.')
        print('  Run morph_research_pipeline/extraction/extract_rgb_windows_fps.py first.')
        return
    print(f'fps_windows dirs: {[str(d) for d in data_dirs]}')

    train_ds = FPSWindowDataset(data_dirs, train_sids)
    val_ds   = FPSWindowDataset(data_dirs, val_sids)

    if len(train_ds) == 0:
        print('ERROR: No training windows found.'); return

    sampler      = SubjectStratifiedSampler(train_ds, BATCH_SIZE, N_SUBJECTS_PER_BATCH)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, sampler=sampler, num_workers=0, collate_fn=collate_pad)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False,   num_workers=0, collate_fn=collate_pad)

    model    = A8Model(latent_dim=LATENT_DIM, in_channels=IN_CHANNELS).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f'A8Model: {n_params:,} parameters')
    print(f'Loss weights: {LAMBDAS}')
    print()

    opt       = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    ckpt_path = CKPT_A8 / 'a8_model.pt'
    log_rows  = []
    best_val  = float('inf')
    patience  = 0

    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        tr_losses, tr_comp = [], {}

        for rgb, gt, sids in train_loader:
            rgb    = rgb.to(device)
            gt     = gt.to(device)
            sids_t = torch.tensor(sids, device=device) if not isinstance(sids, torch.Tensor) else sids.to(device)

            gt_morpho = torch.from_numpy(
                gt_morpho_labels(gt.cpu().numpy()[:, 0, :])
            ).to(device)
            # GT is norm01 [0,1]; Tanh decoder outputs [-1,1]. Centre GT to match.
            gt_c = gt * 2.0 - 1.0

            recon, z, morpho_pred = model(rgb)

            l_morpho  = F.mse_loss(morpho_pred, gt_morpho)               * LAMBDAS['morpho']
            l_pearson = pearson_loss(recon, gt_c)                         * LAMBDAS['pearson']
            l_spec    = spectral_loss(recon, gt_c)                        * LAMBDAS['spectral']
            l_out_div = output_diversity_loss(recon.squeeze(1), sids_t)   * LAMBDAS['output_div']
            l_contr   = subject_contrastive_loss(z, sids_t)               * LAMBDAS['contrastive']
            l_curv    = curvature_loss(recon, gt_c)                       * LAMBDAS['curv']

            loss = l_morpho + l_pearson + l_spec + l_out_div + l_contr + l_curv

            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

            tr_losses.append(loss.item())
            for k, v in [('morpho', l_morpho), ('pearson', l_pearson), ('spec', l_spec),
                          ('out_div', l_out_div), ('contr', l_contr)]:
                tr_comp.setdefault(k, []).append(v.item())

        model.eval()
        vl_losses = []
        with torch.no_grad():
            for rgb, gt, _ in val_loader:
                rgb, gt   = rgb.to(device), gt.to(device)
                gt_c      = gt * 2.0 - 1.0
                recon, z, morpho_pred = model(rgb)
                gt_morpho = torch.from_numpy(
                    gt_morpho_labels(gt.cpu().numpy()[:, 0, :])
                ).to(device)
                vl = (F.mse_loss(morpho_pred, gt_morpho) * LAMBDAS['morpho']
                      + pearson_loss(recon, gt_c)         * LAMBDAS['pearson']
                      + spectral_loss(recon, gt_c)        * LAMBDAS['spectral'])
                vl_losses.append(vl.item())

        avg_tr = float(np.mean(tr_losses))
        avg_vl = float(np.mean(vl_losses))
        log_rows.append({'epoch': epoch, 'train_loss': avg_tr, 'val_loss': avg_vl,
                         **{k: float(np.mean(v)) for k, v in tr_comp.items()}})

        if epoch % 10 == 0:
            comp = ' | '.join(f'{k}={np.mean(v):.3f}' for k, v in tr_comp.items())
            print(f'  Ep {epoch:03d} | Tr {avg_tr:.4f} | Val {avg_vl:.4f} | {comp}')

        if avg_vl < best_val:
            best_val, patience = avg_vl, 0
            torch.save({'model': model.state_dict(), 'epoch': epoch, 'val_loss': avg_vl},
                       ckpt_path)
        else:
            patience += 1
            if patience >= EARLY_STOP:
                print(f'  Early stop epoch {epoch}. Best val: {best_val:.4f}')
                break

    pd.DataFrame(log_rows).to_csv(RESULTS_A8 / 'training_log_a8.csv', index=False)
    print(f'\nA8 training complete. Best val: {best_val:.4f}')
    print(f'Checkpoint: {ckpt_path}')


if __name__ == '__main__':
    main()

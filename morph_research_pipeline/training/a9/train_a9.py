"""
training/a9/train_a9.py — A9 Latent Diffusion Decoder Training
===============================================================
Trains a DDPM in the VAE's 32-dim z-space, conditioned on z_prior from
the frozen V5-B CameraEncoder. Avoids template collapse by design:

  Deterministic networks compute E[z_gt | z_prior] → population mean.
  Diffusion samples from P(z_gt | z_prior) → inherently diverse per subject.

Architecture:
  VAE encoder   (frozen) → z_gt   (32-dim training targets)
  CameraEncoder (frozen, V5-B) → z_prior (32-dim conditioning)
  NoisePredictor (trainable, ~50K params) → predicts ε at each t

Training:
  For each cycle:
    z_gt    = VAE.encoder.mu(GT_PPG)         (frozen)
    z_prior = CameraEncoder(rPPG_pos_cycle)  (frozen)
    t ~ Uniform(0, T)
    z_t = sqrt(ᾱ_t) * z_gt + sqrt(1-ᾱ_t) * ε
    loss = ||ε - ε_θ(z_t, t, z_prior)||²

Inference:
  z_T ~ N(0, I) → DDIM denoise (50 steps) conditioned on z_prior → z_0
  z_0 → VAE decoder → PPG (256 samples)
"""

import os, sys, random
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader, Sampler
from tqdm import tqdm
from pathlib import Path

HERE = Path(__file__).parent
PIPELINE_ROOT = HERE.parent.parent
sys.path.insert(0, str(PIPELINE_ROOT))

from config.paths import (
    UBFC_CYCLES_DIR, STRESS_CYCLES_DIR, FPS2023_CYCLES_DIR,
    FPS2023_60_CYCLES_DIR, CENTAN_CYCLES_DIR,
    CKPT_A9, RESULTS_A9, SPLIT_FILE,
    VAE_CKPT, ENCODER_CKPT,
)
from config.hyperparams import BATCH_SIZE, LEARNING_RATE, N_SUBJECTS_PER_BATCH
from models.encoder import CameraEncoder
from models.vae import PPGVAE
from models.ldm_a9 import LatentDiffusion

LATENT_DIM = 32
MAX_EPOCHS = 300
EARLY_STOP = 30
T_STEPS    = 200


# ── Load frozen Stage 1 models ────────────────────────────────────────────────

def load_frozen_models(device):
    """
    Returns:
      camera_enc: CameraEncoder (V5-B), frozen, maps rPPG_pos → z_prior (32)
      vae:        PPGVAE, frozen
                    .encoder: original VAE encoder  → z_gt (for training targets)
                    .decoder: fine-tuned decoder    → PPG  (for inference)

    Two separate checkpoints are used deliberately:
      Fine-tuned decoder from encoder_B.pt — was jointly trained with CameraEncoder.
      Original VAE encoder from stage1_vae_p4.pt — was trained on GT PPG only,
        maps GT_PPG → z_gt in the same latent space the diffusion model targets.
    Do NOT load_state_dict(full_vae) after loading fine-tuned decoder — that
    overwrites the fine-tuned decoder with the original one.
    """
    state = torch.load(ENCODER_CKPT['B'], map_location=device, weights_only=False)

    camera_enc = CameraEncoder(latent_dim=LATENT_DIM, in_channels=1, morpho_aux=False)
    camera_enc.load_state_dict(state['encoder'], strict=False)
    camera_enc.eval()
    for p in camera_enc.parameters():
        p.requires_grad_(False)

    vae = PPGVAE(latent_dim=LATENT_DIM)

    # Fine-tuned decoder: used for all inference (z → PPG)
    vae.decoder.load_state_dict(state['decoder_finetune'])

    # Original VAE encoder: used for encoding GT PPG → z_gt (training targets only)
    # Load only encoder keys from stage1_vae_p4.pt, do not touch the decoder
    vae_full_state = torch.load(VAE_CKPT, map_location=device, weights_only=False)
    enc_state = {k[len('encoder.'):]: v
                 for k, v in vae_full_state.items() if k.startswith('encoder.')}
    vae.encoder.load_state_dict(enc_state)

    vae.eval()
    for p in vae.parameters():
        p.requires_grad_(False)

    return camera_enc.to(device), vae.to(device)


# ── Dataset ───────────────────────────────────────────────────────────────────

def find_cycle_dirs():
    return [d for d in [UBFC_CYCLES_DIR, STRESS_CYCLES_DIR, FPS2023_CYCLES_DIR,
                        FPS2023_60_CYCLES_DIR, CENTAN_CYCLES_DIR]
            if Path(d).is_dir()]


class A9Dataset(Dataset):
    """
    Pre-computes (z_prior, z_gt) pairs at init. Both are 32-dim vectors.
    z_prior = CameraEncoder(rPPG_pos) — the noisy latent estimate
    z_gt    = VAE.encoder(GT_PPG).mu  — the clean latent target
    """

    def __init__(self, cycle_dirs, split_sids, camera_enc, vae, device):
        self.items    = []   # list of (z_prior: float32 (32,), z_gt: float32 (32,), sid)
        self.sid_list = []

        for d in cycle_dirs:
            npz_files = sorted(Path(d).glob('*_cycles.npz'))
            for npz_f in tqdm(npz_files, desc=Path(d).parent.name, leave=False):
                try:
                    data = np.load(npz_f, allow_pickle=True)
                    sid  = int(data['sid'])
                    if sid not in split_sids:
                        continue

                    pos = data['rppg_pos_cycles']   # (N, 256)
                    gt  = data['gt_cycles']         # (N, 256)

                    if len(gt) < 5:
                        continue

                    N = len(gt)
                    BATCH = 256

                    # z_prior = CameraEncoder(rPPG_pos)  (B, 32)
                    z_prior_list = []
                    for i in range(0, N, BATCH):
                        x = torch.from_numpy(pos[i:i+BATCH]).unsqueeze(1).float().to(device)
                        with torch.no_grad():
                            z_prior_list.append(camera_enc(x).cpu().numpy())
                    z_prior = np.concatenate(z_prior_list, axis=0)   # (N, 32)

                    # z_gt = VAE encoder mu (deterministic)
                    z_gt_list = []
                    for i in range(0, N, BATCH):
                        x = torch.from_numpy(gt[i:i+BATCH]).unsqueeze(1).float().to(device)
                        with torch.no_grad():
                            mu, _ = vae.encoder(x)
                        z_gt_list.append(mu.cpu().numpy())
                    z_gt = np.concatenate(z_gt_list, axis=0)   # (N, 32)

                    for i in range(N):
                        self.items.append((z_prior[i].astype(np.float32),
                                           z_gt[i].astype(np.float32),
                                           sid))
                        self.sid_list.append(sid)

                except Exception:
                    continue

        print(f'Loaded {len(self.items)} z-pairs from {len(set(self.sid_list))} subjects')

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        z_prior, z_gt, sid = self.items[idx]
        return (torch.from_numpy(z_prior).float(),
                torch.from_numpy(z_gt).float(),
                sid)


class A9EvalDataset(Dataset):
    """
    Like A9Dataset but also returns the raw GT cycle for metric computation.
    Returns (z_prior (32,), z_gt (32,), gt_cycle (256,), sid)
    """

    def __init__(self, cycle_dirs, split_sids, camera_enc, vae, device):
        self.items    = []
        self.sid_list = []

        for d in cycle_dirs:
            npz_files = sorted(Path(d).glob('*_cycles.npz'))
            for npz_f in tqdm(npz_files, desc=Path(d).parent.name, leave=False):
                try:
                    data = np.load(npz_f, allow_pickle=True)
                    sid  = int(data['sid'])
                    if sid not in split_sids:
                        continue

                    pos = data['rppg_pos_cycles']
                    gt  = data['gt_cycles']
                    if len(gt) < 5:
                        continue

                    N = len(gt)
                    BATCH = 256
                    z_prior_list = []
                    for i in range(0, N, BATCH):
                        x = torch.from_numpy(pos[i:i+BATCH]).unsqueeze(1).float().to(device)
                        with torch.no_grad():
                            z_prior_list.append(camera_enc(x).cpu().numpy())
                    z_prior = np.concatenate(z_prior_list, axis=0)

                    z_gt_list = []
                    for i in range(0, N, BATCH):
                        x = torch.from_numpy(gt[i:i+BATCH]).unsqueeze(1).float().to(device)
                        with torch.no_grad():
                            mu, _ = vae.encoder(x)
                        z_gt_list.append(mu.cpu().numpy())
                    z_gt = np.concatenate(z_gt_list, axis=0)

                    for i in range(N):
                        self.items.append((z_prior[i].astype(np.float32),
                                           z_gt[i].astype(np.float32),
                                           gt[i].astype(np.float32),
                                           sid))
                        self.sid_list.append(sid)

                except Exception:
                    continue

        print(f'Loaded {len(self.items)} eval z-pairs from {len(set(self.sid_list))} subjects')

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        z_prior, z_gt, gt_cycle, sid = self.items[idx]
        return (torch.from_numpy(z_prior).float(),
                torch.from_numpy(z_gt).float(),
                torch.from_numpy(gt_cycle).float(),
                sid)


class SubjectStratifiedSampler(Sampler):
    def __init__(self, dataset, batch_size, n_subs=16, seed=42):
        self.batch_size  = batch_size
        self.n_subs      = n_subs
        self.rng         = random.Random(seed)
        self.sid_to_idx  = {}
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


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')
    CKPT_A9.mkdir(parents=True, exist_ok=True)
    RESULTS_A9.mkdir(parents=True, exist_ok=True)

    split_df   = pd.read_csv(SPLIT_FILE)
    train_sids = set(split_df[split_df['split'] == 'train']['sid'])
    val_sids   = set(split_df[split_df['split'] == 'val']['sid'])

    print('Loading frozen Stage 1 models (V5-B encoder + VAE)...')
    camera_enc, vae = load_frozen_models(device)

    cycle_dirs = find_cycle_dirs()
    if not cycle_dirs:
        print('ERROR: No cycle directories found.'); return

    print('Building training dataset (pre-computing z-pairs)...')
    train_ds = A9Dataset(cycle_dirs, train_sids, camera_enc, vae, device)
    print('Building validation dataset...')
    val_ds   = A9Dataset(cycle_dirs, val_sids,   camera_enc, vae, device)

    if len(train_ds) == 0:
        print('ERROR: No training cycles.'); return

    sampler      = SubjectStratifiedSampler(train_ds, BATCH_SIZE, N_SUBJECTS_PER_BATCH)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, sampler=sampler,   num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    ldm      = LatentDiffusion(latent_dim=LATENT_DIM, T=T_STEPS).to(device)
    n_params = sum(p.numel() for p in ldm.noise_pred.parameters())
    print(f'NoisePredictor: {n_params:,} parameters  (T={T_STEPS})')

    opt       = torch.optim.Adam(ldm.noise_pred.parameters(), lr=LEARNING_RATE)
    ckpt_path = CKPT_A9 / 'ldm_a9.pt'
    log_rows  = []
    best_val  = float('inf')
    patience  = 0

    for epoch in range(1, MAX_EPOCHS + 1):
        ldm.noise_pred.train()
        tr_losses = []

        for z_prior, z_gt, _ in train_loader:
            z_prior = z_prior.to(device)
            z_gt    = z_gt.to(device)

            loss = ldm(z_gt, z_prior, p_uncond=0.1)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(ldm.noise_pred.parameters(), 1.0)
            opt.step()
            tr_losses.append(loss.item())

        ldm.noise_pred.eval()
        vl_losses = []
        with torch.no_grad():
            for z_prior, z_gt, _ in val_loader:
                vl_losses.append(ldm(z_gt.to(device), z_prior.to(device), p_uncond=0.0).item())

        avg_tr = float(np.mean(tr_losses))
        avg_vl = float(np.mean(vl_losses))
        log_rows.append({'epoch': epoch, 'train_loss': avg_tr, 'val_loss': avg_vl})

        if epoch % 10 == 0:
            print(f'  Ep {epoch:03d} | Tr {avg_tr:.4f} | Val {avg_vl:.4f}')

        if avg_vl < best_val:
            best_val, patience = avg_vl, 0
            torch.save({'model': ldm.noise_pred.state_dict(),
                        'epoch': epoch, 'val_loss': avg_vl}, ckpt_path)
        else:
            patience += 1
            if patience >= EARLY_STOP:
                print(f'  Early stop epoch {epoch}. Best val: {best_val:.4f}')
                break

    pd.DataFrame(log_rows).to_csv(RESULTS_A9 / 'training_log_a9.csv', index=False)
    print(f'\nA9 training complete. Best val: {best_val:.4f}')
    print(f'Checkpoint: {ckpt_path}')


if __name__ == '__main__':
    main()

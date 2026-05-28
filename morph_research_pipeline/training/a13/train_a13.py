"""
training/a13/train_a13.py — A13 Hybrid Morphological Contrastive Training
=========================================================================
Hypothesis (Sun et al. IJCB 2024): Template collapse is partly an optimisation
failure. A subject-level morphological contrastive loss on OUTPUT PPG waveforms
forces the encoder to preserve inter-subject morphological variation.

Architecture:
  rPPG CHROM cycle (1×256)
      → CameraEncoder (fine-tuned from encoder_B.pt)
      → z (32-dim)
      → PPGVAE decoder (frozen, from encoder_B.pt decoder_finetune)
      → PPG_pred (1×256)

Loss = λ_recon * L_pearson(PPG_pred, GT)
     + λ_contrast * L_supcon_pearson(PPG_pred, subject_labels)

L_supcon_cosine (on latent z, not output PPG):
  - Normalize z to unit norm → cosine similarity
  - SupCon: maximize similarity to same-subject positives vs all negatives
  - Temperature τ = 0.3

Batch: SubjectStratifiedSampler — ≥8 subjects × K_CYCLES cycles each.
Early stopping on cross-subject r (PRIMARY objective — not val recon loss).
Success: cross_r < 0.70 on val set (GT = 0.60 on test).

Usage:
    python training/a13/train_a13.py
    (edit LAMBDA_CONTRAST / TEMPERATURE at the top of the file to sweep)
"""

import os, sys, random
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader, Sampler
from itertools import combinations
from scipy.stats import pearsonr
from tqdm import tqdm
from pathlib import Path

HERE = Path(__file__).parent
PIPELINE_ROOT = HERE.parent.parent
sys.path.insert(0, str(PIPELINE_ROOT))

from config.paths import (
    UBFC_CYCLES_DIR, STRESS_CYCLES_DIR, FPS2023_CYCLES_DIR,
    CENTAN_CYCLES_DIR, CKPT_A13, SPLIT_FILE, ENCODER_CKPT, VAE_CKPT,
)
from config.hyperparams import LEARNING_RATE, N_SUBJECTS_PER_BATCH
from models.encoder import CameraEncoder
from models.vae import PPGVAE

LATENT_DIM      = 32
IN_CHANNELS     = 1
MAX_EPOCHS      = 200
EARLY_STOP      = 30
K_CYCLES        = 8       # cycles per subject per batch
TEMPERATURE     = 0.3
LAMBDA_RECON    = 1.0
LAMBDA_CONTRAST = 1.0
N_WARMUP        = 10      # epochs with recon=0: let SupCon cluster z before recon collapses it


CYCLE_DIRS = [
    UBFC_CYCLES_DIR, STRESS_CYCLES_DIR,
    FPS2023_CYCLES_DIR, CENTAN_CYCLES_DIR,
]


# ── Dataset ───────────────────────────────────────────────────────────────────

def find_cycle_dirs():
    return [d for d in CYCLE_DIRS if Path(d).is_dir()]


class A13Dataset(Dataset):
    """POS cycles for given subject IDs. encoder_B.pt was trained on rppg_pos_cycles."""

    def __init__(self, cycle_dirs, split_sids):
        self.items    = []   # (rppg_cycle, gt_cycle, sid)
        self.sid_list = []

        for d in cycle_dirs:
            for npz_path in sorted(Path(d).glob('*_cycles.npz')):
                if '_vmd' in npz_path.name or '_a12_' in npz_path.name:
                    continue
                try:
                    data = np.load(npz_path, allow_pickle=True)
                    sid  = int(data['sid'])
                    if sid not in split_sids:
                        continue
                    gt = data['gt_cycles']
                    rppg = (data['rppg_pos_cycles'] if 'rppg_pos_cycles' in data
                            else data['rppg_cycles'])
                    for i in range(len(gt)):
                        self.items.append((rppg[i].astype(np.float32),
                                          gt[i].astype(np.float32),
                                          sid))
                        self.sid_list.append(sid)
                except Exception:
                    continue

        print(f'A13Dataset: {len(self.items)} cycles from '
              f'{len(set(self.sid_list))} subjects (POS input)')

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        chrom, gt, sid = self.items[idx]
        return (torch.from_numpy(chrom).unsqueeze(0),
                torch.from_numpy(gt).unsqueeze(0),
                sid)


class SubjectStratifiedSampler(Sampler):
    """Each mini-batch draws K_CYCLES cycles from ≥ n_subs subjects."""

    def __init__(self, dataset, n_subs_per_batch=8, k_cycles=8, seed=42):
        self.k    = k_cycles
        self.n    = n_subs_per_batch
        self.rng  = random.Random(seed)
        self.sid_to_idxs = {}
        for idx, sid in enumerate(dataset.sid_list):
            self.sid_to_idxs.setdefault(sid, []).append(idx)
        self.all_sids = list(self.sid_to_idxs.keys())
        self._total = len(dataset)

    def __iter__(self):
        sids = self.all_sids[:]
        self.rng.shuffle(sids)
        indices = []
        for i in range(0, len(sids), self.n):
            group = sids[i:i + self.n]
            if len(group) < 2:
                continue
            for sid in group:
                pool = self.sid_to_idxs[sid][:]
                self.rng.shuffle(pool)
                indices.extend(pool[:self.k])
        return iter(indices)

    def __len__(self):
        return self._total


# ── Losses ────────────────────────────────────────────────────────────────────

def pearson_loss(pred, gt):
    """1 - mean Pearson r(pred, gt). pred, gt: (B, 1, 256)."""
    p = pred[:, 0, :]
    g = gt[:, 0, :]
    p = p - p.mean(dim=1, keepdim=True)
    g = g - g.mean(dim=1, keepdim=True)
    num = (p * g).sum(dim=1)
    den = (p.norm(dim=1) * g.norm(dim=1)).clamp(min=1e-8)
    return (1.0 - (num / den)).mean()


def supcon_cosine_loss(z, sids, temperature):
    """
    SupCon loss on latent z using cosine similarity.

    Applied in latent space (not output space) because output-space SupCon
    produces near-zero gradients when the encoder is already collapsed — all
    PPG predictions look identical, making positive/negative pairs
    indistinguishable. Latent z retains more inter-subject diversity.

    z    : (N, latent_dim)
    sids : list of int — subject IDs per sample
    """
    N = z.shape[0]
    z_n = torch.nn.functional.normalize(z, dim=1)   # (N, D) unit norm
    S   = torch.mm(z_n, z_n.t())                     # (N, N) cosine sim

    sids_t   = z.new_tensor(sids, dtype=torch.long)
    pos_mask = (sids_t.unsqueeze(0) == sids_t.unsqueeze(1))
    eye      = torch.eye(N, dtype=torch.bool, device=z.device)
    pos_mask = pos_mask & ~eye

    if not pos_mask.any():
        return z.new_tensor(0.0)

    logits    = S / temperature
    logits    = logits.masked_fill(eye, -1e9)
    log_denom = torch.logsumexp(logits, dim=1, keepdim=True)
    log_probs = logits - log_denom

    n_pos = pos_mask.float().sum(dim=1).clamp(min=1)
    return -(log_probs * pos_mask.float()).sum(dim=1).div(n_pos).mean()


# ── Validation ────────────────────────────────────────────────────────────────

def compute_val_metrics(encoder, vae, val_ds, device, max_samples=2000):
    """Returns (per_subj_r, cross_subj_r) evaluated on a subset of val cycles."""
    encoder.eval()
    idxs = list(range(len(val_ds)))
    random.shuffle(idxs)
    idxs = idxs[:max_samples]

    recon_by_sid = {}
    gt_by_sid    = {}

    with torch.no_grad():
        for idx in idxs:
            chrom, gt, sid = val_ds[idx]
            sid = int(sid)
            z   = encoder(chrom.unsqueeze(0).to(device))
            ppg = vae.decode(z).squeeze().cpu().numpy()
            recon_by_sid.setdefault(sid, []).append(ppg)
            gt_by_sid.setdefault(sid, []).append(gt.squeeze().numpy())

    cycle_rs = []
    for sid in recon_by_sid:
        for ppg, gt_np in zip(recon_by_sid[sid], gt_by_sid[sid]):
            if np.std(ppg) > 1e-6 and np.std(gt_np) > 1e-6:
                cycle_rs.append(float(pearsonr(ppg, gt_np)[0]))
    per_subj_r = float(np.mean(cycle_rs)) if cycle_rs else 0.0

    means  = {s: np.mean(v, axis=0) for s, v in recon_by_sid.items()}
    sids_s = sorted(means.keys())
    rs_out = [float(pearsonr(means[a], means[b])[0])
              for a, b in combinations(sids_s, 2)
              if np.std(means[a]) > 1e-6 and np.std(means[b]) > 1e-6]
    cross_r = float(np.mean(rs_out)) if rs_out else 1.0

    return per_subj_r, cross_r


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    lc   = LAMBDA_CONTRAST
    temp = TEMPERATURE
    print(f'\nA13 | λ_contrast={lc}  temperature={temp}')

    device    = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    split_df  = pd.read_csv(SPLIT_FILE)
    train_sids = set(split_df[split_df['split'] == 'train']['sid'])
    val_sids   = set(split_df[split_df['split'] == 'val']['sid'])

    dirs = find_cycle_dirs()
    if not dirs:
        print('ERROR: No cycle directories found.'); return

    train_ds = A13Dataset(dirs, train_sids)
    val_ds   = A13Dataset(dirs, val_sids)
    if len(train_ds) == 0:
        print('ERROR: No train cycles found. Re-run extraction.'); return

    sampler = SubjectStratifiedSampler(
        train_ds, n_subs_per_batch=N_SUBJECTS_PER_BATCH, k_cycles=K_CYCLES,
    )
    batch_size   = N_SUBJECTS_PER_BATCH * K_CYCLES
    train_loader = DataLoader(
        train_ds, batch_size=batch_size,
        sampler=sampler, num_workers=0, drop_last=True,
    )

    # Random init — fine-tuning from encoder_B.pt fails because its z space is
    # already collapsed (all z point same direction), giving SupCon zero gradient.
    encoder = CameraEncoder(latent_dim=LATENT_DIM, in_channels=IN_CHANNELS,
                            morpho_aux=False).to(device)
    encoder.train()

    vae = PPGVAE(latent_dim=LATENT_DIM).to(device)
    vae.load_state_dict(torch.load(VAE_CKPT, map_location=device, weights_only=False))
    vae.eval()
    for p in vae.parameters():
        p.requires_grad_(False)

    optimizer = torch.optim.AdamW(
        encoder.parameters(), lr=LEARNING_RATE, weight_decay=1e-4,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=MAX_EPOCHS,
    )

    CKPT_A13.mkdir(parents=True, exist_ok=True)
    suffix    = f'lc{lc}_t{temp}'
    ckpt_path = CKPT_A13 / f'encoder_a13_{suffix}.pt'

    best_cross_r = 1.0
    best_epoch   = 0
    no_improve   = 0

    print(f'Training for up to {MAX_EPOCHS} epochs | early_stop={EARLY_STOP} | '
          f'warmup={N_WARMUP} (recon=0) | device={device}')
    print(f'Train: {len(train_ds)} cycles / Val: {len(val_ds)} cycles\n')

    for epoch in range(1, MAX_EPOCHS + 1):
        encoder.train()
        ep_recon = ep_contrast = ep_total = 0.0
        n_batches = 0

        for chrom, gt, sids in tqdm(train_loader, desc=f'Ep{epoch:03d}', leave=False):
            chrom  = chrom.to(device)
            gt     = gt.to(device)
            sids_l = [int(s) for s in sids]

            z        = encoder(chrom)
            ppg_pred = vae.decode(z)

            l_recon    = pearson_loss(ppg_pred, gt)
            l_contrast = supcon_cosine_loss(z, sids_l, temperature=temp)
            lr_weight  = 0.0 if epoch <= N_WARMUP else LAMBDA_RECON
            loss       = lr_weight * l_recon + lc * l_contrast

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(encoder.parameters(), 1.0)
            optimizer.step()

            ep_recon    += l_recon.item()
            ep_contrast += l_contrast.item()
            ep_total    += loss.item()
            n_batches   += 1

        scheduler.step()

        if n_batches:
            ep_recon    /= n_batches
            ep_contrast /= n_batches
            ep_total    /= n_batches

        val_r, cross_r = compute_val_metrics(encoder, vae, val_ds, device)

        print(f'Ep{epoch:03d} | recon={ep_recon:.4f}  contrast={ep_contrast:.4f}  '
              f'total={ep_total:.4f} | val_r={val_r:.4f}  cross_r={cross_r:.4f}')

        if cross_r < best_cross_r:
            best_cross_r = cross_r
            best_epoch   = epoch
            no_improve   = 0
            torch.save({
                'encoder':          encoder.state_dict(),
                'decoder_finetune': vae.decoder.state_dict(),
                'epoch':            epoch,
                'val_r':            val_r,
                'cross_r':          cross_r,
                'lambda_contrast':  lc,
                'temperature':      temp,
            }, ckpt_path)
            print(f'  Saved best  cross_r={best_cross_r:.4f}  path={ckpt_path.name}')
        else:
            no_improve += 1
            if no_improve >= EARLY_STOP:
                print(f'\nEarly stop at epoch {epoch} (no cross_r improvement '
                      f'for {EARLY_STOP} epochs).')
                break

        if cross_r < 0.70:
            print(f'\nSUCCESS: cross_r={cross_r:.4f} < 0.70 at epoch {epoch}!')
            break

    print(f'\nDone. Best cross_r={best_cross_r:.4f} at epoch {best_epoch}')
    print(f'Checkpoint: {ckpt_path}')


if __name__ == '__main__':
    main()

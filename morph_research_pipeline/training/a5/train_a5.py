"""
training/a5/train_a5.py — A5-v4 Sequential Two-Stage Decoder Training
=======================================================================
Stage 1 (frozen): V5-B CameraEncoder → VAE Decoder → PPG_base
Stage 2 (trainable): RefineNet → residual → PPG_refined = PPG_base + residual

Input to RefineNet: cat(PPG_base, rPPG_best, session_mean) = (3, 256)

Loss (A5-v4):
  1. Pearson correlation on PPG_refined vs GT           (weight=1.0) — shape fidelity
  2. H2/H1 regression on PPG_refined vs GT              (weight=5.0) — harmonic specificity
  3. Pearson diversity on PER-SUBJECT MEANS in batch    (weight=5.0) — anti-collapse at mean level
  4. Residual variance                                  (weight=0.1) — prevents residual=0

Root cause of A5-v2/v3 failure:
  Diversity loss operated on INDIVIDUAL CYCLES. But collapse lives at the
  SUBJECT-MEAN level: individual cycles have heartbeat-level noise (rPPG_best
  varies per cycle) that gives apparent cycle-level diversity (Pearson r < 0.85
  between random heartbeats from different subjects) — the loss fires weakly and
  is satisfied "for free". However this noise cancels when averaging across
  heartbeats per subject, leaving subject-mean outputs identical.
  Evidence: div loss decreased each epoch (0.012 → 0.009) while evaluation
  cross-subject r stayed at 0.94. Model added per-cycle noise, not subject morphology.

A5-v4 fix:
  SubjectStratifiedSampler puts ~8 cycles per subject per batch. We first average
  the RefineNet output over those 8 cycles per subject (batch mean), then apply
  Pearson diversity to the resulting S subject-mean waveforms. This directly
  penalises what evaluation measures: cross-subject Pearson r on mean PPG_refined.
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
    UBFC_CYCLES_DIR, STRESS_CYCLES_DIR, FPS2023_CYCLES_DIR,
    FPS2023_60_CYCLES_DIR, CENTAN_CYCLES_DIR,
    CKPT_A5, RESULTS_A5, SPLIT_FILE,
    VAE_CKPT, ENCODER_CKPT,
)
from config.hyperparams import BATCH_SIZE, LEARNING_RATE, N_SUBJECTS_PER_BATCH
from models.encoder import CameraEncoder
from models.vae import PPGVAE
from models.refinenet_a5 import RefineNetA5

LATENT_DIM  = 32
MAX_EPOCHS  = 300
EARLY_STOP  = 30

LAMBDA_PEARSON   = 1.0   # primary: Pearson on PPG_refined vs GT
LAMBDA_H2H1      = 5.0   # harmonic regression: forces subject-specific H2/H1
LAMBDA_DIVERSITY = 5.0   # subject-mean Pearson diversity: penalise mean-level collapse
LAMBDA_RESIDUAL  = 0.1   # auxiliary: penalise residual variance collapse to zero

DIVERSITY_MARGIN = 0.85  # cross-subject Pearson r (on subject means) above this is penalised


# ── Stage 1: frozen V5-B ─────────────────────────────────────────────────────

def load_stage1(device):
    # encoder_B.pt format: {'encoder': CameraEncoder state_dict,
    #                        'decoder_finetune': PPGDecoder state_dict}
    state = torch.load(ENCODER_CKPT['B'], map_location=device, weights_only=False)

    encoder = CameraEncoder(latent_dim=LATENT_DIM, in_channels=1, morpho_aux=False)
    encoder.load_state_dict(state['encoder'], strict=False)
    encoder.eval()
    for p in encoder.parameters():
        p.requires_grad_(False)

    vae = PPGVAE(latent_dim=LATENT_DIM)
    vae.decoder.load_state_dict(state['decoder_finetune'])
    vae.eval()
    for p in vae.parameters():
        p.requires_grad_(False)

    return encoder.to(device), vae.to(device)


def compute_ppg_base(rppg_np, encoder, vae, device, batch=256):
    """rppg_np: (N, 256) → ppg_base: (N, 256) in [0,1]. Runs in eval mode."""
    N = len(rppg_np)
    out = np.zeros((N, 256), dtype=np.float32)
    for i in range(0, N, batch):
        x = torch.from_numpy(rppg_np[i:i+batch]).unsqueeze(1).float().to(device)
        with torch.no_grad():
            z    = encoder(x)          # (B, 32)
            recon = vae.decode(z)      # (B, 1, 256)
        out[i:i+batch] = recon.cpu().numpy()[:, 0, :]
    return out


# ── Dataset ───────────────────────────────────────────────────────────────────

def _session_mean(pos_cycles, pos_valid):
    """Quality-weighted session mean from valid POS cycles (top 50%)."""
    valid = pos_cycles[pos_valid]
    if len(valid) >= 4:
        rough = valid.mean(axis=0)
        corrs = np.array([float(np.corrcoef(c, rough)[0, 1]) for c in valid])
        top   = valid[corrs >= np.percentile(corrs, 50)]
        return top.mean(axis=0).astype(np.float32)
    if len(valid) > 0:
        return valid.mean(axis=0).astype(np.float32)
    return pos_cycles.mean(axis=0).astype(np.float32)


def _best_rppg(pos, harm, phy, pos_v, harm_v, phy_v, session_mean):
    """Per heartbeat: pick algorithm with highest Pearson r against session_mean."""
    N = len(pos)
    best = np.zeros((N, 256), dtype=np.float32)
    for i in range(N):
        candidates = []
        if pos_v[i]:
            r = float(np.corrcoef(pos[i],  session_mean)[0, 1])
            candidates.append((r, pos[i]))
        if harm_v[i]:
            r = float(np.corrcoef(harm[i], session_mean)[0, 1])
            candidates.append((r, harm[i]))
        if phy_v[i]:
            r = float(np.corrcoef(phy[i],  session_mean)[0, 1])
            candidates.append((r, phy[i]))
        best[i] = max(candidates, key=lambda x: x[0])[1] if candidates else pos[i]
    return best


class A5Dataset(Dataset):
    """
    Loads cycle npz files. Pre-computes PPG_base (Stage 1 output) once at init.
    Returns (inp (3,256), gt (1,256), sid) per heartbeat.
    """

    def __init__(self, cycles_dirs, split_sids, encoder, vae, device):
        self.items    = []
        self.sid_list = []

        for d in cycles_dirs:
            npz_files = sorted(Path(d).glob('*_cycles.npz'))
            for npz_f in tqdm(npz_files, desc=Path(d).parent.name, leave=False):
                try:
                    data   = np.load(npz_f, allow_pickle=True)
                    sid    = int(data['sid'])
                    if sid not in split_sids:
                        continue

                    pos    = data['rppg_pos_cycles']      # (N, 256)
                    harm   = data['rppg_harm_cycles']     # (N, 256)
                    phy    = data['rppg_phybrid_cycles']  # (N, 256)
                    gt     = data['gt_cycles']            # (N, 256)
                    pos_v  = data['rppg_pos_valid']       # (N,) bool
                    harm_v = data['rppg_harm_valid']      # (N,) bool
                    phy_v  = data['rppg_phybrid_valid']   # (N,) bool

                    if len(gt) < 5:
                        continue

                    smean     = _session_mean(pos, pos_v)
                    rppg_best = _best_rppg(pos, harm, phy, pos_v, harm_v, phy_v, smean)
                    # Stage 1 (V5-B) was trained on rppg_pos_cycles — must use pos, not best-of-three
                    ppg_base  = compute_ppg_base(pos, encoder, vae, device)

                    for i in range(len(gt)):
                        inp = np.stack([ppg_base[i], rppg_best[i], smean], axis=0)  # (3, 256)
                        self.items.append((inp.astype(np.float32),
                                           gt[i].astype(np.float32),
                                           sid))
                        self.sid_list.append(sid)
                except Exception as e:
                    continue

        print(f'Loaded {len(self.items)} cycles from {len(set(self.sid_list))} subjects')

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        inp, gt, sid = self.items[idx]
        return (torch.from_numpy(inp).float(),
                torch.from_numpy(gt).unsqueeze(0).float(),
                sid)


class SubjectStratifiedSampler(Sampler):
    def __init__(self, dataset, batch_size, n_subs=16, seed=42):
        self.batch_size = batch_size
        self.n_subs     = n_subs
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


# ── Losses ────────────────────────────────────────────────────────────────────

def pearson_loss(pred, target):
    """1 - mean Pearson r. Sign-sensitive: r(x,-x) = -1 → loss = 2.0."""
    p = pred.view(pred.size(0), -1)
    t = target.view(target.size(0), -1)
    pz = (p - p.mean(1, keepdim=True)) / (p.std(1, keepdim=True) + 1e-8)
    tz = (t - t.mean(1, keepdim=True)) / (t.std(1, keepdim=True) + 1e-8)
    return 1 - (pz * tz).mean(1).mean()


def h2h1_loss(refined, gt):
    """MSE on H2/H1 harmonic ratio — forces subject-specific harmonic structure."""
    # refined, gt: (B, 1, 256)
    p = refined.squeeze(1)   # (B, 256)
    t = gt.squeeze(1)        # (B, 256)
    fft_p = torch.fft.rfft(p, dim=1).abs()   # (B, 129)
    fft_t = torch.fft.rfft(t, dim=1).abs()
    h2h1_p = fft_p[:, 2] / (fft_p[:, 1] + 1e-8)
    h2h1_t = fft_t[:, 2] / (fft_t[:, 1] + 1e-8)
    return F.mse_loss(h2h1_p, h2h1_t)


def diversity_loss(refined, sids, margin=DIVERSITY_MARGIN):
    """
    Penalise cross-subject Pearson r > margin on PER-SUBJECT MEAN waveforms.

    The SubjectStratifiedSampler puts ~n_per cycles per subject per batch.
    We average them to get one mean waveform per subject, then compute the
    Pearson r matrix across subjects. This directly measures what evaluation
    reports as cross-subject r — operating on individual cycles was trivially
    satisfied by per-heartbeat noise that cancels on averaging (A5-v2/v3 failure).
    """
    p = refined.squeeze(1)   # (B, 256)
    sids_t = (sids if isinstance(sids, torch.Tensor) else torch.tensor(sids)).to(refined.device)
    unique_sids = torch.unique(sids_t)
    S = len(unique_sids)
    if S < 2:
        return refined.new_zeros(1).squeeze()

    # Per-subject mean within the batch  (S, 256)
    means = torch.stack([p[sids_t == sid].mean(0) for sid in unique_sids], dim=0)

    # Pearson r matrix on subject means
    mz = (means - means.mean(1, keepdim=True)) / (means.std(1, keepdim=True) + 1e-8)
    pearson_mat = (mz @ mz.t()) / means.size(1)   # (S, S)

    # Penalise all off-diagonal pairs where r > margin
    eye_mask = torch.eye(S, device=refined.device)
    penalty  = F.relu(pearson_mat - margin) * (1 - eye_mask)
    return penalty.sum() / max(S * (S - 1), 1)


def residual_variance_loss(residual):
    """Penalise if residual std collapses toward zero across the batch."""
    std = residual.view(residual.size(0), -1).std(dim=1).mean()
    return F.relu(0.02 - std)  # encourage std >= 0.02


# ── Main ──────────────────────────────────────────────────────────────────────

def find_cycle_dirs():
    return [d for d in [UBFC_CYCLES_DIR, STRESS_CYCLES_DIR, FPS2023_CYCLES_DIR,
                        FPS2023_60_CYCLES_DIR, CENTAN_CYCLES_DIR]
            if Path(d).is_dir()]


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')
    CKPT_A5.mkdir(parents=True, exist_ok=True)
    RESULTS_A5.mkdir(parents=True, exist_ok=True)

    split_df   = pd.read_csv(SPLIT_FILE)
    train_sids = set(split_df[split_df['split'] == 'train']['sid'])
    val_sids   = set(split_df[split_df['split'] == 'val']['sid'])

    print('Loading Stage 1 (frozen V5-B)...')
    encoder_b, vae = load_stage1(device)

    cycle_dirs = find_cycle_dirs()
    if not cycle_dirs:
        print('ERROR: No cycles directories found.'); return

    print('Building training dataset (pre-computing PPG_base)...')
    train_ds = A5Dataset(cycle_dirs, train_sids, encoder_b, vae, device)
    print('Building validation dataset...')
    val_ds   = A5Dataset(cycle_dirs, val_sids,   encoder_b, vae, device)

    if len(train_ds) == 0:
        print('ERROR: No training cycles loaded.'); return

    sampler      = SubjectStratifiedSampler(train_ds, BATCH_SIZE, N_SUBJECTS_PER_BATCH)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, sampler=sampler,   num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    refine    = RefineNetA5().to(device)
    n_params  = sum(p.numel() for p in refine.parameters())
    print(f'RefineNetA5: {n_params:,} parameters')
    print(f'A5-v4 Lambdas: pearson={LAMBDA_PEARSON}, h2h1={LAMBDA_H2H1}, '
          f'subject_mean_diversity={LAMBDA_DIVERSITY} (margin={DIVERSITY_MARGIN}), '
          f'residual_var={LAMBDA_RESIDUAL}')

    opt       = torch.optim.Adam(refine.parameters(), lr=LEARNING_RATE)
    ckpt_path = CKPT_A5 / 'refinenet_a5.pt'
    log_rows  = []
    best_val  = float('inf')
    patience  = 0

    for epoch in range(1, MAX_EPOCHS + 1):
        refine.train()
        tr_losses, tr_pearson, tr_h2h1, tr_div, tr_resvar = [], [], [], [], []

        for inp, gt, sids in train_loader:
            inp  = inp.to(device)   # (B, 3, 256)
            gt   = gt.to(device)    # (B, 1, 256)

            ppg_base    = inp[:, :1, :]           # (B, 1, 256)
            residual    = refine(inp)             # (B, 1, 256)
            ppg_refined = ppg_base + residual     # (B, 1, 256)

            l_pearson  = pearson_loss(ppg_refined, gt)                     * LAMBDA_PEARSON
            l_h2h1     = h2h1_loss(ppg_refined, gt)                        * LAMBDA_H2H1
            l_div      = diversity_loss(ppg_refined, sids)                 * LAMBDA_DIVERSITY
            l_resvar   = residual_variance_loss(residual)                  * LAMBDA_RESIDUAL
            loss       = l_pearson + l_h2h1 + l_div + l_resvar

            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(refine.parameters(), 1.0)
            opt.step()

            tr_losses.append(loss.item())
            tr_pearson.append(l_pearson.item())
            tr_h2h1.append(l_h2h1.item())
            tr_div.append(l_div.item())
            tr_resvar.append(l_resvar.item())

        refine.eval()
        vl_losses = []
        with torch.no_grad():
            for inp, gt, _ in val_loader:
                inp, gt   = inp.to(device), gt.to(device)
                ppg_base  = inp[:, :1, :]
                residual  = refine(inp)
                ppg_refined = ppg_base + residual
                vl = (pearson_loss(ppg_refined, gt) * LAMBDA_PEARSON
                      + h2h1_loss(ppg_refined, gt)  * LAMBDA_H2H1)
                vl_losses.append(vl.item())

        avg_tr  = float(np.mean(tr_losses))
        avg_vl  = float(np.mean(vl_losses))
        log_rows.append({
            'epoch':      epoch,
            'train_loss': avg_tr,
            'val_loss':   avg_vl,
            'pearson':    float(np.mean(tr_pearson)),
            'h2h1':       float(np.mean(tr_h2h1)),
            'diversity':  float(np.mean(tr_div)),
            'resvar':     float(np.mean(tr_resvar)),
        })

        if epoch % 10 == 0:
            print(f'  Ep {epoch:03d} | Tr {avg_tr:.4f} | Val {avg_vl:.4f} | '
                  f'pear={np.mean(tr_pearson):.4f} h2h1={np.mean(tr_h2h1):.4f} '
                  f'div={np.mean(tr_div):.4f} resv={np.mean(tr_resvar):.4f}')

        if avg_vl < best_val:
            best_val, patience = avg_vl, 0
            torch.save({'model': refine.state_dict(), 'epoch': epoch, 'val_loss': avg_vl},
                       ckpt_path)
        else:
            patience += 1
            if patience >= EARLY_STOP:
                print(f'  Early stop epoch {epoch}. Best val: {best_val:.4f}')
                break

    pd.DataFrame(log_rows).to_csv(RESULTS_A5 / 'training_log_a5.csv', index=False)
    print(f'\nA5 training complete. Best val: {best_val:.4f}')
    print(f'Checkpoint: {ckpt_path}')


if __name__ == '__main__':
    main()

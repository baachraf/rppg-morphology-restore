"""
step4_stage1_vae.py — Stage 1: Data Auditing and PPG VAE Training
==================================================================
1. Audits the morphology (notch detection) of the 127k extracted cycles.
2. Creates the subject-level train/val/test split.
3. Trains the VAE to learn the "PPG Shape Dictionary."
"""

import os
import sys
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from pathlib import Path
import gc

# ── path setup ────────────────────────────────────────────────────────────────
HERE = Path(__file__).parent
PIPELINE_ROOT = HERE.parent
sys.path.insert(0, str(PIPELINE_ROOT))

from morph_config import (
    CYCLES_DIR, CHECKPOINTS_DIR, RESULTS_DIR,
    LATENT_DIM, BATCH_SIZE, LEARNING_RATE,
    MAX_EPOCHS_STAGE1, EARLY_STOP_PATIENCE, BETA_KL,
    TRAIN_FRAC, VAL_FRAC, SPLIT_SEED,
    DATALOADER_WORKERS, DATALOADER_PIN_MEMORY,
    VAE_HIGH_QUALITY_ONLY, VAE_MIN_SID, VAE_CKPT_P4
)
from models.vae import PPGVAE, vae_loss

# ── User Task Control ────────────────────────────────────────────────────────
RUN_TRAINING = True  # Set to True to execute the neural network training


# ── Morphological Helper ──────────────────────────────────────────────────────
def detect_notch_simple(cycle):
    """Minimal notch detection for audit during loading."""
    d2 = np.diff(np.diff(cycle))
    # Search for inflection in descending limb [60:180]
    if np.max(d2[60:180]) > 0.005: return True
    return False

# ==============================================================================
# DATASET & SPLITTING
# ==============================================================================

class MultiDatasetPPG(Dataset):
    def __init__(self, cycles_root):
        self.cycles, self.sids, self.datasets = [], [], []
        self.subject_stats = {}
        root = Path(cycles_root)
        all_npz = list(root.rglob("*_cycles.npz"))
        
        print(f"Auditing {len(all_npz)} subjects...")
        for npz_p in tqdm(all_npz, desc="Auditing Morphology"):
            try:
                ds_name = npz_p.parent.name
                data = np.load(npz_p)
                gt = data['gt_cycles']
                sid = int(data['sid'])
                
                if sid not in self.subject_stats:
                    self.subject_stats[sid] = {'count': 0, 'notch_count': 0, 'dataset': ds_name}
                
                for i in range(len(gt)):
                    # Phase 4: skip UBFC subjects when Polymate-only mode is active
                    if VAE_HIGH_QUALITY_ONLY and sid < VAE_MIN_SID:
                        continue
                    c = gt[i].astype(np.float32)
                    self.cycles.append(c)
                    self.sids.append(sid)
                    self.datasets.append(ds_name)
                    self.subject_stats[sid]['count'] += 1
                    if detect_notch_simple(c):
                        self.subject_stats[sid]['notch_count'] += 1
            except: continue

        if VAE_HIGH_QUALITY_ONLY:
            n_subjects = len(self.subject_stats)
            n_cycles   = len(self.cycles)
            print(f'  [Phase 4] Polymate-only mode: {n_subjects} subjects, '
                  f'{n_cycles} cycles (SID >= {VAE_MIN_SID})')
        else:
            print(f'  [Standard] All datasets: {len(self.subject_stats)} subjects, '
                  f'{len(self.cycles)} cycles')

    def __len__(self): return len(self.cycles)
    def __getitem__(self, idx):
        return torch.from_numpy(self.cycles[idx]).unsqueeze(0), self.sids[idx]

def generate_split(dataset):
    """Creates a subject-level split with morphological metadata."""
    all_sids = sorted(list(dataset.subject_stats.keys()))
    rng = np.random.RandomState(SPLIT_SEED)
    rng.shuffle(all_sids)

    n_tr, n_vl = int(len(all_sids)*TRAIN_FRAC), int(len(all_sids)*VAL_FRAC)
    tr_sids, vl_sids, ts_sids = set(all_sids[:n_tr]), set(all_sids[n_tr:n_tr+n_vl]), set(all_sids[n_tr+n_vl:])

    split_rows = []
    for sid, st in dataset.subject_stats.items():
        split = 'train' if sid in tr_sids else ('val' if sid in vl_sids else 'test')
        split_rows.append({
            'sid': sid, 'dataset': st['dataset'], 'cycles': st['count'], 
            'notch_rate': st['notch_count']/st['count'] if st['count']>0 else 0,
            'split': split
        })
    
    df = pd.DataFrame(split_rows)
    out_p = Path(RESULTS_DIR) / 'subject_split_audited.csv'
    df.to_csv(out_p, index=False)
    print(f"Success: Full research split saved to {out_p}")
    return df, {'train': tr_sids, 'val': vl_sids, 'test': ts_sids}

# ==============================================================================
# MAIN EXECUTION
# ==============================================================================

def main():
    print("\nStage 1: Unified Data Audit & Split")
    print("-" * 40)
    
    # 1. Audit and Split (Always runs)
    full_ds = MultiDatasetPPG(CYCLES_DIR)
    split_df, split_map = generate_split(full_ds)
    
    if not RUN_TRAINING:
        print("\nRUN_TRAINING is False. Exiting cleanly after split generation.")
        return

    # 2. Training Loop (Only runs if requested)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nStarting VAE Training on {device}...")
    
    train_idx = [i for i, s in enumerate(full_ds.sids) if s in split_map['train']]
    val_idx   = [i for i, s in enumerate(full_ds.sids) if s in split_map['val']]
    
    train_loader = DataLoader(torch.utils.data.Subset(full_ds, train_idx), batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader   = DataLoader(torch.utils.data.Subset(full_ds, val_idx), batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    model = PPGVAE(latent_dim=LATENT_DIM).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    best_loss = float('inf')
    for epoch in range(1, MAX_EPOCHS_STAGE1 + 1):
        model.train(); tr_l = []
        for x, _ in train_loader:
            x = x.to(device); opt.zero_grad()
            recon, mu, logvar = model(x)
            loss, _, _ = vae_loss(recon, x, mu, logvar, beta=BETA_KL)
            loss.backward(); opt.step(); tr_l.append(loss.item())
            
        model.eval(); vl_l = []
        with torch.no_grad():
            for x, _ in val_loader:
                x = x.to(device)
                recon, mu, logvar = model(x)
                l, _, _ = vae_loss(recon, x, mu, logvar, beta=BETA_KL)
                vl_l.append(l.item())
        
        avg_vl = np.mean(vl_l)
        print(f"Epoch {epoch:03d} | Train: {np.mean(tr_l):.4f} | Val: {avg_vl:.4f}")
        
        if avg_vl < best_loss:
            best_loss = avg_vl
            torch.save(model.state_dict(), Path(CHECKPOINTS_DIR) / VAE_CKPT_P4)

    print(f"\nVAE Training Complete. Model saved: {VAE_CKPT_P4}")

if __name__ == "__main__":
    main()

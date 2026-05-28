"""
step5_stage2_encoders_v6.py — V6 Orthogonal Cascade Training (Production)
=========================================================================
Step 32 config: Full 100-trial Optuna sweep results (composite score 0.4776).
Warmup scheduling for auxiliary losses + spectral L1 + tuned recon_weight.
"""

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
import sys
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
from pathlib import Path

HERE = Path(__file__).parent
PIPELINE_ROOT = HERE.parent.parent
sys.path.insert(0, str(PIPELINE_ROOT))

from morph_config import (
    CYCLES_DIR, CHECKPOINTS_DIR, RESULTS_DIR, SPLIT_FILE,
    LATENT_DIM, BATCH_SIZE, LEARNING_RATE,
    MAX_EPOCHS_STAGE2, EARLY_STOP_PATIENCE,
    V6_CKPT_DIR, V6_RESULTS_DIR,
    MACRO_ENCODER_V6_CKPT, MICRO_ENCODER_V6_CKPT, COND_DECODER_V6_CKPT
)

from models.encoder_v6 import MacroEncoder, MicroEncoder, orthogonal_loss
from models.decoder_v6 import ConditionalDecoder
from training.v5.train_encoders import UnifiedCycleDataset, SubjectStratifiedSampler
from models.metrics import batch_morpho_labels
from models.encoder import subject_contrastive_loss

LAMBDA_GRL         = 0.23502307915207782
LAMBDA_ORTHO       = 0.022394495793177528
LAMBDA_MORPHO      = 2.6525901596034362
LAMBDA_ID_CE       = 0.011169727254027656
LAMBDA_RECON       = 1.0878711978620335
LAMBDA_CONTRASTIVE = 0.5
LAMBDA_SPECTRAL    = 0.02037827763067751
WARMUP_EPOCHS      = 9
GRL_START_EPOCH    = 12
OPTUNA_LR          = 0.00019285457107956743


def spectral_l1_loss(pred, target):
    pred_amp = torch.abs(torch.fft.rfft(pred, dim=-1))
    target_amp = torch.abs(torch.fft.rfft(target, dim=-1))
    return F.l1_loss(pred_amp, target_amp)


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    print(f"Config: GRL={LAMBDA_GRL} ORTHO={LAMBDA_ORTHO} MORPHO={LAMBDA_MORPHO} "
          f"ID_CE={LAMBDA_ID_CE} SPEC={LAMBDA_SPECTRAL} LR={OPTUNA_LR} "
          f"WARMUP={WARMUP_EPOCHS} GRL_START={GRL_START_EPOCH}")

    split_df = pd.read_csv(SPLIT_FILE)
    train_sids = set(split_df[split_df['split']=='train']['sid'])
    val_sids   = set(split_df[split_df['split']=='val']['sid'])

    unique_sids = sorted(list(train_sids.union(val_sids)))
    sid_to_class = {sid: i for i, sid in enumerate(unique_sids)}
    num_subjects = len(unique_sids)

    print(f"Train: {len(train_sids)} subjects | Val: {len(val_sids)} subjects | Classes: {num_subjects}")

    train_ds = UnifiedCycleDataset(CYCLES_DIR, train_sids)
    val_ds   = UnifiedCycleDataset(CYCLES_DIR, val_sids)

    sampler = SubjectStratifiedSampler(train_ds, BATCH_SIZE, 16)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, sampler=sampler, num_workers=0)
    val_loader   = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    macro = MacroEncoder(latent_dim=LATENT_DIM, in_channels=1, num_subjects=num_subjects).to(device)
    micro = MicroEncoder(latent_dim=LATENT_DIM, in_channels=1, num_subjects=num_subjects).to(device)
    decoder = ConditionalDecoder(latent_dim=LATENT_DIM).to(device)

    macro_backbone_params = [p for n, p in macro.named_parameters() if 'id_head' not in n]
    main_optimizer = torch.optim.Adam(
        macro_backbone_params + list(micro.parameters()) + list(decoder.parameters()),
        lr=OPTUNA_LR
    )
    id_head_optimizer = torch.optim.Adam(macro.id_head.parameters(), lr=OPTUNA_LR)

    ce_loss_fn  = nn.CrossEntropyLoss()
    mse_loss_fn = nn.MSELoss()
    l1_loss_fn  = nn.L1Loss()

    best_loss = float('inf')
    patience = 0

    print(f"V6 Step 31 Training (Optuna sweep config, warmup={WARMUP_EPOCHS}, lr={OPTUNA_LR})")

    for epoch in range(1, MAX_EPOCHS_STAGE2 + 1):
        in_warmup = epoch <= WARMUP_EPOCHS
        grl_active = epoch > GRL_START_EPOCH
        effective_grl = LAMBDA_GRL if grl_active else 0.0

        macro.train(); micro.train(); decoder.train()
        tr_losses = []
        macro_id_accs = []

        for gt, g, rppg, sids in train_loader:
            gt, g = gt.to(device), g.to(device)
            x_in = g
            class_ids = torch.tensor([sid_to_class[int(s)] for s in sids], device=device)
            morpho_t = torch.from_numpy(batch_morpho_labels(gt.cpu().numpy()[:, 0, :])).to(device)

            main_optimizer.zero_grad()
            z_macro, _ = macro(x_in)
            z_micro, morpho_preds, id_preds_micro = micro(x_in, grl_lambda=effective_grl)
            recon = decoder(z_macro, z_micro)

            loss_recon = l1_loss_fn(recon, gt) * LAMBDA_RECON

            if in_warmup:
                total_loss = loss_recon
            else:
                loss_ortho = orthogonal_loss(z_macro, z_micro) * LAMBDA_ORTHO
                loss_morpho = mse_loss_fn(morpho_preds, morpho_t) * LAMBDA_MORPHO
                loss_id = ce_loss_fn(id_preds_micro, class_ids) * LAMBDA_ID_CE
                loss_spec = spectral_l1_loss(recon, gt) * LAMBDA_SPECTRAL
                loss_con = subject_contrastive_loss(z_macro, sids.to(device)) * LAMBDA_CONTRASTIVE

                total_loss = loss_recon + loss_ortho + loss_morpho + loss_id + loss_spec + loss_con

            total_loss.backward()
            main_optimizer.step()
            tr_losses.append(total_loss.item())

            id_head_optimizer.zero_grad()
            with torch.no_grad():
                z_macro_det, _ = macro(x_in)
            id_preds_macro = macro.id_head(z_macro_det.detach())
            loss_id_macro = ce_loss_fn(id_preds_macro, class_ids)
            loss_id_macro.backward()
            id_head_optimizer.step()
            macro_id_accs.append((id_preds_macro.argmax(1) == class_ids).float().mean().item())

        macro.eval(); micro.eval(); decoder.eval()
        vl_losses = []
        with torch.no_grad():
            for gt, g, rppg, sids in val_loader:
                gt, g = gt.to(device), g.to(device)
                class_ids = torch.tensor([sid_to_class[int(s)] for s in sids], device=device)
                morpho_t = torch.from_numpy(batch_morpho_labels(gt.cpu().numpy()[:, 0, :])).to(device)
                z_macro, _ = macro(g)
                z_micro, morpho_preds, id_preds_micro = micro(g, grl_lambda=LAMBDA_GRL)
                recon = decoder(z_macro, z_micro)
                total_vl_loss = (
                    l1_loss_fn(recon, gt) * LAMBDA_RECON +
                    orthogonal_loss(z_macro, z_micro) * LAMBDA_ORTHO +
                    mse_loss_fn(morpho_preds, morpho_t) * LAMBDA_MORPHO +
                    ce_loss_fn(id_preds_micro, class_ids) * LAMBDA_ID_CE +
                    spectral_l1_loss(recon, gt) * LAMBDA_SPECTRAL +
                    subject_contrastive_loss(z_macro, sids.to(device)) * LAMBDA_CONTRASTIVE
                )
                vl_losses.append(total_vl_loss.item())

        avg_tr, avg_vl = np.mean(tr_losses), np.mean(vl_losses)
        avg_macro_id = np.mean(macro_id_accs)

        phase_tag = "[warmup]" if in_warmup else ("[GRL:off]" if not grl_active else "[full]")
        if epoch % 5 == 0 or epoch == 1:
            print(f"Epoch {epoch:03d} | Tr: {avg_tr:.4f} | Val: {avg_vl:.4f} | MacroID: {avg_macro_id:.3f} {phase_tag}")

        if avg_vl < best_loss:
            best_loss = avg_vl; patience = 0
            torch.save(macro.state_dict(), Path(V6_CKPT_DIR) / MACRO_ENCODER_V6_CKPT)
            torch.save(micro.state_dict(), Path(V6_CKPT_DIR) / MICRO_ENCODER_V6_CKPT)
            torch.save(decoder.state_dict(), Path(V6_CKPT_DIR) / COND_DECODER_V6_CKPT)
        else:
            patience += 1
            if patience >= EARLY_STOP_PATIENCE:
                print(f"Early stopping at epoch {epoch}. Best Val: {best_loss:.4f}")
                break

    print(f"Training complete. Best Val: {best_loss:.4f}")


if __name__ == "__main__":
    main()

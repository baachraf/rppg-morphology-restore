"""
step6_evaluate_v6.py — V6 Zero-Shot Evaluation
==============================================
Validates that the Orthogonal Biological Cascade successfully tracks
subject-specific vascular morphology (H2/H1) without requiring any
subject calibration (zero-shot).
"""

import os, sys
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from scipy.stats import pearsonr
from tqdm import tqdm
from pathlib import Path

# -- path setup ----------------------------------------------------------------
HERE = Path(__file__).parent
PIPELINE_ROOT = HERE.parent.parent
sys.path.insert(0, str(PIPELINE_ROOT))

from morph_config import (
    CYCLES_DIR, CHECKPOINTS_DIR, RESULTS_DIR,
    LATENT_DIM, BATCH_SIZE, V6_CKPT_DIR, V6_RESULTS_DIR,
    MACRO_ENCODER_V6_CKPT, MICRO_ENCODER_V6_CKPT, COND_DECODER_V6_CKPT
)
from models.encoder_v6 import MacroEncoder, MicroEncoder
from models.decoder_v6 import ConditionalDecoder
from models.metrics import batch_morpho_labels
from training.v5.train_encoders import UnifiedCycleDataset

def shape_r(a, b):
    if np.std(a) < 1e-6 or np.std(b) < 1e-6: return 0.0
    return pearsonr(a, b)[0]

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # 1. Load Split
    split_df = pd.read_csv(Path(RESULTS_DIR)/'subject_split_audited.csv')
    test_sids = set(split_df[split_df['split']=='test']['sid'])
    train_sids = set(split_df[split_df['split']=='train']['sid'])
    val_sids = set(split_df[split_df['split']=='val']['sid'])
    
    unique_sids = sorted(list(train_sids.union(val_sids)))
    num_subjects = len(unique_sids)

    test_ds = UnifiedCycleDataset(CYCLES_DIR, test_sids)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    # 2. Load V6 Models
    macro = MacroEncoder(latent_dim=LATENT_DIM, in_channels=1, num_subjects=num_subjects).to(device)
    micro = MicroEncoder(latent_dim=LATENT_DIM, in_channels=1, num_subjects=num_subjects).to(device)
    decoder = ConditionalDecoder(latent_dim=LATENT_DIM).to(device)

    macro.load_state_dict(torch.load(Path(V6_CKPT_DIR) / MACRO_ENCODER_V6_CKPT, map_location=device, weights_only=True))
    micro.load_state_dict(torch.load(Path(V6_CKPT_DIR) / MICRO_ENCODER_V6_CKPT, map_location=device, weights_only=True))
    decoder.load_state_dict(torch.load(Path(V6_CKPT_DIR) / COND_DECODER_V6_CKPT, map_location=device, weights_only=True))

    macro.eval(); micro.eval(); decoder.eval()

    all_results = []
    
    print("\nRunning V6 Zero-Shot Evaluation Pass...")
    with torch.no_grad():
        for gt, g, rppg, sids in tqdm(test_loader):
            gt_np = gt.numpy()[:,0,:]
            g_t = g.to(device)
            sids_np = sids.numpy()

            morpho_labels_np = batch_morpho_labels(gt_np)

            z_macro, _ = macro(g_t)
            z_micro, morpho_preds, id_preds = micro(g_t)
            recon = decoder(z_macro, z_micro).cpu().numpy()[:,0,:]
            
            morpho_preds_np = morpho_preds.cpu().numpy()

            for i in range(len(gt_np)):
                sid = int(sids_np[i])
                ds_name = split_df[split_df['sid']==sid]['dataset'].iloc[0]
                
                corr = shape_r(recon[i], gt_np[i])
                
                all_results.append({
                    'sid': sid,
                    'dataset': ds_name,
                    'shape_r': corr,
                    'gt_notch_pos': morpho_labels_np[i, 0],
                    'pred_notch_pos': morpho_preds_np[i, 0],
                    'gt_ipa': morpho_labels_np[i, 1],
                    'pred_ipa': morpho_preds_np[i, 1],
                    'gt_rise_time': morpho_labels_np[i, 2],
                    'pred_rise_time': morpho_preds_np[i, 2],
                })

    res_df = pd.DataFrame(all_results)

    print("\n" + "="*60)
    print("FINAL V6 ZERO-SHOT RESULTS: ORTHOGONAL CASCADE")
    print("="*60)
    
    # 1. Shape Restoration 
    summary_r = res_df.groupby('dataset')['shape_r'].mean()
    print("\n[Shape Restoration Accuracy (Pearson r)]")
    print(summary_r.round(4))
    print(f"Overall Mean: {res_df['shape_r'].mean():.4f}")

    # 2. Stiffness tracking
    gt_ipa = res_df['gt_ipa']
    pred_ipa = res_df['pred_ipa']
    
    # Subject level aggregation
    subj_df = res_df.groupby('sid')[['gt_ipa', 'pred_ipa', 'dataset']].first()
    
    # Filter for clinical datasets (stress2023, centan) which have real GT morphology
    clin_df = subj_df[subj_df['dataset'].isin(['stress2023', 'centan'])]
    
    # Check if there's any variance
    if len(clin_df) > 1 and np.std(clin_df['gt_ipa']) > 1e-5 and np.std(clin_df['pred_ipa']) > 1e-5:
        ipa_corr = pearsonr(clin_df['gt_ipa'], clin_df['pred_ipa'])[0]
    else:
        ipa_corr = 0.0
        
    print("\n[Zero-Shot Arterial Stiffness (IPA) Tracking - Clinical GT Only]")
    print(f"Subject-Level Correlation (R): {ipa_corr:.4f}")
    
    print("\n[Stiffness Detail - Clinical Unseen Subjects]")
    print(clin_df[['gt_ipa', 'pred_ipa']].head(10).round(3))

    os.makedirs(V6_RESULTS_DIR, exist_ok=True)
    res_df.to_csv(Path(V6_RESULTS_DIR) / 'v6_zero_shot_eval.csv', index=False)
    print(f"\nSaved evaluation to {Path(V6_RESULTS_DIR) / 'v6_zero_shot_eval.csv'}")

if __name__ == "__main__":
    main()

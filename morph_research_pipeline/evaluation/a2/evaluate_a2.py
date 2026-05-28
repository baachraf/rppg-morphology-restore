"""
evaluation/a2/evaluate_a2.py — A2 Evaluation (Flow Decoder)
==========================================================
Evaluates A2 CameraEncoder + Conditional Flow Decoder on test split.

Outputs:
  results/a2/full_eval_a2.csv
  results/a2/summary_a2.txt
"""

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
import sys
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from scipy.stats import pearsonr
from tqdm import tqdm
from pathlib import Path

HERE = Path(__file__).parent
PIPELINE_ROOT = HERE.parent.parent
sys.path.insert(0, str(PIPELINE_ROOT))

from morph_config import CYCLES_DIR, SPLIT_FILE, BATCH_SIZE, A2_CKPT_DIR, A2_RESULTS_DIR
from models.encoder_a2 import CameraEncoderFlow
from models.flow_a2 import ConditionalFlowDecoder
from models.metrics import batch_morpho_labels, compute_ipa, notch_index
from training.v5.train_encoders import UnifiedCycleDataset

LATENT_DIM = 64
FLOW_STEPS = 10


def shape_r(a, b):
    if np.std(a) < 1e-6 or np.std(b) < 1e-6:
        return 0.0
    return pearsonr(a, b)[0]


def compute_h2_h1(cycle):
    fft = np.abs(np.fft.rfft(cycle))
    if len(fft) < 4 or fft[1] < 1e-8:
        return 0.0
    return fft[2] / fft[1]


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    enc_ckpt = Path(A2_CKPT_DIR) / 'encoder_a2_B.pt'
    flow_ckpt = Path(A2_CKPT_DIR) / 'flow_decoder_a2.pt'

    if not enc_ckpt.exists() or not flow_ckpt.exists():
        print("Missing checkpoints. Run training/a2/train_a2.py first."); return

    split_df = pd.read_csv(SPLIT_FILE)
    test_sids = set(split_df[split_df['split'] == 'test']['sid'])
    test_ds = UnifiedCycleDataset(CYCLES_DIR, test_sids)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    encoder = CameraEncoderFlow(latent_dim=LATENT_DIM, in_channels=1).to(device)
    encoder.load_state_dict(torch.load(enc_ckpt, map_location=device, weights_only=True))
    encoder.eval()

    flow = ConditionalFlowDecoder(latent_dim=LATENT_DIM, hidden_dim=64, n_blocks=6, n_steps=FLOW_STEPS).to(device)
    flow.load_state_dict(torch.load(flow_ckpt, map_location=device, weights_only=True))
    flow.eval()

    results_dir = Path(A2_RESULTS_DIR)
    results_dir.mkdir(parents=True, exist_ok=True)

    all_results = []
    print("\nEvaluating A2 (Flow Decoder)...")
    with torch.no_grad():
        for gt, g, rppg, sids in tqdm(test_loader):
            gt_np = gt.numpy()[:, 0, :]
            g_t = g.to(device)
            sids_np = sids.numpy()

            z = encoder(g_t)
            recon = flow.sample(z, n_steps=FLOW_STEPS).cpu().numpy()[:, 0, :]

            for i in range(len(gt_np)):
                sid = int(sids_np[i])
                ds = split_df[split_df['sid'] == sid]['dataset'].iloc[0]
                r = shape_r(recon[i], gt_np[i])
                gt_ipa = compute_ipa(gt_np[i])
                pred_ipa = compute_ipa(recon[i])
                gt_h2 = compute_h2_h1(gt_np[i])
                pred_h2 = compute_h2_h1(recon[i])
                gt_notch = notch_index(gt_np[i])
                pred_notch = notch_index(recon[i])

                all_results.append({
                    'architecture': 'A2', 'encoder': 'Flow-B', 'sid': sid, 'dataset': ds,
                    'shape_r': r,
                    'gt_ipa': gt_ipa, 'pred_ipa': pred_ipa, 'ipa_error': abs(gt_ipa - pred_ipa),
                    'gt_h2h1': gt_h2, 'pred_h2h1': pred_h2, 'h2h1_error': abs(gt_h2 - pred_h2),
                    'gt_notch': gt_notch, 'pred_notch': pred_notch,
                })

    if not all_results:
        print("No results."); return

    df = pd.DataFrame(all_results)
    df.to_csv(results_dir / 'full_eval_a2.csv', index=False)

    print("\n" + "=" * 60)
    print("A2 (Flow Decoder) EVALUATION RESULTS")
    print("=" * 60)
    subj = df.groupby('sid')['shape_r'].mean()
    print(f"  Cycle-level r: {df['shape_r'].mean():.4f}")
    print(f"  Per-subject r:  {subj.mean():.4f}")
    print(f"  IPA error:      {df['ipa_error'].mean():.4f}")
    print(f"  H2/H1 error:    {df['h2h1_error'].mean():.4f}")
    print(f"  By dataset:")
    for ds in df['dataset'].unique():
        ds_subj = df[df['dataset'] == ds].groupby('sid')['shape_r'].mean()
        print(f"    {ds:12s}: r={ds_subj.mean():.4f} ({ds_subj.shape[0]} subjects)")

    with open(results_dir / 'summary_a2.txt', 'w') as f:
        f.write(df.to_string(index=False))
    print(f"\nSaved to {results_dir / 'full_eval_a2.csv'}")


if __name__ == "__main__":
    main()

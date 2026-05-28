"""
evaluation/a1/evaluate_a1.py — A1 Evaluation (z=64)
====================================================
Evaluates A1 VAE+Encoder on test split. Mirrors evaluation/v6/evaluate.py.

Outputs:
  results/a1/full_eval_a1.csv
  results/a1/summary_a1.txt
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

from morph_config import CYCLES_DIR, SPLIT_FILE, BATCH_SIZE, A1_CKPT_DIR, A1_RESULTS_DIR
from models.vae_a1 import PPGVAEA1
from models.encoder_a1 import CameraEncoderA1
from models.metrics import batch_morpho_labels, compute_ipa, notch_index
from training.v5.train_encoders import UnifiedCycleDataset

LATENT_DIM = 64


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

    split_df = pd.read_csv(SPLIT_FILE)
    test_sids = set(split_df[split_df['split'] == 'test']['sid'])
    test_ds = UnifiedCycleDataset(CYCLES_DIR, test_sids)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    vae = PPGVAEA1(latent_dim=LATENT_DIM).to(device)
    vae.load_state_dict(torch.load(Path(A1_CKPT_DIR) / 'stage1_vae_a1.pt', map_location=device, weights_only=True))

    results_dir = Path(A1_RESULTS_DIR)
    results_dir.mkdir(parents=True, exist_ok=True)

    all_results = []

    for enc_name, in_ch in [('A', 1), ('B', 1), ('C', 2)]:
        ckpt_p = Path(A1_CKPT_DIR) / f'encoder_a1_{enc_name}.pt'
        if not ckpt_p.exists():
            print(f"  Skipping A1-{enc_name} — no checkpoint"); continue

        encoder = CameraEncoderA1(latent_dim=LATENT_DIM, in_channels=in_ch).to(device)
        state = torch.load(ckpt_p, map_location=device, weights_only=True)
        if 'encoder' in state:
            encoder.load_state_dict(state['encoder'])
            if 'decoder_finetune' in state:
                vae.decoder.load_state_dict(state['decoder_finetune'])
        else:
            encoder.load_state_dict(state)
        encoder.eval(); vae.eval()

        print(f"\nEvaluating A1-{enc_name}...")
        with torch.no_grad():
            for gt, g, rppg, sids in tqdm(test_loader, desc=f"A1-{enc_name}"):
                gt_np = gt.numpy()[:, 0, :]
                if enc_name == 'A':
                    x_in = g.to(device)
                elif enc_name == 'B':
                    x_in = rppg.to(device)
                else:
                    x_in = torch.cat([g, rppg], dim=1).to(device)
                sids_np = sids.numpy()

                z, _ = encoder.forward_morpho(x_in)
                recon = vae.decode(z).cpu().numpy()[:, 0, :]

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
                        'architecture': 'A1', 'encoder': enc_name, 'sid': sid, 'dataset': ds,
                        'shape_r': r,
                        'gt_ipa': gt_ipa, 'pred_ipa': pred_ipa, 'ipa_error': abs(gt_ipa - pred_ipa),
                        'gt_h2h1': gt_h2, 'pred_h2h1': pred_h2, 'h2h1_error': abs(gt_h2 - pred_h2),
                        'gt_notch': gt_notch, 'pred_notch': pred_notch,
                    })

    if not all_results:
        print("No results. Run training first."); return

    df = pd.DataFrame(all_results)
    df.to_csv(results_dir / 'full_eval_a1.csv', index=False)

    print("\n" + "=" * 60)
    print("A1 (z=64) EVALUATION RESULTS")
    print("=" * 60)

    for enc in df['encoder'].unique():
        sub = df[df['encoder'] == enc]
        print(f"\n--- A1-{enc} ---")
        print(f"  Cycle-level r: {sub['shape_r'].mean():.4f}")
        subj = sub.groupby('sid')['shape_r'].mean()
        print(f"  Per-subject r:  {subj.mean():.4f}")
        print(f"  IPA error:      {sub['ipa_error'].mean():.4f}")
        print(f"  H2/H1 error:    {sub['h2h1_error'].mean():.4f}")
        print(f"  By dataset:")
        for ds in sub['dataset'].unique():
            ds_sub = sub[sub['dataset'] == ds]
            ds_subj = ds_sub.groupby('sid')['shape_r'].mean()
            print(f"    {ds:12s}: r={ds_subj.mean():.4f} ({ds_subj.shape[0]} subjects)")

    with open(results_dir / 'summary_a1.txt', 'w') as f:
        f.write(df.to_string(index=False))
    print(f"\nSaved to {results_dir / 'full_eval_a1.csv'}")


if __name__ == "__main__":
    main()

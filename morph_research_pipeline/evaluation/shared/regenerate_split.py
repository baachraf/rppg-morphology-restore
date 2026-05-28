"""
regenerate_split.py — Generate correct subject-level split from v2/cycles data.
Scans all NPZ files, extracts unique SIDs, and creates an 80/10/10 split.
"""
import os, sys, glob, random
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm

PIPELINE = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PIPELINE))
from morph_config import CYCLES_DIR, RESULTS_DIR, CYCLE_SAMPLES

SEED = 42
TRAIN_FRAC = 0.70
VAL_FRAC = 0.15

from models.metrics import notch_index

def main():
    v2_dir = Path(CYCLES_DIR)
    datasets_dirs = ['stress2023', 'centan', 'fps2023', 'ubfc']

    sid_data = {}
    for ds in datasets_dirs:
        ds_dir = v2_dir / ds
        if not ds_dir.is_dir():
            print(f'  [{ds}] directory not found, skipping.')
            continue

        npz_files = sorted(ds_dir.glob('*_cycles.npz'))
        print(f'[{ds}] {len(npz_files)} files')

        for npz_f in tqdm(npz_files, desc=ds):
            data = np.load(npz_f)
            if 'sid' not in data:
                continue
            sid = int(data['sid'])
            gt = data['gt_cycles']

            if sid not in sid_data:
                sid_data[sid] = {'dataset': ds, 'cycles': 0, 'notch_count': 0}

            sid_data[sid]['cycles'] += len(gt)

            for i in range(len(gt)):
                idx = notch_index(gt[i])
                if idx >= 0:
                    sid_data[sid]['notch_count'] += 1

    records = []
    for sid, info in sid_data.items():
        records.append({
            'sid': sid,
            'dataset': info['dataset'],
            'cycles': info['cycles'],
            'notch_rate': info['notch_count'] / info['cycles'] if info['cycles'] > 0 else 0,
        })

    df = pd.DataFrame(records)
    print(f'\nTotal unique subjects found: {len(df)}')
    for ds in sorted(df['dataset'].unique()):
        sub = df[df['dataset'] == ds]
        print(f'  {ds}: {len(sub)} subjects, {sub["cycles"].sum()} cycles')

    # Deterministic stratified-ish split per dataset
    random.seed(SEED)
    split_rows = []
    for ds in sorted(df['dataset'].unique()):
        sub = df[df['dataset'] == ds].copy()
        sids = sorted(sub['sid'].tolist())
        random.shuffle(sids)
        n = len(sids)
        n_train = int(n * TRAIN_FRAC)
        n_val = int(n * VAL_FRAC)

        for i, sid in enumerate(sids):
            if i < n_train:
                split = 'train'
            elif i < n_train + n_val:
                split = 'val'
            else:
                split = 'test'
            split_rows.append({'sid': sid, 'dataset': ds, 'split': split})

    split_df = pd.DataFrame(split_rows)
    split_df = split_df.merge(df[['sid', 'cycles', 'notch_rate']], on='sid')

    print(f'\nSplit distribution:')
    print(split_df.groupby(['dataset', 'split']).size().to_string())

    out_path = Path(RESULTS_DIR) / 'subject_split_audited.csv'
    split_df.to_csv(out_path, index=False)
    print(f'\nSaved corrected split to: {out_path}')
    print(f'Total: {len(split_df)} subjects')

    for ds in sorted(split_df['dataset'].unique()):
        sub = split_df[split_df['dataset'] == ds]
        for s in ['train', 'val', 'test']:
            n = len(sub[sub['split'] == s])
            print(f'  {ds}/{s}: {n}')

if __name__ == '__main__':
    main()

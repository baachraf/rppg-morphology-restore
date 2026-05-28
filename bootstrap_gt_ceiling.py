"""
bootstrap_gt_ceiling.py — J6: Bootstrap 95% CI on GT cross-subject r ceiling.

Loads ground-truth contact PPG cycles for the 27 test subjects, computes the
point-estimate GT ceiling (cross-subject r), then bootstraps 1000 times
(resample subjects with replacement) to obtain a 95% CI.

Usage:
    python bootstrap_gt_ceiling.py
"""

import sys, random
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import pearsonr
from itertools import combinations

# --- paths ---
HERE = Path(__file__).parent
PIPELINE_ROOT = HERE / 'morph_research_pipeline'
sys.path.insert(0, str(PIPELINE_ROOT))

from config.paths import (
    SPLIT_FILE,
    UBFC_CYCLES_DIR, STRESS_CYCLES_DIR, FPS2023_CYCLES_DIR, CENTAN_CYCLES_DIR,
)

CYCLE_DIRS = [UBFC_CYCLES_DIR, STRESS_CYCLES_DIR, FPS2023_CYCLES_DIR, CENTAN_CYCLES_DIR]
N_BOOTSTRAP = 1000
SEED = 42


def _pearson(a, b):
    if np.std(a) < 1e-6 or np.std(b) < 1e-6:
        return float('nan')
    return float(pearsonr(a, b)[0])


def cross_subj_r(means: dict) -> float:
    """Mean pairwise Pearson r between subject-mean waveforms (Eq. 1 in paper)."""
    sids = sorted(means.keys())
    pairs = [(a, b) for a, b in combinations(sids, 2)]
    if not pairs:
        return float('nan')
    rs = [_pearson(means[a], means[b]) for a, b in pairs]
    rs = [r for r in rs if not np.isnan(r)]
    return float(np.mean(rs)) if rs else float('nan')


def load_gt_means(test_sids):
    """Load GT contact PPG cycles for test subjects; return {sid: mean_waveform}."""
    gt_means = {}
    for cycle_dir in CYCLE_DIRS:
        cycle_dir = Path(cycle_dir)
        if not cycle_dir.is_dir():
            continue
        for npz_path in sorted(cycle_dir.glob('*_cycles.npz')):
            if '_vmd' in npz_path.name or '_a12_' in npz_path.name:
                continue
            try:
                data = np.load(npz_path, allow_pickle=True)
                sid = int(data['sid'])
                if sid not in test_sids:
                    continue
                gt = data['gt_cycles']  # (N_cycles, 256)
                if len(gt) == 0:
                    continue
                if sid not in gt_means:
                    gt_means[sid] = []
                gt_means[sid].append(gt)
            except Exception as e:
                print(f'  Warning: could not load {npz_path.name}: {e}')

    # Average across all sessions for each subject
    result = {}
    for sid, arrays in gt_means.items():
        all_cycles = np.concatenate(arrays, axis=0)  # (total_cycles, 256)
        result[sid] = all_cycles.mean(axis=0)
    return result


def main():
    rng = np.random.default_rng(SEED)

    # --- Load split ---
    split_df = pd.read_csv(SPLIT_FILE)
    test_sids = set(split_df[split_df['split'] == 'test']['sid'].astype(int))
    print(f'Test subjects: {len(test_sids)}')

    # --- Load GT means ---
    print('Loading GT cycle files ...')
    gt_means = load_gt_means(test_sids)
    found = sorted(gt_means.keys())
    print(f'Found GT means for {len(found)} / {len(test_sids)} test subjects')

    missing = test_sids - set(found)
    if missing:
        print(f'  Missing sids: {sorted(missing)}')

    if len(found) < 2:
        print('ERROR: fewer than 2 test subjects found. Check cycle directories.')
        return

    # --- Point estimate ---
    point_est = cross_subj_r(gt_means)
    print(f'\nGT ceiling (point estimate, N={len(found)} subjects): {point_est:.4f}')
    print(f'  (Documented in paper: 0.601)')

    # --- Bootstrap ---
    print(f'\nRunning {N_BOOTSTRAP} bootstrap iterations (resample subjects) ...')
    sids = np.array(sorted(gt_means.keys()))
    boot_vals = []
    for _ in range(N_BOOTSTRAP):
        sampled = rng.choice(sids, size=len(sids), replace=True)
        boot_means = {i: gt_means[s] for i, s in enumerate(sampled)}
        boot_vals.append(cross_subj_r(boot_means))

    boot_vals = np.array(boot_vals)
    p025 = float(np.percentile(boot_vals, 2.5))
    p975 = float(np.percentile(boot_vals, 97.5))
    b_mean = float(np.mean(boot_vals))

    print(f'\n=== J6 Bootstrap Results ===')
    print(f'  Point estimate     : {point_est:.4f}')
    print(f'  Bootstrap mean     : {b_mean:.4f}')
    print(f'  95% CI (2.5–97.5%): [{p025:.4f}, {p975:.4f}]')
    print(f'  Width of CI        : {p975 - p025:.4f}')
    print(f'\nReport these numbers to the author for incorporation into the paper.')


if __name__ == '__main__':
    main()

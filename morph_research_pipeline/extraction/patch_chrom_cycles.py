"""
patch_chrom_cycles.py — Add rppg_chrom_cycles to existing cycle NPZ files
==========================================================================
The rPPG CSVs already contain rppg_CHROM. The cycle NPZs already have
peak_times (cycle start) and hr (BPM) for every accepted heartbeat, so
cycle end = peak_times[i] + 60/hr[i].

This script uses those existing values to cut CHROM from the CSV and patch
it directly into each NPZ. No re-extraction, no peak detection, no quality
gating, no clock sync needed.

Run once:
    python morph_research_pipeline/extraction/patch_chrom_cycles.py
Then verify:
    python morph_research_pipeline/evaluation/shared/verify_chrom.py
"""

import sys
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.interpolate import pchip_interpolate
from tqdm import tqdm

HERE = Path(__file__).parent
PIPELINE_ROOT = HERE.parent
sys.path.insert(0, str(PIPELINE_ROOT))

from morph_config import (
    UBFC_RPPG_DIR,       UBFC_CYCLES_V4_DIR,
    STRESS_RPPG_DIR,     STRESS_CYCLES_V4_DIR,
    FPS2023_RPPG_DIR,    FPS2023_CYCLES_V4_DIR,
    FPS2023_60_RPPG_DIR, FPS2023_60_CYCLES_V4_DIR,
    CENTAN_RPPG_DIR,     CENTAN_CYCLES_V4_DIR,
)

CYCLE_SAMPLES = 256

DATASETS = [
    (UBFC_RPPG_DIR,       UBFC_CYCLES_V4_DIR),
    (STRESS_RPPG_DIR,     STRESS_CYCLES_V4_DIR),
    (FPS2023_RPPG_DIR,    FPS2023_CYCLES_V4_DIR),
    (FPS2023_60_RPPG_DIR, FPS2023_60_CYCLES_V4_DIR),
    (CENTAN_RPPG_DIR,     CENTAN_CYCLES_V4_DIR),
]


def norm01(x):
    m, M = x.min(), x.max()
    return (x - m) / (M - m + 1e-8)


def pchip_resample(x, n_out):
    n_in = len(x)
    if n_in == n_out:
        return x.astype(np.float32)
    return pchip_interpolate(
        np.linspace(0, 1, n_in), x, np.linspace(0, 1, n_out)
    ).astype(np.float32)


def patch_one(npz_path: Path, rppg_dir: Path) -> str:
    if '_vmd' in npz_path.name or '_a12_' in npz_path.name:
        return 'skip_vmd'

    data = np.load(npz_path, allow_pickle=True)

    if 'rppg_chrom_cycles' in data:
        return 'already_done'

    if 'peak_times' not in data or 'hr' not in data:
        return 'skip_no_timing'

    peak_times = data['peak_times'].astype(np.float64)  # (N,) cycle start times
    hr         = data['hr'].astype(np.float64)           # (N,) BPM
    N = len(peak_times)
    if N == 0:
        return 'skip_empty'

    # Find corresponding rPPG CSV
    stem = npz_path.stem.replace('_cycles', '')
    rppg_path = Path(rppg_dir) / (stem + '_rppg_v2.csv')
    if not rppg_path.exists():
        return 'skip_no_rppg_csv'

    df = pd.read_csv(rppg_path)
    if 'rppg_CHROM' not in df.columns or 'time_sec' not in df.columns:
        return 'skip_no_chrom_col'

    v_times = df['time_sec'].values.astype(np.float64)
    chrom   = df['rppg_CHROM'].values.astype(np.float32)

    chrom_cycles = []
    chrom_valid  = []

    for i in range(N):
        t_s = peak_times[i]
        t_e = t_s + 60.0 / hr[i]          # end = start + cycle duration

        v_idx = np.where((v_times >= t_s) & (v_times <= t_e))[0]
        if len(v_idx) < 5:
            chrom_cycles.append(np.zeros(CYCLE_SAMPLES, dtype=np.float32))
            chrom_valid.append(False)
            continue

        seg = chrom[v_idx]
        if not np.isfinite(seg).all() or np.std(seg) < 1e-6:
            chrom_cycles.append(np.zeros(CYCLE_SAMPLES, dtype=np.float32))
            chrom_valid.append(False)
            continue

        cycle = norm01(pchip_resample(seg, CYCLE_SAMPLES))
        chrom_cycles.append(cycle)
        chrom_valid.append(True)

    chrom_arr = np.array(chrom_cycles, dtype=np.float32)  # (N, 256)
    valid_arr = np.array(chrom_valid,  dtype=bool)         # (N,)

    # Write all existing keys + new CHROM keys back
    save_dict = {k: data[k] for k in data.files}
    save_dict['rppg_chrom_cycles'] = chrom_arr
    save_dict['rppg_chrom_valid']  = valid_arr
    np.savez_compressed(npz_path, **save_dict)

    n_valid = int(valid_arr.sum())
    return f'ok:{n_valid}/{N}'


def main():
    totals = {'ok': 0, 'already': 0, 'skip': 0, 'fail': 0}

    for rppg_dir, cycles_dir in DATASETS:
        cycles_dir = Path(cycles_dir)
        if not cycles_dir.is_dir():
            continue
        npz_files = sorted(cycles_dir.glob('*_cycles.npz'))
        print(f'\n{cycles_dir.parent.name}: {len(npz_files)} files')

        for npz_path in tqdm(npz_files):
            try:
                result = patch_one(npz_path, rppg_dir)
                if result.startswith('ok'):
                    totals['ok'] += 1
                elif result == 'already_done':
                    totals['already'] += 1
                elif result.startswith('skip'):
                    totals['skip'] += 1
                    if result not in ('skip_vmd', 'skip_no_timing'):
                        print(f'  {npz_path.name}: {result}')
            except Exception as e:
                totals['fail'] += 1
                print(f'  FAIL {npz_path.name}: {e}')

    print(f'\nDone — patched={totals["ok"]}  already={totals["already"]}  '
          f'skipped={totals["skip"]}  failed={totals["fail"]}')
    print('Next: python morph_research_pipeline/evaluation/shared/verify_chrom.py')


if __name__ == '__main__':
    main()

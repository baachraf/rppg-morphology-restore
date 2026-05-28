"""
extraction/extract_rgb_windows_a7.py — A7: Physics-Informed RGB Window Extraction
==================================================================================
Key differences from A6:
  - Mean-center only (NO z-score) — preserves amplitude & inter-channel ratios
  - Channel ratios R/G, G/B, R/B as additional features
  - Native resolution (~60 samples for 2s at 30fps, pad/truncate to INPUT_LEN)
  - NO PCHIP resampling of RGB input
  - GT target still PCHIP-resampled to 256 (unchanged)

Output per subject: {stem}_a7_windows.npz
  - rgb_windows: (N, 6, INPUT_LEN) — R, G, B, R/G, G/B, R/B, mean-centered, native res
  - gt_targets:  (N, 256) — average GT PPG cycle template
"""

import os
import sys
import warnings
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from scipy.signal import find_peaks
from scipy.interpolate import pchip_interpolate

HERE = Path(__file__).parent
PIPELINE_ROOT = HERE.parent
sys.path.insert(0, str(PIPELINE_ROOT))

from morph_config import (
    UBFC_CSV_DIR, STRESS_CSV_DIR, FPS2023_CSV_DIR, CENTAN_CSV_DIR,
    UBFC_PPG_DIR, STRESS_PPG_DIR, FPS2023_PPG_DIR, CENTAN_PPG_DIR,
)
from config.paths import DATA_DIR

WINDOW_SEC = 2.0
STEP_SEC = 0.5
INPUT_LEN = 60
RESAMPLE_GT = 256
PATCH_NAMES = ['forehead', 'cheeks_top', 'cheeks_bot', 'nose_chin']
HR_MIN_BPM = 40
HR_MAX_BPM = 150
MIN_GT_SEGS = 5

DATASETS = [
    ('ubfc',       UBFC_CSV_DIR,    UBFC_PPG_DIR),
    ('stress2023', STRESS_CSV_DIR,  STRESS_PPG_DIR),
    ('fps2023',    FPS2023_CSV_DIR, FPS2023_PPG_DIR),
    ('centan',     CENTAN_CSV_DIR,  CENTAN_PPG_DIR),
]


def interp_nan(arr):
    arr = arr.copy().astype(np.float64)
    nans = np.isnan(arr)
    if not nans.any():
        return arr
    valid = ~nans
    if valid.sum() < 2:
        return np.nan_to_num(arr, nan=0.0)
    arr[nans] = np.interp(np.where(nans)[0], np.where(valid)[0], arr[valid])
    return arr


def unwrap_timestamps(ts):
    ts = ts.copy().astype(np.float64)
    jumps = np.diff(ts) < 0
    if not jumps.any():
        return ts
    cum = 0.0
    for i in range(1, len(ts)):
        if ts[i] < ts[i - 1]:
            cum = ts[i - 1]
        ts[i] += cum
    return ts


def pchip_resample(x, n_out):
    n_in = len(x)
    if n_in < 4:
        return np.interp(np.linspace(0, 1, n_out), np.linspace(0, 1, n_in), x)
    t_in = np.linspace(0, 1, n_in)
    t_out = np.linspace(0, 1, n_out)
    return pchip_interpolate(t_in, x, t_out)


def norm01(x):
    m, M = x.min(), x.max()
    return (x - m) / (M - m + 1e-8)


def mean_center(sig):
    out = sig - np.mean(sig)
    return out.astype(np.float32)


def pad_truncate(sig, length):
    if len(sig) >= length:
        return sig[:length]
    return np.pad(sig, (0, length - len(sig)), mode='edge')


def build_template(segs):
    arr = np.array(segs)
    return np.median(arr, axis=0)


def extract_sid(name_clean, ds_name):
    import re
    try:
        if ds_name == 'ubfc':
            if 'ubfc_phys' in name_clean or 'vid_s' in name_clean:
                m = re.search(r'_s(\d+)_', name_clean)
                raw = int(m.group(1)) if m else int(''.join(filter(str.isdigit, name_clean)) or 0)
                return 1000 + raw
            else:
                m = re.search(r'_s(\d+)', name_clean)
                return int(m.group(1)) if m else int(''.join(filter(str.isdigit, name_clean)) or 0)
        elif ds_name == 'stress2023':
            m = re.search(r'sub_?(\d+)', name_clean)
            return 2000 + (int(m.group(1)) if m else 0)
        elif ds_name == 'fps2023':
            if '0316' in name_clean: return 5001
            if '0331' in name_clean: return 5002
            raw_num = int(''.join(filter(str.isdigit, name_clean)) or 0) % 1000
            return 3000 + raw_num
        elif ds_name == 'centan':
            m = re.search(r's(\d+)', name_clean)
            sid = 4000 + (int(m.group(1)) if m else 0)
            if sid == 4004: return 5001
            if sid == 4011: return 5002
            return sid
    except Exception:
        pass
    return 999


def process_subject(csv_path, ppg_npz_path, out_dir, ds_name, sid):
    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        return f'FAIL: csv load {e}'

    if 'time_sec' not in df.columns:
        return 'FAIL: no time_sec'

    df = df[df['time_sec'] >= 0].reset_index(drop=True)
    cam_times = df['time_sec'].values.astype(np.float64)
    if len(cam_times) < 60:
        return 'FAIL: too short'

    patch_cols = {c: [] for c in 'RGB'}
    for ch in 'RGB':
        for pn in PATCH_NAMES:
            col = f'{ch}_patch_Raw_{pn}'
            if col not in df.columns:
                return f'FAIL: missing {col}'
            patch_cols[ch].append(interp_nan(df[col].values.astype(np.float32)))

    R_patches = np.stack(patch_cols['R'], axis=1)
    G_patches = np.stack(patch_cols['G'], axis=1)
    B_patches = np.stack(patch_cols['B'], axis=1)

    R_mean = R_patches.mean(axis=1)
    G_mean = G_patches.mean(axis=1)
    B_mean = B_patches.mean(axis=1)

    try:
        ppg_data = np.load(ppg_npz_path, allow_pickle=True)
        p_vals = interp_nan(ppg_data['ppg_values'])
        p_times = unwrap_timestamps(ppg_data['ppg_times'])
        p_hz = float(ppg_data['ppg_hz'])
    except Exception as e:
        return f'FAIL: ppg load {e}'

    min_dist = int((60.0 / HR_MAX_BPM) * p_hz)
    c_t0, c_t1 = cam_times[0], cam_times[-1]
    p_mask = (p_times >= c_t0) & (p_times <= c_t1)
    p_vals_act = p_vals[p_mask]
    p_times_act = p_times[p_mask]

    if len(p_vals_act) < p_hz * 10:
        return 'FAIL: no GT overlap'

    peaks, _ = find_peaks(p_vals_act, distance=min_dist, prominence=0.01)
    if len(peaks) < MIN_GT_SEGS + 1:
        return 'FAIL: too few GT peaks'

    min_cycle_sec = 60.0 / HR_MAX_BPM
    max_cycle_sec = 60.0 / HR_MIN_BPM

    gt_segs = []
    for i in range(len(peaks) - 1):
        dur = p_times_act[peaks[i + 1]] - p_times_act[peaks[i]]
        if min_cycle_sec <= dur <= max_cycle_sec:
            seg = pchip_resample(norm01(p_vals_act[peaks[i]:peaks[i + 1]]), RESAMPLE_GT)
            gt_segs.append(seg)

    if len(gt_segs) < MIN_GT_SEGS:
        return f'FAIL: only {len(gt_segs)} GT segs'

    gt_template = build_template(gt_segs).astype(np.float32)

    cam_duration = c_t1 - c_t0
    if cam_duration < WINDOW_SEC:
        return 'FAIL: shorter than window'

    n_windows = int((cam_duration - WINDOW_SEC) / STEP_SEC) + 1
    if n_windows < 1:
        return 'FAIL: no windows'

    rgb_windows = []
    gt_targets = []
    win_times = []

    for wi in range(n_windows):
        t_start = c_t0 + wi * STEP_SEC
        t_end = t_start + WINDOW_SEC
        idx = np.where((cam_times >= t_start) & (cam_times <= t_end))[0]
        if len(idx) < 10:
            continue

        r_seg = mean_center(R_mean[idx])
        g_seg = mean_center(G_mean[idx])
        b_seg = mean_center(B_mean[idx])

        eps = 1e-8
        rg_ratio = r_seg / (g_seg + eps)
        gb_ratio = g_seg / (b_seg + eps)
        rb_ratio = r_seg / (b_seg + eps)

        rg_ratio = rg_ratio - np.mean(rg_ratio)
        gb_ratio = gb_ratio - gb_ratio.mean()
        rb_ratio = rb_ratio - np.mean(rb_ratio)

        r_pad = pad_truncate(r_seg, INPUT_LEN)
        g_pad = pad_truncate(g_seg, INPUT_LEN)
        b_pad = pad_truncate(b_seg, INPUT_LEN)
        rg_pad = pad_truncate(rg_ratio.astype(np.float32), INPUT_LEN)
        gb_pad = pad_truncate(gb_ratio.astype(np.float32), INPUT_LEN)
        rb_pad = pad_truncate(rb_ratio.astype(np.float32), INPUT_LEN)

        window = np.stack([r_pad, g_pad, b_pad, rg_pad, gb_pad, rb_pad])

        p_in_window = (p_times_act >= t_start) & (p_times_act <= t_end)
        if p_in_window.sum() > p_hz * 0.5:
            rgb_windows.append(window)
            gt_targets.append(gt_template)
            win_times.append(t_start)

    if len(rgb_windows) < 1:
        return 'FAIL: no valid windows'

    out_path = out_dir / (csv_path.stem + '_a7_windows.npz')
    np.savez_compressed(
        out_path,
        rgb_windows=np.array(rgb_windows),
        gt_targets=np.array(gt_targets),
        win_times=np.array(win_times, dtype=np.float64),
        gt_template=gt_template,
        sid=sid,
        ppg_hz=p_hz,
        dataset=ds_name,
        window_sec=WINDOW_SEC,
        step_sec=STEP_SEC,
        input_len=INPUT_LEN,
        n_channels=6,
    )
    return f'OK: {len(rgb_windows)} windows'


def main():
    print(f'\nextract_rgb_windows_a7.py — A7 Physics-Informed RGB Window Extraction')
    print(f'  Window: {WINDOW_SEC}s | Step: {STEP_SEC}s | Input: {INPUT_LEN} samples (6ch)')
    print(f'  Preprocessing: mean-center only (no z-score, no resample)')
    print()

    for ds_name, csv_dir, ppg_dir in DATASETS:
        csv_dir = Path(csv_dir)
        ppg_dir = Path(ppg_dir)
        if not csv_dir.is_dir():
            print(f'  [{ds_name}] parsed dir not found, skipping.')
            continue

        out_dir = csv_dir.parent / 'a7_windows'
        out_dir.mkdir(parents=True, exist_ok=True)

        csv_files = sorted(csv_dir.glob('*.csv'))

        print(f'[{ds_name}] {len(csv_files)} subjects')
        ok, fail = 0, 0

        for csv_f in tqdm(csv_files, desc=ds_name):
            ppg_candidates = list(ppg_dir.glob(f'{csv_f.stem}_ppg*.npz'))
            if not ppg_candidates:
                fail += 1
                continue

            sid = extract_sid(csv_f.stem.lower(), ds_name)
            result = process_subject(csv_f, ppg_candidates[0], out_dir, ds_name, sid)
            if result.startswith('OK'):
                ok += 1
            else:
                fail += 1

        print(f'  {ds_name}: {ok} OK, {fail} FAIL')

    print('\nDone.')


if __name__ == '__main__':
    main()

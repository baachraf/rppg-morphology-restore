"""
extraction/extract_vmd_rppg.py — VMD-Peak-Detected Cardiac Cycles
==================================================================
New rPPG extraction method for A12.

Instead of using rPPG (CHROM/POS) peak times to define cycle boundaries,
this script runs VMD on the session-level raw G channel and detects peaks
in the resulting cardiac mode. Those VMD-native peaks define cycle boundaries
— producing phase-consistent cycles (diagnostic confirmed r=+0.374 vs
rPPG per-cycle r=-0.048).

Output per session: {stem}_a12_cycles.npz
  vmd_g_cycles  : (N, 256)  — VMD-G cardiac mode, VMD-peak-aligned, z-scored
  gt_cycles     : (N, 256)  — matched GT PPG cycles (from existing _cycles.npz)
  sid           : int
  peak_times    : (N,) float64 — VMD peak timestamps (camera seconds)

Matching: each VMD peak is matched to the nearest GT peak within ±0.35 s.
Cycles with no GT match are discarded.

Usage:
  python morph_research_pipeline/extraction/extract_vmd_rppg.py
  python morph_research_pipeline/extraction/extract_vmd_rppg.py --overwrite
"""

import sys
import json
import warnings
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.interpolate import interp1d
from scipy.interpolate import pchip_interpolate
from scipy.signal import butter, filtfilt, find_peaks
from tqdm import tqdm

HERE          = Path(__file__).parent
PIPELINE_ROOT = HERE.parent
sys.path.insert(0, str(PIPELINE_ROOT))

from config.paths import (
    UBFC_CSV_DIR,    UBFC_CYCLES_DIR,
    STRESS_CSV_DIR,  STRESS_CYCLES_DIR,
    FPS2023_CSV_DIR, FPS2023_CYCLES_DIR,
    FPS2023_60_CSV_DIR, FPS2023_60_CYCLES_DIR,
    CENTAN_CSV_DIR,  CENTAN_CYCLES_DIR,
)
from config.hyperparams import PATCH_NAMES

try:
    from vmdpy import VMD as _vmd_fn
    VMD_AVAILABLE = True
except ImportError:
    VMD_AVAILABLE = False
    print('[extract_vmd_rppg] WARNING: vmdpy not found — install with: pip install vmdpy')

FREQ_LO       = 0.5
FREQ_HI       = 8.0
CYCLE_SAMPLES = 256
TARGET_FPS    = 30.0
PEAK_MATCH_TOL = 0.35   # seconds — max GT-VMD peak distance to accept a match
VMD_PARAMS_FILE = PIPELINE_ROOT / 'config' / 'vmd_params.json'

DEFAULT_K     = 5
DEFAULT_ALPHA = 2000
DEFAULT_TAU   = 0.0

DATASET_PAIRS = [
    (UBFC_CYCLES_DIR,       UBFC_CSV_DIR),
    (STRESS_CYCLES_DIR,     STRESS_CSV_DIR),
    (FPS2023_CYCLES_DIR,    FPS2023_CSV_DIR),
    (FPS2023_60_CYCLES_DIR, FPS2023_60_CSV_DIR),
    (CENTAN_CYCLES_DIR,     CENTAN_CSV_DIR),
]


def load_vmd_params():
    if VMD_PARAMS_FILE.exists():
        try:
            p     = json.loads(VMD_PARAMS_FILE.read_text())
            K     = int(p.get('K', DEFAULT_K))
            alpha = float(p.get('alpha', DEFAULT_ALPHA))
            tau   = float(p.get('tau', DEFAULT_TAU))
            print(f'[VMD params] K={K}, alpha={alpha}, tau={tau}')
            return K, alpha, tau
        except Exception:
            pass
    print(f'[VMD params] Using defaults K={DEFAULT_K}, alpha={DEFAULT_ALPHA}')
    return DEFAULT_K, DEFAULT_ALPHA, DEFAULT_TAU


def _interp_nan(arr):
    out = arr.copy().astype(np.float64)
    nan = np.isnan(out)
    if nan.all():
        return np.zeros_like(out)
    if nan.any():
        idx = np.arange(len(out))
        out[nan] = np.interp(idx[nan], idx[~nan], out[~nan])
    return out


def resample_uniform(times, sig, fps):
    t0, t1 = times[0], times[-1]
    n_out   = max(2, int((t1 - t0) * fps))
    t_uni   = np.linspace(t0, t1, n_out)
    f       = interp1d(times, sig, kind='linear',
                       bounds_error=False, fill_value='extrapolate')
    return t_uni, f(t_uni).astype(np.float32)


def pchip_resample(x, n_out):
    n_in = len(x)
    if n_in == n_out:
        return x.astype(np.float32)
    return pchip_interpolate(np.linspace(0, 1, n_in), x,
                              np.linspace(0, 1, n_out)).astype(np.float32)


def bandpass(sig, fps, lo=FREQ_LO, hi=FREQ_HI, order=3):
    nyq  = fps / 2.0
    hi_c = min(hi, nyq * 0.95)
    if lo >= hi_c or len(sig) < 20:
        return sig.copy().astype(np.float32)
    b, a = butter(order, [lo / nyq, hi_c / nyq], btype='band')
    return filtfilt(b, a, sig).astype(np.float32)


def vmd_cardiac(signal_1d, fps, K, alpha, tau):
    """VMD → sum cardiac modes. Falls back to bandpass on failure."""
    if not VMD_AVAILABLE or len(signal_1d) < K * 10:
        return bandpass(signal_1d, fps)
    try:
        u, u_hat, omega = _vmd_fn(signal_1d.astype(np.float64),
                                   alpha, tau, K, DC=0, init=1, tol=1e-7)
        center_freqs  = omega[:, -1] * fps
        cardiac_mask  = (center_freqs >= FREQ_LO) & (center_freqs <= FREQ_HI)
        if not cardiac_mask.any():
            closest = np.argmin(np.abs(center_freqs - 1.2))
            cardiac_mask[closest] = True
        return u[cardiac_mask].sum(axis=0).astype(np.float32)
    except Exception:
        return bandpass(signal_1d, fps)


def detect_vmd_peaks(cardiac_sig, fps):
    """
    Detect cardiac cycle peaks in session-level VMD cardiac mode.
    Returns array of sample indices.
    """
    min_dist   = max(1, int(fps * 0.4))   # min 0.4 s between peaks (150 BPM max)
    prominence = cardiac_sig.std() * 0.3
    peaks, _   = find_peaks(cardiac_sig, distance=min_dist, prominence=prominence)
    return peaks


def process_session(cycles_npz, csv_path, K, alpha, tau, overwrite):
    stem    = cycles_npz.stem.replace('_cycles', '')
    out_npz = cycles_npz.parent / (stem + '_a12_cycles.npz')

    if not overwrite and out_npz.exists():
        return 'skip'

    # ── load existing GT cycles + GT peak times ───────────────────────────────
    try:
        cyc_data   = np.load(cycles_npz, allow_pickle=True)
        gt_all   = cyc_data['gt_cycles'].astype(np.float32)   # (N_gt, 256)
        sid      = int(cyc_data['sid'])
        if 'peak_times' not in cyc_data:
            return 'FAIL: peak_times missing (re-run extract_cycles.py for this session)'
        gt_peaks = cyc_data['peak_times'].astype(np.float64)   # (N_gt,) seconds
    except Exception as e:
        return f'FAIL_cycles: {e}'

    if len(gt_all) < 5:
        return 'FAIL: too few GT cycles'

    if not csv_path.exists():
        return f'FAIL: CSV not found: {csv_path}'

    # ── load raw RGB ──────────────────────────────────────────────────────────
    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        return f'FAIL_csv: {e}'

    if 'time_sec' not in df.columns:
        return 'FAIL: no time_sec column'

    df = df[df['time_sec'] >= 0].reset_index(drop=True)
    if len(df) < 30:
        return 'FAIL: too few frames'

    times_raw = df['time_sec'].values.astype(np.float64)

    try:
        G_patches = np.stack(
            [_interp_nan(df[f'G_patch_Raw_{p}'].values) for p in PATCH_NAMES], axis=1
        )
    except KeyError as e:
        return f'FAIL: missing patch column {e}'

    # ── resample G to uniform grid ────────────────────────────────────────────
    native_fps = 1.0 / (np.median(np.diff(times_raw)) + 1e-9)
    fps        = float(np.clip(min(native_fps, TARGET_FPS), 5.0, 120.0))

    G_mean         = G_patches.mean(axis=1).astype(np.float64)
    Gn             = G_mean / (G_mean.mean() + 1e-8)
    t_uni, Gn_uni  = resample_uniform(times_raw, Gn, fps)

    # ── VMD on session-level Gn ───────────────────────────────────────────────
    cardiac_sig = vmd_cardiac(Gn_uni, fps, K, alpha, tau)

    # ── detect peaks in VMD cardiac mode ─────────────────────────────────────
    vmd_peak_idx = detect_vmd_peaks(cardiac_sig, fps)
    if len(vmd_peak_idx) < 3:
        return 'FAIL: too few VMD peaks detected'

    vmd_peak_times = t_uni[vmd_peak_idx]

    # ── match VMD peaks to nearest GT peak ────────────────────────────────────
    matched_vmd_g  = []
    matched_gt     = []
    matched_ptimes = []

    for i, vt in enumerate(vmd_peak_times):
        diffs = np.abs(gt_peaks - vt)
        j     = int(np.argmin(diffs))
        if diffs[j] > PEAK_MATCH_TOL:
            continue   # no GT match within tolerance

        # define cycle window: vmd_peak[i] → vmd_peak[i+1]
        t_s = vmd_peak_times[i]
        t_e = vmd_peak_times[i + 1] if (i + 1) < len(vmd_peak_times) else t_s + 1.0

        v_idx = np.where((t_uni >= t_s) & (t_uni <= t_e))[0]
        if len(v_idx) < 5:
            continue

        seg = cardiac_sig[v_idx].astype(np.float32)
        std = seg.std()
        if std > 1e-8:
            seg = (seg - seg.mean()) / std
        else:
            continue   # flat segment — skip

        matched_vmd_g.append(pchip_resample(seg, CYCLE_SAMPLES))
        matched_gt.append(gt_all[j])
        matched_ptimes.append(vt)

    if len(matched_vmd_g) < 5:
        return f'FAIL: only {len(matched_vmd_g)} matched cycles'

    vmd_g_cycles = np.stack(matched_vmd_g, axis=0).astype(np.float32)  # (N, 256)
    gt_cycles    = np.stack(matched_gt,    axis=0).astype(np.float32)  # (N, 256)
    peak_times   = np.array(matched_ptimes, dtype=np.float64)

    try:
        np.savez_compressed(
            out_npz,
            sid          = sid,
            vmd_g_cycles = vmd_g_cycles,
            gt_cycles    = gt_cycles,
            peak_times   = peak_times,
        )
    except Exception as e:
        return f'FAIL_save: {e}'

    return f'ok:{len(vmd_g_cycles)}'


def main():
    K, alpha, tau = load_vmd_params()

    print('\nextract_vmd_rppg — VMD-Peak-Detected Cycle Extraction (A12)')
    print(f'  VMD: K={K}, alpha={alpha}, tau={tau}')
    print(f'  Peak match tolerance: ±{PEAK_MATCH_TOL} s')
    print(f'  Output: {{stem}}_a12_cycles.npz')

    counts = {'ok': 0, 'skip': 0, 'fail': 0}
    total_cycles = 0

    for cycles_dir, csv_dir in DATASET_PAIRS:
        cycles_dir = Path(cycles_dir)
        csv_dir    = Path(csv_dir)
        if not cycles_dir.is_dir():
            continue

        npz_files = [f for f in sorted(cycles_dir.glob('*_cycles.npz'))
                     if '_a12_cycles' not in f.name]
        if not npz_files:
            continue

        ds_name = cycles_dir.parent.name
        print(f'\n  {ds_name} — {len(npz_files)} sessions')

        for npz_f in tqdm(npz_files, desc=ds_name, leave=False):
            stem     = npz_f.stem.replace('_cycles', '')
            csv_path = csv_dir / (stem + '.csv')
            r = process_session(npz_f, csv_path, K, alpha, tau, overwrite=True)
            if r.startswith('ok'):
                counts['ok'] += 1
                n = int(r.split(':')[1]) if ':' in r else 0
                total_cycles += n
            elif r == 'skip':
                counts['skip'] += 1
            else:
                counts['fail'] += 1
                tqdm.write(f'  [{stem}] {r}')

    print(f'\nDone.  ok={counts["ok"]}  skip={counts["skip"]}  fail={counts["fail"]}')
    print(f'Total VMD-aligned cycles saved: {total_cycles}')
    print('\nNext:')
    print('  python morph_research_pipeline/training/a12/train_a12.py')


if __name__ == '__main__':
    main()

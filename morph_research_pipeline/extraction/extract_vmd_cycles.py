"""
extraction/extract_vmd_cycles.py — VMD-Based Cardiac Mode Extraction from Raw RGB
==================================================================================
Applies Variational Mode Decomposition (VMD) to raw RGB patch traces and extracts
per-cycle cardiac-band features in the 0.5–8 Hz range.

Why VMD instead of rPPG (CHROM/POS):
  - rPPG extraction (CHROM/POS) bandpass-filters to 0.5–8 Hz BUT the algorithm
    also applies cross-channel normalization and spectral weighting that destroys
    relative harmonic amplitudes (H2/H1 ratio). A10 confirmed: forward model
    PPG→rPPG achieves Pearson r=0.112 — rPPG carries almost no morphological info.
  - VMD decomposes raw RGB into adaptive frequency modes WITHOUT a fixed spectral
    basis. Cardiac modes selected by center-frequency in [0.5, 8] Hz preserve
    the full harmonic structure before any spectral destruction.
  - Upper limit 8 Hz (not the old v1 limit of 4 Hz) preserves H2 and partial H3.

Produces two feature sets saved per session as *_vmd.npz:
  vmd_3ch_cycles : (N, 3, 256)  — cardiac VMD modes from Rn, Gn, Bn
  vmd_6ch_cycles : (N, 6, 256)  — + CHROM intermediates Xs, Ys and POS_raw composite

VMD parameters: loaded from config/vmd_params.json (written by optuna_vmd_params.py).
Default if file absent: K=5, alpha=2000, tau=0.0

Output: saves {subject_stem}_vmd.npz alongside existing *_cycles.npz files.

Requires:  pip install vmdpy

Usage:
  python morph_research_pipeline/extraction/extract_vmd_cycles.py
  python morph_research_pipeline/extraction/extract_vmd_cycles.py --overwrite
"""

import os
import sys
import json
import argparse
import warnings
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.interpolate import interp1d, pchip_interpolate
from scipy.signal import butter, filtfilt
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
    print('[extract_vmd] WARNING: vmdpy not found. Install with: pip install vmdpy')
    print('[extract_vmd] Falling back to bandpass-only mode (no VMD decomposition).')

# ══════════════════════════════════════════════════════════════════════════════
# CONFIG
# ══════════════════════════════════════════════════════════════════════════════

FREQ_LO       = 0.5    # cardiac band low (Hz)
FREQ_HI       = 8.0    # cardiac band high — preserves H2/H3 at 60–180 BPM
CYCLE_SAMPLES = 256
TARGET_FPS    = 30.0
VMD_PARAMS_FILE = PIPELINE_ROOT / 'config' / 'vmd_params.json'

# Default VMD parameters (overridden by vmd_params.json if present)
DEFAULT_K     = 5
DEFAULT_ALPHA = 2000
DEFAULT_TAU   = 0.0

# Dataset directory pairs: (cycles_dir, parsed_csv_dir)
DATASET_PAIRS = [
    (UBFC_CYCLES_DIR,      UBFC_CSV_DIR),
    (STRESS_CYCLES_DIR,    STRESS_CSV_DIR),
    (FPS2023_CYCLES_DIR,   FPS2023_CSV_DIR),
    (FPS2023_60_CYCLES_DIR, FPS2023_60_CSV_DIR),
    (CENTAN_CYCLES_DIR,    CENTAN_CSV_DIR),
]

# ══════════════════════════════════════════════════════════════════════════════
# LOAD VMD PARAMS
# ══════════════════════════════════════════════════════════════════════════════

def load_vmd_params():
    if VMD_PARAMS_FILE.exists():
        try:
            p = json.loads(VMD_PARAMS_FILE.read_text())
            K     = int(p.get('K', DEFAULT_K))
            alpha = float(p.get('alpha', DEFAULT_ALPHA))
            tau   = float(p.get('tau', DEFAULT_TAU))
            print(f'[VMD params] Loaded from {VMD_PARAMS_FILE}: K={K}, alpha={alpha}, tau={tau}')
            return K, alpha, tau
        except Exception as e:
            print(f'[VMD params] Failed to load {VMD_PARAMS_FILE}: {e} — using defaults')
    else:
        print(f'[VMD params] {VMD_PARAMS_FILE} not found — using defaults K={DEFAULT_K}, alpha={DEFAULT_ALPHA}')
        print('[VMD params] Run optuna_vmd_params.py first for tuned parameters.')
    return DEFAULT_K, DEFAULT_ALPHA, DEFAULT_TAU

# ══════════════════════════════════════════════════════════════════════════════
# SIGNAL UTILITIES
# ══════════════════════════════════════════════════════════════════════════════

def _interp_nan(arr):
    out = arr.copy().astype(np.float64)
    nan = np.isnan(out)
    if nan.all():
        return np.zeros_like(out)
    if nan.any():
        idx = np.arange(len(out))
        out[nan] = np.interp(idx[nan], idx[~nan], out[~nan])
    return out


def resample_uniform(times, channels_dict, fps):
    t0, t1 = times[0], times[-1]
    n_out = max(2, int((t1 - t0) * fps))
    t_uni = np.linspace(t0, t1, n_out)
    out = {}
    for k, v in channels_dict.items():
        f = interp1d(times, v, kind='linear', bounds_error=False, fill_value='extrapolate')
        out[k] = f(t_uni).astype(np.float32)
    return t_uni, out


def bandpass_signal(sig, fps, lo=FREQ_LO, hi=FREQ_HI, order=3):
    nyq = fps / 2.0
    hi_c = min(hi, nyq * 0.95)
    if lo >= hi_c or len(sig) < 20:
        return sig.copy()
    b, a = butter(order, [lo / nyq, hi_c / nyq], btype='band')
    return filtfilt(b, a, sig).astype(np.float32)


def pchip_resample(x, n_out):
    n_in = len(x)
    if n_in == n_out:
        return x.astype(np.float32)
    return pchip_interpolate(np.linspace(0, 1, n_in), x,
                              np.linspace(0, 1, n_out)).astype(np.float32)

# ══════════════════════════════════════════════════════════════════════════════
# VMD CARDIAC EXTRACTION
# ══════════════════════════════════════════════════════════════════════════════

def vmd_cardiac_signal(signal_1d, fps, K, alpha, tau):
    """
    Apply VMD, select cardiac modes (center freq in [FREQ_LO, FREQ_HI] Hz),
    and return their sum. Falls back to bandpass if VMD fails or is unavailable.

    Returns (cardiac_signal, n_cardiac_modes).
    """
    if not VMD_AVAILABLE or len(signal_1d) < K * 10:
        return bandpass_signal(signal_1d, fps), 0

    try:
        u, u_hat, omega = _vmd_fn(signal_1d.astype(np.float64),
                                   alpha, tau, K, DC=0, init=1, tol=1e-7)
        # omega: (K, n_iters), last column = final center freqs in [0, 0.5] normalized
        center_freqs = omega[:, -1] * fps      # convert to Hz
        cardiac_mask = (center_freqs >= FREQ_LO) & (center_freqs <= FREQ_HI)

        if not cardiac_mask.any():
            # Fallback: pick mode closest to 1.2 Hz (typical resting HR)
            closest = np.argmin(np.abs(center_freqs - 1.2))
            cardiac_mask[closest] = True

        cardiac = u[cardiac_mask].sum(axis=0).astype(np.float32)
        return cardiac, int(cardiac_mask.sum())

    except Exception:
        return bandpass_signal(signal_1d, fps), 0


def build_6ch_signals(R_mean, G_mean, B_mean):
    """
    Compute 6 illumination-normalized channels from mean R, G, B.

    Returns dict with keys: 'Rn', 'Gn', 'Bn', 'Xs', 'Ys', 'POS_raw'
    """
    eps = 1e-8
    Rn  = R_mean / (R_mean.mean() + eps)
    Gn  = G_mean / (G_mean.mean() + eps)
    Bn  = B_mean / (B_mean.mean() + eps)

    Xs  = 3.0 * Rn - 2.0 * Gn
    Ys  = 1.5 * Rn + Gn - 1.5 * Bn

    # POS skin-orthogonal projection (Wang 2017), no bandpass
    P       = np.array([[0, 1, -1], [-2, 1, 1]], dtype=np.float64)
    n       = np.stack([Rn, Gn, Bn], axis=1)          # (T, 3)
    proj    = n @ P.T                                   # (T, 2)
    s1, s2  = proj[:, 0], proj[:, 1]
    alpha   = s1.std() / (s2.std() + eps)
    POS_raw = (s1 + alpha * s2).astype(np.float32)

    return {
        'Rn': Rn.astype(np.float32),
        'Gn': Gn.astype(np.float32),
        'Bn': Bn.astype(np.float32),
        'Xs': Xs.astype(np.float32),
        'Ys': Ys.astype(np.float32),
        'POS_raw': POS_raw,
    }

# ══════════════════════════════════════════════════════════════════════════════
# PER-SESSION EXTRACTION
# ══════════════════════════════════════════════════════════════════════════════

def process_session(cycles_npz, csv_path, K, alpha, tau, overwrite):
    """
    Extract VMD cardiac cycles for one session.

    Returns 'ok', 'skip', or error string.
    """
    stem    = cycles_npz.stem.replace('_cycles', '')
    out_npz = cycles_npz.parent / (stem + '_vmd.npz')

    if not overwrite and out_npz.exists():
        return 'skip'

    # ── load cycles NPZ for peak_times ────────────────────────────────────────
    try:
        cyc_data   = np.load(cycles_npz, allow_pickle=True)
        peak_times = cyc_data['peak_times'].astype(np.float64)  # (N,) camera seconds
        sid        = int(cyc_data['sid'])
        n_cycles   = len(cyc_data['gt_cycles'])
    except Exception as e:
        return f'FAIL_cycles_load: {e}'

    if len(peak_times) < n_cycles:
        return 'FAIL: peak_times shorter than gt_cycles'

    # ── load parsed CSV ───────────────────────────────────────────────────────
    if not csv_path.exists():
        return f'FAIL: CSV not found: {csv_path}'

    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        return f'FAIL_csv_load: {e}'

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
        R_patches = np.stack(
            [_interp_nan(df[f'R_patch_Raw_{p}'].values) for p in PATCH_NAMES], axis=1
        )
        B_patches = np.stack(
            [_interp_nan(df[f'B_patch_Raw_{p}'].values) for p in PATCH_NAMES], axis=1
        )
    except KeyError as e:
        return f'FAIL: missing patch column {e}'

    # ── resample to uniform grid ──────────────────────────────────────────────
    native_fps = 1.0 / (np.median(np.diff(times_raw)) + 1e-9)
    fps        = float(np.clip(min(native_fps, TARGET_FPS), 5.0, 120.0))

    R_mean = R_patches.mean(axis=1).astype(np.float64)
    G_mean = G_patches.mean(axis=1).astype(np.float64)
    B_mean = B_patches.mean(axis=1).astype(np.float64)

    all_ch = {'R': R_mean, 'G': G_mean, 'B': B_mean}
    t_uni, ch_uni = resample_uniform(times_raw, all_ch, fps)

    R_u = ch_uni['R'].astype(np.float64)
    G_u = ch_uni['G'].astype(np.float64)
    B_u = ch_uni['B'].astype(np.float64)

    # ── build 6-channel illumination-normalized signals ───────────────────────
    ch6 = build_6ch_signals(R_u, G_u, B_u)   # dict of (T,) arrays

    # ── apply VMD to each channel ─────────────────────────────────────────────
    ch_cardiac = {}
    ch_n_modes = {}
    for ch_name, sig in ch6.items():
        cardiac, n_modes = vmd_cardiac_signal(sig, fps, K, alpha, tau)
        ch_cardiac[ch_name] = cardiac
        ch_n_modes[ch_name] = n_modes

    # ── slice per-cycle using peak_times ─────────────────────────────────────
    vmd_3ch_list = []   # [Rn, Gn, Bn]
    vmd_6ch_list = []   # [Rn, Gn, Bn, Xs, Ys, POS_raw]

    for i in range(n_cycles):
        t_s = peak_times[i]
        t_e = peak_times[i + 1] if (i + 1) < len(peak_times) else t_s + 1.0

        v_idx = np.where((t_uni >= t_s) & (t_uni <= t_e))[0]
        if len(v_idx) < 5:
            # Not enough frames — use zero-padded fallback
            vmd_3ch_list.append(np.zeros((3, CYCLE_SAMPLES), dtype=np.float32))
            vmd_6ch_list.append(np.zeros((6, CYCLE_SAMPLES), dtype=np.float32))
            continue

        def _extract_and_resample(sig_full):
            seg = sig_full[v_idx].astype(np.float32)
            # z-score within cycle (mean-center, unit variance)
            std = seg.std()
            if std > 1e-8:
                seg = (seg - seg.mean()) / std
            return pchip_resample(seg, CYCLE_SAMPLES)

        Rn_cyc  = _extract_and_resample(ch_cardiac['Rn'])
        Gn_cyc  = _extract_and_resample(ch_cardiac['Gn'])
        Bn_cyc  = _extract_and_resample(ch_cardiac['Bn'])
        Xs_cyc  = _extract_and_resample(ch_cardiac['Xs'])
        Ys_cyc  = _extract_and_resample(ch_cardiac['Ys'])
        POS_cyc = _extract_and_resample(ch_cardiac['POS_raw'])

        vmd_3ch_list.append(np.stack([Rn_cyc, Gn_cyc, Bn_cyc], axis=0))  # (3, 256)
        vmd_6ch_list.append(np.stack([Rn_cyc, Gn_cyc, Bn_cyc,
                                       Xs_cyc, Ys_cyc, POS_cyc], axis=0))  # (6, 256)

    vmd_3ch = np.stack(vmd_3ch_list, axis=0).astype(np.float32)  # (N, 3, 256)
    vmd_6ch = np.stack(vmd_6ch_list, axis=0).astype(np.float32)  # (N, 6, 256)

    # ── save ──────────────────────────────────────────────────────────────────
    try:
        np.savez_compressed(
            out_npz,
            sid              = sid,
            vmd_3ch_cycles   = vmd_3ch,
            vmd_6ch_cycles   = vmd_6ch,
            vmd_K            = K,
            vmd_alpha        = alpha,
            vmd_tau          = tau,
            n_cardiac_modes  = np.array([ch_n_modes[c] for c in
                                          ['Rn', 'Gn', 'Bn', 'Xs', 'Ys', 'POS_raw']]),
        )
    except Exception as e:
        return f'FAIL_save: {e}'

    return 'ok'

# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--overwrite', action='store_true',
                        help='Overwrite existing _vmd.npz files')
    args = parser.parse_args()

    K, alpha, tau = load_vmd_params()

    print(f'\nextract_vmd_cycles — VMD Cardiac Mode Extraction')
    print(f'  VMD: K={K}, alpha={alpha}, tau={tau}')
    print(f'  Cardiac band: {FREQ_LO}–{FREQ_HI} Hz')
    print(f'  Overwrite: {args.overwrite}')
    if not VMD_AVAILABLE:
        print('  [!] vmdpy not installed — using bandpass fallback for all sessions')

    results = {'ok': 0, 'skip': 0, 'fail': 0}
    fail_log = []

    for cycles_dir, csv_dir in DATASET_PAIRS:
        cycles_dir = Path(cycles_dir)
        csv_dir    = Path(csv_dir)

        if not cycles_dir.is_dir():
            continue

        npz_files = sorted(cycles_dir.glob('*_cycles.npz'))
        if not npz_files:
            continue

        ds_name = cycles_dir.parent.name
        print(f'\n  Dataset: {ds_name} — {len(npz_files)} sessions')

        for npz_f in tqdm(npz_files, desc=ds_name, leave=False):
            stem     = npz_f.stem.replace('_cycles', '')
            csv_path = csv_dir / (stem + '.csv')

            r = process_session(npz_f, csv_path, K, alpha, tau, args.overwrite)
            if r == 'ok':
                results['ok'] += 1
            elif r == 'skip':
                results['skip'] += 1
            else:
                results['fail'] += 1
                fail_log.append(f'{npz_f.name}: {r}')

    print(f'\nDone — ok={results["ok"]}  skip={results["skip"]}  fail={results["fail"]}')
    if fail_log:
        print('\nFailed sessions:')
        for f in fail_log[:20]:
            print(f'  {f}')


if __name__ == '__main__':
    main()

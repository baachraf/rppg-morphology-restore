"""
morph_extract_rppg_v2.py — Frame-by-Frame Continuous rPPG Extraction
=====================================================================
Produces a continuous rPPG signal for every subject by sliding a
3-second window ONE FRAME AT A TIME over the RGB patch traces.

Key design decisions vs v1:
  1. Resample RGB to a uniform grid FIRST (irregular camera timing is fixed
     before any algorithm runs — v1 skipped this step).
  2. Stride = 1 frame. Each output sample uses a fresh 3-second window.
     No stitching gaps.
  3. Light bandpass 0.5–8 Hz only (preserves H2/H3, removes DC drift and
     above-Nyquist noise). No wavelet, no Savgol, no phase rectification.
  4. Optional conditional harmonic boost: applied ONLY when the dominant
     spectral peak SNR >= SNR_BOOST_THRESHOLD_DB. Avoids boosting noise.
  5. Four outputs per subject:
       rppg_POS        — colour-plane projection, global mean patches
       rppg_POS_harm   — POS with conditional harmonic boost (H2/H3 amplified)
       rppg_GRAW       — raw green channel mean, unprocessed (Encoder A input)
       rppg_PHybrid    — patch-PCA hybrid (your previous journal algorithm)
       rppg_CHROM      — CHROM chrominance algorithm (de Haan & Jeanne 2013)
  6. SQI (spectral SNR in dB) saved per frame for downstream quality gating.

Output: one CSV per subject in the dataset rppg directory.
Columns: time_sec | rppg_POS | rppg_POS_harm | rppg_GRAW | rppg_PHybrid | rppg_CHROM | sqi_POS

Post-processing (noisy segment gating, cycle cutting) is NOT done here.
That is a separate downstream step.
"""

import os
import sys
import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from scipy import signal as sp_signal
from scipy.interpolate import interp1d
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm

warnings.filterwarnings('ignore')

HERE         = Path(__file__).parent
PIPELINE_ROOT = HERE.parent
sys.path.insert(0, str(PIPELINE_ROOT))

from morph_config import (
    UBFC_CSV_DIR,    UBFC_RPPG_DIR,
    STRESS_CSV_DIR,  STRESS_RPPG_DIR,
    FPS2023_CSV_DIR, FPS2023_RPPG_DIR,
    FPS2023_60_CSV_DIR, FPS2023_60_RPPG_DIR,
    CENTAN_CSV_DIR,  CENTAN_RPPG_DIR,
    PATCH_NAMES, EXTRACT_WORKERS,
)

# ── try to import PHybrid from the shared DSP library ─────────────────────────
try:
    _tbme_shared = Path(__file__).resolve().parents[3] / 'PATCH_PCA_CodecStudy' / 'shared'
    sys.path.insert(0, str(_tbme_shared))
    from rppg_dsp import patch_pca_master_hybrid, DSPConfig
    PHYBRID_AVAILABLE = True
except ImportError:
    PHYBRID_AVAILABLE = False
    print('[rppg_v2] WARNING: rppg_dsp not found — PHybrid output will be NaN.')

# ══════════════════════════════════════════════════════════════════════════════
# CONFIG
# ══════════════════════════════════════════════════════════════════════════════

TARGET_FPS            = 30.0    # uniform grid target (clips to video native if lower)
WIN_SEC               = 3.0     # sliding window size in seconds
BP_LO_HZ              = 0.5     # bandpass low cut
BP_HI_HZ              = 8.0     # bandpass high cut (preserves H2/H3 at 60-180 BPM)
BP_ORDER              = 3       # Butterworth order (zero-phase via filtfilt)
SNR_BOOST_THRESHOLD_DB = 4.0    # min spectral SNR (dB) to apply harmonic boost
HARM_GAIN             = 1.5     # gain applied to H1/H2/H3 when boost is active
HARM_BANDWIDTH_FRAC   = 0.10    # ±10% around each harmonic
MIN_VIDEO_SEC         = WIN_SEC + 2.0
OVERWRITE             = False
PILOT_MODE            = False
PILOT_LIMIT           = 3

# ══════════════════════════════════════════════════════════════════════════════
# SIGNAL UTILITIES
# ══════════════════════════════════════════════════════════════════════════════

def _interp_nan(arr: np.ndarray) -> np.ndarray:
    out = arr.copy().astype(np.float64)
    nan = np.isnan(out)
    if nan.all():
        return np.zeros_like(out)
    if nan.any():
        idx = np.arange(len(out))
        out[nan] = np.interp(idx[nan], idx[~nan], out[~nan])
    return out


def resample_to_uniform(times: np.ndarray,
                        channels: dict,
                        target_fps: float) -> tuple:
    """
    Interpolate all channels to a uniform timestamp grid.
    Returns (uniform_times, uniform_channels_dict).
    """
    t0, t1 = times[0], times[-1]
    n_out   = max(2, int((t1 - t0) * target_fps))
    t_uni   = np.linspace(t0, t1, n_out)
    out     = {}
    for key, arr in channels.items():
        f     = interp1d(times, arr, kind='linear',
                         bounds_error=False, fill_value='extrapolate')
        out[key] = f(t_uni).astype(np.float32)
    return t_uni, out


def bandpass(sig: np.ndarray, fps: float,
             lo: float = BP_LO_HZ, hi: float = BP_HI_HZ,
             order: int = BP_ORDER) -> np.ndarray:
    nyq  = fps / 2.0
    hi_c = min(hi, nyq * 0.95)
    if lo >= hi_c:
        return sig
    b, a = sp_signal.butter(order, [lo / nyq, hi_c / nyq], btype='band')
    if len(sig) <= 3 * max(len(a), len(b)):
        return sig
    return sp_signal.filtfilt(b, a, sig).astype(np.float32)


def compute_sqi(seg: np.ndarray, fps: float) -> float:
    """
    Spectral SNR in dB: power at dominant HR peak vs surrounding noise floor.
    Returns 0.0 if signal is too short or flat.
    """
    if len(seg) < int(fps * 2):
        return 0.0
    freqs, psd = sp_signal.welch(seg, fs=fps,
                                  nperseg=min(len(seg), int(fps * 2)))
    hr_mask  = (freqs >= 0.7) & (freqs <= 3.5)
    if not hr_mask.any():
        return 0.0
    psd_hr   = psd[hr_mask]
    peak_pow = psd_hr.max()
    noise    = np.median(psd_hr)
    if noise < 1e-12:
        return 0.0
    return float(10.0 * np.log10(peak_pow / noise))


def harmonic_boost(sig: np.ndarray, fps: float,
                   gain: float = HARM_GAIN,
                   bw_frac: float = HARM_BANDWIDTH_FRAC) -> np.ndarray:
    """
    Boost H1/H2/H3 in the frequency domain.
    Only called when SNR is already verified by the caller.
    """
    n      = len(sig)
    S      = np.fft.rfft(sig)
    freqs  = np.fft.rfftfreq(n, d=1.0 / fps)
    hr_mask = (freqs >= 0.7) & (freqs <= 3.5)
    if not hr_mask.any():
        return sig
    psd    = np.abs(S) ** 2
    f_dom  = freqs[hr_mask][np.argmax(psd[hr_mask])]
    for h in range(1, 4):
        hf  = f_dom * h
        if hf > fps / 2:
            break
        lo  = hf * (1.0 - bw_frac)
        hi  = hf * (1.0 + bw_frac)
        mask = (freqs >= lo) & (freqs <= hi)
        S[mask] *= gain
    return np.fft.irfft(S, n=n).astype(np.float32)

# ══════════════════════════════════════════════════════════════════════════════
# rPPG ALGORITHMS
# ══════════════════════════════════════════════════════════════════════════════

def _pos(R: np.ndarray, G: np.ndarray, B: np.ndarray) -> np.ndarray:
    """
    POS algorithm (Wang 2017) on (T,) global-mean channels.
    Normalise → project onto skin-orthogonal plane → adaptive mix.
    """
    rgb  = np.stack([R, G, B], axis=1)          # (T, 3)
    mu   = rgb.mean(axis=0) + 1e-6
    n    = rgb / mu                              # (T, 3) normalised
    # Projection matrix from Wang 2017
    P    = np.array([[0, 1, -1], [-2, 1, 1]], dtype=np.float64)
    proj = n @ P.T                               # (T, 2)
    s1, s2 = proj[:, 0], proj[:, 1]
    alpha  = (s1.std() / (s2.std() + 1e-8))
    return (s1 + alpha * s2).astype(np.float32)


def _graw(G: np.ndarray) -> np.ndarray:
    """Raw green channel — no colour-space transformation."""
    return G.astype(np.float32)


def _phybrid(G_patches: np.ndarray,
             R_patches: np.ndarray,
             B_patches: np.ndarray,
             fps: float) -> np.ndarray:
    """
    Patch-PCA hybrid. Requires rppg_dsp.patch_pca_master_hybrid.
    G/R/B_patches are (T, N_patches).
    """
    if not PHYBRID_AVAILABLE:
        return np.full(len(G_patches), np.nan, dtype=np.float32)
    cfg = DSPConfig(target_fps=fps, win_sec=WIN_SEC)
    try:
        out = patch_pca_master_hybrid(G_patches, cfg, R_patches, B_patches)
        return out.astype(np.float32)
    except Exception:
        return np.full(len(G_patches), np.nan, dtype=np.float32)


def _chrom(R: np.ndarray, G: np.ndarray, B: np.ndarray) -> np.ndarray:
    """CHROM algorithm (de Haan & Jeanne, TBME 2013). Operates on global-mean channels."""
    mu_R = R.mean() + 1e-6
    mu_G = G.mean() + 1e-6
    mu_B = B.mean() + 1e-6
    Rn = R / mu_R
    Gn = G / mu_G
    Bn = B / mu_B
    Xs = 3.0 * Rn - 2.0 * Gn
    Ys = 1.5 * Rn + Gn - 1.5 * Bn
    alpha = np.std(Xs) / (np.std(Ys) + 1e-8)
    return (Xs - alpha * Ys).astype(np.float32)

# ══════════════════════════════════════════════════════════════════════════════
# SLIDING WINDOW EXTRACTION
# ══════════════════════════════════════════════════════════════════════════════

def sliding_extract(G_patches: np.ndarray,
                    R_patches: np.ndarray,
                    B_patches: np.ndarray,
                    fps:       float) -> dict:
    """
    Slide a WIN_SEC window one frame at a time.
    For each frame t >= win_len, extract one rPPG sample from window [t-win_len, t].

    Returns dict with keys: 'pos', 'graw', 'phybrid', 'chrom', 'sqi_pos'
    Each array has length T (NaN for the warm-up period).
    """
    T       = len(G_patches)
    win_len = int(WIN_SEC * fps)

    if T < win_len + 1:
        nan = np.full(T, np.nan, dtype=np.float32)
        return {'pos': nan, 'graw': nan, 'phybrid': nan, 'chrom': nan, 'sqi_pos': nan}

    # Global-mean channels for POS and G-raw
    R_mean = R_patches.mean(axis=1)   # (T,)
    G_mean = G_patches.mean(axis=1)
    B_mean = B_patches.mean(axis=1)

    pos_raw   = np.full(T, np.nan, dtype=np.float32)
    graw_raw  = np.full(T, np.nan, dtype=np.float32)
    phy_raw   = np.full(T, np.nan, dtype=np.float32)
    chrom_raw = np.full(T, np.nan, dtype=np.float32)
    sqi_arr   = np.full(T, np.nan, dtype=np.float32)

    for t in range(win_len, T):
        sl = slice(t - win_len, t)

        # POS — take last sample of window output
        pos_seg    = _pos(R_mean[sl], G_mean[sl], B_mean[sl])
        pos_raw[t] = pos_seg[-1]

        # G-raw — just the last sample
        graw_raw[t] = G_mean[t]

        # PHybrid — last sample
        if PHYBRID_AVAILABLE:
            phy_seg    = _phybrid(G_patches[sl], R_patches[sl], B_patches[sl], fps)
            phy_raw[t] = phy_seg[-1]

        # CHROM — last sample
        chrom_seg     = _chrom(R_mean[sl], G_mean[sl], B_mean[sl])
        chrom_raw[t]  = chrom_seg[-1]

        # SQI on current POS window (fast — uses welch on win_len samples)
        sqi_arr[t] = compute_sqi(pos_seg, fps)

    return {
        'pos':     pos_raw,
        'graw':    graw_raw,
        'phybrid': phy_raw,
        'chrom':   chrom_raw,
        'sqi_pos': sqi_arr,
    }

# ══════════════════════════════════════════════════════════════════════════════
# POST-PROCESSING (LIGHT)
# ══════════════════════════════════════════════════════════════════════════════

def postprocess(sig: np.ndarray, fps: float,
                apply_harm: bool = True) -> tuple:
    """
    1. Fill NaN warm-up with zeros (they will be masked by SQI later).
    2. Bandpass 0.5–8 Hz.
    3. Conditional harmonic boost based on global signal SNR.

    Returns (sig_bp, sig_harm) — bandpassed only and boosted variant.
    """
    filled = sig.copy()
    nan_mask = np.isnan(filled)
    filled[nan_mask] = 0.0

    sig_bp   = bandpass(filled, fps)
    sig_bp[nan_mask] = np.nan          # restore NaN for warm-up frames

    sig_harm = sig_bp.copy()
    if apply_harm:
        valid = ~nan_mask
        if valid.sum() > int(fps * 5):
            snr = compute_sqi(sig_bp[valid], fps)
            if snr >= SNR_BOOST_THRESHOLD_DB:
                boosted              = harmonic_boost(sig_bp[valid], fps)
                sig_harm[valid]      = boosted
            # else: sig_harm stays equal to sig_bp (no boost)

    return sig_bp, sig_harm

# ══════════════════════════════════════════════════════════════════════════════
# PER-SUBJECT PROCESSING
# ══════════════════════════════════════════════════════════════════════════════

def process_csv(csv_path: Path, output_dir: Path, overwrite: bool) -> str:
    out_path = output_dir / (csv_path.stem + '_rppg_v2.csv')
    if not overwrite and out_path.exists():
        return 'skip'

    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        return f'FAIL_LOAD: {e}'

    # ── check minimum length ──────────────────────────────────────────────────
    if 'time_sec' not in df.columns:
        return 'FAIL: no time_sec column'

    # FPS2023 CSVs contain negative-timestamp rows where the PPG sensor was
    # running before the camera started. Drop those — they have no RGB data.
    df = df[df['time_sec'] >= 0].reset_index(drop=True)

    times_raw = df['time_sec'].values.astype(np.float64)
    if len(times_raw) < 2:
        return 'FAIL: no valid frames after t>=0 filter'
    dur = times_raw[-1] - times_raw[0]
    if dur < MIN_VIDEO_SEC:
        return f'FAIL: too short ({dur:.1f}s)'

    # ── load & interpolate NaN in raw patches ─────────────────────────────────
    try:
        G_patches = np.stack(
            [_interp_nan(df[f'G_patch_Raw_{p}'].values) for p in PATCH_NAMES], axis=1
        ).astype(np.float32)
        R_patches = np.stack(
            [_interp_nan(df[f'R_patch_Raw_{p}'].values) for p in PATCH_NAMES], axis=1
        ).astype(np.float32)
        B_patches = np.stack(
            [_interp_nan(df[f'B_patch_Raw_{p}'].values) for p in PATCH_NAMES], axis=1
        ).astype(np.float32)
    except KeyError as e:
        return f'FAIL: missing patch column {e}'

    # ── step 1: resample to uniform grid ─────────────────────────────────────
    native_fps = 1.0 / (np.median(np.diff(times_raw)) + 1e-9)
    native_fps = float(np.clip(native_fps, 5.0, 120.0))
    fps        = min(native_fps, TARGET_FPS)

    all_channels = {}
    for pi, pname in enumerate(PATCH_NAMES):
        all_channels[f'G_{pname}'] = G_patches[:, pi]
        all_channels[f'R_{pname}'] = R_patches[:, pi]
        all_channels[f'B_{pname}'] = B_patches[:, pi]

    t_uni, ch_uni = resample_to_uniform(times_raw, all_channels, fps)

    G_u = np.stack([ch_uni[f'G_{p}'] for p in PATCH_NAMES], axis=1)
    R_u = np.stack([ch_uni[f'R_{p}'] for p in PATCH_NAMES], axis=1)
    B_u = np.stack([ch_uni[f'B_{p}'] for p in PATCH_NAMES], axis=1)

    # ── step 2: sliding-window extraction ─────────────────────────────────────
    raw = sliding_extract(G_u, R_u, B_u, fps)

    # ── step 3: bandpass + conditional harmonic boost ─────────────────────────
    pos_bp,   pos_harm  = postprocess(raw['pos'],     fps, apply_harm=True)
    graw_bp,  _         = postprocess(raw['graw'],    fps, apply_harm=False)
    phy_bp,   _         = postprocess(raw['phybrid'], fps, apply_harm=False)
    chrom_bp, _         = postprocess(raw['chrom'],   fps, apply_harm=False)

    # ── step 4: save ──────────────────────────────────────────────────────────
    out_df = pd.DataFrame({
        'time_sec':      t_uni,
        'rppg_POS':      pos_bp,       # bandpass only
        'rppg_POS_harm': pos_harm,     # bandpass + conditional harmonic boost
        'rppg_GRAW':     graw_bp,      # raw G, bandpass only
        'rppg_PHybrid':  phy_bp,       # patch-PCA hybrid, bandpass only
        'rppg_CHROM':    chrom_bp,     # CHROM (de Haan & Jeanne 2013), bandpass only
        'sqi_POS':       raw['sqi_pos'],
    })
    output_dir.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out_path, index=False)
    return 'ok'

# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    dir_map = {
        Path(UBFC_CSV_DIR):    Path(UBFC_RPPG_DIR),
        Path(STRESS_CSV_DIR):  Path(STRESS_RPPG_DIR),
        Path(FPS2023_CSV_DIR): Path(FPS2023_RPPG_DIR),
        Path(FPS2023_60_CSV_DIR): Path(FPS2023_60_RPPG_DIR),
        Path(CENTAN_CSV_DIR):  Path(CENTAN_RPPG_DIR),
    }

    jobs = []
    for in_dir, out_dir in dir_map.items():
        if not in_dir.is_dir():
            continue
        files = sorted(f for f in in_dir.glob('*.csv')
                       if not f.name.endswith('_rppg.csv')
                       and not f.name.endswith('_rppg_v2.csv'))
        if PILOT_MODE:
            files = files[:PILOT_LIMIT]
        for f in files:
            jobs.append((f, out_dir))

    print(f'\nmorph_extract_rppg_v2 — Frame-by-Frame Continuous Extraction')
    print(f'  Window: {WIN_SEC}s | Stride: 1 frame | Bandpass: {BP_LO_HZ}–{BP_HI_HZ} Hz')
    print(f'  Harmonic boost threshold: {SNR_BOOST_THRESHOLD_DB} dB')
    print(f'  PHybrid available: {PHYBRID_AVAILABLE}')
    print(f'  Jobs: {len(jobs)}\n')

    results = {'ok': 0, 'skip': 0, 'fail': 0}

    with ProcessPoolExecutor(max_workers=EXTRACT_WORKERS) as ex:
        futs = {
            ex.submit(process_csv, f, od, OVERWRITE): f
            for f, od in jobs
        }
        for fut in tqdm(as_completed(futs), total=len(futs)):
            r = fut.result()
            if r == 'ok':
                results['ok'] += 1
            elif r == 'skip':
                results['skip'] += 1
            else:
                results['fail'] += 1
                print(f'  {futs[fut].name}: {r}')

    print(f'\nDone — ok={results["ok"]}  skip={results["skip"]}  fail={results["fail"]}')


if __name__ == '__main__':
    main()

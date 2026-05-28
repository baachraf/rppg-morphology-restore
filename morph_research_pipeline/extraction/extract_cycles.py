"""
extract_cycles.py — Per-Algorithm Gated Cycle Extraction
=========================================================
Gate 2 is now INDEPENDENT per algorithm (POS, POS-harm, PHybrid, CHROM).
A heartbeat is kept if ANY algorithm passes its quality gate.
All four algorithm cycles are saved for each kept heartbeat, together
with per-algorithm validity flags so A5/A13 can do best-of-four selection
without index misalignment.

New keys added to output npz:
  rppg_pos_valid      : (N,) bool — POS passed its own quality gate
  rppg_harm_valid     : (N,) bool — POS-harm passed its own quality gate
  rppg_phybrid_valid  : (N,) bool — PHybrid passed its own quality gate
  rppg_chrom_valid    : (N,) bool — CHROM passed its own quality gate
  peak_times          : (N,) float64 — GT peak timestamp (seconds) for each cycle

All arrays are index-aligned: row i of every array is the same heartbeat.

Backward compatibility: rppg_pos_cycles, rppg_harm_cycles, g_cycles,
  rppg_phybrid_cycles, gt_cycles, sqi, hr keys are unchanged.
  V5/V6 training code continues to work without modification.

FORCE_REEXTRACT = True  → overwrite existing files (needed for first run)
"""

import os
import sys
import re
import warnings
warnings.filterwarnings('ignore', message='invalid value encountered in divide',
                        category=RuntimeWarning)
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from scipy.signal import find_peaks
from scipy.interpolate import pchip_interpolate

HERE          = Path(__file__).parent
PIPELINE_ROOT = HERE.parent
sys.path.insert(0, str(PIPELINE_ROOT))

from morph_config import (
    UBFC_CSV_DIR,    UBFC_RPPG_DIR,    UBFC_PPG_DIR,    UBFC_CYCLES_V4_DIR,
    STRESS_CSV_DIR,  STRESS_RPPG_DIR,  STRESS_PPG_DIR,  STRESS_CYCLES_V4_DIR,
    FPS2023_CSV_DIR, FPS2023_RPPG_DIR, FPS2023_PPG_DIR, FPS2023_CYCLES_V4_DIR,
    FPS2023_60_CSV_DIR, FPS2023_60_RPPG_DIR, FPS2023_60_PPG_DIR, FPS2023_60_CYCLES_V4_DIR,
    CENTAN_CSV_DIR,  CENTAN_RPPG_DIR,  CENTAN_PPG_DIR,  CENTAN_CYCLES_V4_DIR,
    RESULTS_DIR,
)

# ══════════════════════════════════════════════════════════════════════════════
# CONFIG
# ══════════════════════════════════════════════════════════════════════════════

CYCLE_SAMPLES = 256
MIN_CYCLE_SEC = 0.45    # ~133 BPM max
MAX_CYCLE_SEC = 1.20    # ~50 BPM min

# Gate 1 — GT template correlation (strict: contact PPG is clean)
GT_THRESHOLDS = {
    'ubfc':      0.70,
    'stress2023':0.85,
    'fps2023':   0.85,
    'centan':    0.90,
}

# Gate 2 — rPPG template correlation (lenient: camera signal is noisy)
# Lowered from 0.40-0.50: V2 rPPG wide bandpass introduces noise that inflates
# template noise floor, causing spurious rejections at higher thresholds.
RPPG_THRESHOLDS = {
    'ubfc':      0.20,
    'stress2023':0.25,
    'fps2023':   0.25,
    'centan':    0.25,
}

# Gate 3 — minimum mean SQI (dB) in the cycle window
SQI_THRESHOLD = 3.0

PILOT_MODE      = False
PILOT_LIMIT     = 5
FORCE_REEXTRACT = False

AUDIT_LOG = []

# ══════════════════════════════════════════════════════════════════════════════
# UTILITIES
# ══════════════════════════════════════════════════════════════════════════════

def norm01(x):
    m, M = x.min(), x.max()
    return (x - m) / (M - m + 1e-8)

def pchip_resample(x, n_out):
    n_in = len(x)
    if n_in == n_out:
        return x
    return pchip_interpolate(np.linspace(0, 1, n_in), x,
                             np.linspace(0, 1, n_out))

def interp_nan(x):
    s = pd.Series(x)
    return s.interpolate(method='linear').bfill().ffill().values

def unwrap_timestamps(t):
    dt = np.diff(t, prepend=t[0])
    dt[dt < 0] = 0.001
    return np.cumsum(dt)

def build_template(cycles: list) -> np.ndarray:
    """Mean of top-75th-percentile cycles by correlation to rough mean."""
    if not cycles:
        return None
    arr  = np.array(cycles)
    mean = arr.mean(axis=0)
    corrs = np.array([np.corrcoef(c, mean)[0, 1] for c in arr])
    top   = arr[corrs >= np.percentile(corrs, 75)]
    return top.mean(axis=0) if len(top) > 0 else mean

# ══════════════════════════════════════════════════════════════════════════════
# PER-SUBJECT PROCESSING
# ══════════════════════════════════════════════════════════════════════════════

def process_subject(csv_path, rppg_v2_path, ppg_npz_path,
                    output_path, sid, ds_name):

    stats = {
        'subject':       csv_path.stem,
        'dataset':       ds_name,
        'sid':           sid,
        'candidates':    0,
        'fail_bpm':      0,
        'fail_gt':       0,
        'fail_rppg':     0,   # all three algorithms failed
        'fail_sqi':      0,
        'fail_sync':     0,
        'ok':            0,
        'extra_vs_pos':  0,   # cycles saved only because harm or phybrid passed (POS failed)
        'status':        '',
    }

    # ── load files ────────────────────────────────────────────────────────────
    try:
        ppg_data   = np.load(ppg_npz_path, allow_pickle=True)
        df_rppg_v2 = pd.read_csv(rppg_v2_path)
    except Exception as e:
        stats['status'] = f'Load error: {e}'
        AUDIT_LOG.append(stats)
        return f'FAIL: {e}'

    # ── GT PPG ────────────────────────────────────────────────────────────────
    p_vals  = interp_nan(ppg_data['ppg_values'])
    p_times = unwrap_timestamps(ppg_data['ppg_times'])
    p_hz    = float(ppg_data['ppg_hz'])

    # auto-orient GT PPG (systolic peak should be positive)
    min_dist = int(MIN_CYCLE_SEC * p_hz)
    def _orient_score(sig):
        peaks, _ = find_peaks(sig, distance=min_dist, prominence=0.01)
        if len(peaks) < 5:
            return 0.0
        segs = [pchip_resample(norm01(sig[peaks[i]:peaks[i+1]]), 128)
                for i in range(min(20, len(peaks) - 1))]
        return np.argmax(np.mean(segs, axis=0))

    if _orient_score(p_vals) < 64 and _orient_score(-p_vals) > 64:
        p_vals = -p_vals

    # ── rPPG v2 signal ────────────────────────────────────────────────────────
    if 'time_sec' not in df_rppg_v2.columns or 'rppg_POS' not in df_rppg_v2.columns:
        stats['status'] = 'rppg_v2 missing columns'
        AUDIT_LOG.append(stats)
        return 'FAIL: rppg_v2 missing columns'

    v_times   = df_rppg_v2['time_sec'].values.astype(np.float64)
    rppg_pos  = interp_nan(df_rppg_v2['rppg_POS'].values.astype(np.float32))
    rppg_harm = interp_nan(df_rppg_v2['rppg_POS_harm'].values.astype(np.float32)) \
                if 'rppg_POS_harm' in df_rppg_v2.columns else rppg_pos.copy()
    rppg_graw  = interp_nan(df_rppg_v2['rppg_GRAW'].values.astype(np.float32)) \
                 if 'rppg_GRAW' in df_rppg_v2.columns else np.full_like(rppg_pos, np.nan)
    rppg_phy   = interp_nan(df_rppg_v2['rppg_PHybrid'].values.astype(np.float32)) \
                 if 'rppg_PHybrid' in df_rppg_v2.columns else np.full_like(rppg_pos, np.nan)
    rppg_chrom = interp_nan(df_rppg_v2['rppg_CHROM'].values.astype(np.float32)) \
                 if 'rppg_CHROM' in df_rppg_v2.columns else np.full_like(rppg_pos, np.nan)
    sqi_pos   = df_rppg_v2['sqi_POS'].values.astype(np.float32) \
                if 'sqi_POS' in df_rppg_v2.columns else np.full_like(rppg_pos, 10.0)

    # ── align GT and camera timelines ─────────────────────────────────────────
    v_t0, v_t1 = v_times[0], v_times[-1]
    p_mask     = (p_times >= v_t0) & (p_times <= v_t1)
    p_vals_act = p_vals[p_mask]
    p_times_act = p_times[p_mask]

    if len(p_vals_act) < p_hz * 10:
        stats['status'] = 'No overlap'
        AUDIT_LOG.append(stats)
        return 'FAIL: No overlap'

    # ── GT peak detection ─────────────────────────────────────────────────────
    peaks, _ = find_peaks(p_vals_act, distance=min_dist, prominence=0.01)
    stats['candidates'] = max(0, len(peaks) - 1)

    # ── build GT template ─────────────────────────────────────────────────────
    valid_gt_segs = []
    for i in range(len(peaks) - 1):
        dur = p_times_act[peaks[i+1]] - p_times_act[peaks[i]]
        if MIN_CYCLE_SEC <= dur <= MAX_CYCLE_SEC:
            seg = pchip_resample(
                norm01(p_vals_act[peaks[i]:peaks[i+1]]), CYCLE_SAMPLES)
            valid_gt_segs.append(seg)

    if len(valid_gt_segs) < 5:
        stats['status'] = 'Too few valid GT segs'
        AUDIT_LOG.append(stats)
        return 'FAIL: Too few valid GT segs'

    gt_template   = build_template(valid_gt_segs)
    gt_threshold  = GT_THRESHOLDS.get(ds_name, 0.75)
    rppg_threshold = RPPG_THRESHOLDS.get(ds_name, 0.45)

    # ── build per-algorithm rPPG templates (rough pass, no quality gate) ─────
    rough_pos, rough_harm, rough_phy, rough_chrom = [], [], [], []
    for i in range(len(peaks) - 1):
        t_s = p_times_act[peaks[i]]
        t_e = p_times_act[peaks[i + 1]]
        if not (MIN_CYCLE_SEC <= (t_e - t_s) <= MAX_CYCLE_SEC):
            continue
        v_idx = np.where((v_times >= t_s) & (v_times <= t_e))[0]
        if len(v_idx) < 5:
            continue
        raw_pos   = rppg_pos[v_idx]
        raw_harm  = rppg_harm[v_idx]
        raw_phy   = rppg_phy[v_idx]
        raw_chrom = rppg_chrom[v_idx]
        if np.isfinite(raw_pos).all():
            rough_pos.append(norm01(pchip_resample(raw_pos,   CYCLE_SAMPLES)))
        if np.isfinite(raw_harm).all():
            rough_harm.append(norm01(pchip_resample(raw_harm,  CYCLE_SAMPLES)))
        if np.isfinite(raw_phy).all():
            rough_phy.append(norm01(pchip_resample(raw_phy,   CYCLE_SAMPLES)))
        if np.isfinite(raw_chrom).all():
            rough_chrom.append(norm01(pchip_resample(raw_chrom, CYCLE_SAMPLES)))

    if len(rough_pos) < 5:
        stats['status'] = 'Too few rPPG segs for template'
        AUDIT_LOG.append(stats)
        return 'FAIL: Too few rPPG segs for template'

    # one template per algorithm — each builds from its own top-25% cycles
    tmpl_pos   = build_template(rough_pos)
    tmpl_harm  = build_template(rough_harm)
    tmpl_phy   = build_template(rough_phy)
    tmpl_chrom = build_template(rough_chrom)

    # ── per-algorithm gated extraction ───────────────────────────────────────
    gt_cycles         = []
    rppg_pos_out      = []
    rppg_harm_out     = []
    rppg_graw_out     = []
    rppg_phy_out      = []
    rppg_chrom_out    = []
    pos_valid_out     = []
    harm_valid_out    = []
    phy_valid_out     = []
    chrom_valid_out   = []
    peak_times_out    = []
    sqi_out           = []
    hr_list           = []

    for i in range(len(peaks) - 1):
        t_s = p_times_act[peaks[i]]
        t_e = p_times_act[peaks[i + 1]]
        dur = t_e - t_s

        # Gate 4 — HR range
        if not (MIN_CYCLE_SEC <= dur <= MAX_CYCLE_SEC):
            stats['fail_bpm'] += 1
            continue

        # Gate 1 — GT template correlation
        gt_seg = pchip_resample(
            norm01(p_vals_act[peaks[i]:peaks[i+1]]), CYCLE_SAMPLES)
        if np.corrcoef(gt_seg, gt_template)[0, 1] < gt_threshold:
            stats['fail_gt'] += 1
            continue

        # find corresponding rPPG indices
        v_idx = np.where((v_times >= t_s) & (v_times <= t_e))[0]
        if len(v_idx) < 5:
            stats['fail_sync'] += 1
            continue

        # Gate 2 — INDEPENDENT per algorithm (non-finite segments auto-fail)
        raw_pos   = rppg_pos[v_idx]
        raw_harm  = rppg_harm[v_idx]
        raw_phy   = rppg_phy[v_idx]
        raw_chrom = rppg_chrom[v_idx]

        if np.isfinite(raw_pos).all() and tmpl_pos is not None:
            seg_pos = norm01(pchip_resample(raw_pos, CYCLE_SAMPLES))
            pos_ok  = float(np.corrcoef(seg_pos, tmpl_pos)[0, 1]) >= rppg_threshold
        else:
            seg_pos = np.zeros(CYCLE_SAMPLES, dtype=np.float32)
            pos_ok  = False

        if np.isfinite(raw_harm).all() and tmpl_harm is not None:
            seg_harm = norm01(pchip_resample(raw_harm, CYCLE_SAMPLES))
            harm_ok  = float(np.corrcoef(seg_harm, tmpl_harm)[0, 1]) >= rppg_threshold
        else:
            seg_harm = np.zeros(CYCLE_SAMPLES, dtype=np.float32)
            harm_ok  = False

        if np.isfinite(raw_phy).all() and tmpl_phy is not None:
            seg_phy = norm01(pchip_resample(raw_phy, CYCLE_SAMPLES))
            phy_ok  = float(np.corrcoef(seg_phy, tmpl_phy)[0, 1]) >= rppg_threshold
        else:
            seg_phy = np.zeros(CYCLE_SAMPLES, dtype=np.float32)
            phy_ok  = False

        if np.isfinite(raw_chrom).all() and tmpl_chrom is not None:
            seg_chrom = norm01(pchip_resample(raw_chrom, CYCLE_SAMPLES))
            chrom_ok  = float(np.corrcoef(seg_chrom, tmpl_chrom)[0, 1]) >= rppg_threshold
        else:
            seg_chrom = np.zeros(CYCLE_SAMPLES, dtype=np.float32)
            chrom_ok  = False

        # keep heartbeat if ANY algorithm is clean
        if not (pos_ok or harm_ok or phy_ok or chrom_ok):
            stats['fail_rppg'] += 1
            continue

        # Gate 3 — SQI
        valid_sqi = sqi_pos[v_idx][~np.isnan(sqi_pos[v_idx])]
        mean_sqi  = float(valid_sqi.mean()) if len(valid_sqi) > 0 else 0.0
        if mean_sqi < SQI_THRESHOLD:
            stats['fail_sqi'] += 1
            continue

        # track extra cycles gained by multi-algorithm gate
        if not pos_ok:
            stats['extra_vs_pos'] += 1

        # ── all gates passed ──────────────────────────────────────────────────
        gt_cycles.append(gt_seg.astype(np.float32))
        rppg_pos_out.append(seg_pos.astype(np.float32))
        rppg_harm_out.append(seg_harm.astype(np.float32))
        raw_graw = rppg_graw[v_idx]
        if np.isfinite(raw_graw).all():
            graw_seg = norm01(pchip_resample(raw_graw, CYCLE_SAMPLES)).astype(np.float32)
        else:
            graw_seg = np.zeros(CYCLE_SAMPLES, dtype=np.float32)
        rppg_graw_out.append(graw_seg)
        rppg_phy_out.append(seg_phy.astype(np.float32))
        rppg_chrom_out.append(seg_chrom.astype(np.float32))
        pos_valid_out.append(pos_ok)
        harm_valid_out.append(harm_ok)
        phy_valid_out.append(phy_ok)
        chrom_valid_out.append(chrom_ok)
        peak_times_out.append(float(t_s))
        sqi_out.append(mean_sqi)
        hr_list.append(60.0 / dur)

    stats['ok'] = len(gt_cycles)
    if stats['ok'] < 5:
        stats['status'] = f'Only {stats["ok"]} ok'
        AUDIT_LOG.append(stats)
        return f'FAIL: {stats["status"]}'

    np.savez_compressed(
        output_path,
        gt_cycles           = np.array(gt_cycles),
        rppg_pos_cycles     = np.array(rppg_pos_out),
        rppg_harm_cycles    = np.array(rppg_harm_out),
        g_cycles            = np.array(rppg_graw_out),
        rppg_phybrid_cycles = np.array(rppg_phy_out),
        rppg_chrom_cycles   = np.array(rppg_chrom_out),
        # per-algorithm validity flags (True = passed quality gate)
        rppg_pos_valid      = np.array(pos_valid_out,   dtype=bool),
        rppg_harm_valid     = np.array(harm_valid_out,  dtype=bool),
        rppg_phybrid_valid  = np.array(phy_valid_out,   dtype=bool),
        rppg_chrom_valid    = np.array(chrom_valid_out, dtype=bool),
        # timestamp of GT peak for each cycle (seconds) — for debugging / future matching
        peak_times          = np.array(peak_times_out, dtype=np.float64),
        sqi                 = np.array(sqi_out,  dtype=np.float32),
        hr                  = np.array(hr_list,  dtype=np.float32),
        sid                 = sid,
        ppg_hz              = p_hz,
        dataset             = ds_name,
    )

    stats['status'] = 'OK'
    AUDIT_LOG.append(stats)
    return f'OK: {stats["ok"]} cycles'

# ══════════════════════════════════════════════════════════════════════════════
# SID EXTRACTION (same logic as v3)
# ══════════════════════════════════════════════════════════════════════════════

def extract_sid(name_clean, ds_name):
    try:
        if ds_name == 'ubfc':
            if 'ubfc_phys' in name_clean or 'vid_s' in name_clean:
                m = re.search(r'_s(\d+)_', name_clean)
                raw = int(m.group(1)) if m else int(''.join(filter(str.isdigit, name_clean)) or 0)
                return 1000 + raw
            else:
                # ubfc_rppg_s01 → _s(\d+) → 1; fallback: all digits → int('01')=1
                m = re.search(r'_s(\d+)', name_clean)
                return int(m.group(1)) if m else int(''.join(filter(str.isdigit, name_clean)) or 0)
        elif ds_name == 'stress2023':
            m = re.search(r'sub_?(\d+)', name_clean)
            return 2000 + (int(m.group(1)) if m else 0)
        elif ds_name == 'fps2023':
            # VIP cross-dataset subjects recorded in two labs
            if '0316' in name_clean: return 5001
            if '0331' in name_clean: return 5002
            # Per-session SID: 3000 + (all_digits % 1000)
            # poly_0323_Session_1 → digits '03231' → %1000=231 → 3231
            # poly_0404_Session_1 → digits '04041' → %1000=41  → 3041
            raw_num = int(''.join(filter(str.isdigit, name_clean)) or 0) % 1000
            return 3000 + raw_num
        elif ds_name == 'centan':
            m = re.search(r's(\d+)', name_clean)
            sid = 4000 + (int(m.group(1)) if m else 0)
            # VIP overrides: centan_s04 and centan_s11 are same humans as 0316/0331
            if sid == 4004: return 5001
            if sid == 4011: return 5002
            return sid
    except Exception:
        pass
    return 999

# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    datasets = [
        ('ubfc',       UBFC_CSV_DIR,       UBFC_RPPG_DIR,       UBFC_PPG_DIR,       UBFC_CYCLES_V4_DIR),
        ('stress2023', STRESS_CSV_DIR,     STRESS_RPPG_DIR,     STRESS_PPG_DIR,     STRESS_CYCLES_V4_DIR),
        ('fps2023',    FPS2023_CSV_DIR,    FPS2023_RPPG_DIR,    FPS2023_PPG_DIR,    FPS2023_CYCLES_V4_DIR),
        ('fps2023_60', FPS2023_60_CSV_DIR, FPS2023_60_RPPG_DIR, FPS2023_60_PPG_DIR, FPS2023_60_CYCLES_V4_DIR),
        ('centan',     CENTAN_CSV_DIR,     CENTAN_RPPG_DIR,     CENTAN_PPG_DIR,     CENTAN_CYCLES_V4_DIR),
    ]

    print('\nmorph_extract_cycles_v4 — Dual-Gated Cycle Extraction')
    print(f'  GT thresholds:   {GT_THRESHOLDS}')
    print(f'  rPPG thresholds: {RPPG_THRESHOLDS}')
    print(f'  SQI threshold:   {SQI_THRESHOLD} dB\n')

    for ds_name, csv_dir, rppg_v2_dir, ppg_dir, out_dir in datasets:
        if not os.path.isdir(csv_dir):
            print(f'  [{ds_name}] csv_dir not found, skipping.')
            continue

        rppg_v2_dir = Path(rppg_v2_dir)
        ppg_dir_p   = Path(ppg_dir)
        out_dir_p   = Path(out_dir)
        out_dir_p.mkdir(parents=True, exist_ok=True)

        csv_files = sorted(Path(csv_dir).glob('*.csv'))
        if PILOT_MODE:
            csv_files = csv_files[:PILOT_LIMIT]

        print(f'[{ds_name}] {len(csv_files)} subjects found')

        for csv_f in tqdm(csv_files, desc=ds_name):
            # find rppg_v2 file
            rppg_f = rppg_v2_dir / (csv_f.stem + '_rppg_v2.csv')
            if not rppg_f.exists():
                AUDIT_LOG.append({'subject': csv_f.stem, 'dataset': ds_name,
                                  'status': 'no rppg_v2 file', 'ok': 0,
                                  'candidates': 0, 'fail_bpm': 0, 'fail_gt': 0,
                                  'fail_rppg': 0, 'fail_sqi': 0, 'fail_sync': 0})
                continue

            # find GT PPG npz (ppg/ dir, not parsed/)
            ppg_candidates = list(ppg_dir_p.glob(f'{csv_f.stem}_ppg*.npz'))
            if not ppg_candidates:
                AUDIT_LOG.append({'subject': csv_f.stem, 'dataset': ds_name,
                                  'status': 'no ppg npz', 'ok': 0,
                                  'candidates': 0, 'fail_bpm': 0, 'fail_gt': 0,
                                  'fail_rppg': 0, 'fail_sqi': 0, 'fail_sync': 0})
                continue

            sid   = extract_sid(csv_f.stem.lower(), ds_name)
            out_f = out_dir_p / (csv_f.stem + '_cycles.npz')

            if out_f.exists() and not FORCE_REEXTRACT:
                continue  # already extracted, skip

            process_subject(csv_f, rppg_f, ppg_candidates[0],
                            out_f, sid, ds_name)

    # ── audit report ──────────────────────────────────────────────────────────
    if not AUDIT_LOG:
        print('No subjects processed.')
        return

    df = pd.DataFrame(AUDIT_LOG)

    print('\n' + '='*75)
    print('  CYCLE YIELD AUDIT — PER DATASET')
    print('='*75)

    total_row = {'dataset': 'TOTAL', 'subjects': 0, 'candidates': 0,
                 'fail_bpm': 0, 'fail_gt': 0, 'fail_rppg': 0,
                 'fail_sqi': 0, 'fail_sync': 0, 'ok': 0, 'extra_vs_pos': 0}

    for ds in ['ubfc', 'stress2023', 'fps2023', 'fps2023_60', 'centan']:
        sub = df[df['dataset'] == ds]
        if sub.empty:
            continue
        ok_sub    = sub[sub['status'] == 'OK']
        n_subj    = len(ok_sub)
        cands     = int(sub['candidates'].sum())
        f_bpm     = int(sub['fail_bpm'].sum())
        f_gt      = int(sub['fail_gt'].sum())
        f_rppg    = int(sub['fail_rppg'].sum())
        f_sqi     = int(sub['fail_sqi'].sum())
        f_sync    = int(sub['fail_sync'].sum())
        n_ok      = int(sub['ok'].sum())
        pct       = 100.0 * n_ok / cands if cands > 0 else 0.0

        extra = int(sub['extra_vs_pos'].sum()) if 'extra_vs_pos' in sub.columns else 0
        print(f'\n  {ds.upper()} ({n_subj} subjects with cycles)')
        print(f'    Candidates:               {cands:>6}')
        print(f'    Fail HR range:            {f_bpm:>6}')
        print(f'    Fail GT gate:             {f_gt:>6}')
        print(f'    Fail rPPG gate (all alg): {f_rppg:>6}')
        print(f'    Fail SQI gate:            {f_sqi:>6}')
        print(f'    Fail sync:                {f_sync:>6}')
        print(f'    Accepted:                 {n_ok:>6}  ({pct:.1f}%)')
        print(f'    Extra vs POS-only gate:   {extra:>6}  (cycles gained by multi-alg)')

        for k in ['candidates', 'fail_bpm', 'fail_gt', 'fail_rppg',
                  'fail_sqi', 'fail_sync', 'ok', 'extra_vs_pos']:
            total_row[k] += int(sub[k].sum()) if k in sub.columns else 0
        total_row['subjects'] += n_subj

    total_cands = total_row['candidates']
    total_ok    = total_row['ok']
    total_pct   = 100.0 * total_ok / total_cands if total_cands > 0 else 0.0

    total_extra = total_row['extra_vs_pos']
    print(f'\n  {"="*40}')
    print(f'  TOTAL ({total_row["subjects"]} subjects)')
    print(f'    Candidates:             {total_cands:>6}')
    print(f'    Accepted:               {total_ok:>6}  ({total_pct:.1f}%)')
    print(f'    Extra vs POS-only gate: {total_extra:>6}  ({100*total_extra/max(total_ok,1):.1f}% gain)')
    print('='*75)

    # save audit CSV
    audit_path = Path(RESULTS_DIR) / 'shared' / 'cycle_extraction_audit.csv'
    df.to_csv(audit_path, index=False)
    print(f'\nAudit saved: {audit_path}')


if __name__ == '__main__':
    main()

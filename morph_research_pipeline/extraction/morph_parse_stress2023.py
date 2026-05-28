"""
morph_parse_stress2023.py — Parse Stress 2023 Dataset (.dat files, 500Hz PPG)
==============================================================================
Loads pre-cropped face arrays from .dat files (zlib+pickle), applies MediaPipe
FaceMesh to extract patch signals, synchronizes with 500Hz PPG via alignment map.

OUTPUTS TWO FILES PER SESSION:
  1. CSV — patch RGB signals at video FPS (one row per frame, same format as
     UBFC parsers). gt_ppg column is interpolated to video FPS for rPPG
     extraction compatibility. This file feeds morph_extract_rppg.py unchanged.
  2. Companion NPZ (same name, _ppg500.npz) — the FULL 500Hz PPG signal at
     native rate with timestamps. This is the morphology ground truth for
     Section 3 cycle extraction. NOT downsampled.

Key features:
  - Face already cropped in .dat files (no face detection needed)
  - PPG at 500 Hz saved at native rate in companion NPZ
  - CSV gt_ppg interpolated to video FPS (for rPPG algo compat only)
  - Uses same face_vision_degraded helpers as TBME pipeline
  - Only 'none' variant (no codec degradations)
  - Checkpoint/resume: reads existing CSV, skips processed frames, appends

Run:
    python morph_parse_stress2023.py
"""

import os
import sys
import traceback
import multiprocessing as mpc
from multiprocessing import Pool
from pathlib import Path
import zlib
import pickle
from datetime import datetime

import cv2
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
from tqdm import tqdm

HERE         = Path(__file__).parent
PROJECT_ROOT = HERE.parent.parent
TBME_SHARED  = PROJECT_ROOT.parent / 'PATCH_PCA_CodecStudy' / 'shared'
sys.path.insert(0, str(TBME_SHARED))
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(HERE.parent))

from morph_config import (
    STRESS2023_ROOT, STRESS2023_VIDEO_DIR, STRESS2023_HRDATA_DIR,
    STRESS2023_ALIGNMENT_MAP, STRESS2023_SID_OFFSET, STRESS2023_PPG_HZ,
    STRESS_CSV_DIR, MORPH_STANDALONE_ROOT, PARSE_WORKERS,
)

try:
    from face_vision_degraded import (
        FaceVisionConfig,
        extract_patch_signals,
        calibrate_rule_a_thresholds,
        skin_filter,
        _mahal_mean,
        _update_ewma,
        _fresh_patch_states,
        _dist,
        PATCH_NAMES,
        EXCL_REGIONS,
        FACE_OVAL_IDX,
        _MODEL_POINTS,
    )
except ImportError as e:
    print(f'ERROR: Cannot import face_vision_degraded.py from {TBME_SHARED}')
    print(f'  {e}')
    sys.exit(1)

import mediapipe as mp

CFG = FaceVisionConfig(
    skin_detect       = True,
    skin_margin       = 30,
    skin_calib_frames = 30,
    rule_a_mxmi_diff  = 15,
    rule_a_abs_diff   = 15,
    alpha_fast        = 2.0 / (50.0  + 1.0),
    alpha_slow        = 2.0 / (300.0 + 1.0),
    calib_min_frames  = 135,
    yaw_delta_thresh  = 2.0,
    mar_thresh        = 0.25,
    ippc_buffer_len   = 90,
    gui               = False,
    std_col_suffix    = 'Raw',
    save_ippc_xcorr   = True,
    save_mahal_global = False,
    frame_transform   = None,
)

VARIANTS = [('none', None)]


def time_str_to_seconds(t_str):
    try:
        if '.' in t_str:
            hms, ms = t_str.split('.')
            ms = ms[:3]
            t_str = f'{hms}.{ms}'
        dt = datetime.strptime(t_str, '%H:%M:%S.%f')
    except ValueError:
        dt = datetime.strptime(t_str, '%H:%M:%S')
    return dt.hour * 3600 + dt.minute * 60 + dt.second + dt.microsecond / 1_000_000


def load_hr_csv(hr_csv_path, hr_start_sec):
    df = pd.read_csv(hr_csv_path, skiprows=3)
    df.columns = df.columns.str.strip()
    clock_sec = df['CLOCK'].apply(time_str_to_seconds).values
    ppg_times = clock_sec - hr_start_sec
    ppg_values = df['PPG'].values.astype(np.float32)
    return ppg_values, ppg_times


def interpolate_ppg(frame_times, ppg_values, ppg_times):
    if not np.all(np.diff(ppg_times) > 0):
        sort_idx = np.argsort(ppg_times)
        ppg_times = ppg_times[sort_idx]
        ppg_values = ppg_values[sort_idx]
    interp_fn = interp1d(ppg_times, ppg_values, kind='linear',
                         bounds_error=False, fill_value=np.nan)
    return interp_fn(frame_times).astype(np.float32)


def iter_dat_frames(dat_path):
    with open(dat_path, 'rb') as f:
        compressed = f.read()
        data = pickle.loads(zlib.decompress(compressed))
    keys = sorted(data.keys(), key=lambda x: int(x))
    for k in keys:
        yield k, data[k]


CHECKPOINT_EVERY = 500


def process_session(session_row, overwrite=False):
    subject = session_row['subject']
    session = session_row['session']
    hr_csv  = session_row['hr_csv']
    hr_start_str = session_row['hr_start']
    video_start_str = session_row['video_start']

    try:
        subj_num = int(subject.split('_')[1])
    except (IndexError, ValueError):
        print(f'ERROR: Cannot parse subject number from {subject}')
        return None, session, 'FAIL: bad subject', 0

    sid = subj_num + STRESS2023_SID_OFFSET

    out_csv = os.path.join(STRESS_CSV_DIR, f'stress2023_{subject}_{session}.csv')
    ppg500_path = os.path.join(STRESS_CSV_DIR,
                               f'stress2023_{subject}_{session}_ppg500.npz')

    resume_from = 0
    all_rows = []

    if overwrite:
        for f in [out_csv, ppg500_path]:
            if os.path.exists(f):
                os.remove(f)
    else:
        if os.path.exists(out_csv):
            try:
                existing_df = pd.read_csv(out_csv)
                if len(existing_df) > 0 and 'frame' in existing_df.columns:
                    last_frame = int(existing_df['frame'].max())
                    resume_from = last_frame + 1
                    all_rows = existing_df.to_dict('records')
                    print(f'  [RESUME CHECK] {os.path.basename(out_csv)}: {len(existing_df)} rows, last_frame={last_frame}')
                    if os.path.exists(ppg500_path):
                        print(f'  [SKIP] {os.path.basename(out_csv)} complete (ppg500 exists)')
                        return sid, session, 'SKIP', len(all_rows)
                    print(f'  [RESUME] {os.path.basename(out_csv)} from frame {resume_from}')
                else:
                    for f in [out_csv, ppg500_path]:
                        if os.path.exists(f):
                            os.remove(f)
            except Exception:
                for f in [out_csv, ppg500_path]:
                    if os.path.exists(f):
                        os.remove(f)

    if resume_from > 0 and not os.path.exists(out_csv):
        resume_from = 0
        all_rows = []

    date = session_row['date']
    exp = 'Experiment_1'
    session_dir = os.path.join(STRESS2023_VIDEO_DIR, date, exp, subject, session)
    if not os.path.isdir(session_dir):
        print(f'ERROR: Session directory not found: {session_dir}')
        return sid, session, 'FAIL: no session dir', 0

    dat_files = sorted(Path(session_dir).glob('*.dat'))
    if not dat_files:
        print(f'ERROR: No .dat files in {session_dir}')
        return sid, session, 'FAIL: no .dat files', 0

    hr_csv_path = os.path.join(STRESS2023_HRDATA_DIR, hr_csv)
    if not os.path.exists(hr_csv_path):
        print(f'ERROR: HR CSV not found: {hr_csv_path}')
        return sid, session, 'FAIL: no HR CSV', 0

    hr_start_sec = time_str_to_seconds(hr_start_str)
    video_start_sec = time_str_to_seconds(video_start_str)

    try:
        ppg_values, ppg_times = load_hr_csv(hr_csv_path, hr_start_sec)
    except Exception as e:
        print(f'ERROR: Failed to load HR CSV {hr_csv_path}: {e}')
        return sid, session, f'FAIL: HR CSV load {e}', 0

    face_mesh = mp.solutions.face_mesh.FaceMesh(
        static_image_mode=False, max_num_faces=1, refine_landmarks=True,
        min_detection_confidence=0.7, min_tracking_confidence=0.7)

    skin_thresholds, calib_pixel_pool, calib_done, prev_yaw = {}, [], False, None
    variant_states = {v[0]: {p: _fresh_patch_states() for p in PATCH_NAMES} for v in VARIANTS}
    for v in VARIANTS:
        variant_states[v[0]]['IPPC'] = {}
        variant_states[v[0]]['global_mahal'] = {'mu': None, 'C': None}

    frame_idx = 0
    rows_since_save = 0

    for dat_path in dat_files:
        for frame_key, frame_data in iter_dat_frames(dat_path):
            if frame_idx < resume_from:
                frame_idx += 1
                continue

            full_time_str = frame_data.get('FullTime', '')
            is_face_detected = frame_data.get('isFaceDetected', False)
            face_array = frame_data.get('Face', None)

            if full_time_str:
                frame_time_sec = time_str_to_seconds(full_time_str) - hr_start_sec
            else:
                frame_time_sec = (video_start_sec - hr_start_sec) + frame_idx / 30.0

            gt_ppg = interpolate_ppg(np.array([frame_time_sec]), ppg_values, ppg_times)[0]

            row = {
                'frame': frame_idx,
                'time_sec': frame_time_sec,
                'face_detected': 1 if is_face_detected else 0,
                'gt_ppg': gt_ppg,
                'gt_bpm': np.nan,
                'skin_detect_active': int(calib_done),
                'subject_id': sid,
                'camera': 'stress2023',
                'condition': 'none',
                'subject': subject,
                'session': session,
                'task': 1,
                's_num': subj_num,
            }

            yaw = pitch = roll = t_x = t_y = t_z = mar = ear = np.nan
            lm_pts = None

            if is_face_detected and face_array is not None and face_array.size > 0:
                h, w = face_array.shape[:2]
                rgb_face = cv2.cvtColor(face_array, cv2.COLOR_BGR2RGB)
                res = face_mesh.process(rgb_face)

                if res.multi_face_landmarks:
                    lm_pts = np.array([(l.x * w, l.y * h) for l in res.multi_face_landmarks[0].landmark])
                    img_pts = lm_pts[[1, 152, 225, 445, 230, 450]].astype(np.float64)
                    cam_matrix = np.array([[w, 0, w/2], [0, w, h/2], [0, 0, 1]], dtype=np.float64)
                    success, rvec, tvec = cv2.solvePnP(_MODEL_POINTS, img_pts, cam_matrix, np.zeros((4,1)))
                    if success:
                        t_x, t_y, t_z = tvec.flatten()
                        euler_angles = cv2.decomposeProjectionMatrix(np.hstack((cv2.Rodrigues(rvec)[0], tvec)))[-1]
                        roll, pitch, yaw = euler_angles.flatten()
                    mar = _dist(lm_pts[13], lm_pts[14]) / _dist(lm_pts[78], lm_pts[308])
                    ear = (_dist(lm_pts[160], lm_pts[144]) + _dist(lm_pts[158], lm_pts[153])) / (2 * _dist(lm_pts[33], lm_pts[133]))

                    head_stable = False
                    if prev_yaw is None or abs(yaw - prev_yaw) < CFG.yaw_delta_thresh:
                        head_stable = True
                    prev_yaw = yaw

                    if not calib_done and CFG.skin_detect:
                        mask_cal = np.zeros((h, w), dtype=np.uint8)
                        cv2.fillConvexPoly(mask_cal, cv2.convexHull(lm_pts[FACE_OVAL_IDX].astype(int)), 1)
                        for ex in EXCL_REGIONS:
                            cv2.fillConvexPoly(mask_cal, cv2.convexHull(lm_pts[ex].astype(int)), 0)
                        cal_pix = rgb_face[mask_cal == 1]
                        if len(cal_pix) > 100:
                            calib_pixel_pool.append(cal_pix[np.random.choice(len(cal_pix), min(len(cal_pix), 500))])
                        if len(calib_pixel_pool) > CFG.skin_calib_frames:
                            calibrate_rule_a_thresholds(np.vstack(calib_pixel_pool), skin_thresholds, CFG)
                            calib_done = True

                    for vname, transform_fn in VARIANTS:
                        rgb_v = rgb_face
                        mask_f = np.zeros((h, w), dtype=np.uint8)
                        cv2.fillConvexPoly(mask_f, cv2.convexHull(lm_pts[FACE_OVAL_IDX].astype(int)), 1)
                        for ex in EXCL_REGIONS:
                            cv2.fillConvexPoly(mask_f, cv2.convexHull(lm_pts[ex].astype(int)), 0)
                        pix_g = rgb_v[mask_f == 1].astype(np.float32)
                        row['Pixels_global'] = len(pix_g)
                        if len(pix_g) > 10:
                            mg, sg, cov = pix_g.mean(axis=0), pix_g.std(axis=0), np.cov(pix_g.T)
                            row['Raw_R_global'], row['Raw_G_global'], row['Raw_B_global'] = round(float(mg[0]), 4), round(float(mg[1]), 4), round(float(mg[2]), 4)
                            row['Std_R_global'], row['Std_G_global'], row['Std_B_global'] = round(float(sg[0]), 4), round(float(sg[1]), 4), round(float(sg[2]), 4)
                            vals, vecs = np.linalg.eigh(cov)
                            idx = np.argsort(vals)[::-1]
                            vals, vecs = vals[idx], vecs[:, idx]
                            row['eigval_1'], row['eigval_2'], row['eigval_3'] = round(float(vals[0]), 4), round(float(vals[1]), 4), round(float(vals[2]), 4)
                            row['u1_r'], row['u1_g'], row['u1_b'] = vecs[:, 0]
                            row['u2_r'], row['u2_g'], row['u2_b'] = vecs[:, 1]
                            row['u3_r'], row['u3_g'], row['u3_b'] = vecs[:, 2]
                            mst = variant_states[vname]['global_mahal']
                            if mst['mu'] is None:
                                mst['mu'], mst['C'] = mg.copy(), cov.copy()
                            else:
                                mst['mu'] = (1 - CFG.alpha_fast) * mst['mu'] + CFG.alpha_fast * mg
                                mst['C']  = (1 - CFG.alpha_fast) * mst['C']  + CFG.alpha_fast * cov
                            m_mn = _mahal_mean(pix_g, mst['mu'], mst['C'])
                            row['Mahal_R_global'], row['Mahal_G_global'], row['Mahal_B_global'] = round(float(m_mn[0]), 4), round(float(m_mn[1]), 4), round(float(m_mn[2]), 4)

                        patch_signals = extract_patch_signals(rgb_v, lm_pts, h, w, variant_states[vname], head_stable, skin_thresholds, CFG)
                        row.update(patch_signals)
                        row['degradation'] = vname
                else:
                    lm_pts = None
            else:
                lm_pts = None

            if lm_pts is None:
                for col in ['yaw', 'pitch', 'roll', 't_x', 't_y', 't_z', 'mar', 'ear',
                            'Pixels_global', 'Raw_R_global', 'Raw_G_global', 'Raw_B_global',
                            'Std_R_global', 'Std_G_global', 'Std_B_global',
                            'eigval_1', 'eigval_2', 'eigval_3',
                            'u1_r', 'u1_g', 'u1_b', 'u2_r', 'u2_g', 'u2_b', 'u3_r', 'u3_g', 'u3_b',
                            'Mahal_R_global', 'Mahal_G_global', 'Mahal_B_global']:
                    if col not in row:
                        row[col] = np.nan
                for pname in PATCH_NAMES:
                    for suffix in ['Pixels_Raw', 'R_patch_Raw', 'G_patch_Raw', 'B_patch_Raw',
                                   'Std_R_Raw', 'Std_G_Raw', 'Std_B_Raw',
                                   'G_patch_EWMA_Fast', 'G_patch_EWMA_Slow',
                                   'G_patch_EWMA_Gated', 'G_patch_IPPC',
                                   'Pixels_KDTree', 'G_patch_EWMA_KDTree']:
                        col = f'{suffix}_{pname}'
                        if col not in row:
                            row[col] = np.nan
                for i in range(len(PATCH_NAMES)):
                    for j in range(i+1, len(PATCH_NAMES)):
                        col = f'IPPC_xcorr_{PATCH_NAMES[i]}_{PATCH_NAMES[j]}'
                        row[col] = np.nan
                row['degradation'] = 'none'

            all_rows.append(row)
            frame_idx += 1
            rows_since_save += 1

            if rows_since_save >= CHECKPOINT_EVERY:
                pd.DataFrame(all_rows).to_csv(out_csv, index=False)
                print(f'  [CKPT] {os.path.basename(out_csv)} frame {frame_idx} ({len(all_rows)} total rows)')
                rows_since_save = 0

    face_mesh.close()

    if all_rows:
        df_out = pd.DataFrame(all_rows)
        df_out.to_csv(out_csv, index=False)

        np.savez_compressed(ppg500_path,
                            ppg_values=ppg_values,
                            ppg_times=ppg_times,
                            ppg_hz=STRESS2023_PPG_HZ,
                            hr_start_sec=hr_start_sec,
                            subject=subject,
                            session=session,
                            subject_id=sid)

        status = f'OK {len(all_rows)} rows | ppg500 saved'
        if resume_from > 0:
            status = f'OK {len(all_rows)} rows (resumed from {resume_from}) | ppg500 saved'
    else:
        status = 'FAIL: no rows'

    return sid, session, status, len(all_rows)


def build_jobs(overwrite=False, limit=0):
    if not os.path.exists(STRESS2023_ALIGNMENT_MAP):
        print(f'ERROR: Alignment map not found: {STRESS2023_ALIGNMENT_MAP}')
        return []

    df_align = pd.read_csv(STRESS2023_ALIGNMENT_MAP)
    n_before = len(df_align)
    df_align = df_align[df_align['hr_csv'] != 'N/A']
    n_after = len(df_align)
    if n_before > n_after:
        print(f'  Note: filtered out {n_before - n_after} sessions with missing HR data')
    if limit > 0:
        df_align = df_align.head(limit)
    jobs = []
    for _, row in df_align.iterrows():
        jobs.append((row.to_dict(), overwrite))
    return jobs


def _worker_stress2023(args):
    session_row, overwrite = args
    try:
        return process_session(session_row, overwrite)
    except Exception as e:
        subject = session_row.get('subject', 'unknown')
        session = session_row.get('session', 'unknown')
        return STRESS2023_SID_OFFSET, session, f'FAIL: {e}\n{traceback.format_exc()[:400]}', 0


def main():
    workers = PARSE_WORKERS
    overwrite = False
    limit = 0

    os.makedirs(STRESS_CSV_DIR, exist_ok=True)

    jobs = build_jobs(overwrite, limit)
    if not jobs:
        print('No Stress 2023 sessions found to process.')
        return

    print(f'\nStress 2023 parser — {len(jobs)} sessions')
    print(f'  Workers  : {workers}')
    print(f'  Overwrite: {overwrite}')
    print(f'  Output   : {STRESS_CSV_DIR}')
    print(f'  Checkpoint every {CHECKPOINT_EVERY} frames\n')

    log_rows = []
    n_ok = n_fail = n_skip = 0

    w = min(workers, max(1, len(jobs)))
    if workers == 1:
        bar = tqdm(jobs, desc='Stress 2023', unit='session')
        for args in bar:
            res = _worker_stress2023(args)
            sid, session, status, n = res
            if 'FAIL' in str(status): n_fail += 1
            elif status == 'SKIP': n_skip += 1
            else: n_ok += 1
            bar.set_postfix(ok=n_ok, skip=n_skip, fail=n_fail)
            log_rows.append({'dataset': 'stress2023', 'sid': sid, 'session': session,
                             'status': status, 'n_rows': n})
    else:
        with Pool(processes=w) as pool:
            bar = tqdm(pool.imap_unordered(_worker_stress2023, jobs),
                       total=len(jobs), desc='Stress 2023', unit='session')
            for res in bar:
                sid, session, status, n = res
                if 'FAIL' in str(status): n_fail += 1
                elif status == 'SKIP': n_skip += 1
                else: n_ok += 1
                bar.set_postfix(ok=n_ok, skip=n_skip, fail=n_fail)
                log_rows.append({'dataset': 'stress2023', 'sid': sid, 'session': session,
                                 'status': status, 'n_rows': n})

    print(f'\n  OK={n_ok}  SKIP={n_skip}  FAIL={n_fail}')

    df_log = pd.DataFrame(log_rows)
    log_path = os.path.join(MORPH_STANDALONE_ROOT, 'parse_log_stress2023.csv')
    df_log.to_csv(log_path, index=False)
    print(f'\nParse log: {log_path}')
    print(f'CSVs in  : {STRESS_CSV_DIR}')
    print('\nNext step: python morph_extract_rppg.py')


if __name__ == '__main__':
    mpc.freeze_support()
    main()

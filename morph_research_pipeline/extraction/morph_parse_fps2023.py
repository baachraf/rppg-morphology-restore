"""
morph_parse_fps2023.py — Parse FPS_Analysis_CENTAN Dataset
===========================================================
Real .avi videos (640x480, 30 FPS, cam1 only) + PKL PPG (500 Hz).
Uses per-frame wall-clock timestamps from FT_*.csv for exact PPG sync.

OUTPUTS per session:
  1. CSV — patch RGB signals at video FPS (same format as UBFC parsers)
  2. Companion _ppg500.npz — full 500 Hz PPG at native rate (morphology GT)

Key features:
  - Frame-by-frame MediaPipe processing (no process_video dependency)
  - Checkpoint/resume: reads existing CSV, skips processed frames, appends
  - Real wall-clock timestamps from FT_*.csv files
  - Only cam1_30FPS, only 'none' variant
  - No argparse — settings hardcoded in main()

Run:
    python morph_parse_fps2023.py
"""

import os
import sys
import traceback
import multiprocessing as mpc
from multiprocessing import Pool
from pathlib import Path
from datetime import datetime
import pickle

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
    FPS2023_ROOT, FPS2023_VIDEO_DIR, FPS2023_DATA_DIR, FPS2023_SID_OFFSET, FPS2023_PPG_HZ,
    FPS2023_SUB_MAP, FPS2023_CSV_DIR, MORPH_STANDALONE_ROOT, PARSE_WORKERS,
)

try:
    from face_vision_degraded import (
        FaceVisionConfig,
        extract_patch_signals,
        calibrate_rule_a_thresholds,
        _mahal_mean,
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
CHECKPOINT_EVERY = 500


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


def load_pkl_subject(pkl_path):
    with open(pkl_path, 'rb') as f:
        data = pickle.load(f)
    return data


def load_frame_times(ft_csv_path):
    if not os.path.exists(ft_csv_path):
        print(f'  [ERROR] FT file missing: {ft_csv_path}')
        return np.array([], dtype=np.float64)
    try:
        with open(ft_csv_path, 'r') as f:
            raw_lines = f.readlines()
    except Exception as e:
        print(f'  [ERROR] FT file read error: {ft_csv_path} — {e}')
        return np.array([], dtype=np.float64)
    if len(raw_lines) == 0:
        print(f'  [ERROR] FT file EMPTY: {ft_csv_path}')
        return np.array([], dtype=np.float64)
    print(f'  [DEBUG] FT file {os.path.basename(ft_csv_path)}: {len(raw_lines)} lines, first={raw_lines[0].strip()[:80]}')
    try:
        header = raw_lines[0].strip().split(',')[0]
        times = [header]
        for line in raw_lines[1:]:
            t = line.strip().split(',')[0]
            if t:
                times.append(t)
        return np.array([time_str_to_seconds(t) for t in times], dtype=np.float64)
    except Exception as e:
        print(f'  [ERROR] FT file parse error: {ft_csv_path} — {e}')
        print(f'  [DEBUG] first 3 lines: {[l.strip()[:80] for l in raw_lines[:3]]}')
        return np.array([], dtype=np.float64)


def find_video_parts(session_dir):
    for root, dirs, files in os.walk(session_dir):
        if 'RGB_Vid' in root:
            avi_files = sorted([f for f in files if f.endswith('.avi')])
            if avi_files:
                return root, avi_files
    return None, []


def find_ft_dir(session_dir):
    for root, dirs, files in os.walk(session_dir):
        if 'Frame_Full_Time' in root:
            return root
    return None


def _process_frame(frame_idx, frame_bgr, frame_time_sec, gt_ppg_val,
                   sid, session_name, sub_code,
                   face_mesh, skin_thresholds, calib_pixel_pool, calib_done,
                   prev_yaw, variant_states):
    row = {
        'frame': frame_idx,
        'time_sec': frame_time_sec,
        'face_detected': 0,
        'gt_ppg': gt_ppg_val,
        'gt_bpm': np.nan,
        'skin_detect_active': int(calib_done),
        'subject_id': sid,
        'camera': 'fps2023_30fps',
        'condition': 'none',
        'sub_code': sub_code,
        'session': session_name,
    }

    h, w = frame_bgr.shape[:2]
    rgb_frame = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    res = face_mesh.process(rgb_frame)

    yaw = pitch = roll = t_x = t_y = t_z = mar = ear = np.nan
    lm_pts = None

    if res.multi_face_landmarks:
        lm_pts = np.array([(l.x * w, l.y * h)
                           for l in res.multi_face_landmarks[0].landmark])
        row['face_detected'] = 1

        img_pts = lm_pts[[1, 152, 225, 445, 230, 450]].astype(np.float64)
        cam_matrix = np.array([[w, 0, w/2], [0, w, h/2], [0, 0, 1]], dtype=np.float64)
        success, rvec, tvec = cv2.solvePnP(_MODEL_POINTS, img_pts, cam_matrix,
                                           np.zeros((4,1)))
        if success:
            t_x, t_y, t_z = tvec.flatten()
            euler_angles = cv2.decomposeProjectionMatrix(
                np.hstack((cv2.Rodrigues(rvec)[0], tvec)))[-1]
            roll, pitch, yaw = euler_angles.flatten()
        mar = _dist(lm_pts[13], lm_pts[14]) / _dist(lm_pts[78], lm_pts[308])
        ear = (_dist(lm_pts[160], lm_pts[144]) + _dist(lm_pts[158], lm_pts[153])) / \
              (2 * _dist(lm_pts[33], lm_pts[133]))

        head_stable = (prev_yaw is None or abs(yaw - prev_yaw) < CFG.yaw_delta_thresh)
        prev_yaw = yaw

        if not calib_done and CFG.skin_detect:
            mask_cal = np.zeros((h, w), dtype=np.uint8)
            cv2.fillConvexPoly(mask_cal,
                               cv2.convexHull(lm_pts[FACE_OVAL_IDX].astype(int)), 1)
            for ex in EXCL_REGIONS:
                cv2.fillConvexPoly(mask_cal,
                                   cv2.convexHull(lm_pts[ex].astype(int)), 0)
            cal_pix = rgb_frame[mask_cal == 1]
            if len(cal_pix) > 100:
                calib_pixel_pool.append(
                    cal_pix[np.random.choice(len(cal_pix),
                                             min(len(cal_pix), 500))])
            if len(calib_pixel_pool) > CFG.skin_calib_frames:
                calibrate_rule_a_thresholds(np.vstack(calib_pixel_pool),
                                            skin_thresholds, CFG)
                calib_done = True

        for vname, transform_fn in VARIANTS:
            rgb_v = rgb_frame
            mask_f = np.zeros((h, w), dtype=np.uint8)
            cv2.fillConvexPoly(mask_f,
                               cv2.convexHull(lm_pts[FACE_OVAL_IDX].astype(int)), 1)
            for ex in EXCL_REGIONS:
                cv2.fillConvexPoly(mask_f,
                                   cv2.convexHull(lm_pts[ex].astype(int)), 0)
            pix_g = rgb_v[mask_f == 1].astype(np.float32)
            row['Pixels_global'] = len(pix_g)
            if len(pix_g) > 10:
                mg = pix_g.mean(axis=0)
                sg = pix_g.std(axis=0)
                cov = np.cov(pix_g.T)
                row['Raw_R_global'] = round(float(mg[0]), 4)
                row['Raw_G_global'] = round(float(mg[1]), 4)
                row['Raw_B_global'] = round(float(mg[2]), 4)
                row['Std_R_global'] = round(float(sg[0]), 4)
                row['Std_G_global'] = round(float(sg[1]), 4)
                row['Std_B_global'] = round(float(sg[2]), 4)
                vals, vecs = np.linalg.eigh(cov)
                idx = np.argsort(vals)[::-1]
                vals, vecs = vals[idx], vecs[:, idx]
                row['eigval_1'] = round(float(vals[0]), 4)
                row['eigval_2'] = round(float(vals[1]), 4)
                row['eigval_3'] = round(float(vals[2]), 4)
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
                row['Mahal_R_global'] = round(float(m_mn[0]), 4)
                row['Mahal_G_global'] = round(float(m_mn[1]), 4)
                row['Mahal_B_global'] = round(float(m_mn[2]), 4)

            patch_signals = extract_patch_signals(
                rgb_v, lm_pts, h, w, variant_states[vname],
                head_stable, skin_thresholds, CFG)
            row.update(patch_signals)
            row['degradation'] = vname
    else:
        lm_pts = None

    if lm_pts is None:
        for col in ['yaw', 'pitch', 'roll', 't_x', 't_y', 't_z', 'mar', 'ear',
                     'Pixels_global', 'Raw_R_global', 'Raw_G_global', 'Raw_B_global',
                     'Std_R_global', 'Std_G_global', 'Std_B_global',
                     'eigval_1', 'eigval_2', 'eigval_3',
                     'u1_r', 'u1_g', 'u1_b', 'u2_r', 'u2_g', 'u2_b',
                     'u3_r', 'u3_g', 'u3_b',
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

    return row, prev_yaw, calib_done


def process_one_session(args):
    sub_code, session_name, pkl_data, session_idx, overwrite = args

    sid = int(sub_code) + FPS2023_SID_OFFSET

    out_csv = os.path.join(FPS2023_CSV_DIR, f'poly_{sub_code}_{session_name}.csv')
    ppg500_path = os.path.join(FPS2023_CSV_DIR,
                               f'poly_{sub_code}_{session_name}_ppg500.npz')

    resume_from = 0
    all_rows = []

    if overwrite:
        for f in [out_csv, ppg500_path]:
            if os.path.exists(f):
                os.remove(f)
    else:
        if os.path.exists(out_csv) and os.path.exists(ppg500_path):
            try:
                existing_df = pd.read_csv(out_csv)
                if len(existing_df) > 0 and 'frame' in existing_df.columns:
                    last_frame = int(existing_df['frame'].max())
                    print(f'  [SKIP] {os.path.basename(out_csv)} complete ({len(existing_df)} rows, ppg500 exists)')
                    return sid, session_name, 'SKIP', len(existing_df)
                else:
                    for f in [out_csv, ppg500_path]:
                        if os.path.exists(f):
                            os.remove(f)
            except Exception:
                for f in [out_csv, ppg500_path]:
                    if os.path.exists(f):
                        os.remove(f)
        elif os.path.exists(out_csv) and not os.path.exists(ppg500_path):
            try:
                existing_df = pd.read_csv(out_csv)
                if len(existing_df) > 0 and 'frame' in existing_df.columns:
                    resume_from = int(existing_df['frame'].max()) + 1
                    all_rows = existing_df.to_dict('records')
                    print(f'  [RESUME CHECK] {os.path.basename(out_csv)}: {len(existing_df)} rows, last_frame={resume_from-1}')
                    print(f'  [RESUME] {os.path.basename(out_csv)} from frame {resume_from}')
            except Exception:
                if os.path.exists(out_csv):
                    os.remove(out_csv)
                resume_from = 0
                all_rows = []

    if resume_from > 0 and not os.path.exists(out_csv):
        resume_from = 0
        all_rows = []

    sess_key = f'sess{session_idx}'
    if sess_key not in pkl_data:
        return sid, session_name, f'FAIL: {sess_key} not in PKL', 0

    ppg_values = pkl_data[sess_key]['ppg'].astype(np.float32)
    ppg_fil    = pkl_data[sess_key].get('ppg_fil', ppg_values).astype(np.float32)
    ppg_times  = np.arange(len(ppg_values), dtype=np.float64) / FPS2023_PPG_HZ

    video_folder = None
    for entry in sorted(os.listdir(FPS2023_VIDEO_DIR)):
        if sub_code in entry and 'cam1_30FPS' in entry:
            candidate = os.path.join(FPS2023_VIDEO_DIR, entry, session_name)
            if os.path.isdir(candidate):
                video_folder = candidate
                break

    if video_folder is None:
        return sid, session_name, 'FAIL: no video folder', 0

    rgb_dir, avi_files = find_video_parts(video_folder)
    if not rgb_dir or not avi_files:
        return sid, session_name, 'FAIL: no .avi files', 0

    ft_dir = find_ft_dir(video_folder)

    all_frame_times = []
    frame_sources = []
    for avi_name in avi_files:
        avi_path = os.path.join(rgb_dir, avi_name)
        cap = cv2.VideoCapture(avi_path)
        n_part = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()

        ft_name = 'FT_' + avi_name.replace('.avi', '.csv')
        ft_path = os.path.join(ft_dir, ft_name) if ft_dir else None
        if ft_path and os.path.exists(ft_path):
            times = load_frame_times(ft_path)
            if len(times) == 0:
                print(f'  [ERROR] No timestamps from {ft_name} for {avi_name}, skipping part')
                continue
        else:
            print(f'  [WARN] No FT file for {avi_name}, using synthetic 30fps timestamps')
            t0 = all_frame_times[-1] + 0.033 if all_frame_times else 0.0
            times = np.array([t0 + i / 30.0 for i in range(n_part)])

        for i in range(min(n_part, len(times))):
            all_frame_times.append(times[i])
            frame_sources.append((avi_path, i))

    n_frames = len(frame_sources)
    if n_frames == 0:
        return sid, session_name, 'FAIL: 0 frames', 0

    frame_times_arr = np.array(all_frame_times, dtype=np.float64)
    frame_times_rel = frame_times_arr - frame_times_arr[0]

    ppg_times_shifted = ppg_times - ppg_times[0]
    interp_fn = interp1d(ppg_times_shifted, ppg_values, kind='linear',
                         bounds_error=False, fill_value=np.nan)
    gt_ppg_all = interp_fn(frame_times_rel).astype(np.float32)

    face_mesh = mp.solutions.face_mesh.FaceMesh(
        static_image_mode=False, max_num_faces=1, refine_landmarks=True,
        min_detection_confidence=0.7, min_tracking_confidence=0.7)

    skin_thresholds = {}
    calib_pixel_pool = []
    calib_done = False
    prev_yaw = None
    variant_states = {v[0]: {p: _fresh_patch_states() for p in PATCH_NAMES}
                      for v in VARIANTS}
    for v in VARIANTS:
        variant_states[v[0]]['IPPC'] = {}
        variant_states[v[0]]['global_mahal'] = {'mu': None, 'C': None}

    rows_since_save = 0
    current_cap = None
    current_avi = None

    if resume_from > 0:
        print(f'  [SKIP-FRAMES] {os.path.basename(out_csv)} skipping {resume_from}/{n_frames} frames...')

    for global_idx in range(n_frames):
        avi_path, _ = frame_sources[global_idx]

        if current_avi != avi_path:
            if current_cap is not None:
                current_cap.release()
            current_cap = cv2.VideoCapture(avi_path)
            current_avi = avi_path

        ret, frame_bgr = current_cap.read()
        if not ret:
            print(f'  [WARN] {os.path.basename(out_csv)} cap.read() failed at frame {global_idx}/{n_frames}')
            break

        if global_idx < resume_from:
            continue

        if global_idx == resume_from:
            print(f'  [PROCESSING] {os.path.basename(out_csv)} now processing from frame {resume_from}/{n_frames}')

        row, prev_yaw, calib_done = _process_frame(
            global_idx, frame_bgr, frame_times_rel[global_idx],
            gt_ppg_all[global_idx],
            sid, session_name, sub_code,
            face_mesh, skin_thresholds, calib_pixel_pool, calib_done,
            prev_yaw, variant_states)

        all_rows.append(row)
        rows_since_save += 1

        if rows_since_save >= CHECKPOINT_EVERY:
            pd.DataFrame(all_rows).to_csv(out_csv, index=False)
            print(f'  [CKPT] {os.path.basename(out_csv)} frame {global_idx}/{n_frames} ({len(all_rows)} total rows)')
            rows_since_save = 0

    if current_cap is not None:
        current_cap.release()
    face_mesh.close()

    if all_rows:
        pd.DataFrame(all_rows).to_csv(out_csv, index=False)

        np.savez_compressed(ppg500_path,
                            ppg_values=ppg_values,
                            ppg_fil=ppg_fil,
                            ppg_times=ppg_times_shifted,
                            ppg_hz=FPS2023_PPG_HZ,
                            sub_code=sub_code,
                            session=session_name,
                            subject_id=sid,
                            ppg_peaks=pkl_data[sess_key].get('ppg_peak', np.array([])),
                            ecg_peaks=pkl_data[sess_key].get('ecg_peak', np.array([])))

        status = f'OK {len(all_rows)} rows | ppg500 saved'
        if resume_from > 0:
            status = f'OK {len(all_rows)} rows (resumed from {resume_from}) | ppg500 saved'
    else:
        status = 'FAIL: no rows'

    return sid, session_name, status, len(all_rows)


def build_jobs(overwrite=False):
    jobs = []
    for date_code, pkl_name in sorted(FPS2023_SUB_MAP.items()):
        pkl_path = os.path.join(FPS2023_DATA_DIR, pkl_name)
        if not os.path.exists(pkl_path):
            print(f'  WARN: PKL not found: {pkl_path}')
            continue

        pkl_data = load_pkl_subject(pkl_path)
        n_sess = len(pkl_data)

        video_dirs = []
        if os.path.isdir(FPS2023_VIDEO_DIR):
            for entry in sorted(os.listdir(FPS2023_VIDEO_DIR)):
                if date_code in entry and 'cam1_30FPS' in entry:
                    full = os.path.join(FPS2023_VIDEO_DIR, entry)
                    if os.path.isdir(full):
                        for sess in sorted(os.listdir(full)):
                            sess_full = os.path.join(full, sess)
                            if os.path.isdir(sess_full) and sess.startswith('Session_'):
                                video_dirs.append((sess, int(sess.replace('Session_', ''))))

        video_dirs.sort(key=lambda x: x[1])

        for sess_name, sess_idx in video_dirs:
            if sess_idx > n_sess:
                continue
            jobs.append((date_code, sess_name, pkl_data, sess_idx, overwrite))

    return jobs


def main():
    workers   = PARSE_WORKERS
    overwrite = False

    os.makedirs(FPS2023_CSV_DIR, exist_ok=True)

    jobs = build_jobs(overwrite)
    if not jobs:
        print('No Polymate sessions found.')
        return

    print(f'\nPolymate parser — {len(jobs)} sessions (cam1_30FPS only)')
    print(f'  Workers : {workers}')
    print(f'  Output  : {FPS2023_CSV_DIR}')
    print(f'  PPG     : {FPS2023_PPG_HZ} Hz from PKL')
    print(f'  Checkpoint every {CHECKPOINT_EVERY} frames\n')

    log_rows = []
    n_ok = n_fail = n_skip = 0

    w = min(workers, max(1, len(jobs)))
    if workers == 1:
        bar = tqdm(jobs, desc='Polymate', unit='session')
        for args in bar:
            sid, session, status, n = process_one_session(args)
            if 'FAIL' in str(status):
                n_fail += 1
            elif status == 'SKIP':
                n_skip += 1
            else:
                n_ok += 1
            bar.set_postfix(ok=n_ok, skip=n_skip, fail=n_fail)
            log_rows.append({'dataset': 'fps2023', 'sid': sid, 'session': session,
                             'status': status, 'n_rows': n})
    else:
        with Pool(processes=w) as pool:
            bar = tqdm(pool.imap_unordered(process_one_session, jobs),
                       total=len(jobs), desc='Polymate', unit='session')
            for sid, session, status, n in bar:
                if 'FAIL' in str(status):
                    n_fail += 1
                elif status == 'SKIP':
                    n_skip += 1
                else:
                    n_ok += 1
                bar.set_postfix(ok=n_ok, skip=n_skip, fail=n_fail)
                log_rows.append({'dataset': 'fps2023', 'sid': sid, 'session': session,
                                 'status': status, 'n_rows': n})

    print(f'\n  OK={n_ok}  SKIP={n_skip}  FAIL={n_fail}')

    df_log = pd.DataFrame(log_rows)
    log_path = os.path.join(MORPH_STANDALONE_ROOT, 'parse_log_fps2023.csv')
    df_log.to_csv(log_path, index=False)
    print(f'Parse log: {log_path}')
    print(f'CSVs in  : {FPS2023_CSV_DIR}')
    print('\nNext step: python morph_extract_rppg.py')


if __name__ == '__main__':
    mpc.freeze_support()
    main()

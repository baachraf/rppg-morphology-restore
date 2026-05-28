"""
morph_parse_ubfc.py — Standalone Video Parser (UBFC-rPPG + UBFC-PHYS)
=====================================================================
MULTI-PROCESS STABILITY MODE:
  - spawn context  : each worker is a fresh Python process → no shared TFLite state
  - maxtasksperchild=1 : worker exits after ONE subject → OS reclaims all memory
  - PARSE_WORKERS from morph_config controls pool size (default 2)
  - Full checkpoint/resume: reads partial CSV, seeks video to last_frame+1
"""

import os
import sys
import traceback
from pathlib import Path
import gc
import multiprocessing

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
    UBFC_RPPG_RAW_ROOT, UBFC_PHYS_RAW_ROOT,
    UBFC_CSV_DIR, MORPH_STANDALONE_ROOT, DATA_ROOT,
    UBFC_RPPG_SUBJECT_PREFIX,
    UBFC_RPPG_VIDEO_FILE, UBFC_RPPG_GT_FILE,
    UBFC_RPPG_SID_OFFSET,
    UBFC_PHYS_TASKS, UBFC_PHYS_SID_OFFSET, UBFC_PHYS_BVP_HZ,
    PARSE_WORKERS,
)

try:
    from face_vision_degraded import (
        FaceVisionConfig, extract_patch_signals, calibrate_rule_a_thresholds,
        _mahal_mean, _fresh_patch_states, _dist, PATCH_NAMES, EXCL_REGIONS,
        FACE_OVAL_IDX, _MODEL_POINTS,
    )
except ImportError as e:
    print(f'ERROR: Cannot import face_vision_degraded.py from {TBME_SHARED}')
    sys.exit(1)

CFG = FaceVisionConfig(
    skin_detect=True, skin_margin=30, skin_calib_frames=30, alpha_fast=0.04, alpha_slow=0.006,
    calib_min_frames=135, yaw_delta_thresh=2.0, mar_thresh=0.25, ippc_buffer_len=90, gui=False,
    std_col_suffix='Raw', save_ippc_xcorr=True, save_mahal_global=False,
)

CHECKPOINT_EVERY = 500


# ──────────────────────────────────────────────────────────────────────────────
# Ground-truth loaders
# ──────────────────────────────────────────────────────────────────────────────

def load_gt_ubfc(gt_path, n_frames, fps_nom):
    with open(gt_path, 'r') as f:
        lines = f.readlines()
    ppg_raw = np.array([float(x) for x in lines[0].strip().split()], dtype=np.float64)
    ts_raw  = np.array([float(x) for x in lines[2].strip().split()], dtype=np.float64) if len(lines) > 2 else None
    if ts_raw is not None and len(ts_raw) > 1:
        f_times = interp1d(np.linspace(0, n_frames - 1, len(ts_raw)), ts_raw,
                           fill_value="extrapolate")(np.arange(n_frames))
        gt_ppg  = interp1d(ts_raw, ppg_raw, bounds_error=False,
                           fill_value="extrapolate")(f_times).astype(np.float32)
    else:
        f_times = np.arange(n_frames) / fps_nom
        gt_ppg  = interp1d(np.linspace(0, 1, len(ppg_raw)), ppg_raw)(
                      np.linspace(0, 1, n_frames)).astype(np.float32)
    return gt_ppg, f_times


def load_bvp_phys(bvp_path, n_frames, fps_nom):
    bvp_raw = np.loadtxt(bvp_path, dtype=np.float64)
    b_times = np.arange(len(bvp_raw)) / UBFC_PHYS_BVP_HZ
    f_times = np.arange(n_frames) / fps_nom
    gt_ppg  = interp1d(b_times, bvp_raw, bounds_error=False,
                       fill_value="extrapolate")(f_times).astype(np.float32)
    return gt_ppg, f_times


# ──────────────────────────────────────────────────────────────────────────────
# Per-frame processor (unchanged from single-process version)
# ──────────────────────────────────────────────────────────────────────────────

def _process_frame(frame_idx, bgr, t, gt, sid, cam, meta,
                   mesh, skin_thresh, pool, cal_done, prev_yaw, states):
    row = {'frame': frame_idx, 'time_sec': t, 'face_detected': 0,
           'gt_ppg': gt, 'subject_id': sid, 'camera': cam, **meta}
    h, w = bgr.shape[:2]
    rgb  = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    res  = mesh.process(rgb)
    if res.multi_face_landmarks:
        pts = np.array([(l.x * w, l.y * h) for l in res.multi_face_landmarks[0].landmark])
        row['face_detected'] = 1
        mask = np.zeros((h, w), dtype=np.uint8)
        if CFG.skin_detect:
            cv2.fillConvexPoly(mask, cv2.convexHull(pts[FACE_OVAL_IDX].astype(int)), 1)
            for ex in EXCL_REGIONS:
                cv2.fillConvexPoly(mask, cv2.convexHull(pts[ex].astype(int)), 0)
            if not cal_done:
                pix = rgb[mask == 1]
                if len(pix) > 100:
                    pool.append(pix[np.random.choice(len(pix), min(len(pix), 500))])
                if len(pool) > CFG.skin_calib_frames:
                    calibrate_rule_a_thresholds(np.vstack(pool), skin_thresh, CFG)
                    cal_done = True
        mg = rgb[mask == 1].mean(axis=0) if cal_done else rgb.mean(axis=(0, 1))
        row['Raw_R_global'], row['Raw_G_global'], row['Raw_B_global'] = mg
        row.update(extract_patch_signals(rgb, pts, h, w, states['none'], True, skin_thresh, CFG))
    return row, 0.0, cal_done


# ──────────────────────────────────────────────────────────────────────────────
# Worker — runs in a dedicated subprocess (spawn, maxtasksperchild=1)
# ──────────────────────────────────────────────────────────────────────────────

def worker_ubfc(args):
    """
    Returns (sid, status_str, n_new_frames).
    Imports mediapipe inside the function so each spawned process owns its own
    TFLite interpreter — no shared native state, no leak accumulation.
    """
    sid, subj_dir, out_csv, overwrite, is_phys, task, s_num = args

    import mediapipe as mp  # local import: each subprocess gets its own TFLite

    # ── Already fully done? ───────────────────────────────────────────────────
    npz_glob = list(Path(UBFC_CSV_DIR).glob(
        f"{Path(out_csv).stem}_ppg{'64' if is_phys else '*'}.npz"))
    if not overwrite and os.path.exists(out_csv) and npz_glob:
        return (sid, 'SKIP', 0)

    # ── Resume detection — read only the frame column (O(n) but single column) ─
    resume_from = 0
    csv_exists  = not overwrite and os.path.exists(out_csv)
    if csv_exists:
        try:
            frames_col = pd.read_csv(out_csv, usecols=['frame'])
            if len(frames_col) > 0:
                resume_from = int(frames_col['frame'].max()) + 1
            else:
                csv_exists = False  # empty file → treat as fresh start
        except Exception:
            resume_from = 0
            csv_exists  = False

    # append=True  → new rows go after existing data, no rewrite, no RAM copy
    # append=False → fresh file, write header on first batch
    appending    = csv_exists and resume_from > 0
    write_mode   = 'a' if appending else 'w'
    write_header = not appending

    try:
        v_path  = os.path.join(subj_dir,
                               UBFC_RPPG_VIDEO_FILE if not is_phys
                               else f'vid_s{s_num}_T{task}.avi')
        gt_path = os.path.join(subj_dir,
                               UBFC_RPPG_GT_FILE if not is_phys
                               else f'bvp_s{s_num}_T{task}.csv')

        cap = cv2.VideoCapture(v_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        n_f = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        if is_phys:
            gt_ppg, f_times = load_bvp_phys(gt_path, n_f, fps)
            p_native        = np.loadtxt(gt_path, dtype=np.float32)
            p_t_native      = np.arange(len(p_native)) / 64.0
        else:
            gt_ppg, f_times = load_gt_ubfc(gt_path, n_f, fps)

        # ── Already fully processed (all frames in CSV, NPZ missing) ─────────
        if resume_from >= n_f:
            tag   = 64 if is_phys else int(round(fps))
            npz_p = out_csv.replace('.csv', f'_ppg{tag}.npz')
            if not os.path.exists(npz_p):
                np.savez_compressed(
                    npz_p,
                    ppg_values = p_native   if is_phys else gt_ppg,
                    ppg_times  = p_t_native if is_phys else f_times,
                    ppg_hz     = 64.0       if is_phys else fps,
                    subject_id = sid,
                )
            cap.release()
            return (sid, 'SKIP', 0)

        # ── Seek to resume point ──────────────────────────────────────────────
        if resume_from > 0:
            cap.set(cv2.CAP_PROP_POS_FRAMES, resume_from)

        mesh     = mp.solutions.face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1)
        skin_t   = {}
        pool_cal = []
        cal_done = False
        states   = {'none': {p: _fresh_patch_states() for p in PATCH_NAMES}}
        states['none']['IPPC'] = {}

        # Only the current batch lives in RAM — flushed every CHECKPOINT_EVERY frames
        rows  = []
        n_new = 0
        for i in range(resume_from, n_f):
            ret, bgr = cap.read()
            if not ret:
                break
            row, _, cal_done = _process_frame(
                i, bgr, f_times[i], gt_ppg[i], sid,
                'ubfc', {'task': task} if is_phys else {},
                mesh, skin_t, pool_cal, cal_done, 0.0, states,
            )
            rows.append(row)
            n_new += 1
            if len(rows) % CHECKPOINT_EVERY == 0:
                pd.DataFrame(rows).to_csv(out_csv, mode=write_mode, header=write_header, index=False)
                write_mode   = 'a'     # all subsequent writes are appends
                write_header = False
                rows = []              # discard batch — it's on disk

        cap.release()
        mesh.close()
        del mesh, skin_t, pool_cal, states
        gc.collect()

        if n_new == 0 and not appending:
            return (sid, 'FAIL: no rows', 0)

        # Flush remaining rows (tail batch smaller than CHECKPOINT_EVERY)
        if rows:
            pd.DataFrame(rows).to_csv(out_csv, mode=write_mode, header=write_header, index=False)

        tag   = 64 if is_phys else int(round(fps))
        npz_p = out_csv.replace('.csv', f'_ppg{tag}.npz')
        np.savez_compressed(
            npz_p,
            ppg_values = p_native   if is_phys else gt_ppg,
            ppg_times  = p_t_native if is_phys else f_times,
            ppg_hz     = 64.0       if is_phys else fps,
            subject_id = sid,
        )
        return (sid, 'OK', n_new)

    except Exception:
        return (sid, f'FAIL: {traceback.format_exc()}', 0)


# ──────────────────────────────────────────────────────────────────────────────
# Job builder
# ──────────────────────────────────────────────────────────────────────────────

def _build_jobs():
    os.makedirs(UBFC_CSV_DIR, exist_ok=True)
    jobs = []

    # UBFC-rPPG
    for e in sorted(os.listdir(UBFC_RPPG_RAW_ROOT)):
        if e.startswith(UBFC_RPPG_SUBJECT_PREFIX):
            sn  = int(e[len(UBFC_RPPG_SUBJECT_PREFIX):])
            sid = sn + UBFC_RPPG_SID_OFFSET
            jobs.append((sid,
                         os.path.join(UBFC_RPPG_RAW_ROOT, e),
                         os.path.join(UBFC_CSV_DIR, f'ubfc_rppg_s{sn:02d}.csv'),
                         False, False, 0, sn))

    # UBFC-PHYS
    for e in sorted(os.listdir(UBFC_PHYS_RAW_ROOT)):
        if e.startswith('s'):
            sn  = int(e[1:])
            sid = sn + UBFC_PHYS_SID_OFFSET
            for t in UBFC_PHYS_TASKS:
                jobs.append((sid,
                             os.path.join(UBFC_PHYS_RAW_ROOT, e),
                             os.path.join(UBFC_CSV_DIR, f'ubfc_phys_s{sn:02d}_T{t}.csv'),
                             False, True, t, sn))
    return jobs


# ──────────────────────────────────────────────────────────────────────────────
# Entry point  — MUST be guarded for Windows spawn safety
# ──────────────────────────────────────────────────────────────────────────────

def main():
    jobs     = _build_jobs()
    n_workers = max(1, min(PARSE_WORKERS, len(jobs)))

    print(f"\nStandalone UBFC Parser — Multi-Process Stability Mode")
    print(f"  workers={n_workers} (maxtasksperchild=1)  "
          f"jobs={len(jobs)}  checkpoint_every={CHECKPOINT_EVERY}")

    # spawn = fresh interpreter per worker, no fork, no shared TFLite state
    # maxtasksperchild=1 = worker exits after each subject → OS frees all memory
    ctx = multiprocessing.get_context('spawn')
    n_ok = n_fail = n_skip = 0

    with ctx.Pool(n_workers, maxtasksperchild=1) as pool:
        bar = tqdm(total=len(jobs), desc="Subjects")
        try:
            for sid, status, n_new in pool.imap_unordered(worker_ubfc, jobs):
                if status == 'OK':
                    n_ok += 1
                    tqdm.write(f"  [OK]   sid={sid}  +{n_new} frames")
                elif status == 'SKIP':
                    n_skip += 1
                else:
                    n_fail += 1
                    tqdm.write(f"  [FAIL] sid={sid}: {status}")
                bar.update(1)
                bar.set_postfix(ok=n_ok, skip=n_skip, fail=n_fail)
        except KeyboardInterrupt:
            print("\n[INTERRUPTED] Terminating workers — partial CSVs are safe to resume.")
            pool.terminate()
            pool.join()
        finally:
            bar.close()

    print(f"\nDone — OK={n_ok}  SKIP={n_skip}  FAIL={n_fail}")


if __name__ == '__main__':
    main()

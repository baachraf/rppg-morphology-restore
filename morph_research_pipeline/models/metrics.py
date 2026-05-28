"""
models/metrics.py — Morphological Feature Extraction
=====================================================
Provides cycle-level feature extraction used for:
  - Auxiliary head supervision labels (training)
  - IPA metric in evaluation
  - Notch position detection via baseline-deviation method

All functions accept a 1-D numpy array of length 256 (normalised to [0,1]).

CHANGELOG 2026-05-13:
  - Completely replaced d2-based notch detection with baseline-deviation method
  - Notch defined as: maximum positive deviation from a straight line drawn
    from systolic peak to cycle end, normalised by signal amplitude
"""

import numpy as np


def _rectify_peak(cycle: np.ndarray) -> int:
    """Find the systolic peak position, handling misaligned cycles."""
    pk = int(np.argmax(cycle))
    n = len(cycle)

    if pk <= 5 or pk >= n - 20:
        # Peak at boundary — cycle is shifted. Find true systolic peak
        # in the first 40% of the cycle (real PPG peak is at 15-25%).
        mid = max(n // 3, 30)
        pk = int(np.argmax(cycle[:mid]))
    return pk


def notch_index(cycle: np.ndarray, min_deviation: float = 0.02,
                localization_thresh: float = 1.2) -> int:
    """
    Locate the dicrotic notch using baseline-deviation method.

    Method:
      1. Find systolic peak (with wrap-around handling)
      2. Define descending limb [peak, ~85% of cycle]
      3. Draw a straight line from peak value to end-of-descending-limb value
      4. The notch is the point of maximum positive deviation above this line
      5. Two checks:
         a. deviation > min_deviation * cycle_amplitude (real feature)
         b. max_deviation / mean_deviation > localization_thresh
            (sharp/localized notch, not broad sinusoidal bulge)

    Returns sample index or -1 if no significant notch found.
    """
    amp = float(np.max(cycle) - np.min(cycle))
    if amp < 1e-8:
        return -1

    pk = _rectify_peak(cycle)
    end = int(len(cycle) * 0.85)
    if end <= pk + 10:
        return -1

    y = cycle[pk:end].astype(np.float64)
    baseline = np.linspace(y[0], y[-1], len(y))
    deviation = y - baseline

    max_dev = float(np.max(deviation))
    if max_dev < min_deviation * amp:
        return -1

    mean_abs_dev = float(np.mean(np.abs(deviation)))
    if mean_abs_dev > 1e-12:
        localization = max_dev / mean_abs_dev
        if localization < localization_thresh:
            return -1

    idx_in_segment = int(np.argmax(deviation))
    return pk + idx_in_segment


def notch_confidence(cycle: np.ndarray) -> float:
    """
    Returns 0-1 confidence score for the presence of a dicrotic notch.

    Uses same baseline-deviation method as notch_index but returns a
    continuous score combining deviation magnitude and localization.

    0.0 = smooth monotonic descent (sinusoidal/no notch)
    0.0-0.5 = weak possible notch
    0.5-1.0 = clear notch
    """
    amp = float(np.max(cycle) - np.min(cycle))
    if amp < 1e-8:
        return 0.0

    pk = _rectify_peak(cycle)
    end = int(len(cycle) * 0.85)
    if end <= pk + 10:
        return 0.0

    y = cycle[pk:end].astype(np.float64)
    baseline = np.linspace(y[0], y[-1], len(y))
    deviation = y - baseline

    max_dev = float(np.max(deviation))
    norm_dev = max_dev / (amp + 1e-8)

    mean_abs_dev = float(np.mean(np.abs(deviation)))
    if mean_abs_dev > 1e-12:
        localization = max_dev / mean_abs_dev
    else:
        localization = 1.0

    dev_score = np.clip(norm_dev / 0.06, 0.0, 1.0)
    loc_score = np.clip((localization - 0.8) / 1.2, 0.0, 1.0)
    return float(dev_score * loc_score)


def compute_ipa(cycle: np.ndarray) -> float:
    """
    Inflection Point Area (IPA) — PaPaGei foundation model metric.

    IPA = systolic_area / (systolic_area + diastolic_area)
    where the split point is the dicrotic notch.
    Returns 0.5 (neutral) if no notch is detected.

    Clinical reference: real PPG ~ 0.55–0.65; rPPG (no notch) ~ 0.90–1.0
    """
    idx = notch_index(cycle)
    if idx < 0:
        return 0.5
    systolic = float(np.trapz(cycle[:idx]))
    diastolic = float(np.trapz(cycle[idx:]))
    denom = systolic + diastolic
    if denom < 1e-6:
        return 0.5
    return systolic / denom


def compute_rise_time(cycle: np.ndarray) -> float:
    """
    Normalised systolic rise time: sample index of peak / 256.

    Real PPG: rapid systolic upstroke, peak at ~15–25% of cycle.
    rPPG (sinusoidal): peak at ~50%.

    Returns value in [0, 1].
    """
    return float(np.argmax(cycle)) / len(cycle)


def extract_morpho_labels(cycle: np.ndarray) -> np.ndarray:
    """
    Extract the 3-element morphological label vector for aux head supervision.

    Returns np.ndarray of shape (3,) with values in [0, 1]:
      [0] notch_pos  — normalised notch sample index (0.5 if no notch)
      [1] ipa        — Inflection Point Area
      [2] rise_time  — normalised systolic peak position
    """
    idx = notch_index(cycle)
    notch_pos = float(idx) / len(cycle) if idx >= 0 else 0.5
    return np.array([notch_pos, compute_ipa(cycle), compute_rise_time(cycle)],
                    dtype=np.float32)


def batch_morpho_labels(cycles: np.ndarray) -> np.ndarray:
    """
    Apply extract_morpho_labels to a batch of cycles.

    cycles: (N, 256) numpy array
    Returns: (N, 3) numpy array
    """
    return np.stack([extract_morpho_labels(c) for c in cycles], axis=0)

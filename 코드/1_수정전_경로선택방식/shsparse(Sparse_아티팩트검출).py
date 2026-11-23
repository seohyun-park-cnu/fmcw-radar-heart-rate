'''jysparse - Sparse Movement Cancellation for Radar-based Heartbeat Extraction'''
# Made by Jaeyoung Choi
# Version 1.0
#
# Theoretical Background
# ─────────────────────────────────────────────────────────────────────────────
# Short-term body movement artifacts in FMCW radar phase signals are SPARSE
# in the time domain: they occur infrequently but with large instantaneous
# magnitude changes. This module exploits that sparsity to detect and suppress
# movement noise *before* the Moving Average (MA) filter stage, improving
# heartbeat estimation accuracy in dynamic environments (e.g., driving).
#
# Insertion point in the existing pipeline:
#   DACM → SMA #1 (trend removal) → dacm_diff
#     ↓
#   [NEW] detect_sparse_movement()  →  preprocess_signal()
#     ↓
#   SMA #2 (smoothing) → BPF → STFT → HR estimation
#
# Design principles for IoT / real-time use
# ─────────────────────────────────────────────────────────────────────────────
# • No iterative optimization — only sliding-window statistics (O(n·W))
# • Adaptive MAD-based threshold: robust to outliers, no manual tuning per test
# • Linear interpolation: avoids spectral leakage artifacts from hard zeroing
# • All parameters are exposed for easy sensitivity analysis

import numpy as np
from jymas import sma


# ─────────────────────────────────────────────────────────────────────────────
# Step 1: Sparse Movement Detection
# ─────────────────────────────────────────────────────────────────────────────

def detect_sparse_movement(signal, fps=20, win_size=5,
                           threshold_factor=3.0, min_gap=3):
    """
    Identify short-term body movement artifacts in a trend-removed DACM signal.

    Algorithm
    ---------
    ①  First-order difference  →  highlights sudden phase jumps
    ②  Sliding-window RMS energy of the derivative
    ③  Adaptive threshold = median(energy) + k × 1.4826 × MAD(energy)
           1.4826: normalisation constant so that MAD ≈ std for Gaussian data
           k (threshold_factor): lower → more sensitive, higher → conservative
    ④  Binary mask: True where local energy exceeds threshold
    ⑤  Gap-filling: merge detections separated by ≤ min_gap samples
    ⑥  Guard band: extend each flagged segment by ±1 sample (edge transitions)

    Parameters
    ----------
    signal           : 1-D ndarray   dacm_diff  (= dacm − SMA(dacm))
    fps              : int            sampling rate in frames/sec  (default 20)
    win_size         : int            sliding window length [samples]
                                      recommended: fps//4 ~ fps//2  (0.25–0.5 s)
    threshold_factor : float          k in the MAD threshold formula
                                      typical range: 2.5–4.0
    min_gap          : int            maximum gap [samples] between two flagged
                                      segments that will be merged into one

    Returns
    -------
    mask : bool 1-D ndarray, True = movement artifact present
    """
    length = len(signal)
    half_w = win_size // 2

    # ① First-order difference (instantaneous rate of change)
    diff    = np.empty(length)
    diff[0] = 0.0
    diff[1:] = np.diff(signal)

    # ② Sliding-window RMS energy of the derivative
    energy = np.empty(length)
    for i in range(length):
        lo = max(0, i - half_w)
        hi = min(length, i + half_w + 1)
        energy[i] = np.sqrt(np.mean(diff[lo:hi] ** 2))

    # ③ Adaptive MAD-based threshold
    med = np.median(energy)
    mad = np.median(np.abs(energy - med))
    thr = med + threshold_factor * 1.4826 * mad

    # ④ Initial binary mask
    mask = (energy > thr)

    # ⑤ Gap-filling: merge segments closer than min_gap samples
    i = 0
    while i < length:
        if mask[i]:
            # locate end of current flagged segment
            j = i + 1
            while j < length and mask[j]:
                j += 1
            # scan forward for the next flagged segment
            k = j
            while k < length and not mask[k]:
                k += 1
            # if gap is short enough, fill it
            if k < length and (k - j) <= min_gap:
                mask[j:k] = True
            i = k
        else:
            i += 1

    # ⑥ Guard band: extend by ±1 sample to cover signal transition edges
    guard        = np.zeros(length, dtype=bool)
    guard[1:]   |= mask[:-1]
    guard[:-1]  |= mask[1:]
    mask        |= guard

    return mask


# ─────────────────────────────────────────────────────────────────────────────
# Step 2: Signal Suppression / Interpolation
# ─────────────────────────────────────────────────────────────────────────────

def preprocess_signal(signal, mask, method='interpolate', max_interp_len=None):
    """
    Remove movement artifacts identified by detect_sparse_movement().

    Two suppression strategies:
    • 'zero'        — set flagged samples to 0 (fast, may introduce spectral leakage)
    • 'interpolate' — linearly interpolate from the nearest clean boundaries
                      (recommended: preserves signal continuity and reduces
                       spectral leakage from abrupt transitions)

    Parameters
    ----------
    signal        : 1-D ndarray   input signal  (dacm_diff)
    mask          : bool 1-D arr  from detect_sparse_movement()
    method        : str           'zero' or 'interpolate'
    max_interp_len: int or None   if a flagged segment is longer than this,
                                  fall back to zero-fill to avoid unreliable
                                  long interpolations.  None = always interpolate.

    Returns
    -------
    cleaned : 1-D ndarray, same length as input, artifacts replaced
    """
    length  = len(signal)
    cleaned = signal.copy()

    # ── zero suppression ────────────────────────────────────────────────────
    if method == 'zero':
        cleaned[mask] = 0.0
        return cleaned

    # ── linear interpolation ─────────────────────────────────────────────────
    i = 0
    while i < length:
        if not mask[i]:
            i += 1
            continue

        seg_start = i
        while i < length and mask[i]:
            i += 1
        seg_end = i          # first clean sample *after* the flagged segment
        seg_len = seg_end - seg_start

        # fall back to zero-fill if segment is too long to interpolate reliably
        if max_interp_len is not None and seg_len > max_interp_len:
            cleaned[seg_start:seg_end] = 0.0
            continue

        left_val  = cleaned[seg_start - 1] if seg_start > 0      else 0.0
        right_val = cleaned[seg_end]        if seg_end   < length else 0.0

        if seg_start == 0:
            # segment at signal start: replicate right boundary
            cleaned[seg_start:seg_end] = right_val
        elif seg_end == length:
            # segment at signal end: replicate left boundary
            cleaned[seg_start:seg_end] = left_val
        else:
            # interior segment: linear ramp between boundaries
            cleaned[seg_start:seg_end] = np.linspace(
                left_val, right_val, seg_len + 2)[1:-1]

    return cleaned


# ─────────────────────────────────────────────────────────────────────────────
# Step 3: Integrated Heartbeat Extraction Pipeline
# ─────────────────────────────────────────────────────────────────────────────

def main_heartrate_process(
        target,
        fps              = 20,
        measuretime      = 60,
        ma_size          = 2,
        bp_low           = 0.8,           # bandpass low  cutoff [Hz]
        bp_high          = 2.1,           # bandpass high cutoff [Hz]
        stft_W           = 100,           # STFT window length [samples]
        stft_L           = 99,            # STFT overlap      [samples]
        sparse_win       = 5,             # energy detection window [samples]
        sparse_thresh    = 3.0,           # MAD threshold multiplier
        suppress_method  = 'interpolate', # 'zero' or 'interpolate'
        max_interp_len   = None,
        use_sparse       = True,          # set False for baseline comparison
        kalman_R         = 0.01):
    """
    Integrated pipeline:
        DACM → Trend Removal → [Sparse Cancellation] → MA → BPF → STFT → HR

    Extends the existing for_article_MAs_for.py processing flow by inserting
    Sparse Movement Cancellation between the first SMA (trend removal) and
    the second SMA (heartbeat smoothing).

    Parameters
    ----------
    target          : complex 1-D ndarray
                      Extracted radar target signal after MTI + Peak Search
                      (same `target` array as in for_article_MAs_for.py)
    fps             : int     frame rate [fps]         (default 20)
    measuretime     : float   recording duration [s]   (default 60)
    ma_size         : int     SMA window size parameter (passed to sma())
    bp_low/bp_high  : float   bandpass filter cutoffs [Hz]
    stft_W/stft_L   : int     STFT window and overlap [samples]
    sparse_win      : int     sliding window size for energy calculation
    sparse_thresh   : float   k in "median + k·1.4826·MAD"
    suppress_method : str     'zero' or 'interpolate'
    max_interp_len  : int or None  max segment length for interpolation
    use_sparse      : bool    False → skip sparse cancellation (baseline)
    kalman_R        : float   Kalman filter measurement noise covariance

    Returns
    -------
    dict
        'hr_estim'        : 1-D ndarray  heartbeat estimate [BPM], Kalman-filtered
        'hr_time'         : 1-D ndarray  time axis for hr_estim [s]
        'hr_bpm'          : 1-D ndarray  BPM axis for the STFT
        'dacm'            : 1-D ndarray  raw DACM phase signal
        'dacm_diff'       : 1-D ndarray  trend-removed residual (before sparse)
        'dacm_diff_clean' : 1-D ndarray  after sparse artifact suppression
        'movement_mask'   : bool ndarray True = movement artifact at that sample
        'detection_ratio' : float        fraction of signal flagged as movement
    """
    from jyvital7_1 import (dacm_phase_unwrapping, bandpass_filter,
                             phase_stft, stft_time, stft_bpm,
                             heartbeat_estimation)

    numFrame = len(target)

    # ── Stage 1: DACM Phase Unwrapping ───────────────────────────────────────
    dacm = dacm_phase_unwrapping(target)

    # ── Stage 2: Trend Removal  (SMA pass #1) ────────────────────────────────
    dacm_trend = sma(dacm, size=ma_size)
    dacm_diff  = dacm - dacm_trend        # residual: HR signal + movement noise

    # ── Stage 3: Sparse Movement Cancellation (NEW) ───────────────────────────
    if use_sparse:
        movement_mask   = detect_sparse_movement(
            dacm_diff,
            fps              = fps,
            win_size         = sparse_win,
            threshold_factor = sparse_thresh
        )
        dacm_diff_clean = preprocess_signal(
            dacm_diff, movement_mask,
            method         = suppress_method,
            max_interp_len = max_interp_len
        )
    else:
        movement_mask   = np.zeros(numFrame, dtype=bool)
        dacm_diff_clean = dacm_diff.copy()

    detection_ratio = float(np.sum(movement_mask)) / numFrame

    # ── Stage 4: Smoothing MA  (SMA pass #2, on cleaned signal) ──────────────
    dacm_diff_ma = sma(dacm_diff_clean, size=ma_size)

    # ── Stage 5: Bandpass Filter  (0.8–2.1 Hz, 4th-order Butterworth) ─────────
    dacm_diff_hr = bandpass_filter(
        dacm_diff_ma, lowcut=bp_low, highcut=bp_high, fps=fps)

    # ── Stage 6: STFT + Heartbeat Estimation ──────────────────────────────────
    hr_stft  = phase_stft(dacm_diff_hr, stft_W, stft_L, numFrame)
    hr_time  = stft_time(numFrame, stft_W, stft_L, fps)
    hr_bpm   = stft_bpm(numFrame, measuretime, fps)
    hr_estim = heartbeat_estimation(
        hr_stft, hr_bpm, len(hr_time), kalmanR=kalman_R)

    return {
        'hr_estim'        : hr_estim,
        'hr_time'         : hr_time,
        'hr_bpm'          : hr_bpm,
        'dacm'            : dacm,
        'dacm_diff'       : dacm_diff,
        'dacm_diff_clean' : dacm_diff_clean,
        'movement_mask'   : movement_mask,
        'detection_ratio' : detection_ratio,
    }

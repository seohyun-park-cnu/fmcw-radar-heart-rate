"""
external_baselines_v2.py
──────────────────────────────────────────────────────────────────────────────
External baseline comparison (full rewrite): 9 baselines + 2 proposed systems.

Baselines:
  B1  EDACM          — Wang et al., IEEE Trans. I&M (2014)
  B2  VMD            — Qu et al., IEEE Sensors J. (2023)
  B3  DWT            — Li & Lin, IEEE Trans. MTT (2018)
  B4  EEMD           — Xu et al., Sensors (2022)
  B5  RLS            — Huang et al., Micromachines (2022)
  B6  Adaptive Notch — Tazen et al., IEEE Access (2023)
  B7  SSA            — Singular Spectrum Analysis
  B8  CEEMDAN        — Torres et al., ICASSP (2011); Measurement (2025)
  B9  EWT            — Gilles (2013); Measurement (2025)

Proposed:
  Batch  — 3-way gate routing (Raw / MA_old / Sparse_new)
  RT     — SPARQRealtime frame-by-frame

Outputs:
  - paper/external_baselines_v2.json
  - Console report with overall, cross-type, per-participant, per-category tables
  - Wilcoxon signed-rank tests
  - Consistency analysis

Usage:
  cd HB_Revision
  python external_baselines_v2.py
"""

import os, sys, json, time, warnings, datetime
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
warnings.filterwarnings('ignore')

import numpy as np
from scipy import stats
from scipy.io import loadmat
from scipy.signal import welch, iirnotch, filtfilt as scipy_filtfilt
from vmdpy import VMD
import pywt
from PyEMD import EEMD as PyEEMD
from PyEMD import CEEMDAN as PyCEEMDAN
import ewtpy

from jyvital7_1 import (MTI, dacm_phase_unwrapping, bandpass_filter,
                         phase_stft, stft_time, stft_bpm, heartbeat_estimation)
from jymas import sma
from shsparse import detect_sparse_movement, preprocess_signal
from sparq_realtime import SPARQRealtime

# ─────────────────────────────── Config ───────────────────────────────────────
numFrame    = 1200
measuretime = 60
fps         = numFrame // measuretime   # 20
quarter     = 4
radarFpsQ   = fps // quarter            # 5
ubpulseFpsQ = 256 // quarter            # 64
MAX_INTERP  = 20
DET_GUARD   = 0.03
STFT_W, STFT_L = 100, 99

K1_OLD, K2_OLD = 2, 2
K1_NEW, K2_NEW = 5, 2

# Gate thresholds (from eval_3way_gate.py)
THR_MAD_STD      = 0.508
THR_HR_POWER     = 0.042
THR_DETREND_5V2  = 1.509

BEST_SPARSE_PARAMS = {
    'static':         {'thresh': 4.0, 'win': 3, 'guard': False},
    'cough':          {'thresh': 6.0, 'win': 3, 'guard': False},
    'disturb':        {'thresh': 4.0, 'win': 3, 'guard': False},
    'transition':     {'thresh': 3.0, 'win': 3, 'guard': False},
    'water':          {'thresh': 3.0, 'win': 7, 'guard': False},
    'arm':            {'thresh': 2.5, 'win': 3, 'guard': False},
    'head':           {'thresh': 5.0, 'win': 3, 'guard': True},
    'swing':          {'thresh': 7.0, 'win': 7, 'guard': True},
    'shoulder_shrug': {'thresh': 3.0, 'win': 7, 'guard': False},
}

CATS_ORDER = ['static', 'arm', 'head', 'cough', 'shoulder_shrug',
              'disturb', 'swing', 'transition', 'water']
TRIALS     = ['1st', '2nd', '3rd']
DATA_DIR   = './data/Participant Data'
PERSONS    = ['SH', 'BG', 'JY', 'HJ', 'DY', 'ET', 'JB', 'JH', 'JHL', 'YW']

SIGNAL_TYPES = {
    'SH': 'Standard', 'BG': 'Standard', 'JHL': 'Standard', 'YW': 'Standard',
    'JY': 'Drift', 'JB': 'Drift',
    'HJ': 'Clean', 'ET': 'Clean',
    'DY': 'Mid-drift', 'JH': 'Mixed',
}

OUT_DIR = './paper'
os.makedirs(OUT_DIR, exist_ok=True)

hr_time_axis = stft_time(numFrame, STFT_W, STFT_L, fps)
hr_bpm_axis  = stft_bpm(numFrame, measuretime, fps)

# ─────────────────────────────── Common Helpers ───────────────────────────────
def extract_gt(mat_hr):
    bpm = np.squeeze(mat_hr['saveDataHeartRateUbpulse1'], axis=0).astype(float)
    L = len(bpm); ss = 100 // fps
    bpm[bpm > 150] = np.nan
    vi = np.where(~np.isnan(bpm))[0]; ni = np.where(np.isnan(bpm))[0]
    if len(ni) > 0 and len(vi) > 0: bpm[ni] = np.interp(ni, vi, bpm[vi])
    m = np.zeros(L)
    for i in range(256 * ss, L): m[i] = np.mean(bpm[i - 256 * ss:i])
    q = m[np.arange(0, L, ubpulseFpsQ)]
    return q[quarter * radarFpsQ:]

def extract_target(d):
    """Target cell fixed for the whole recording.
    range bin = argmax of the time-averaged, channel-averaged |MTI|;
    angle bin = argmax of the time-averaged |angle FFT| at that range bin."""
    rfft = MTI(d[:, [6, 4, 3, 2], :]); rm = np.mean(np.abs(rfft), 1)
    rp = int(np.argmax(rm.mean(0)))
    dm = np.fft.fftshift(np.fft.fft(rfft, axis=1), 1)
    dp = int(np.argmax(np.abs(dm[:, :, rp]).mean(0)))
    return dm[:, dp, rp]

def downsample_hr(e):
    return e[np.arange(0, len(e), radarFpsQ)][:-1]

def calc_mae(x, y):
    n = min(len(x), len(y))
    if n == 0: return float('nan')
    x = np.array(x[:n], dtype=float); y = np.array(y[:n], dtype=float)
    valid = ~(np.isnan(x) | np.isnan(y))
    if np.sum(valid) == 0: return float('nan')
    return float(np.abs(x[valid] - y[valid]).mean())

def common_backend(signal, kalman_r=0.10):
    """BPF → STFT → Kalman → downsample. Common to all baselines."""
    filtered = bandpass_filter(signal, 0.8, 2.1, fps)
    stft_out = phase_stft(filtered, STFT_W, STFT_L, numFrame)
    hr_estim = heartbeat_estimation(stft_out, hr_bpm_axis, len(hr_time_axis), kalmanR=kalman_r)
    return downsample_hr(hr_estim)

def compute_gate_features(dacm):
    """Compute features for 3-way gate routing."""
    d = np.diff(dacm)
    diff_mad = float(np.median(np.abs(d - np.median(d))))
    diff_std = float(np.std(d))
    mad_std_ratio = diff_mad / diff_std if diff_std > 0 else 0.0

    N = len(dacm)
    try:
        freqs, psd = welch(dacm, fs=fps, nperseg=min(256, N), noverlap=min(128, N//2))
        total_power = np.sum(psd)
        if total_power > 0:
            hr_mask = (freqs >= 0.8) & (freqs <= 2.1)
            hr_power_ratio = float(np.sum(psd[hr_mask]) / total_power)
        else:
            hr_power_ratio = 0.0
    except Exception:
        hr_power_ratio = 0.0

    sma5_det = dacm - sma(dacm, size=5)
    sma2_det = dacm - sma(dacm, size=2)
    std5 = float(np.std(sma5_det))
    std2 = float(np.std(sma2_det))
    detrend_ratio_5v2 = std5 / std2 if std2 > 0 else 1.0

    return {
        'mad_std_ratio': mad_std_ratio,
        'hr_power_ratio': hr_power_ratio,
        'detrend_ratio_5v2': detrend_ratio_5v2,
    }

# ─────────────────────────── B1: EDACM ───────────────────────────────────────
def edacm_phase_unwrapping(target):
    """Extended DACM: I/Q cross-differentiation for phase rate estimation."""
    I = target.real
    Q = target.imag
    N = len(target)
    phase_rate = np.zeros(N)
    for i in range(2, N):
        dI1 = I[i] - I[i-1]
        dQ1 = Q[i] - Q[i-1]
        dI2 = I[i] - I[i-2]
        dQ2 = Q[i] - Q[i-2]
        num = I[i] * (dQ1 + dQ2) - Q[i] * (dI1 + dI2)
        den = I[i]**2 + Q[i]**2
        if den > 1e-12:
            phase_rate[i] = num / (2.0 * den)
        else:
            phase_rate[i] = 0.0
    phase = np.cumsum(phase_rate)
    return phase

def run_edacm(target):
    """B1: EDACM phase → BPF → STFT → Kalman."""
    phase = edacm_phase_unwrapping(target)
    return common_backend(phase)

# ─────────────────────────── B2: VMD (improved) ──────────────────────────────
def run_vmd(target):
    """B2: DACM → VMD(K=4, α=2000) → HR-band mode sum → BPF → STFT → Kalman."""
    dacm = dacm_phase_unwrapping(target)

    alpha = 2000
    tau = 0
    K = 4
    DC = 0
    init = 1
    tol = 1e-7

    try:
        u, u_hat, omega = VMD(dacm, alpha, tau, K, DC, init, tol)
    except Exception:
        return common_backend(dacm)

    # Improved mode selection: sum modes with >30% HR-band energy ratio
    hr_band = (0.8, 2.1)
    selected_modes = []
    for i in range(K):
        freqs, psd = welch(u[i], fs=fps, nperseg=min(256, len(u[i])), noverlap=128)
        total_e = np.sum(psd)
        if total_e > 0:
            hr_mask = (freqs >= hr_band[0]) & (freqs <= hr_band[1])
            hr_ratio = np.sum(psd[hr_mask]) / total_e
            if hr_ratio >= 0.30:
                selected_modes.append(i)

    if len(selected_modes) == 0:
        # Fallback: pick mode with highest HR-band energy
        best_mode, best_energy = 0, -1
        for i in range(K):
            freqs, psd = welch(u[i], fs=fps, nperseg=min(256, len(u[i])), noverlap=128)
            hr_mask = (freqs >= hr_band[0]) & (freqs <= hr_band[1])
            hr_energy = np.sum(psd[hr_mask])
            if hr_energy > best_energy:
                best_energy = hr_energy
                best_mode = i
        selected_modes = [best_mode]

    combined = np.sum(u[selected_modes], axis=0)
    return common_backend(combined)

# ─────────────────────────── B3: DWT (BayesShrink) ───────────────────────────
def _bayes_shrink_threshold(detail_coeffs, sigma_noise):
    """BayesShrink: level-adaptive soft threshold."""
    sigma_y2 = np.var(detail_coeffs)
    sigma_x2 = max(sigma_y2 - sigma_noise**2, 0)
    if sigma_x2 == 0:
        return np.max(np.abs(detail_coeffs))  # threshold everything
    return sigma_noise**2 / np.sqrt(sigma_x2)

def run_dwt(target):
    """B3: DACM → DWT(db4, level=4) → BayesShrink/VisuShrink (best) → BPF → STFT → Kalman."""
    dacm = dacm_phase_unwrapping(target)

    wavelet = 'db4'
    level = 4
    coeffs = pywt.wavedec(dacm, wavelet, level=level)

    # Noise estimate from finest detail coefficients
    sigma = np.median(np.abs(coeffs[-1])) / 0.6745

    # --- VisuShrink ---
    thr_visu = sigma * np.sqrt(2 * np.log(len(dacm)))
    coeffs_visu = [coeffs[0]]
    for i in range(1, len(coeffs)):
        coeffs_visu.append(pywt.threshold(coeffs[i], thr_visu, mode='soft'))
    denoised_visu = pywt.waverec(coeffs_visu, wavelet)[:len(dacm)]

    # --- BayesShrink ---
    coeffs_bayes = [coeffs[0]]
    for i in range(1, len(coeffs)):
        thr_b = _bayes_shrink_threshold(coeffs[i], sigma)
        coeffs_bayes.append(pywt.threshold(coeffs[i], thr_b, mode='soft'))
    denoised_bayes = pywt.waverec(coeffs_bayes, wavelet)[:len(dacm)]

    # Pick whichever gives lower BPF energy in non-HR bands (cleaner)
    # Simple heuristic: use BayesShrink as default (generally better)
    y_bayes = common_backend(denoised_bayes)
    y_visu = common_backend(denoised_visu)

    # Choose by checking which has less out-of-band energy
    freqs_b, psd_b = welch(denoised_bayes, fs=fps, nperseg=min(256, len(dacm)))
    freqs_v, psd_v = welch(denoised_visu, fs=fps, nperseg=min(256, len(dacm)))
    hr_mask_b = (freqs_b >= 0.8) & (freqs_b <= 2.1)
    hr_mask_v = (freqs_v >= 0.8) & (freqs_v <= 2.1)
    total_b = np.sum(psd_b); total_v = np.sum(psd_v)
    hr_ratio_b = np.sum(psd_b[hr_mask_b]) / total_b if total_b > 0 else 0
    hr_ratio_v = np.sum(psd_v[hr_mask_v]) / total_v if total_v > 0 else 0

    return y_bayes if hr_ratio_b >= hr_ratio_v else y_visu

# ─────────────────────────── B4: EEMD ────────────────────────────────────────
def run_eemd(target):
    """B4: DACM → EEMD(trials=50, noise=0.05) → HR-band IMF → BPF → STFT → Kalman."""
    dacm = dacm_phase_unwrapping(target)

    eemd = PyEEMD(trials=50, noise_width=0.05)
    try:
        imfs = eemd.eemd(dacm)
    except Exception:
        return common_backend(dacm)

    if imfs is None or len(imfs) == 0:
        return common_backend(dacm)

    # Select IMF with max HR-band energy
    hr_band = (0.8, 2.1)
    best_imf, best_energy = 0, -1
    for i in range(len(imfs)):
        freqs, psd = welch(imfs[i], fs=fps, nperseg=min(256, len(imfs[i])), noverlap=128)
        hr_mask = (freqs >= hr_band[0]) & (freqs <= hr_band[1])
        hr_energy = np.sum(psd[hr_mask])
        if hr_energy > best_energy:
            best_energy = hr_energy
            best_imf = i

    return common_backend(imfs[best_imf])

# ─────────────────────────── B5: RLS ─────────────────────────────────────────
def run_rls(target):
    """B5: DACM → RLS(order=8, λ=0.999, ref=SMA trend) → residual → BPF → STFT → Kalman."""
    dacm = dacm_phase_unwrapping(target)

    # Reference signal: low-frequency trend estimated by SMA(k=5)
    reference = sma(dacm, size=5)

    # RLS adaptive filter
    order = 8
    lam = 0.999
    delta = 1.0  # P initialization scale

    N = len(dacm)
    w = np.zeros(order)
    P = delta * np.eye(order)
    y_hat = np.zeros(N)

    for n in range(order, N):
        x = reference[n-order:n][::-1]  # tap-delay vector
        y_hat[n] = np.dot(w, x)
        e = dacm[n] - y_hat[n]  # error = residual (contains heartbeat)
        Px = P @ x
        denom = lam + x @ Px
        k = Px / denom
        w = w + k * e
        P = (P - np.outer(k, x @ P)) / lam

    residual = dacm - y_hat
    return common_backend(residual)

# ─────────────────────────── B6: Adaptive Notch ──────────────────────────────
def run_adaptive_notch(target):
    """B6: DACM → respiration rate estimation → cascade notch (1-5th harmonic) → BPF → STFT → Kalman."""
    dacm = dacm_phase_unwrapping(target)

    # Estimate respiration rate from DACM (0.1-0.5 Hz band)
    freqs, psd = welch(dacm, fs=fps, nperseg=min(512, len(dacm)), noverlap=256)
    resp_mask = (freqs >= 0.1) & (freqs <= 0.5)
    if np.any(resp_mask) and np.sum(psd[resp_mask]) > 0:
        resp_freqs = freqs[resp_mask]
        resp_psd = psd[resp_mask]
        resp_rate = resp_freqs[np.argmax(resp_psd)]
    else:
        resp_rate = 0.25  # default ~15 breaths/min

    # Cascade notch filter for harmonics that fall in HR band (0.7-2.2 Hz)
    filtered = dacm.copy()
    Q = 30.0
    for n in range(1, 6):
        harmonic_freq = resp_rate * n
        if 0.7 <= harmonic_freq <= 2.2:
            # Normalize frequency
            w0 = harmonic_freq / (fps / 2.0)
            if 0 < w0 < 1:
                b, a = iirnotch(w0, Q)
                filtered = scipy_filtfilt(b, a, filtered)

    return common_backend(filtered)

# ─────────────────────────── B7: SSA ─────────────────────────────────────────
def _ssa_antidiag_avg(mat, N):
    """Vectorized anti-diagonal averaging for SSA reconstruction."""
    L, K = mat.shape
    result = np.zeros(N)
    for n in range(N):
        i_start = max(0, n - K + 1)
        i_end = min(L, n + 1)
        result[n] = np.mean([mat[i, n - i] for i in range(i_start, i_end)])
    return result

def run_ssa(target):
    """B7: DACM → SSA(L=100, select HR-band components) → BPF → STFT → Kalman."""
    dacm = dacm_phase_unwrapping(target)
    N = len(dacm)
    L = min(100, N // 3)  # window length (~5s at 20Hz)
    K = N - L + 1

    # Build trajectory (Hankel) matrix using stride tricks (fast)
    from numpy.lib.stride_tricks import as_strided
    X = as_strided(dacm, shape=(L, K),
                   strides=(dacm.strides[0], dacm.strides[0])).copy()

    # Truncated SVD (only first 20 components needed)
    n_comp = min(20, min(L, K))
    U, s, Vt = np.linalg.svd(X, full_matrices=False)
    U = U[:, :n_comp]; s = s[:n_comp]; Vt = Vt[:n_comp, :]

    # Select components with significant HR-band energy
    hr_band = (0.8, 2.1)
    selected = []
    comp_series = {}
    for c in range(n_comp):
        Xc = s[c] * np.outer(U[:, c], Vt[c, :])
        rc = _ssa_antidiag_avg(Xc, N)
        comp_series[c] = rc

        freqs, psd = welch(rc, fs=fps, nperseg=min(256, N), noverlap=min(128, N // 2))
        total_e = np.sum(psd)
        if total_e > 0:
            hr_mask = (freqs >= hr_band[0]) & (freqs <= hr_band[1])
            hr_ratio = np.sum(psd[hr_mask]) / total_e
            if hr_ratio >= 0.25:
                selected.append(c)

    if len(selected) == 0:
        # Fallback: use component with highest HR energy
        best_c, best_e = 0, -1
        for c in range(n_comp):
            rc = comp_series[c]
            freqs, psd = welch(rc, fs=fps, nperseg=min(256, N), noverlap=min(128, N // 2))
            hr_mask = (freqs >= hr_band[0]) & (freqs <= hr_band[1])
            hr_e = np.sum(psd[hr_mask])
            if hr_e > best_e:
                best_e = hr_e
                best_c = c
        selected = [best_c]

    # Reconstruct selected components
    # Sum selected component time series
    reconstructed = np.zeros(N)
    for c in selected:
        reconstructed += comp_series[c]

    return common_backend(reconstructed)

# ─────────────────────── B8: CEEMDAN ─────────────────────────────────────────
def run_ceemdan(target):
    """B8: DACM → CEEMDAN(trials=50, epsilon=0.05) → HR-band IMF sum → BPF → STFT → Kalman."""
    dacm = dacm_phase_unwrapping(target)

    ceemdan = PyCEEMDAN(trials=50, epsilon=0.05)
    try:
        imfs = ceemdan.ceemdan(dacm)
    except Exception:
        return common_backend(dacm)

    if imfs is None or len(imfs) == 0:
        return common_backend(dacm)

    # Select IMFs with significant HR-band energy (>= 30% ratio)
    hr_band = (0.8, 2.1)
    selected = []
    for i in range(len(imfs)):
        freqs, psd = welch(imfs[i], fs=fps, nperseg=min(256, len(imfs[i])), noverlap=128)
        total_e = np.sum(psd)
        if total_e > 0:
            hr_mask = (freqs >= hr_band[0]) & (freqs <= hr_band[1])
            hr_ratio = np.sum(psd[hr_mask]) / total_e
            if hr_ratio >= 0.30:
                selected.append(i)

    if len(selected) == 0:
        # Fallback: pick IMF with highest HR-band energy
        best_imf, best_energy = 0, -1
        for i in range(len(imfs)):
            freqs, psd = welch(imfs[i], fs=fps, nperseg=min(256, len(imfs[i])), noverlap=128)
            hr_mask = (freqs >= hr_band[0]) & (freqs <= hr_band[1])
            hr_energy = np.sum(psd[hr_mask])
            if hr_energy > best_energy:
                best_energy = hr_energy
                best_imf = i
        selected = [best_imf]

    combined = np.sum(imfs[selected], axis=0)
    return common_backend(combined)

# ─────────────────────── B9: EWT ─────────────────────────────────────────────
def run_ewt(target):
    """B9: DACM → EWT(N=8) → HR-band component sum → BPF → STFT → Kalman."""
    dacm = dacm_phase_unwrapping(target)

    try:
        ewt_coeffs, mfb, boundaries = ewtpy.EWT1D(dacm, N=8)
    except Exception:
        return common_backend(dacm)

    if ewt_coeffs is None or ewt_coeffs.shape[1] == 0:
        return common_backend(dacm)

    n_comp = ewt_coeffs.shape[1]

    # Select components with significant HR-band energy (>= 25% ratio)
    hr_band = (0.8, 2.1)
    selected = []
    for i in range(n_comp):
        comp = ewt_coeffs[:, i]
        freqs, psd = welch(comp, fs=fps, nperseg=min(256, len(comp)), noverlap=128)
        total_e = np.sum(psd)
        if total_e > 0:
            hr_mask = (freqs >= hr_band[0]) & (freqs <= hr_band[1])
            hr_ratio = np.sum(psd[hr_mask]) / total_e
            if hr_ratio >= 0.25:
                selected.append(i)

    if len(selected) == 0:
        # Fallback: pick component with highest HR-band energy
        best_comp, best_energy = 0, -1
        for i in range(n_comp):
            comp = ewt_coeffs[:, i]
            freqs, psd = welch(comp, fs=fps, nperseg=min(256, len(comp)), noverlap=128)
            hr_mask = (freqs >= hr_band[0]) & (freqs <= hr_band[1])
            hr_energy = np.sum(psd[hr_mask])
            if hr_energy > best_energy:
                best_energy = hr_energy
                best_comp = i
        selected = [best_comp]

    combined = np.sum(ewt_coeffs[:, selected], axis=1)
    return common_backend(combined)

# ─────────────────────── Proposed: Batch (3-way gate) ────────────────────────
def run_proposed_batch(target, cat):
    """Proposed Batch: 3-way gate → Raw / MA_old / Sparse_new → common backend."""
    dacm = dacm_phase_unwrapping(target)
    feats = compute_gate_features(dacm)

    # Gate routing
    if feats['mad_std_ratio'] > THR_MAD_STD:
        # Raw path
        sig = dacm.copy()
    elif (feats['hr_power_ratio'] > THR_HR_POWER
          or feats['detrend_ratio_5v2'] < THR_DETREND_5V2):
        # MA_old path
        dd = dacm - sma(dacm, size=K1_OLD)
        sig = sma(dd, size=K2_OLD)
    else:
        # Sparse_new path
        p = BEST_SPARSE_PARAMS[cat]
        dacm_c = dacm.copy()
        apply_sparse = True
        if p['guard']:
            mp = detect_sparse_movement(dacm_c, fps=fps, win_size=p['win'],
                                        threshold_factor=p['thresh'])
            if float(np.sum(mp)) / numFrame < DET_GUARD:
                apply_sparse = False
        if apply_sparse:
            mask = detect_sparse_movement(dacm_c, fps=fps, win_size=p['win'],
                                          threshold_factor=p['thresh'])
            dacm_c = preprocess_signal(dacm_c, mask, method='interpolate',
                                       max_interp_len=MAX_INTERP)
        dd = dacm_c - sma(dacm_c, size=K1_NEW)
        sig = sma(dd, size=K2_NEW)

    return common_backend(sig)

# ─────────────────────── Proposed: RT ────────────────────────────────────────
def run_proposed_rt(target):
    """Proposed RT: SPARQRealtime frame-by-frame processing."""
    rt = SPARQRealtime(fps=fps, kalman_R=0.10, sparse_thresh=4.0, sparse_win=3)
    bpm_per_frame = np.full(len(target), np.nan)
    for i, sample in enumerate(target):
        bpm = rt.process_frame(sample)
        if bpm is not None:
            bpm_per_frame[i] = bpm
    ds = np.arange(0, len(target), radarFpsQ)
    bpm_ds = bpm_per_frame[ds]
    if len(bpm_ds) > 0 and np.isnan(bpm_ds[-1]):
        bpm_ds = bpm_ds[:-1]
    return bpm_ds

# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════
if __name__ == '__main__':
    t_start = time.time()
    print("=" * 80)
    print("  External Baseline Comparison v2 (9 Baselines + 2 Proposed)")
    print("=" * 80)

    ALGO_NAMES = ['EDACM', 'VMD', 'DWT', 'EEMD', 'RLS', 'AdaptNotch', 'SSA',
                  'CEEMDAN', 'EWT', 'Batch', 'RT']
    ALGO_FUNCS = {
        'EDACM': lambda t, c: run_edacm(t),
        'VMD':   lambda t, c: run_vmd(t),
        'DWT':   lambda t, c: run_dwt(t),
        'EEMD':  lambda t, c: run_eemd(t),
        'RLS':   lambda t, c: run_rls(t),
        'AdaptNotch': lambda t, c: run_adaptive_notch(t),
        'SSA':   lambda t, c: run_ssa(t),
        'CEEMDAN': lambda t, c: run_ceemdan(t),
        'EWT':   lambda t, c: run_ewt(t),
        'Batch': lambda t, c: run_proposed_batch(t, c),
        'RT':    lambda t, c: run_proposed_rt(t),
    }

    # ── Build test cases ──
    TEST_CASES = []
    for person in PERSONS:
        for cat in CATS_ORDER:
            for tr in TRIALS:
                TEST_CASES.append({
                    'name': f'{cat}_{person}_{tr}', 'cat': cat, 'person': person,
                    'prefix': f'{DATA_DIR}/{cat}_{person}_{tr}',
                })

    # ── [1/4] Load data ──
    print(f"\n[1/4] Loading data ({len(TEST_CASES)} candidates)...")
    loaded = []
    for tc in TEST_CASES:
        try:
            mr  = loadmat(f"{tc['prefix']}_radar.mat")
            mhr = loadmat(f"{tc['prefix']}_ubpulse_heartbeat.mat")
            t   = extract_target(mr['saveDataRadar1'])
            gt  = extract_gt(mhr)
            if len(gt) == 0: raise ValueError("GT empty")
            loaded.append({'name': tc['name'], 'cat': tc['cat'],
                           'person': tc['person'], 'target': t, 'gt': gt})
        except Exception as e:
            print(f"  [SKIP] {tc['name']}: {e}")

    N = len(loaded)
    print(f"  {N} cases loaded")

    # ── [2/4] Run all algorithms ──
    print(f"\n[2/4] Running {len(ALGO_NAMES)} algorithms x {N} cases...")
    # results[i] = {'name': ..., 'cat': ..., 'person': ..., 'type': ..., 'mae_ALGO': ...}
    results = []
    mae_arrays = {algo: [] for algo in ALGO_NAMES}

    for i, d in enumerate(loaded):
        nm = d['name']; gt = d['gt']; target = d['target']; cat = d['cat']
        row = {'name': nm, 'cat': cat, 'person': d['person'],
               'type': SIGNAL_TYPES.get(d['person'], 'Unknown')}

        for algo in ALGO_NAMES:
            try:
                y = ALGO_FUNCS[algo](target, cat)
                m = calc_mae(gt, y)
            except Exception as e:
                m = float('nan')
            row[f'mae_{algo}'] = m
            mae_arrays[algo].append(m)

        results.append(row)

        if (i + 1) % 25 == 0 or i == N - 1:
            elapsed = time.time() - t_start
            print(f"  {i+1:3d}/{N} done  ({elapsed:.0f}s)")

    # Convert to numpy arrays
    for algo in ALGO_NAMES:
        mae_arrays[algo] = np.array(mae_arrays[algo])

    # ═══════════════════════════════════════════════════════════════════════════
    # [3/4] Analysis & Output
    # ═══════════════════════════════════════════════════════════════════════════
    print(f"\n[3/4] Analysis")
    SEP = "=" * 90
    DIV = "-" * 88

    # ── Overall Table ──
    print(f"\n{SEP}")
    print("  Table: Overall Performance Comparison")
    print(SEP)
    print(f"  {'Method':<16s} {'MAE':>7s} {'RMSE':>7s} {'Median':>7s} "
          f"{'<=5':>6s} {'<=10':>6s} {'<=15':>6s}")
    print(f"  {DIV}")

    for algo in ALGO_NAMES:
        arr = mae_arrays[algo]
        valid = arr[~np.isnan(arr)]
        if len(valid) == 0: continue
        mae_val = np.mean(valid)
        rmse_val = np.sqrt(np.mean(valid**2))
        median_val = np.median(valid)
        le5  = np.mean(valid <= 5) * 100
        le10 = np.mean(valid <= 10) * 100
        le15 = np.mean(valid <= 15) * 100
        marker = " ***" if algo in ('Batch', 'RT') else ""
        print(f"  {algo:<16s} {mae_val:7.2f} {rmse_val:7.2f} {median_val:7.2f} "
              f"{le5:5.1f}% {le10:5.1f}% {le15:5.1f}%{marker}")

    # ── Wilcoxon Signed-Rank Tests ──
    print(f"\n{SEP}")
    print("  Wilcoxon Signed-Rank Tests")
    print(SEP)

    baselines = ['EDACM', 'VMD', 'DWT', 'EEMD', 'RLS', 'AdaptNotch', 'SSA', 'CEEMDAN', 'EWT']
    proposed  = ['Batch', 'RT']

    for prop in proposed:
        print(f"\n  --- vs {prop} ---")
        print(f"  {'Baseline':<16s} {'p-value':>12s} {'sig':>4s} {'mean_diff':>10s} {'median_diff':>12s}")
        print(f"  {DIV}")
        for bl in baselines:
            diff = mae_arrays[bl] - mae_arrays[prop]
            valid_mask = ~(np.isnan(diff))
            diff_valid = diff[valid_mask]
            if len(diff_valid) < 10:
                print(f"  {bl:<16s} {'N/A':>12s}")
                continue
            try:
                stat, p_val = stats.wilcoxon(diff_valid, alternative='two-sided')
            except Exception:
                p_val = 1.0
            sig = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else "ns"
            print(f"  {bl:<16s} {p_val:12.2e} {sig:>4s} {np.mean(diff_valid):+10.2f} "
                  f"{np.median(diff_valid):+12.2f}")

    # ── Cross-Type Analysis ──
    print(f"\n{SEP}")
    print("  Cross-Type Analysis (Key Claim: consistency)")
    print(SEP)

    type_order = ['Clean', 'Drift', 'Standard', 'Mid-drift', 'Mixed']
    type_groups = {}
    for r in results:
        t = r['type']
        if t not in type_groups:
            type_groups[t] = {algo: [] for algo in ALGO_NAMES}
        for algo in ALGO_NAMES:
            type_groups[t][algo].append(r[f'mae_{algo}'])

    header = f"  {'Type':<12s}"
    for algo in ALGO_NAMES:
        header += f" {algo:>10s}"
    print(header)
    print(f"  {DIV}")

    type_means = {algo: {} for algo in ALGO_NAMES}
    for t in type_order:
        if t not in type_groups: continue
        g = type_groups[t]
        line = f"  {t:<12s}"
        for algo in ALGO_NAMES:
            vals = [v for v in g[algo] if not np.isnan(v)]
            m = np.mean(vals) if vals else float('nan')
            type_means[algo][t] = m
            line += f" {m:10.2f}"
        print(line)

    # Range (consistency metric)
    print(f"\n  {'Range':<12s}", end="")
    for algo in ALGO_NAMES:
        vals = [v for v in type_means[algo].values() if not np.isnan(v)]
        if len(vals) >= 2:
            rng = max(vals) - min(vals)
        else:
            rng = float('nan')
        print(f" {rng:10.2f}", end="")
    print()

    # ── Per-Participant Table ──
    print(f"\n{SEP}")
    print("  Per-Participant MAE")
    print(SEP)

    header = f"  {'ID':<5s} {'Type':<10s}"
    for algo in ALGO_NAMES:
        header += f" {algo:>10s}"
    print(header)
    print(f"  {DIV}")

    for person in PERSONS:
        pr = [r for r in results if r['person'] == person]
        if not pr: continue
        t = SIGNAL_TYPES.get(person, '?')
        line = f"  {person:<5s} {t:<10s}"
        for algo in ALGO_NAMES:
            vals = [r[f'mae_{algo}'] for r in pr if not np.isnan(r[f'mae_{algo}'])]
            m = np.mean(vals) if vals else float('nan')
            line += f" {m:10.2f}"
        print(line)

    # ── Per-Category Table ──
    print(f"\n{SEP}")
    print("  Per-Category MAE")
    print(SEP)

    header = f"  {'Category':<16s} {'N':>3s}"
    for algo in ALGO_NAMES:
        header += f" {algo:>10s}"
    print(header)
    print(f"  {DIV}")

    for cat in CATS_ORDER:
        cr = [r for r in results if r['cat'] == cat]
        if not cr: continue
        line = f"  {cat:<16s} {len(cr):3d}"
        for algo in ALGO_NAMES:
            vals = [r[f'mae_{algo}'] for r in cr if not np.isnan(r[f'mae_{algo}'])]
            m = np.mean(vals) if vals else float('nan')
            line += f" {m:10.2f}"
        print(line)

    # ═══════════════════════════════════════════════════════════════════════════
    # [4/4] Save JSON
    # ═══════════════════════════════════════════════════════════════════════════
    print(f"\n[4/4] Saving results...")

    output = {
        '_meta': {
            'run_at': datetime.datetime.now().isoformat(),
            'total_cases': N,
            'algorithms': ALGO_NAMES,
            'common_backend': {
                'bpf': '0.8-2.1 Hz, 4th Butterworth',
                'stft': f'W={STFT_W}, L={STFT_L}',
                'kalman_R': 0.10,
                'downsample': f'radarFpsQ={radarFpsQ}',
            },
        },
        'overall': {},
        'cross_type': {},
        'per_participant': {},
        'per_category': {},
        'wilcoxon': {},
        'case_results': [],
    }

    # Overall
    for algo in ALGO_NAMES:
        arr = mae_arrays[algo]
        valid = arr[~np.isnan(arr)]
        if len(valid) == 0: continue
        output['overall'][algo] = {
            'mean_mae': float(np.mean(valid)),
            'rmse': float(np.sqrt(np.mean(valid**2))),
            'median_mae': float(np.median(valid)),
            'le5': float(np.mean(valid <= 5) * 100),
            'le10': float(np.mean(valid <= 10) * 100),
            'le15': float(np.mean(valid <= 15) * 100),
        }

    # Cross-type
    for t in type_order:
        if t not in type_groups: continue
        g = type_groups[t]
        output['cross_type'][t] = {}
        for algo in ALGO_NAMES:
            vals = [v for v in g[algo] if not np.isnan(v)]
            output['cross_type'][t][algo] = float(np.mean(vals)) if vals else None

    # Cross-type range
    output['cross_type_range'] = {}
    for algo in ALGO_NAMES:
        vals = [v for t, v in type_means[algo].items() if not np.isnan(v)]
        output['cross_type_range'][algo] = float(max(vals) - min(vals)) if len(vals) >= 2 else None

    # Per-participant
    for person in PERSONS:
        pr = [r for r in results if r['person'] == person]
        if not pr: continue
        output['per_participant'][person] = {'type': SIGNAL_TYPES.get(person, '?')}
        for algo in ALGO_NAMES:
            vals = [r[f'mae_{algo}'] for r in pr if not np.isnan(r[f'mae_{algo}'])]
            output['per_participant'][person][algo] = float(np.mean(vals)) if vals else None

    # Per-category
    for cat in CATS_ORDER:
        cr = [r for r in results if r['cat'] == cat]
        if not cr: continue
        output['per_category'][cat] = {'n': len(cr)}
        for algo in ALGO_NAMES:
            vals = [r[f'mae_{algo}'] for r in cr if not np.isnan(r[f'mae_{algo}'])]
            output['per_category'][cat][algo] = float(np.mean(vals)) if vals else None

    # Wilcoxon
    for prop in proposed:
        output['wilcoxon'][prop] = {}
        for bl in baselines:
            diff = mae_arrays[bl] - mae_arrays[prop]
            valid_mask = ~np.isnan(diff)
            diff_valid = diff[valid_mask]
            if len(diff_valid) < 10:
                output['wilcoxon'][prop][bl] = {'p_value': None, 'sig': 'N/A'}
                continue
            try:
                stat, p_val = stats.wilcoxon(diff_valid, alternative='two-sided')
            except Exception:
                p_val = 1.0
            sig = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else "ns"
            output['wilcoxon'][prop][bl] = {
                'p_value': float(p_val),
                'sig': sig,
                'mean_diff': float(np.mean(diff_valid)),
                'median_diff': float(np.median(diff_valid)),
            }

    # Case results (compact)
    for r in results:
        case = {'name': r['name'], 'cat': r['cat'], 'person': r['person'], 'type': r['type']}
        for algo in ALGO_NAMES:
            case[algo] = round(r[f'mae_{algo}'], 4) if not np.isnan(r[f'mae_{algo}']) else None
        output['case_results'].append(case)

    out_path = f'{OUT_DIR}/external_baselines_v2.json'
    with open(out_path, 'w') as f:
        json.dump(output, f, indent=2, default=str)
    print(f"  Saved: {out_path}")

    elapsed = time.time() - t_start
    print(f"\n{'=' * 80}")
    print(f"  Done!  ({elapsed:.0f}s)")
    print(f"{'=' * 80}")

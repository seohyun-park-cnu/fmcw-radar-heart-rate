"""Evaluate 8 external baselines + 3 internal + gate/oracle under the CANONICAL
15-second non-overlapping block MAE protocol (no GT smoothing).

Canonical protocol (from make_tables_AB_singleparam.py):
  - GT: >150 BPM interpolated, NO smoothing, decimated to 4 Hz, first 5s dropped
  - Metric: 15s non-overlapping block mean MAE
  - Sparse: single global param (thr=3.0, win=7, guard=False)
  - Gate thresholds: tau1=0.508, tau2=0.042, tau3=1.509

Run:  cd IoT && python3 eval/eval_external_block15s.py
"""
import os, sys, json, time, warnings
warnings.filterwarnings('ignore')

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import numpy as np
from scipy.io import loadmat
from scipy.signal import welch, iirnotch, filtfilt as scipy_filtfilt
from scipy.stats import wilcoxon
from vmdpy import VMD
import pywt
from PyEMD import EEMD as PyEEMD, CEEMDAN as PyCEEMDAN
import ewtpy

from jyvital7_1 import (MTI, dacm_phase_unwrapping, bandpass_filter,
                         phase_stft, stft_time, stft_bpm, heartbeat_estimation)
from jymas import sma
from shsparse import detect_sparse_movement, preprocess_signal

# ── Constants ──
numFrame, measuretime = 1200, 60
fps = numFrame // measuretime  # 20
quarter = 4
radarFpsQ = fps // quarter     # 5
ubpulseFpsQ = 256 // quarter   # 64
ds_fps = fps // radarFpsQ      # 4

BLK_S = 15
BLK_W = BLK_S * ds_fps         # 60 samples
BLK_STRIDE = BLK_W             # non-overlapping

BP_LOW, BP_HIGH = 0.8, 2.1
STFT_W, STFT_L = 100, 99
MAX_INTERP = 20
DET_GUARD = 0.03
KALMAN_R = 0.10

K1_OLD, K2_OLD = 2, 2
K1_NEW, K2_NEW = 5, 2

# Gate thresholds
THR_MAD_STD = 0.508
THR_HR_POWER = 0.042
THR_DETREND_5V2 = 1.509

# Single global Sparse param
SP_THR, SP_WIN, SP_GUARD = 3.0, 7, False

AZ = [6, 4, 3, 2]
CATS = ['static', 'arm', 'head', 'cough', 'shoulder_shrug',
        'disturb', 'swing', 'transition', 'water']
TRIALS = ['1st', '2nd', '3rd']
PERSONS = ['SH', 'BG', 'JY', 'HJ', 'ET', 'JB', 'JH', 'JHL', 'YW', 'SJ']
DATA = "/root/workspace/2026-1/IoT_data/Participant Data"

hr_time_axis = stft_time(numFrame, STFT_W, STFT_L, fps)
hr_bpm_axis = stft_bpm(numFrame, measuretime, fps)


# ── GT: canonical (no smoothing) ──
def extract_gt(mhr):
    bpm = np.squeeze(mhr['saveDataHeartRateUbpulse1'], 0).astype(float)
    L = len(bpm)
    bpm[bpm > 150] = np.nan
    vi = np.where(~np.isnan(bpm))[0]
    ni = np.where(np.isnan(bpm))[0]
    if len(ni) and len(vi):
        bpm[ni] = np.interp(ni, vi, bpm[vi])
    return bpm[np.arange(0, L, ubpulseFpsQ)][quarter * radarFpsQ:]


def extract_target(d):
    """Target cell fixed for the whole recording.
    range bin = argmax of the time-averaged, channel-averaged |MTI|;
    angle bin = argmax of the time-averaged |angle FFT| at that range bin."""
    rfft = MTI(d[:, AZ, :]); rm = np.mean(np.abs(rfft), 1)
    rp = int(np.argmax(rm.mean(0)))
    dm = np.fft.fftshift(np.fft.fft(rfft, axis=1), 1)
    dp = int(np.argmax(np.abs(dm[:, :, rp]).mean(0)))
    return dm[:, dp, rp]


# ── Block MAE evaluation ──
def blocks(x):
    return np.array([np.mean(x[s:s + BLK_W])
                     for s in range(0, len(x) - BLK_W + 1, BLK_STRIDE)])


def block_errors(track, gb):
    yb = blocks(track)
    n = min(len(gb), len(yb))
    return np.abs(gb[:n] - yb[:n])


# ── Common backend ──
def downsample_hr(e):
    return e[np.arange(0, len(e), radarFpsQ)][:-1]


def common_backend(signal):
    f = bandpass_filter(signal, BP_LOW, BP_HIGH, fps)
    stft_out = phase_stft(f, STFT_W, STFT_L, numFrame)
    hr_est = heartbeat_estimation(stft_out, hr_bpm_axis, len(hr_time_axis),
                                  kalmanR=KALMAN_R)
    return downsample_hr(hr_est)


# ── Gate features & routing ──
def gate_feats(dacm):
    dd = np.diff(dacm)
    r_ms = float(np.median(np.abs(dd - np.median(dd)))) / (np.std(dd) + 1e-12)
    fr, psd = welch(dacm, fs=fps, nperseg=min(256, len(dacm)),
                    noverlap=min(128, len(dacm) // 2))
    tp = np.sum(psd)
    r_hr = float(np.sum(psd[(fr >= 0.8) & (fr <= 2.1)]) / tp) if tp > 0 else 0.0
    r_dt = float(np.std(dacm - sma(dacm, size=5)) /
                 (np.std(dacm - sma(dacm, size=2)) + 1e-12))
    return r_ms, r_hr, r_dt


def route_of(dacm):
    a, b, c = gate_feats(dacm)
    if a > THR_MAD_STD:
        return "Raw"
    elif b > THR_HR_POWER or c < THR_DETREND_5V2:
        return "MA"
    else:
        return "Sparse"


# ── Internal pipelines ──
def sig_raw(dacm):
    return dacm


def sig_ma(dacm):
    return sma(dacm - sma(dacm, K1_OLD), K2_OLD)


def sig_sparse(dacm):
    dc = np.asarray(dacm, float).copy()
    ap = True
    if SP_GUARD:
        det = detect_sparse_movement(dc, fps=fps, win_size=SP_WIN,
                                     threshold_factor=SP_THR)
        if float(np.sum(det)) / len(dc) < DET_GUARD:
            ap = False
    if ap:
        m = detect_sparse_movement(dc, fps=fps, win_size=SP_WIN,
                                   threshold_factor=SP_THR)
        dc = preprocess_signal(dc, m, method='interpolate',
                               max_interp_len=MAX_INTERP)
    return sma(dc - sma(dc, K1_NEW), K2_NEW)


# ── External baselines (functions copied from external_baselines_v2.py) ──

def run_vmd(dacm):
    try:
        u, _, _ = VMD(dacm, 2000, 0, 4, 0, 1, 1e-7)
    except Exception:
        return common_backend(dacm)
    hr_band = (0.8, 2.1)
    selected = []
    for i in range(len(u)):
        fr, psd = welch(u[i], fs=fps, nperseg=min(256, len(u[i])), noverlap=128)
        tp = np.sum(psd)
        if tp > 0:
            hr_r = np.sum(psd[(fr >= hr_band[0]) & (fr <= hr_band[1])]) / tp
            if hr_r >= 0.30:
                selected.append(i)
    if not selected:
        best_i, best_e = 0, -1
        for i in range(len(u)):
            fr, psd = welch(u[i], fs=fps, nperseg=min(256, len(u[i])), noverlap=128)
            he = np.sum(psd[(fr >= 0.8) & (fr <= 2.1)])
            if he > best_e:
                best_e, best_i = he, i
        selected = [best_i]
    return common_backend(np.sum(u[selected], axis=0))


def run_dwt(dacm):
    wavelet, level = 'db4', 4
    coeffs = pywt.wavedec(dacm, wavelet, level=level)
    sigma = np.median(np.abs(coeffs[-1])) / 0.6745
    # BayesShrink
    new_coeffs = [coeffs[0]]
    for i in range(1, len(coeffs)):
        sy2 = np.var(coeffs[i])
        sx2 = max(sy2 - sigma**2, 0)
        thr = sigma**2 / np.sqrt(sx2) if sx2 > 0 else np.max(np.abs(coeffs[i]))
        new_coeffs.append(pywt.threshold(coeffs[i], thr, mode='soft'))
    denoised = pywt.waverec(new_coeffs, wavelet)[:len(dacm)]
    return common_backend(denoised)


def run_eemd(dacm):
    eemd = PyEEMD(trials=50, noise_width=0.05)
    try:
        imfs = eemd.eemd(dacm)
    except Exception:
        return common_backend(dacm)
    if imfs is None or len(imfs) == 0:
        return common_backend(dacm)
    best_i, best_e = 0, -1
    for i in range(len(imfs)):
        fr, psd = welch(imfs[i], fs=fps, nperseg=min(256, len(imfs[i])), noverlap=128)
        he = np.sum(psd[(fr >= 0.8) & (fr <= 2.1)])
        if he > best_e:
            best_e, best_i = he, i
    return common_backend(imfs[best_i])


def run_rls(dacm):
    reference = sma(dacm, size=5)
    order, lam, delta = 8, 0.999, 1.0
    N = len(dacm)
    w = np.zeros(order)
    P = delta * np.eye(order)
    y_hat = np.zeros(N)
    for n in range(order, N):
        x = reference[n - order:n][::-1]
        y_hat[n] = np.dot(w, x)
        e = dacm[n] - y_hat[n]
        Px = P @ x
        k = Px / (lam + x @ Px)
        w = w + k * e
        P = (P - np.outer(k, x @ P)) / lam
    return common_backend(dacm - y_hat)


def run_adaptive_notch(dacm):
    fr, psd = welch(dacm, fs=fps, nperseg=min(512, len(dacm)), noverlap=256)
    resp_mask = (fr >= 0.1) & (fr <= 0.5)
    if np.any(resp_mask) and np.sum(psd[resp_mask]) > 0:
        resp_rate = fr[resp_mask][np.argmax(psd[resp_mask])]
    else:
        resp_rate = 0.25
    filtered = dacm.copy()
    Q = 30.0
    for n in range(1, 6):
        hf = resp_rate * n
        if 0.7 <= hf <= 2.2:
            w0 = hf / (fps / 2.0)
            if 0 < w0 < 1:
                b, a = iirnotch(w0, Q)
                filtered = scipy_filtfilt(b, a, filtered)
    return common_backend(filtered)


def _ssa_antidiag_avg(mat, N):
    L, K = mat.shape
    result = np.zeros(N)
    for n in range(N):
        i_s = max(0, n - K + 1)
        i_e = min(L, n + 1)
        result[n] = np.mean([mat[i, n - i] for i in range(i_s, i_e)])
    return result


def run_ssa(dacm):
    N = len(dacm)
    L = min(100, N // 3)
    K = N - L + 1
    from numpy.lib.stride_tricks import as_strided
    X = as_strided(dacm, shape=(L, K),
                   strides=(dacm.strides[0], dacm.strides[0])).copy()
    n_comp = min(20, min(L, K))
    U, s, Vt = np.linalg.svd(X, full_matrices=False)
    U = U[:, :n_comp]; s = s[:n_comp]; Vt = Vt[:n_comp, :]
    comp_series = {}
    selected = []
    for c in range(n_comp):
        Xc = s[c] * np.outer(U[:, c], Vt[c, :])
        rc = _ssa_antidiag_avg(Xc, N)
        comp_series[c] = rc
        fr, psd = welch(rc, fs=fps, nperseg=min(256, N), noverlap=min(128, N // 2))
        tp = np.sum(psd)
        if tp > 0:
            hr_r = np.sum(psd[(fr >= 0.8) & (fr <= 2.1)]) / tp
            if hr_r >= 0.25:
                selected.append(c)
    if not selected:
        best_c, best_e = 0, -1
        for c in range(n_comp):
            fr, psd = welch(comp_series[c], fs=fps, nperseg=min(256, N),
                            noverlap=min(128, N // 2))
            he = np.sum(psd[(fr >= 0.8) & (fr <= 2.1)])
            if he > best_e:
                best_e, best_c = he, c
        selected = [best_c]
    reconstructed = np.zeros(N)
    for c in selected:
        reconstructed += comp_series[c]
    return common_backend(reconstructed)


def run_ceemdan(dacm):
    ceemdan = PyCEEMDAN(trials=50, epsilon=0.05)
    try:
        imfs = ceemdan.ceemdan(dacm)
    except Exception:
        return common_backend(dacm)
    if imfs is None or len(imfs) == 0:
        return common_backend(dacm)
    selected = []
    for i in range(len(imfs)):
        fr, psd = welch(imfs[i], fs=fps, nperseg=min(256, len(imfs[i])), noverlap=128)
        tp = np.sum(psd)
        if tp > 0:
            hr_r = np.sum(psd[(fr >= 0.8) & (fr <= 2.1)]) / tp
            if hr_r >= 0.30:
                selected.append(i)
    if not selected:
        best_i, best_e = 0, -1
        for i in range(len(imfs)):
            fr, psd = welch(imfs[i], fs=fps, nperseg=min(256, len(imfs[i])), noverlap=128)
            he = np.sum(psd[(fr >= 0.8) & (fr <= 2.1)])
            if he > best_e:
                best_e, best_i = he, i
        selected = [best_i]
    return common_backend(np.sum(imfs[selected], axis=0))


def run_ewt(dacm):
    try:
        ewt_coeffs, mfb, boundaries = ewtpy.EWT1D(dacm, N=8)
    except Exception:
        return common_backend(dacm)
    if ewt_coeffs is None or ewt_coeffs.shape[1] == 0:
        return common_backend(dacm)
    n_comp = ewt_coeffs.shape[1]
    selected = []
    for i in range(n_comp):
        comp = ewt_coeffs[:, i]
        fr, psd = welch(comp, fs=fps, nperseg=min(256, len(comp)), noverlap=128)
        tp = np.sum(psd)
        if tp > 0:
            hr_r = np.sum(psd[(fr >= 0.8) & (fr <= 2.1)]) / tp
            if hr_r >= 0.25:
                selected.append(i)
    if not selected:
        best_i, best_e = 0, -1
        for i in range(n_comp):
            comp = ewt_coeffs[:, i]
            fr, psd = welch(comp, fs=fps, nperseg=min(256, len(comp)), noverlap=128)
            he = np.sum(psd[(fr >= 0.8) & (fr <= 2.1)])
            if he > best_e:
                best_e, best_i = he, i
        selected = [best_i]
    return common_backend(np.sum(ewt_coeffs[:, selected], axis=1))


# ── Algorithm registry ──
# Note: all external baselines take DACM as input (not raw target)
# because our canonical pipeline extracts DACM first
EXTERNAL_ALGOS = {
    'VMD': run_vmd,
    'DWT': run_dwt,
    'EEMD': run_eemd,
    'RLS': run_rls,
    'AdaptNotch': run_adaptive_notch,
    'SSA': run_ssa,
    'CEEMDAN': run_ceemdan,
    'EWT': run_ewt,
}


# ═══════════════════════════════════════════════════════════════════════════════
if __name__ == '__main__':
    t0 = time.time()
    print("=" * 70)
    print("  External Baseline Evaluation — 15s Block MAE (canonical)")
    print("=" * 70)

    # ── [1] Load data ──
    print("\n[1/3] Loading data...")
    records = []
    for p in PERSONS:
        for cat in CATS:
            for tr in TRIALS:
                nm = f"{cat}_{p}_{tr}"
                try:
                    mr = loadmat(f"{DATA}/{nm}_radar.mat")
                    mhr = loadmat(f"{DATA}/{nm}_ubpulse_heartbeat.mat")
                    gt = extract_gt(mhr)
                    if len(gt) == 0:
                        raise ValueError("empty GT")
                    target = extract_target(mr['saveDataRadar1'])
                    dacm = dacm_phase_unwrapping(target)
                    gb = blocks(gt)
                    route = route_of(dacm)

                    # Pre-compute internal pipelines
                    hr_raw = common_backend(sig_raw(dacm))
                    hr_ma = common_backend(sig_ma(dacm))
                    hr_sp = common_backend(sig_sparse(dacm))

                    e_raw = block_errors(hr_raw, gb)
                    e_ma = block_errors(hr_ma, gb)
                    e_sp = block_errors(hr_sp, gb)

                    records.append({
                        'name': nm, 'person': p, 'cat': cat,
                        'dacm': dacm, 'gb': gb, 'route': route,
                        'e_Raw': e_raw, 'e_MA': e_ma, 'e_Sparse': e_sp,
                    })
                except Exception as e:
                    pass

    print(f"  Loaded {len(records)} recordings")

    # ── [2] Run external baselines ──
    print(f"\n[2/3] Running {len(EXTERNAL_ALGOS)} external baselines x {len(records)} recordings...")
    ext_errors = {algo: [] for algo in EXTERNAL_ALGOS}

    for i, rec in enumerate(records):
        dacm = rec['dacm']
        gb = rec['gb']
        for algo, func in EXTERNAL_ALGOS.items():
            try:
                hr_track = func(dacm)
                errs = block_errors(hr_track, gb)
            except Exception:
                errs = np.array([np.nan])
            ext_errors[algo].append(errs)

        if (i + 1) % 30 == 0 or i == len(records) - 1:
            elapsed = time.time() - t0
            print(f"  {i+1:3d}/{len(records)} done ({elapsed:.0f}s)")

    # ── [3] Compute metrics ──
    print(f"\n[3/3] Computing metrics...")

    def all_block_errs(scheme):
        return np.concatenate([r[f'e_{scheme}'] for r in records])

    def percat_mae(scheme, cat):
        errs = [r[f'e_{scheme}'] for r in records if r['cat'] == cat]
        if not errs:
            return np.nan
        return float(np.mean(np.concatenate(errs)))

    # Compute gate and oracle errors
    for rec in records:
        n = min(len(rec['e_Raw']), len(rec['e_MA']), len(rec['e_Sparse']))
        errs = {k: rec[f'e_{k}'][:n] for k in ['Raw', 'MA', 'Sparse']}
        rec['e_gate'] = errs[rec['route']]
        rec['e_oracle'] = np.min(np.stack([errs[k] for k in errs]), axis=0)

    # Store external errors in records
    for algo in EXTERNAL_ALGOS:
        for i, rec in enumerate(records):
            rec[f'e_{algo}'] = ext_errors[algo][i]

    # Build summary
    ALL_SCHEMES = ['Raw', 'MA', 'Sparse', 'gate', 'oracle'] + list(EXTERNAL_ALGOS.keys())
    summary = {}
    for s in ALL_SCHEMES:
        ae = np.concatenate([r[f'e_{s}'] for r in records])
        ae = ae[~np.isnan(ae)]
        pc = [percat_mae(s, c) for c in CATS]
        summary[s] = {
            'MAE': float(np.mean(ae)),
            'RMSE': float(np.sqrt(np.mean(ae**2))),
            'WorstMAE': float(np.nanmax(pc)),
            'condStd': float(np.nanstd(pc)),
            'le10': float(np.mean(ae <= 10) * 100),
            'le15': float(np.mean(ae <= 15) * 100),
            'percat': {c: float(v) for c, v in zip(CATS, pc)},
        }

    # Wilcoxon tests (gate vs each)
    gate_all = np.concatenate([r['e_gate'] for r in records])
    for s in ALL_SCHEMES:
        if s == 'gate' or s == 'oracle':
            summary[s]['p_vs_gate'] = None
            continue
        other = np.concatenate([r[f'e_{s}'] for r in records])
        n = min(len(gate_all), len(other))
        g, o = gate_all[:n], other[:n]
        valid = ~(np.isnan(g) | np.isnan(o))
        if np.sum(valid) > 10:
            try:
                _, p = wilcoxon(g[valid], o[valid])
                summary[s]['p_vs_gate'] = float(p)
            except Exception:
                summary[s]['p_vs_gate'] = None
        else:
            summary[s]['p_vs_gate'] = None

    # ── Print results ──
    print(f"\n{'='*80}")
    print(f"  15s Block MAE Results ({len(records)} recordings)")
    print(f"{'='*80}")
    print(f"  {'Method':<14s} {'MAE':>7s} {'RMSE':>7s} {'Worst':>7s} {'cStd':>6s} "
          f"{'<=10':>6s} {'<=15':>6s} {'p':>10s}")
    print(f"  {'-'*72}")
    for s in ALL_SCHEMES:
        d = summary[s]
        p_str = f"{d['p_vs_gate']:.1e}" if d['p_vs_gate'] is not None else "---"
        marker = " ***" if s in ('gate', 'oracle') else ""
        print(f"  {s:<14s} {d['MAE']:7.3f} {d['RMSE']:7.3f} {d['WorstMAE']:7.3f} "
              f"{d['condStd']:6.3f} {d['le10']:5.1f}% {d['le15']:5.1f}% {p_str:>10s}{marker}")

    # ── Save ──
    route_dist = {}
    for r in records:
        route_dist[r['route']] = route_dist.get(r['route'], 0) + 1

    out = {
        'protocol': '15s_block_MAE_nosmooth',
        'n_recordings': len(records),
        'sparse_param': {'thr': SP_THR, 'win': SP_WIN, 'guard': SP_GUARD},
        'gate_thresholds': {'tau1': THR_MAD_STD, 'tau2': THR_HR_POWER, 'tau3': THR_DETREND_5V2},
        'route_dist': route_dist,
        'summary': summary,
    }

    out_path = "/root/workspace/2026-1/IoT/results/external_block15s.json"
    with open(out_path, 'w') as f:
        json.dump(out, f, indent=1, ensure_ascii=False)
    print(f"\nSaved: {out_path}")
    print(f"Total time: {time.time() - t0:.0f}s")

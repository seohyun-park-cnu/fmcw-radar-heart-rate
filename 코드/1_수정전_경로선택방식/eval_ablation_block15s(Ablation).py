"""Ablation study under canonical 15s block MAE protocol.

Configurations (building up from baseline to full system):
  A. MA-only, R=0.01   — original Choi et al. baseline
  B. MA-only, R=0.10   — improved Kalman only
  C. Sparse-only, R=0.10 — add pre-diff sparse (single global param)
  D. 2-way gate, R=0.10  — Raw vs Sparse routing (Gate 1 only)
  E. 3-way gate, R=0.10  — full system (Raw/MA/Sparse routing)
  F. Oracle, R=0.10       — upper bound (per-block best pipeline)

Run: cd IoT && python3 eval/eval_ablation_block15s.py
"""
import sys, json
import numpy as np
import scipy.io
from scipy.signal import welch
sys.path.insert(0, "/root/workspace/2026-1/IoT/src")
from jyvital7_1 import (MTI, dacm_phase_unwrapping, bandpass_filter, phase_stft,
                         stft_time, stft_bpm, heartbeat_estimation)
from jymas import sma
from shsparse import detect_sparse_movement, preprocess_signal

numFrame, measuretime = 1200, 60
fps = numFrame // measuretime
quarter, radarFpsQ, ubpulseFpsQ = 4, fps // 4, 256 // 4
ds_fps = fps // radarFpsQ
BLK_W = 15 * ds_fps
BLK_STRIDE = BLK_W
BP_LOW, BP_HIGH, STFT_W, STFT_L, MAX_INTERP, DET_GUARD = 0.8, 2.1, 100, 99, 20, 0.03
K1_OLD, K2_OLD, K1_NEW, K2_NEW = 2, 2, 5, 2
THR_MAD_STD, THR_HR_POWER, THR_DETREND_5V2 = 0.508, 0.042, 1.509
SP_THR, SP_WIN, SP_GUARD = 3.0, 7, False
AZ = [6, 4, 3, 2]
CATS = ['static', 'arm', 'head', 'cough', 'shoulder_shrug', 'disturb', 'swing', 'transition', 'water']
TRIALS = ['1st', '2nd', '3rd']
PERSONS = ['SH', 'BG', 'JY', 'HJ', 'ET', 'JB', 'JH', 'JHL', 'YW', 'SJ']
DATA = "/root/workspace/2026-1/IoT_data/Participant Data"

hr_time_axis = stft_time(numFrame, STFT_W, STFT_L, fps)
hr_bpm_axis = stft_bpm(numFrame, measuretime, fps)


def extract_gt(mhr):
    bpm = np.squeeze(mhr['saveDataHeartRateUbpulse1'], 0).astype(float)
    L = len(bpm); bpm[bpm > 150] = np.nan
    vi = np.where(~np.isnan(bpm))[0]; ni = np.where(np.isnan(bpm))[0]
    if len(ni) and len(vi): bpm[ni] = np.interp(ni, vi, bpm[vi])
    return bpm[np.arange(0, L, ubpulseFpsQ)][quarter * radarFpsQ:]


def extract_target(d):
    nF = d.shape[0]; rfft = MTI(d[:, AZ, :]); rm = np.mean(np.abs(rfft), 1)
    rp = np.array([0] + [int(np.argmax(rm[i])) for i in range(1, nF)])
    dm = np.fft.fftshift(np.fft.fft(rfft, axis=1), 1)
    dp = np.array([0] + [int(np.argmax(np.abs(dm[i, :, rp[i]]))) for i in range(1, nF)])
    return np.array([dm[i, dp[i], rp[i]] for i in range(nF)])


def downsample_hr(e): return e[np.arange(0, len(e), radarFpsQ)][:-1]


def compute_hr(sig, kalman_r=0.10):
    f = bandpass_filter(sig, BP_LOW, BP_HIGH, fps)
    return downsample_hr(heartbeat_estimation(
        phase_stft(f, STFT_W, STFT_L, numFrame), hr_bpm_axis, len(hr_time_axis), kalmanR=kalman_r))


def blocks(x): return np.array([np.mean(x[s:s + BLK_W]) for s in range(0, len(x) - BLK_W + 1, BLK_STRIDE)])
def berr(track, gb): yb = blocks(track); n = min(len(gb), len(yb)); return np.abs(gb[:n] - yb[:n])

def sig_ma(d): return sma(d - sma(d, K1_OLD), K2_OLD)

def sig_sparse(dacm):
    dc = np.asarray(dacm, float).copy(); ap = True
    if SP_GUARD:
        det = detect_sparse_movement(dc, fps=fps, win_size=SP_WIN, threshold_factor=SP_THR)
        if float(np.sum(det)) / len(dc) < DET_GUARD: ap = False
    if ap:
        m = detect_sparse_movement(dc, fps=fps, win_size=SP_WIN, threshold_factor=SP_THR)
        dc = preprocess_signal(dc, m, method='interpolate', max_interp_len=MAX_INTERP)
    return sma(dc - sma(dc, K1_NEW), K2_NEW)


def gate_feats(dacm):
    dd = np.diff(dacm)
    r_ms = float(np.median(np.abs(dd - np.median(dd)))) / (np.std(dd) + 1e-12)
    fr, psd = welch(dacm, fs=fps, nperseg=min(256, len(dacm)), noverlap=min(128, len(dacm) // 2))
    tp = np.sum(psd); r_hr = float(np.sum(psd[(fr >= 0.8) & (fr <= 2.1)]) / tp) if tp > 0 else 0.0
    r_dt = float(np.std(dacm - sma(dacm, size=5)) / (np.std(dacm - sma(dacm, size=2)) + 1e-12))
    return r_ms, r_hr, r_dt


# ── Load data & compute all pipeline outputs ──
print("Loading 270 recordings...")
cache = []
for p in PERSONS:
    for cat in CATS:
        for tr in TRIALS:
            nm = f"{cat}_{p}_{tr}"
            try:
                mr = scipy.io.loadmat(f"{DATA}/{nm}_radar.mat")
                mhr = scipy.io.loadmat(f"{DATA}/{nm}_ubpulse_heartbeat.mat")
                gt = extract_gt(mhr)
                if len(gt) == 0: raise ValueError
                dacm = dacm_phase_unwrapping(extract_target(mr['saveDataRadar1']))
            except Exception:
                continue

            gb = blocks(gt)
            r_ms, r_hr, r_dt = gate_feats(dacm)

            # Compute all needed HR tracks
            hr_raw_r010 = compute_hr(dacm, kalman_r=0.10)
            hr_ma_r001  = compute_hr(sig_ma(dacm), kalman_r=0.01)
            hr_ma_r010  = compute_hr(sig_ma(dacm), kalman_r=0.10)
            hr_sp_r010  = compute_hr(sig_sparse(dacm), kalman_r=0.10)

            # Block errors for each config
            e_raw_r010 = berr(hr_raw_r010, gb)
            e_ma_r001  = berr(hr_ma_r001, gb)
            e_ma_r010  = berr(hr_ma_r010, gb)
            e_sp_r010  = berr(hr_sp_r010, gb)

            # Route
            if r_ms > THR_MAD_STD:
                route3 = "Raw"
            elif r_hr > THR_HR_POWER or r_dt < THR_DETREND_5V2:
                route3 = "MA"
            else:
                route3 = "Sparse"

            # 2-way route (Gate 1 only: Raw vs Sparse)
            route2 = "Raw" if r_ms > THR_MAD_STD else "Sparse"

            cache.append({
                "cat": cat,
                "e_ma_r001": e_ma_r001, "e_ma_r010": e_ma_r010,
                "e_raw_r010": e_raw_r010, "e_sp_r010": e_sp_r010,
                "route2": route2, "route3": route3,
            })
print(f"Loaded {len(cache)} recordings\n")


# ── Compute ablation configs ──
def config_errors(cache, config_name):
    all_errs = []
    for r in cache:
        if config_name == "A_MA_R001":
            e = r["e_ma_r001"]
        elif config_name == "B_MA_R010":
            e = r["e_ma_r010"]
        elif config_name == "C_Sparse_R010":
            e = r["e_sp_r010"]
        elif config_name == "D_2way_R010":
            errs = {"Raw": r["e_raw_r010"], "Sparse": r["e_sp_r010"]}
            n = min(len(errs["Raw"]), len(errs["Sparse"]))
            e = errs[r["route2"]][:n]
        elif config_name == "E_3way_R010":
            errs = {"Raw": r["e_raw_r010"], "MA": r["e_ma_r010"], "Sparse": r["e_sp_r010"]}
            n = min(len(v) for v in errs.values())
            e = errs[r["route3"]][:n]
        elif config_name == "F_Oracle_R010":
            errs = {"Raw": r["e_raw_r010"], "MA": r["e_ma_r010"], "Sparse": r["e_sp_r010"]}
            n = min(len(v) for v in errs.values())
            e = np.min(np.stack([errs[k][:n] for k in errs]), axis=0)
        all_errs.append(e)
    return all_errs


def compute_metrics(all_errs, cache):
    flat = np.concatenate(all_errs)
    percat = {}
    for c in CATS:
        cat_e = [all_errs[i] for i in range(len(cache)) if cache[i]["cat"] == c]
        if cat_e:
            percat[c] = float(np.mean(np.concatenate(cat_e)))
    pc = list(percat.values())
    return {
        "MAE": float(np.mean(flat)),
        "RMSE": float(np.sqrt(np.mean(flat**2))),
        "WorstMAE": float(max(pc)),
        "condStd": float(np.std(pc)),
        "percat": percat,
    }


CONFIGS = [
    ("A_MA_R001", "MA only, R=0.01 (Choi baseline)"),
    ("B_MA_R010", "MA only, R=0.10"),
    ("C_Sparse_R010", "Sparse only, R=0.10"),
    ("D_2way_R010", "2-way gate (Raw/Sparse), R=0.10"),
    ("E_3way_R010", "3-way gate (Raw/MA/Sparse), R=0.10"),
    ("F_Oracle_R010", "Oracle (per-block best), R=0.10"),
]

results = {}
print(f"{'Config':<45s} {'MAE':>7s} {'RMSE':>7s} {'Worst':>7s} {'cStd':>6s}")
print("-" * 78)
prev_mae = None
for cfg_id, cfg_desc in CONFIGS:
    errs = config_errors(cache, cfg_id)
    m = compute_metrics(errs, cache)
    delta = f"{m['MAE'] - prev_mae:+.3f}" if prev_mae is not None else "---"
    print(f"  {cfg_desc:<43s} {m['MAE']:7.3f} {m['RMSE']:7.3f} {m['WorstMAE']:7.3f} {m['condStd']:6.3f}  ({delta})")
    prev_mae = m['MAE']
    results[cfg_id] = {"desc": cfg_desc, **m}

# Decomposition
a = results["A_MA_R001"]["MAE"]
e = results["E_3way_R010"]["MAE"]
total = a - e
print(f"\nTotal improvement: {a:.3f} -> {e:.3f} = {total:.3f} BPM")
for i in range(1, len(CONFIGS) - 1):  # skip Oracle
    prev_id = CONFIGS[i-1][0]
    cur_id = CONFIGS[i][0]
    d = results[prev_id]["MAE"] - results[cur_id]["MAE"]
    share = d / total * 100 if total > 0 else 0
    print(f"  {CONFIGS[i][1]:<43s}  delta={d:+.3f} ({share:.1f}%)")

out_path = "/root/workspace/2026-1/IoT/results/ablation_block15s.json"
json.dump(results, open(out_path, "w"), indent=1, ensure_ascii=False)
print(f"\nSaved: {out_path}")

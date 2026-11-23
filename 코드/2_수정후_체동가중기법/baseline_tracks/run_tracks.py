"""Dump per-recording time series (phase, GT, HR tracks of the three paths, impulse mask) for qualitative analysis.
Pipeline functions are identical to run_tables.py (canonical). Usage: python run_tracks.py <out.npz>"""
import os, sys, json
import numpy as np, scipy.io
from scipy.signal import welch
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, ".."))
import core   # registers jyvital7_1, jymas, shsparse from ../../1_수정전_경로선택방식
from jyvital7_1 import (MTI, dacm_phase_unwrapping, bandpass_filter, phase_stft, stft_time, stft_bpm, heartbeat_estimation)
from jymas import sma
from shsparse import detect_sparse_movement, preprocess_signal

numFrame, measuretime = 1200, 60
fps = 20; quarter, radarFpsQ, ubpulseFpsQ = 4, 5, 64
BP_LOW, BP_HIGH, STFT_W, STFT_L, MAX_INTERP = 0.8, 2.1, 100, 99, 20
KALMAN_R = 0.10
T1, T2, T3 = 0.508, 0.042, 1.509
AZ = [6, 4, 3, 2]
CATS = ['static', 'arm', 'head', 'cough', 'shoulder_shrug', 'disturb', 'swing', 'transition', 'water']
TRIALS = ['1st', '2nd', '3rd']
PERSONS = ['SH', 'BG', 'JY', 'HJ', 'ET', 'JB', 'JH', 'JHL', 'YW', 'SJ']
DATA = os.path.join(_HERE, "..", "..", "..", "데이터")
THR, WIN = 3.0, 7

def extract_gt(mhr):
    bpm = np.squeeze(mhr['saveDataHeartRateUbpulse1'], 0).astype(float); L = len(bpm)
    bpm[bpm > 150] = np.nan
    vi = np.where(~np.isnan(bpm))[0]; ni = np.where(np.isnan(bpm))[0]
    if len(ni) and len(vi): bpm[ni] = np.interp(ni, vi, bpm[vi])
    return bpm[np.arange(0, L, ubpulseFpsQ)][quarter * radarFpsQ:]

def extract_target(d):
    nF = d.shape[0]; rfft = MTI(d[:, AZ, :]); rm = np.mean(np.abs(rfft), 1)
    rp = np.array([0] + [int(np.argmax(rm[i])) for i in range(1, nF)])
    dm = np.fft.fftshift(np.fft.fft(rfft, axis=1), 1)
    dp = np.array([0] + [int(np.argmax(np.abs(dm[i, :, rp[i]]))) for i in range(1, nF)])
    return np.array([dm[i, dp[i], rp[i]] for i in range(nF)]), rp

def compute_hr(sig):
    f = bandpass_filter(sig, BP_LOW, BP_HIGH, fps)
    ht = stft_time(numFrame, STFT_W, STFT_L, fps); hb = stft_bpm(numFrame, measuretime, fps)
    e = heartbeat_estimation(phase_stft(f, STFT_W, STFT_L, numFrame), hb, len(ht), kalmanR=KALMAN_R)
    return e[np.arange(0, len(e), radarFpsQ)][:-1]

def sig_ma(d): return sma(d - sma(d, 2), 2)
def sig_sparse(sig):
    dc = np.asarray(sig, float).copy()
    m = detect_sparse_movement(dc, fps=fps, win_size=WIN, threshold_factor=THR)
    dc2 = preprocess_signal(dc, m, method='interpolate', max_interp_len=MAX_INTERP)
    return sma(dc2 - sma(dc2, 5), 2), np.asarray(m).astype(bool), dc2

def gate_feats(dacm):
    dd = np.diff(dacm)
    r_ms = float(np.median(np.abs(dd - np.median(dd)))) / (np.std(dd) + 1e-12)
    fr, psd = welch(dacm, fs=fps, nperseg=min(256, len(dacm)), noverlap=min(128, len(dacm) // 2))
    tp = np.sum(psd); r_hr = float(np.sum(psd[(fr >= 0.8) & (fr <= 2.1)]) / tp) if tp > 0 else 0.0
    r_dt = float(np.std(dacm - sma(dacm, size=5)) / (np.std(dacm - sma(dacm, size=2)) + 1e-12))
    return r_ms, r_hr, r_dt

out = {}; meta = []
for p in PERSONS:
    for cat in CATS:
        for tr in TRIALS:
            nm = f"{cat}_{p}_{tr}"
            try:
                mr = scipy.io.loadmat(f"{DATA}/{nm}_radar.mat"); mhr = scipy.io.loadmat(f"{DATA}/{nm}_ubpulse_heartbeat.mat")
                gt = extract_gt(mhr)
                if len(gt) == 0: raise ValueError
                tgt, rp = extract_target(mr['saveDataRadar1'])
                dacm = dacm_phase_unwrapping(tgt)
            except Exception as ex:
                print("skip", nm, ex); continue
            a, b, c = gate_feats(dacm)
            route = "Raw" if a > T1 else ("MA" if (b > T2 or c < T3) else "Sparse")
            ssp, mask, interp = sig_sparse(dacm)
            i = len(meta)
            out[f"dacm_{i}"] = np.asarray(dacm, np.float32); out[f"gt_{i}"] = gt.astype(np.float32)
            out[f"Raw_{i}"] = compute_hr(dacm).astype(np.float32)
            out[f"MA_{i}"] = compute_hr(sig_ma(dacm)).astype(np.float32)
            out[f"Sparse_{i}"] = compute_hr(ssp).astype(np.float32)
            out[f"mask_{i}"] = mask; out[f"interp_{i}"] = np.asarray(interp, np.float32); out[f"rp_{i}"] = rp.astype(np.int16)
            meta.append({"p": p, "cat": cat, "tr": tr, "nm": nm, "route": route, "feats": [a, b, c]})
    print(p, len(meta), flush=True)
np.savez_compressed(sys.argv[1], **out)
json.dump(meta, open(sys.argv[1].replace(".npz", "_meta.json"), "w", encoding="utf-8"))
print("done", len(meta))

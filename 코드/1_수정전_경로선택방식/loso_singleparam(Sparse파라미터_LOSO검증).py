"""LOSO (Leave-One-Subject-Out) generalization check for the fixed single Sparse
parameter, on the canonical batch pipeline (STFT 5s, no-smooth GT, 15s non-overlap).

Two questions:
 (1) FIXED config per-subject: with the globally-chosen Sparse param (3.0,7,False)
     and fixed gate thresholds held constant, what is each held-out subject's gate
     MAE? (does any subject collapse?)
 (2) SELECTION generalization (true LOSO): for each held-out subject, RE-SELECT the
     Sparse param by grid on the OTHER 9 subjects (min train gate MAE), apply to the
     held-out subject. Report selected param per fold + held-out gate MAE. If folds
     consistently pick the global param and held-out MAE stays ~7.5, the fixed param
     generalizes.
Gate thresholds are kept fixed (chosen on pooled data) — a known mild optimism,
disclosed. Only the Sparse param is LOSO-validated (per user request).

Pipeline funcs copied verbatim from canonical (no reimplementation).
Run: cd IoT && python3 analysis/loso_singleparam.py
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
BLK_S = 15; W = BLK_S * ds_fps; STRIDE = W
BP_LOW, BP_HIGH, STFT_W, STFT_L, MAX_INTERP, DET_GUARD = 0.8, 2.1, 100, 99, 20, 0.03
K1_OLD, K2_OLD, K1_NEW, K2_NEW, KALMAN_R = 2, 2, 5, 2, 0.10
THR_MAD_STD, THR_HR_POWER, THR_DETREND_5V2 = 0.508, 0.042, 1.509
AZ = [6, 4, 3, 2]
CATS = ['static', 'arm', 'head', 'cough', 'shoulder_shrug', 'disturb', 'swing', 'transition', 'water']
TRIALS = ['1st', '2nd', '3rd']
PERSONS = ['SH', 'BG', 'JY', 'HJ', 'ET', 'JB', 'JH', 'JHL', 'YW', 'SJ']
DATA = "/root/workspace/2026-1/IoT_data/Participant Data"
GRID = [(t, w, g) for t in (3.0, 4.0, 5.0, 6.0) for w in (3, 7) for g in (False, True)]
GLOBAL = (3.0, 7, False)


def extract_gt(mhr):
    bpm = np.squeeze(mhr['saveDataHeartRateUbpulse1'], 0).astype(float); L = len(bpm)
    bpm[bpm > 150] = np.nan
    vi = np.where(~np.isnan(bpm))[0]; ni = np.where(np.isnan(bpm))[0]
    if len(ni) and len(vi): bpm[ni] = np.interp(ni, vi, bpm[vi])
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


def downsample_hr(e): return e[np.arange(0, len(e), radarFpsQ)][:-1]
def compute_hr(sig):
    f = bandpass_filter(sig, BP_LOW, BP_HIGH, fps)
    ht = stft_time(numFrame, STFT_W, STFT_L, fps); hb = stft_bpm(numFrame, measuretime, fps)
    return downsample_hr(heartbeat_estimation(phase_stft(f, STFT_W, STFT_L, numFrame), hb, len(ht), kalmanR=KALMAN_R))
def sig_ma(d): return sma(d - sma(d, K1_OLD), K2_OLD)
def sig_sparse_param(sig, thr, win, guard):
    dc = np.asarray(sig, float).copy(); ap = True
    if guard and float(np.sum(detect_sparse_movement(dc, fps=fps, win_size=win, threshold_factor=thr))) / len(dc) < DET_GUARD: ap = False
    if ap:
        m = detect_sparse_movement(dc, fps=fps, win_size=win, threshold_factor=thr)
        dc = preprocess_signal(dc, m, method='interpolate', max_interp_len=MAX_INTERP)
    return sma(dc - sma(dc, K1_NEW), K2_NEW)
def gate_feats(dacm):
    dd = np.diff(dacm); r_ms = float(np.median(np.abs(dd - np.median(dd)))) / (np.std(dd) + 1e-12)
    fr, psd = welch(dacm, fs=fps, nperseg=min(256, len(dacm)), noverlap=min(128, len(dacm) // 2))
    tp = np.sum(psd); r_hr = float(np.sum(psd[(fr >= 0.8) & (fr <= 2.1)]) / tp) if tp > 0 else 0.0
    r_dt = float(np.std(dacm - sma(dacm, size=5)) / (np.std(dacm - sma(dacm, size=2)) + 1e-12))
    return r_ms, r_hr, r_dt
def route_of(dacm):
    a, b, c = gate_feats(dacm)
    return "Raw" if a > THR_MAD_STD else ("MA" if (b > THR_HR_POWER or c < THR_DETREND_5V2) else "Sparse")
def blocks(x): return np.array([np.mean(x[s:s + W]) for s in range(0, len(x) - W + 1, STRIDE)])
def berr(track, gb): yb = blocks(track); n = min(len(gb), len(yb)); return np.abs(gb[:n] - yb[:n])


# ---- cache: per recording label-independent parts + Sparse err for EVERY grid param ----
recs = []
for p in PERSONS:
    for cat in CATS:
        for tr in TRIALS:
            nm = f"{cat}_{p}_{tr}"
            try:
                mr = scipy.io.loadmat(f"{DATA}/{nm}_radar.mat"); mhr = scipy.io.loadmat(f"{DATA}/{nm}_ubpulse_heartbeat.mat")
                gt = extract_gt(mhr)
                if len(gt) == 0: raise ValueError
                dacm = dacm_phase_unwrapping(extract_target(mr['saveDataRadar1']))
            except Exception:
                continue
            gb = blocks(gt)
            r = {"subj": p, "cat": cat, "route": route_of(dacm),
                 "Raw": berr(compute_hr(dacm), gb), "MA": berr(compute_hr(sig_ma(dacm)), gb), "Sparse": {}}
            for prm in GRID:
                r["Sparse"][prm] = berr(compute_hr(sig_sparse_param(dacm, *prm)), gb)
            recs.append(r)
    print(f"  cached subject {p}: total recs so far {len(recs)}", flush=True)
print(f"cached recordings: {len(recs)}", flush=True)


def gate_err_list(subset, prm):
    """flatten gate block errors over subset for a given Sparse param."""
    out = []
    for r in subset:
        errs = {"Raw": r["Raw"], "MA": r["MA"], "Sparse": r["Sparse"][prm]}
        n = min(len(errs["Raw"]), len(errs["MA"]), len(errs["Sparse"]))
        out.extend(errs[r["route"]][:n].tolist())
    return out
def mae(subset, prm): return float(np.mean(gate_err_list(subset, prm)))


# ---- (1) fixed global config, per-subject held-out gate MAE ----
print("\n=== (1) FIXED global param (3.0,7,False) — per-subject gate MAE ===", flush=True)
per_subj_fixed = {}
for p in PERSONS:
    sub = [r for r in recs if r["subj"] == p]
    if not sub: continue
    per_subj_fixed[p] = mae(sub, GLOBAL)
    print(f"  {p:4s}  n={len(sub):2d}  gate MAE={per_subj_fixed[p]:.3f}", flush=True)
fixed_vals = list(per_subj_fixed.values())
print(f"  mean={np.mean(fixed_vals):.3f}  worst-subj={max(fixed_vals):.3f}  std={np.std(fixed_vals):.3f}")

# ---- (2) LOSO param re-selection ----
print("\n=== (2) LOSO: re-select Sparse param on 9 train subjects, test on held-out ===", flush=True)
loso = {}
for p in PERSONS:
    train = [r for r in recs if r["subj"] != p]
    test = [r for r in recs if r["subj"] == p]
    if not test: continue
    sel = min(GRID, key=lambda prm: mae(train, prm))       # re-select on train
    held = mae(test, sel)
    loso[p] = {"sel": list(sel), "heldout_mae": held, "same_as_global": (sel == GLOBAL)}
    print(f"  holdout {p:4s}  sel={sel}  held-out gate MAE={held:.3f}  {'(=global)' if sel==GLOBAL else '(DIFF)'}", flush=True)
held_vals = [v["heldout_mae"] for v in loso.values()]
n_same = sum(1 for v in loso.values() if v["same_as_global"])
print(f"\n  LOSO held-out gate MAE: mean={np.mean(held_vals):.3f}  worst={max(held_vals):.3f}  std={np.std(held_vals):.3f}")
print(f"  folds selecting global param (3.0,7,False): {n_same}/{len(loso)}")
print(f"  in-sample (pooled) gate MAE for reference: {mae(recs, GLOBAL):.3f}")

json.dump({"global_param": list(GLOBAL),
           "fixed_per_subject": per_subj_fixed,
           "fixed_summary": {"mean": float(np.mean(fixed_vals)), "worst": float(max(fixed_vals)), "std": float(np.std(fixed_vals))},
           "loso": loso,
           "loso_summary": {"mean_heldout": float(np.mean(held_vals)), "worst_heldout": float(max(held_vals)),
                            "std_heldout": float(np.std(held_vals)), "folds_same_as_global": n_same, "n_folds": len(loso)},
           "insample_pooled": mae(recs, GLOBAL)},
          open("/root/workspace/2026-1/IoT/results/loso_singleparam.json", "w"), indent=1, ensure_ascii=False)
print("\nsaved: results/loso_singleparam.json")

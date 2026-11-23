# -*- coding: utf-8 -*-
"""Diagnostics on the V1 pipeline: bin hopping, DACM form, Kalman=cumulative mean, null tests, zero-fill."""
import sys, json, numpy as np
from scipy.signal import lfilter, welch, medfilt
from scipy import stats
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import core
S = core.SP
from core import load, gt_device, blocks
import jyvital7_1 as jv
from jymas import sma
from shsparse import detect_sparse_movement, preprocess_signal

fps, NF = 20, 1200
AZ = [6, 4, 3, 2]
T1, T2, T3 = 0.508, 0.042, 1.509
radar, hr, ppg, meta = load()
N = len(meta)
BPM = jv.stft_bpm(NF, 60, fps)
BAND = (BPM >= 0) & (BPM <= 180)
HB = (BPM >= 48) & (BPM <= 126)


def mti(d):
    a = 0.01
    cl = lfilter([a], [1, -(1 - a)], d[1:], axis=0)
    out = np.zeros_like(d); out[1:] = d[1:] - cl
    return out


def targets(d):
    r = mti(d[:, AZ, :].astype(complex)); rm = np.mean(np.abs(r), 1)
    rp = np.argmax(rm, 1); rp[0] = 0
    dm = np.fft.fftshift(np.fft.fft(r, axis=1), 1)
    dp = np.array([0] + [int(np.argmax(np.abs(dm[i, :, rp[i]]))) for i in range(1, NF)])
    can = dm[np.arange(NF), dp, rp]
    rpf = stats.mode(rp[1:], keepdims=False).mode
    dpf = int(np.argmax(np.mean(np.abs(dm[1:, :, rpf]), 0)))
    fix = dm[:, dpf, rpf].copy(); fix[0] = fix[1]
    return can, fix, rp, dp


def stft_mag(sig):
    f = jv.bandpass_filter(sig, 0.8, 2.1, fps)
    return np.abs(jv.phase_stft(f, 100, 99, NF))  # (1200, 1101)


def ds(e): return e[::5][:-1]


def est_canon(M):
    pk = BPM[BAND][np.argmax(M[BAND], 0)]
    return ds(jv.kalman_filter(pk, R=0.10)), pk


def est_cummean(pk): return ds(np.cumsum(pk) / np.arange(1, len(pk) + 1))


def est_accum(M, whiten=False):
    P = np.cumsum(M[HB] ** 2, 1)
    if whiten:
        out = np.zeros(P.shape[1])
        for j in range(0, P.shape[1], 5):
            bg = np.exp(medfilt(np.log(P[:, j] + 1e-30), 41)); out[j:j + 5] = BPM[HB][np.argmax(P[:, j] / bg)]
        return ds(out)
    return ds(BPM[HB][np.argmax(P, 0)])


def p_ma(d): return sma(d - sma(d, 2), 2)


def p_sp(d, max_interp=20):
    m = detect_sparse_movement(d.copy(), fps=fps, win_size=7, threshold_factor=3.0)
    c = preprocess_signal(d.copy(), m, method='interpolate', max_interp_len=max_interp)
    return sma(c - sma(c, 5), 2), m


def feats(d):
    dd = np.diff(d); r_ms = np.median(np.abs(dd - np.median(dd))) / (np.std(dd) + 1e-12)
    fr, psd = welch(d, fs=fps, nperseg=256, noverlap=128)
    r_hr = psd[(fr >= 0.8) & (fr <= 2.1)].sum() / psd.sum()
    r_dt = np.std(d - sma(d, 5)) / (np.std(d - sma(d, 2)) + 1e-12)
    return r_ms, r_hr, r_dt


def route(d):
    a, b, c = feats(d)
    return "B" if a > T1 else ("D" if (b > T2 or c < T3) else "I")


def be(tr, gb): return np.abs(blocks(tr)[:len(gb)] - gb)


R = {}
def add(k, i, v): R.setdefault(k, [None] * N)[i] = v

hop = []; longseg = []; dacm_chk = []; psds = []
for i in range(N):
    d = radar[i]
    can, fix, rp, dp = targets(d)
    g = gt_device(hr[i]); gb = blocks(g); add("gt", i, gb); add("gtm", i, g.mean())
    ph_can = jv.dacm_phase_unwrapping(can)
    ph_fix = jv.dacm_phase_unwrapping(fix)
    ph_atan = np.unwrap(np.angle(fix))
    # --- bin hopping
    ch = np.zeros(NF, bool); ch[2:] = (rp[2:] != rp[1:-1]) | (dp[2:] != dp[1:-1])
    dphi = np.abs(np.diff(ph_can, prepend=ph_can[0]))
    top = dphi >= np.percentile(dphi, 95)
    sig_sp, mask = p_sp(ph_can)
    hop.append((ch.mean(), (rp[2:] != rp[1:-1]).mean(), ch[top].mean(), ch[mask].mean() if mask.any() else np.nan, mask.mean(), len(np.unique(rp[1:]))))
    # --- long segments zero-filled
    e = np.diff(np.concatenate([[0], mask.astype(int), [0]])); st = np.where(e == 1)[0]; en = np.where(e == -1)[0]
    L = en - st; longseg.append(((L > 20).sum(), L[L > 20].sum() / NF))
    # --- DACM vs true phase increment on canonical target
    tr = np.angle(can[2:] * np.conj(can[1:-1])); dd = np.diff(ph_can)[1:]
    ar = np.abs(can[1:-1]) / (np.abs(can[2:]) + 1e-12)
    dacm_chk.append((np.corrcoef(tr, dd)[0, 1], np.mean((ar > 2) | (ar < 0.5)), np.mean(np.abs(tr) > np.pi / 2), np.sum(np.abs(dd)) / np.sum(np.abs(tr))))
    f_, p_ = welch(ph_can - np.linspace(ph_can[0], ph_can[-1], NF), fs=fps, nperseg=1024, noverlap=512); psds.append(p_ / p_.sum())
    for tag, ph in (("can", ph_can), ("fix", ph_fix), ("atan", ph_atan)):
        sigs = {"B": ph, "D": p_ma(ph), "I": p_sp(ph)[0]}
        rt = route(ph); add(tag + "_route", i, rt)
        for k, s in sigs.items():
            M = stft_mag(s); tk, pk = est_canon(M)
            add(f"{tag}_{k}", i, be(tk, gb)); add(f"{tag}_{k}_m", i, tk.mean())
            if tag == "can":
                add(f"cm_{k}", i, be(est_cummean(pk), gb))
                add(f"acc_{k}", i, be(est_accum(M), gb)); add(f"acc_{k}_m", i, est_accum(M).mean())
                add(f"accw_{k}", i, be(est_accum(M, True), gb)); add(f"accw_{k}_m", i, est_accum(M, True).mean())
                add(f"pkstd_{k}", i, pk.std()); add(f"pkmed_{k}", i, np.median(pk))
        if tag == "can":
            add("I_nozero", i, be(est_canon(stft_mag(p_sp(ph, None)[0]))[0], gb))
    if i % 27 == 26: print(i + 1, flush=True)

out = {}
def P(k, v): out[k] = v; print(k, v, flush=True)
gt = np.array(R["gt"]); gtm = np.array(R["gtm"])
def mae(k): return float(np.mean(R[k]))
def slope(k): return [float(x) for x in np.round(np.polyfit(gtm, np.array(R[k]), 1)[0:1].tolist() + [np.corrcoef(gtm, np.array(R[k]))[0, 1]], 3)]
def gate(tag): return float(np.mean([R[f"{tag}_{R[tag + '_route'][i]}"][i] for i in range(N)]))
for tag in ("can", "fix", "atan"):
    P(f"{tag} MAE B/D/I/gate", [round(mae(f"{tag}_{k}"), 3) for k in "BDI"] + [round(gate(tag), 3)])
    P(f"{tag} mean est B/D/I", [round(float(np.mean(R[f'{tag}_{k}_m'])), 1) for k in "BDI"])
    P(f"{tag} slope,corr B/D/I", [slope(f"{tag}_{k}_m") for k in "BDI"])
    P(f"{tag} routes", {k: R[tag + "_route"].count(k) for k in "BDI"})
    P(f"{tag} oracle", float(np.mean(np.min(np.stack([np.array(R[f"{tag}_{k}"]) for k in "BDI"]), 0))))
P("cummean==kalman MAE B/D/I", [round(mae(f"cm_{k}"), 3) for k in "BDI"])
P("spectral accumulation MAE B/D/I", [round(mae(f"acc_{k}"), 3) for k in "BDI"])
P("accum slope,corr", [slope(f"acc_{k}_m") for k in "BDI"])
P("whitened accumulation MAE B/D/I", [round(mae(f"accw_{k}"), 3) for k in "BDI"])
P("whitened accum slope,corr", [slope(f"accw_{k}_m") for k in "BDI"])
P("whitened accum mean est", [round(float(np.mean(R[f'accw_{k}_m'])), 1) for k in "BDI"])
P("frame-peak std within recording (median) B/D/I", [round(float(np.median(R[f'pkstd_{k}'])), 1) for k in "BDI"])
P("Impulse no zero-fill MAE", round(mae("I_nozero"), 3))
h = np.array(hop, float)
P("bin change rate (any, range) mean", [round(float(h[:, 0].mean()), 3), round(float(h[:, 1].mean()), 3)])
P("bin change rate among top5% |dphi| / among detected impulses / detect ratio", [round(float(np.nanmean(h[:, j])), 3) for j in (2, 3, 4)])
P("n distinct range bins per recording (median, max)", [float(np.median(h[:, 5])), float(h[:, 5].max())])
P("recs with zero rng hop", int((h[:, 1] == 0).sum()))
P("corr(bin change rate, R_MS route==B)", float(np.corrcoef(h[:, 0], [r == "B" for r in R["can_route"]])[0, 1]))
ls = np.array(longseg, float)
P("recs with >20-sample segs zero-filled / mean share of samples zeroed (those recs)", [int((ls[:, 0] > 0).sum()), round(float(ls[ls[:, 0] > 0, 1].mean()), 3)])
dc = np.array(dacm_chk)
P("DACM vs true dphi: median corr, frac amp-ratio outside[0.5,2], frac |dphi|>pi/2, sum|dacm|/sum|true|", [round(float(np.median(dc[:, j])), 3) for j in range(4)])

# --- null 1: shuffled references (other subjects)
rng = np.random.default_rng(0); pers = np.array([m[0] for m in meta])
gate_e = np.array([R[f"can_{R['can_route'][i]}"][i] for i in range(N)])
def trk_block_est(k): return None
# need block estimates -> recompute from err is impossible; store est blocks instead via gt +/-: skip, use est mean per recording
gm = np.array([R[f"can_{R['can_route'][i]}_m"][i] for i in range(N)])
true_rec = float(np.mean(np.abs(gm - gtm)))
sh = []
for _ in range(200):
    perm = rng.permutation(N)
    ok = pers[perm] != pers
    sh.append(np.mean(np.abs(gm[ok] - gtm[perm][ok])))
P("recording-level |est-ref|: true pairing vs other-subject pairing (mean of 200 perms)", [round(true_rec, 2), round(float(np.mean(sh)), 2)])
P("recording-level |79-ref|", round(float(np.mean(np.abs(79 - gtm))), 2))

# --- null 2: heartbeat-free coloured noise with the median PSD shape
f_ = np.fft.rfftfreq(1024, 1 / fps); med = np.median(np.array(psds), 0)
fr = np.fft.rfftfreq(NF, 1 / fps); amp = np.sqrt(np.interp(fr, f_, medfilt(med, 31))); amp[0] = 0
nl = {k: [] for k in "BDI"}
for _ in range(60):
    x = np.fft.irfft(amp * np.exp(1j * rng.uniform(0, 2 * np.pi, len(fr))), NF)
    for k, s in (("B", x), ("D", p_ma(x)), ("I", p_sp(x)[0])):
        nl[k].append(est_canon(stft_mag(s))[0][20:].mean())
P("heartbeat-free coloured noise -> mean estimate B/D/I", [round(float(np.mean(nl[k])), 1) for k in "BDI"])
P("  (std across realisations)", [round(float(np.std(nl[k])), 1) for k in "BDI"])
json.dump(out, open(S + "/diag1.json", "w"), indent=1, ensure_ascii=False)
np.savez(S + "/diag1_R.npz", **{k: np.array(v) for k, v in R.items()})

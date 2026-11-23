# -*- coding: utf-8 -*-
"""Ablation / control variants of the new estimator: block-protocol fused spectra (no reference used here).
usage: python nm_ablate.py <L> <hop> <p> <ww> <out.npz>"""
import sys
from nm_common import *
from scipy.ndimage import uniform_filter1d
from scipy.signal import lfilter
from nm_blocks import fuse, TB
import jyvital7_1 as jv
from shsparse import detect_sparse_movement, preprocess_signal
L, HOP, PW, WW = int(sys.argv[1]), int(sys.argv[2]), float(sys.argv[3]), int(sys.argv[4])
radar, hr, ppg, meta = load(); rng = np.random.default_rng(0)
AZ = [6, 4, 3, 2]
def F_of(v, p=PW, ww=WW, L_=L, hop=HOP): return fuse(seg_psd(v, seg=L_, hop=hop)[0], p, ww)
def nowhite(v):
    P = seg_psd(v, seg=L, hop=HOP)[0]; wt = 1.0 / (P[:, :, BAND].mean(-1, keepdims=True) ** PW + 1e-30)
    Pm = (P * wt).sum(0) / wt.sum(0); return (Pm / Pm[:, BAND].mean(1, keepdims=True)).mean(0)
def interp_cells(x):
    ph = np.cumsum(phase_vel(x), 0); out = np.zeros_like(ph)
    for j in range(ph.shape[1]):
        m = detect_sparse_movement(ph[:, j].copy(), fps=20, win_size=7, threshold_factor=3.0)
        out[:, j] = preprocess_signal(ph[:, j].copy(), m, method='interpolate', max_interp_len=None)
    v = np.zeros_like(out); v[1:] = np.diff(out, axis=0); return v
def canon_dacm(d):
    a = 0.01; dd = d[:, AZ, :]; cl = lfilter([a], [1, -(1 - a)], dd[1:], axis=0); r = np.zeros_like(dd); r[1:] = dd[1:] - cl
    rp = int(np.argmax(np.mean(np.abs(r), 1).mean(0))); dm = np.fft.fftshift(np.fft.fft(r, axis=1), 1)   # target cell fixed for the recording
    dp = int(np.argmax(np.abs(dm[:, :, rp]).mean(0)))
    ph = jv.dacm_phase_unwrapping(dm[:, dp, rp]); v = np.zeros((len(d), 1)); v[1:, 0] = np.diff(ph); return v
def mti_cells(d, lo=2, hi=6):
    r0 = centre_bin(d); bins = [r for r in range(r0 - lo, r0 + hi + 1) if 1 <= r <= 31]; dd = d[:, :, bins]
    a = 0.01; cl = lfilter([a], [1, -(1 - a)], dd[1:], axis=0); r = np.zeros_like(dd); r[1:] = dd[1:] - cl
    return r.transpose(0, 2, 1).reshape(len(d), -1)
def surrogate(v):
    X = np.fft.rfft(v, axis=0); k = max(3, int(round(61 * len(v) / 1200)) | 1)
    A = np.sqrt(median_filter(np.abs(X) ** 2, (k, 1), mode="nearest"))
    s = np.fft.irfft(A * np.exp(1j * rng.uniform(0, 2 * np.pi, A.shape)), len(v), axis=0)
    env = np.sqrt(uniform_filter1d(v ** 2, 20, axis=0)); return s * env / (np.sqrt(uniform_filter1d(s ** 2, 20, axis=0)) + 1e-12)
NAMES = ["default", "p0", "p0.5", "p2", "cell_1", "cell_bin", "cell_az4", "ext_11", "ext_22", "ext_48", "nowhite", "interp", "canon1", "agc", "mti", "null_shape", "null_white", "L200_p0"]
out = {k: np.zeros((270, 3, len(G)), np.float32) for k in NAMES}
for i in range(270):
    for b, T in enumerate(TB):
        d = radar[i][:T].astype(complex); x, rb, ch = cells(d); v = phase_vel(x); r0 = centre_bin(d)
        P = seg_psd(v, seg=L, hop=HOP)[0]
        o = lambda k, F: out[k].__setitem__((i, b), F)
        o("default", fuse(P, PW, WW)); o("p0", fuse(P, 0, WW)); o("p0.5", fuse(P, 0.5, WW)); o("p2", fuse(P, 2, WW))
        o("cell_1", fuse(P[:, (rb == r0) & (ch == 6)], PW, WW)); o("cell_bin", fuse(P[:, rb == r0], PW, WW)); o("cell_az4", fuse(P[:, np.isin(ch, AZ)], PW, WW))
        o("ext_11", fuse(P[:, np.abs(rb - r0) <= 1], PW, WW)); o("ext_22", fuse(P[:, np.abs(rb - r0) <= 2], PW, WW))
        o("ext_48", F_of(phase_vel(cells(d, 4, 8)[0]))); o("nowhite", nowhite(v)); o("interp", F_of(interp_cells(x)))
        o("canon1", F_of(canon_dacm(d))); o("mti", F_of(phase_vel(mti_cells(d))))
        o("agc", fuse(seg_psd(v / np.sqrt(uniform_filter1d(v ** 2, 20, axis=0) + 1e-12), seg=200, hop=100)[0], 0, WW))
        o("null_shape", F_of(surrogate(v))); o("null_white", F_of(rng.standard_normal(v.shape) * np.sqrt(uniform_filter1d(v ** 2, 20, axis=0))))
        o("L200_p0", fuse(seg_psd(v, seg=200, hop=100)[0], 0, 81))
    if i % 27 == 26: print(i + 1, flush=True)
np.savez(sys.argv[5], **out)

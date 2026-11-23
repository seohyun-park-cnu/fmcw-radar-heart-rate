# -*- coding: utf-8 -*-
"""Shared pieces for the new-method development (reference-free front end + spectra)."""
import sys, numpy as np
from scipy.ndimage import median_filter
from scipy.signal import butter, sosfiltfilt, hilbert
import os
# cache folder: set NM_CACHE, default <this folder>/cache (large intermediate files; safe to delete)
S = os.environ.get("NM_CACHE", os.path.join(os.path.dirname(os.path.abspath(__file__)), "cache")).replace("\\", "/")
os.makedirs(S, exist_ok=True)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import core
core.SP = S
from core import load, gt_device, ppg_hr_track, blocks, PERSONS, CATS
FPS = 20
DEV = ['SH', 'JY', 'ET', 'JH', 'YW']          # design subjects (even index) - fixed before any result of the new method was seen
HOLD = ['BG', 'HJ', 'JB', 'JHL', 'SJ']        # untouched until the final run
G = np.arange(40, 380.25, 0.5)                # BPM grid (covers up to the 3rd harmonic of 126)
NFFT = 4096
FR = np.fft.rfftfreq(NFFT, 1 / FPS) * 60
BAND = (G >= 48) & (G <= 126); Gb = G[BAND]


def centre_bin(d):
    """range bin with the largest AC power (reference-free)."""
    x = d - d.mean(0)
    return int(np.argmax(np.mean(np.abs(x[20:]) ** 2, (0, 1))[1:16])) + 1


def cells(d, lo=2, hi=6):
    """AC part of the cells around the centre bin: returns (T, ncell) complex, and (bin, ch) index arrays."""
    r0 = centre_bin(d); bins = [r for r in range(r0 - lo, r0 + hi + 1) if 1 <= r <= 31]
    x = d[:, :, bins].astype(complex); x = x - x.mean(0)
    return x.transpose(0, 2, 1).reshape(len(d), -1), np.repeat(bins, d.shape[1]), np.tile(np.arange(d.shape[1]), len(bins))


def phase_vel(x):
    v = np.zeros(x.shape); v[1:] = np.angle(x[1:] * np.conj(x[:-1])); return v


def seg_psd(v, seg=200, hop=100, t0=0, t1=None):
    """periodograms of Hann segments: v (T, n) -> (nseg, n, len(G))."""
    t1 = v.shape[0] if t1 is None else t1
    st = np.arange(t0, t1 - seg + 1, hop); w = np.hanning(seg)[:, None]
    out = np.zeros((len(st), v.shape[1], len(G)), np.float32)
    pos = G / (FR[1] - FR[0]); i0 = np.floor(pos).astype(int); fr = (pos - i0)[:, None]
    for k, s in enumerate(st):
        X = v[s:s + seg]; P = np.abs(np.fft.rfft((X - X.mean(0)) * w, NFFT, axis=0)) ** 2
        out[k] = (P[i0] * (1 - fr) + P[i0 + 1] * fr).T
    return out, st


def whiten(P, width=81):
    """divide by the running median along frequency (last axis)."""
    sz = [1] * (P.ndim - 1) + [width]
    return P / (median_filter(P, sz, mode="nearest") + 1e-30)


def harm(F, w2=1.0, w3=0.0):
    """harmonic summation on the 48..126 grid from a spectrum on G (last axis)."""
    s = F[..., BAND].copy()
    if w2: s = s + w2 * np.apply_along_axis(lambda a: np.interp(2 * Gb, G, a), -1, F)
    if w3: s = s + w3 * np.apply_along_axis(lambda a: np.interp(3 * Gb, G, a), -1, F)
    return s


_I2 = None
def hsum(F, w=(1.0, 1.0, 0.0)):
    """weighted harmonic summation on the 48..126 grid: w[0]*F(f) + w[1]*F(2f) + w[2]*F(3f). F on G (last axis), G step 0.5 BPM."""
    k0 = np.where(BAND)[0]; g0 = int(round((Gb[0] - G[0]) * 2))
    s = w[0] * F[..., k0]
    if w[1]: s = s + w[1] * F[..., (np.round((2 * Gb - G[0]) * 2)).astype(int)]
    if len(w) > 2 and w[2]: s = s + w[2] * F[..., (np.round((3 * Gb - G[0]) * 2)).astype(int)]
    return s

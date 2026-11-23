# -*- coding: utf-8 -*-
"""Shared helpers for the development experiments (canonical pipeline pieces + references)."""
import os, sys, json, numpy as np
from scipy.signal import butter, filtfilt, find_peaks, sosfiltfilt
import importlib.util, glob
_HERE = os.path.dirname(os.path.abspath(__file__)).replace("\\", "/")
CODE = _HERE + "/../1_수정전_경로선택방식"          # signal-processing modules of the earlier pipeline
def _load(name):
    path = glob.glob(f"{CODE}/{name}(*.py")[0]
    spec = importlib.util.spec_from_file_location(name, path); m = importlib.util.module_from_spec(spec); sys.modules[name] = m; spec.loader.exec_module(m); return m
for _n in ("jymas", "jyvital7_1", "shsparse"): _load(_n)
SP = os.environ.get("NM_CACHE", _HERE + "/cache").replace("\\", "/")   # large intermediate files; safe to delete
os.makedirs(SP, exist_ok=True)
REPRO = _HERE + "/baseline_tracks"
FPS = 20
CATS = ['static', 'arm', 'head', 'cough', 'shoulder_shrug', 'disturb', 'swing', 'transition', 'water']
PERSONS = ['SH', 'BG', 'JY', 'HJ', 'ET', 'JB', 'JH', 'JHL', 'YW', 'SJ']

def load():
    z = np.load(SP + "/all270.npz")
    return z["radar"], z["hr"], z["ppg"], [tuple(m) for m in z["meta"]]

def gt_device(h):
    """canonical reference: device HR (256 Hz) -> >150 interpolated -> 4 Hz -> drop first 5 s. returns 220 samples (t = 5.0 .. 59.75 s)"""
    bpm = h.astype(float).copy(); bpm[bpm > 150] = np.nan
    vi = np.where(~np.isnan(bpm))[0]; ni = np.where(np.isnan(bpm))[0]
    if len(ni) and len(vi): bpm[ni] = np.interp(ni, vi, bpm[vi])
    return bpm[::64][20:]

def ppg_beats(ppg, fs=256):
    """beat times (s) from the raw PPG waveform: 0.5-4 Hz band-pass, systolic peaks with adaptive prominence."""
    x = ppg.astype(float); x = x - np.mean(x)
    sos = butter(3, [0.5 / (fs / 2), 4.0 / (fs / 2)], btype="band", output="sos")
    y = sosfiltfilt(sos, x)
    s = 1.4826 * np.median(np.abs(y - np.median(y))) + 1e-9
    pk, _ = find_peaks(y, distance=int(fs * 0.33), prominence=0.5 * s)
    return pk / fs, y

def ppg_hr_track(ppg, fs=256, tgrid=None, win=8.0):
    """HR(t) from the PPG waveform by spectral peak in a sliding window ending at t (robust to missed beats)."""
    if tgrid is None: tgrid = np.arange(5.0, 60.0, 0.25)
    x = ppg.astype(float); x = x - np.mean(x)
    sos = butter(3, [0.5 / (fs / 2), 4.0 / (fs / 2)], btype="band", output="sos")
    y = sosfiltfilt(sos, x)[::8]; f2 = fs // 8      # 32 Hz
    n = 8192; fr = np.fft.rfftfreq(n, 1 / f2) * 60; band = (fr >= 40) & (fr <= 180)
    out = np.zeros(len(tgrid)); q = np.zeros(len(tgrid))
    for i, t in enumerate(tgrid):
        e = int(round(t * f2)); s = max(0, e - int(win * f2)); seg = y[s:e]
        P = np.abs(np.fft.rfft((seg - seg.mean()) * np.hanning(len(seg)), n)) ** 2
        k = np.argmax(P[band]); out[i] = fr[band][k]
        pk = fr[band][k]; q[i] = P[band][(fr[band] > pk - 6) & (fr[band] < pk + 6)].sum() / P[band].sum()
    return out, q

def blocks(x, W=60): return np.array([np.mean(x[s:s + W]) for s in range(0, len(x) - W + 1, W)])

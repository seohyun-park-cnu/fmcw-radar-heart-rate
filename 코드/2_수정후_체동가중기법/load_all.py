# -*- coding: utf-8 -*-
"""Cache all 270 recordings (raw radar cube, device HR, raw PPG) into one npz. Usage: python load_all.py <out.npz>"""
import os, sys, numpy as np, scipy.io
CATS = ['static', 'arm', 'head', 'cough', 'shoulder_shrug', 'disturb', 'swing', 'transition', 'water']
TRIALS = ['1st', '2nd', '3rd']
PERSONS = ['SH', 'BG', 'JY', 'HJ', 'ET', 'JB', 'JH', 'JHL', 'YW', 'SJ']
DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "데이터")
R, H, P, meta = [], [], [], []
for p in PERSONS:
    for c in CATS:
        for t in TRIALS:
            nm = f"{c}_{p}_{t}"
            r = scipy.io.loadmat(f"{DATA}/{nm}_radar.mat")['saveDataRadar1']
            h = scipy.io.loadmat(f"{DATA}/{nm}_ubpulse_heartbeat.mat")['saveDataHeartRateUbpulse1'].ravel()
            g = scipy.io.loadmat(f"{DATA}/{nm}_ubpulse_ppg.mat")['saveDataPpgUbpulse1'].ravel()
            if r.shape != (1200, 7, 32) or len(h) != 15360 or len(g) != 15360: print("shape!", nm, r.shape, len(h), len(g))
            R.append(r.astype(np.complex64)); H.append(h.astype(np.uint8)); P.append(g.astype(np.uint16)); meta.append((p, c, t, nm))
    print(p, len(meta), flush=True)
os.makedirs(os.path.dirname(os.path.abspath(sys.argv[1])), exist_ok=True)
np.savez(sys.argv[1], radar=np.array(R), hr=np.array(H), ppg=np.array(P), meta=np.array(meta))

# -*- coding: utf-8 -*-
"""Causal check: inject a synthetic heartbeat-like displacement at a RANDOM rate into the real recordings.
If the estimator measures periodic chest displacement, its output must follow the injected rate once the injected amplitude is large enough.
Recording level (60 s). usage: python nm_inject.py <L> <hop> <p> <ww> <h1,h2,h3>"""
import sys, json
from nm_common import *
from nm_blocks import fuse
L, HOP, PW, WW = int(sys.argv[1]), int(sys.argv[2]), float(sys.argv[3]), int(sys.argv[4]); H = tuple(float(a) for a in sys.argv[5].split(","))
radar, hr, ppg, meta = load(); Z = np.load(S + "/refs.npz"); ref = Z["gp"].mean(1); rng = np.random.default_rng(7)
LAM = 5000.0  # um
def beat_wave(f_bpm, n=1200, jitter=0.03):
    """unit peak-to-peak pulse train (Gaussian pulse, sigma 60 ms) with 3 % beat-to-beat jitter."""
    t = np.arange(n) / FPS; ibi = 60.0 / f_bpm; bt = np.cumsum(ibi * (1 + jitter * rng.standard_normal(int(70 / ibi)))) - rng.uniform(0, ibi)
    w = np.exp(-0.5 * ((t[:, None] - bt[None, :]) / 0.06) ** 2).sum(1); return (w - w.min()) / (w.max() - w.min())
AMPS = [0, 25, 50, 100, 200, 400]
res = {a: [] for a in AMPS}; rows = []
for i in range(270):
    d = radar[i].astype(complex); x, rb, ch = cells(d)
    cand = np.arange(52, 123); cand = cand[np.abs(cand - ref[i]) >= 15]; f = float(rng.choice(cand)); w = beat_wave(f)
    for a in AMPS:
        v = phase_vel(x * np.exp(1j * 4 * np.pi * a * w / LAM)[:, None])
        e = Gb[int(np.argmax(hsum(fuse(seg_psd(v, seg=L, hop=HOP)[0], PW, WW), H)))]; res[a].append(e)
    rows.append((ref[i], f))
    if i % 27 == 26: print(i + 1, flush=True)
rows = np.array(rows); out = {}
print("amp (um p-p) | follows injected rate (within 5 BPM) | still at true HR (within 5 BPM) | corr(est, injected)")
for a in AMPS:
    e = np.array(res[a]); out[a] = [float(np.mean(np.abs(e - rows[:, 1]) <= 5)), float(np.mean(np.abs(e - rows[:, 0]) <= 5)), float(np.corrcoef(e, rows[:, 1])[0, 1])]
    print(f"  {a:4d}   {out[a][0]:.3f}   {out[a][1]:.3f}   {out[a][2]:+.2f}")
json.dump(out, open(S + "/nm_inject.json", "w"), indent=1)

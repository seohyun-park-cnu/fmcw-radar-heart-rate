# -*- coding: utf-8 -*-
"""DEV subjects only: which representation exposes the heartbeat? recording-level (10..60 s), no prior."""
from nm_common import *
radar, hr, ppg, meta = load(); Z = np.load(S + "/refs.npz"); subj = Z["subj"]; ref = Z["gp"].mean(1)
idx = [i for i in range(270) if subj[i] in DEV]
sos_hi = butter(4, [2.5 / 10, 8.0 / 10], btype="band", output="sos")
res = {}
def put(k, i, s): res.setdefault(k, {})[i] = Gb[int(np.argmax(s))]
for n, i in enumerate(idx):
    d = radar[i].astype(complex)
    x, rb, ch = cells(d); v = phase_vel(x)
    P, st = seg_psd(v, t0=0)                                  # (11, ncell, nf)
    Pm = P.mean(0); W = whiten(Pm); nb = len(np.unique(rb))
    r0 = centre_bin(d); c0 = rb == r0
    put("1cell(ch6) fund", i, W[(rb == r0) & (ch == 6)][0][BAND])
    put("centre bin 7ch fund", i, W[c0].mean(0)[BAND])
    put("all cells fund (no whiten)", i, (Pm / Pm[:, BAND].mean(1, keepdims=True)).mean(0)[BAND])
    put("all cells fund", i, W.mean(0)[BAND])
    put("all cells fund+h2", i, harm(W.mean(0), 1, 0))
    put("all cells fund+h2+h3", i, harm(W.mean(0), 1, 1))
    put("all cells h2 only", i, harm(W.mean(0), 1, 0) - W.mean(0)[BAND])
    put("all cells geo-mean fund+h2", i, harm(np.exp(np.log(W + 1e-30).mean(0)), 1, 0))
    # robust over time: median over segments, then whiten
    Wmed = whiten(np.median(P, 0)); put("segment-median fund+h2", i, harm(Wmed.mean(0), 1, 0))
    # whiten each segment separately, then average (each segment contributes equally -> motion bursts cannot dominate)
    Ws = whiten(P); put("per-segment whiten fund+h2", i, harm(Ws.mean((0, 1)), 1, 0))
    put("per-segment whiten, seg-median fund+h2", i, harm(np.median(Ws, 0).mean(0), 1, 0))
    # motion-aware segment weights: inverse broadband power of the segment (per cell)
    wt = 1.0 / (P[:, :, BAND].mean(-1, keepdims=True) + 1e-30); Pw = (P * wt).sum(0) / wt.sum(0)
    put("inverse-power segment weights fund+h2", i, harm(whiten(Pw).mean(0), 1, 0))
    # eigen-beamformed channels per range bin, then phase
    yb = []
    for r in np.unique(rb):
        y = x[:, rb == r]; w = np.linalg.eigh(y[20:].conj().T @ y[20:])[1][:, -1]; yb.append(y @ w.conj())
    Pb, _ = seg_psd(phase_vel(np.array(yb).T)); put("eig-beamformed bins fund+h2", i, harm(whiten(Pb.mean(0)).mean(0), 1, 0))
    # coherent: power-weighted mean of phase velocities across cells
    pw = np.mean(np.abs(x) ** 2, 0); vc = (v * pw).sum(1, keepdims=True) / pw.sum()
    Pc, _ = seg_psd(vc); put("coherent power-weighted fund+h2", i, harm(whiten(Pc.mean(0))[0], 1, 0))
    # envelope of the 2.5-8 Hz band of the phase velocity
    env = np.abs(hilbert(sosfiltfilt(sos_hi, v, axis=0), axis=0)); Pe, _ = seg_psd(env)
    put("high-band envelope fund", i, whiten(Pe.mean(0)).mean(0)[BAND])
    put("high-band envelope + phase fund+h2", i, whiten(Pe.mean(0)).mean(0)[BAND] / np.mean(whiten(Pe.mean(0)).mean(0)[BAND]) + harm(W.mean(0), 1, 0) / np.mean(harm(W.mean(0), 1, 0)))
    # amplitude-weighted cells (weight = cell power)
    put("power-weighted cells fund+h2", i, harm((W * pw[:, None]).sum(0) / pw.sum(), 1, 0))
    # cell weight = own spectral peakiness (max/median in band)
    pk = W[:, BAND].max(1) ** 2; put("peakiness-weighted cells fund+h2", i, harm((W * pk[:, None]).sum(0) / pk.sum(), 1, 0))
    # wider / narrower range extents
    for lo, hi in ((0, 0), (1, 1), (2, 2), (2, 4), (4, 8)):
        x2, _, _ = cells(d, lo, hi); P2, _ = seg_psd(phase_vel(x2)); put(f"extent -{lo}..+{hi} fund+h2", i, harm(whiten(P2.mean(0)).mean(0), 1, 0))
    if n % 27 == 26: print(n + 1, flush=True)
r = ref[idx]
print(f"{'representation':44s}  hit5   hit10  MAE    corr  slope   (DEV, n={len(idx)}; const79: hit5 {np.mean(np.abs(79-r)<=5):.2f} MAE {np.abs(79-r).mean():.2f})")
for k, v in res.items():
    e = np.array([v[i] for i in idx]); a = np.abs(e - r)
    print(f"{k:44s}  {np.mean(a<=5):.3f}  {np.mean(a<=10):.3f}  {a.mean():5.2f}  {np.corrcoef(e, r)[0,1]:5.2f}  {np.polyfit(r, e, 1)[0]:5.2f}")

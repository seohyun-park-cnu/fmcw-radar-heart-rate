# -*- coding: utf-8 -*-
"""Block-protocol fused spectra for a grid of configurations (no reference used here).
Block b (b=0,1,2) is evaluated on [5+15b, 20+15b] s; its spectrum uses ONLY frames [0, T_b), T_b = (20+15b) s:
clutter mean, centre bin, segment weights and whitening are all computed from that prefix (strictly causal per block).
usage: python nm_blocks.py <out.npz> <all|dev> <hop: half|N samples> <L list e.g. 40,60>"""
import sys
from nm_common import *
TB = (400, 700, 1000)
def name(c): return f"L{c['L']}_p{c['p']}_w{c['ww']}_{c['mode']}"
def fuse(P, p, ww=81):
    """P (nseg, ncell, nf) -> fused whitened spectrum on G. segment weight = (in-band power of the segment)^-p, per cell."""
    wt = 1.0 / (P[:, :, BAND].mean(-1, keepdims=True) ** p + 1e-30) if p else np.ones((P.shape[0], P.shape[1], 1))
    return whiten((P * wt).sum(0) / wt.sum(0), ww).mean(0)
if __name__ == "__main__":
    radar, hr, ppg, meta = load(); subj = np.array([m[0] for m in meta])
    which = sys.argv[2]; hop = sys.argv[3]; LS = [int(a) for a in sys.argv[4].split(",")]
    PS = [float(a) for a in sys.argv[5].split(",")] if len(sys.argv) > 5 else [0, 0.5, 1, 2]
    MODES = sys.argv[6].split(",") if len(sys.argv) > 6 else ["cum", "trail"]
    idx = [i for i in range(270) if which == "all" or subj[i] in DEV]
    GRID = [dict(L=L, p=p, ww=ww, mode=mode) for L in LS for p in PS for ww in (81, 161) for mode in MODES]
    out = {name(c): np.zeros((270, 3, len(G)), np.float32) for c in GRID}
    for n, i in enumerate(idx):
        for b, T in enumerate(TB):
            d = radar[i][:T].astype(complex); x, rb, ch = cells(d); v = phase_vel(x)
            for L in LS:
                P, st = seg_psd(v, seg=L, hop=L // 2 if hop == "half" else int(hop))
                tr = st >= T - 400
                for c in GRID:
                    if c["L"] == L: out[name(c)][i, b] = fuse(P if c["mode"] == "cum" else P[tr], c["p"], c["ww"])
        if n % 27 == 26: print(n + 1, flush=True)
    np.savez(sys.argv[1], **out)

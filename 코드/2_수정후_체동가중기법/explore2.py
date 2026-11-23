# -*- coding: utf-8 -*-
"""DEV only: variants of motion-aware weighting. recording level, no prior."""
from nm_common import *
from scipy.ndimage import uniform_filter1d
radar, hr, ppg, meta = load(); Z = np.load(S + "/refs.npz"); subj = Z["subj"]; ref = Z["gp"].mean(1)
idx = [i for i in range(270) if subj[i] in DEV]
res = {}
def put(k, i, s): res.setdefault(k, {})[i] = Gb[int(np.argmax(s))]
def wfuse(P, p=1.0, on="band", joint=False, pre_white=False, w3=0):
    pw = P[:, :, BAND].mean(-1, keepdims=True) if on == "band" else P.mean(-1, keepdims=True)
    wt = 1.0 / (pw ** p + 1e-30)
    if pre_white:
        W = whiten(P)
        F = (W * wt).sum((0, 1)) / wt.sum() if joint else ((W * wt).sum(0) / wt.sum(0)).mean(0)
        return harm(F, 1, w3)
    if joint:
        Wc = whiten((P * wt).sum(0) / wt.sum(0)); cw = wt.mean(0) / wt.mean(0).sum() * P.shape[1]
        return harm((Wc * cw).mean(0), 1, w3)
    return harm(whiten((P * wt).sum(0) / wt.sum(0)).mean(0), 1, w3)
for n, i in enumerate(idx):
    d = radar[i].astype(complex); x, rb, ch = cells(d); v = phase_vel(x)
    for L in (60, 100, 200, 300):
        P, _ = seg_psd(v, seg=L, hop=L // 2)
        for p in (0, 0.5, 1, 2, 4):
            put(f"L={L/20:.0f}s p={p}", i, wfuse(P, p))
        if L == 200:
            put("L=10s p=1 weight on total power", i, wfuse(P, 1, on="tot"))
            put("L=10s p=1 +h3(0.5)", i, wfuse(P, 1, w3=0.5))
            put("L=10s p=1 whiten per segment", i, wfuse(P, 1, pre_white=True))
            put("L=10s p=1 whiten per segment, joint cell weights", i, wfuse(P, 1, pre_white=True, joint=True))
            put("L=10s p=2 whiten per segment", i, wfuse(P, 2, pre_white=True))
        if L == 100:
            put("L=5s p=1 whiten per segment", i, wfuse(P, 1, pre_white=True))
            put("L=5s p=2 whiten per segment", i, wfuse(P, 2, pre_white=True))
    # time-domain normalisation (AGC) by local RMS, then plain Welch
    for win in (20, 40, 80):
        s = np.sqrt(uniform_filter1d(v ** 2, win, axis=0) + 1e-12); P, _ = seg_psd(v / s)
        put(f"AGC local RMS {win/20:.0f}s", i, harm(whiten(P.mean(0)).mean(0), 1, 0))
    # clipping at k*MAD (per cell)
    for k in (1.0, 2.0, 3.0):
        m = 1.4826 * np.median(np.abs(v - np.median(v, 0)), 0) + 1e-12; P, _ = seg_psd(np.clip(v, -k * m, k * m))
        put(f"clip {k} MAD", i, harm(whiten(P.mean(0)).mean(0), 1, 0))
        if k == 2.0:
            for p in (1, 2): put(f"clip 2 MAD + p={p}", i, wfuse(P, p))
    put("sign(v)", i, harm(whiten(seg_psd(np.sign(v))[0].mean(0)).mean(0), 1, 0))
    if n % 27 == 26: print(n + 1, flush=True)
r = ref[idx]
print(f"{'variant':50s}  hit5   hit10  MAE    corr  slope")
for k, v in res.items():
    e = np.array([v[i] for i in idx]); a = np.abs(e - r)
    print(f"{k:50s}  {np.mean(a<=5):.3f}  {np.mean(a<=10):.3f}  {a.mean():5.2f}  {np.corrcoef(e, r)[0,1]:5.2f}  {np.polyfit(r, e, 1)[0]:5.2f}")

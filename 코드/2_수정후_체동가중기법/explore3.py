# -*- coding: utf-8 -*-
"""DEV only: sanity checks on the short-segment inverse-power weighting (recording level, no prior)."""
from nm_common import *
from scipy.ndimage import uniform_filter1d
radar, hr, ppg, meta = load(); Z = np.load(S + "/refs.npz"); subj = Z["subj"]; ref = Z["gp"].mean(1)
idx = [i for i in range(270) if subj[i] in DEV]; rng = np.random.default_rng(0)
def fuse(P, p=1, ww=81):
    wt = 1.0 / (P[:, :, BAND].mean(-1, keepdims=True) ** p + 1e-30)
    return whiten((P * wt).sum(0) / wt.sum(0), ww).mean(0)
E = {}
def put(k, i, F, w2=1.0): E.setdefault(k, {})[i] = Gb[int(np.argmax(harm(F, w2, 0)))]
Fs = {}
for n, i in enumerate(idx):
    d = radar[i].astype(complex); x, rb, ch = cells(d); v = phase_vel(x)
    for L in (40, 60, 100):
        P, _ = seg_psd(v, seg=L, hop=L // 2)
        for ww in (81, 161, 321):
            F = fuse(P, 1, ww); put(f"L={L/20:.0f}s w={ww//2}BPM", i, F)
            if L == 60 and ww == 81: Fs[i] = F; put("L=3s fund only", i, F, 0.0); put("L=3s h2=0.5", i, F, 0.5)
    P, _ = seg_psd(v, seg=60, hop=30)
    put("L=3s hop 0.5s", i, fuse(seg_psd(v, seg=60, hop=10)[0]))
    # null A: broad-shape-only surrogate (narrow lines removed by a 60-BPM median filter of the full-record spectrum, random phases, same local RMS envelope)
    X = np.fft.rfft(v, axis=0); A = np.sqrt(median_filter(np.abs(X) ** 2, (41, 1), mode="nearest"))      # 41 bins * 1 BPM... (rfft bin = 1 BPM for 1200 frames)
    A = np.sqrt(median_filter(np.abs(X) ** 2, (61, 1), mode="nearest"))
    sur = np.fft.irfft(A * np.exp(1j * rng.uniform(0, 2 * np.pi, A.shape)), len(v), axis=0)
    env = np.sqrt(uniform_filter1d(v ** 2, 20, axis=0)); env_s = np.sqrt(uniform_filter1d(sur ** 2, 20, axis=0)) + 1e-12
    put("NULL broad-shape surrogate (L=3s)", i, fuse(seg_psd(sur * env / env_s, seg=60, hop=30)[0]))
    # null B: time-reversed order of 3-s segments does not matter (incoherent) -> instead scramble samples within cells (white)
    put("NULL white noise with same envelope (L=3s)", i, fuse(seg_psd(rng.standard_normal(v.shape) * env, seg=60, hop=30)[0]))
    # single-channel / single-bin versions of the L=3 s estimator
    r0 = centre_bin(d)
    put("L=3s 1 cell (centre, ch6)", i, fuse(P[:, (rb == r0) & (ch == 6)]))
    put("L=3s centre bin 7 ch", i, fuse(P[:, rb == r0]))
    put("L=3s 4 az ch all bins", i, fuse(P[:, np.isin(ch, [6, 4, 3, 2])]))
    put("L=3s p=0 (no weighting)", i, fuse(P, 0))
    if n % 27 == 26: print(n + 1, flush=True)
r = ref[idx]; sj = subj[idx]
def within(e):
    a, b = e.copy(), r.copy()
    for p in DEV: m = sj == p; a[m] -= a[m].mean(); b[m] -= b[m].mean()
    return np.polyfit(b, a, 1)[0], np.corrcoef(a, b)[0, 1]
print(f"{'variant':44s}  hit5   MAE    corr  slope | within-subj slope corr | mean est  sd est")
for k, v in E.items():
    e = np.array([v[i] for i in idx]); a = np.abs(e - r); w = within(e)
    print(f"{k:44s}  {np.mean(a<=5):.3f}  {a.mean():5.2f}  {np.corrcoef(e, r)[0,1]:5.2f}  {np.polyfit(r, e, 1)[0]:5.2f} |  {w[0]:5.2f} {w[1]:5.2f} | {e.mean():5.1f} {e.std():5.1f}")
e = np.array([E["L=3s w=40BPM"][i] for i in idx])
print("ref mean %.1f sd %.1f ; within-subject ref sd %.1f" % (r.mean(), r.std(), np.mean([r[sj == p].std() for p in DEV])))
for p in DEV: m = sj == p; print(p, "ref %.1f est %.1f  MAE %.2f hit5 %.2f  within corr %.2f" % (r[m].mean(), e[m].mean(), np.abs(e - r)[m].mean(), np.mean(np.abs(e - r)[m] <= 5), np.corrcoef(e[m], r[m])[0, 1]))
sh = [np.mean(np.abs(e - r[rng.permutation(len(r))])) for _ in range(500)]; print("shuffled pairing MAE %.2f (true %.2f)" % (np.mean(sh), np.abs(e - r).mean()))
np.savez(S + "/explore3_F.npz", F=np.array([Fs[i] for i in idx]), idx=np.array(idx), ref=r, est=e)

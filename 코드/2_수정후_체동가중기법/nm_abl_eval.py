# -*- coding: utf-8 -*-
"""Ablation table (fixed DEV configuration, all 270 recordings, device reference, 15-s blocks)."""
import sys, json
from nm_common import *
Z = np.load(S + "/refs.npz"); subj = Z["subj"]; gd = np.array([blocks(g)[:3] for g in Z["gd"]]); A = np.load(S + "/abl.npz")
H = tuple(float(a) for a in sys.argv[1].split(","))
def st(F, w=H):
    e = Gb[np.argmax(hsum(F, w), -1)]; a = np.abs(e - gd); em, gm = e.mean(1), gd.mean(1); x, y = em.copy(), gm.copy()
    for p in PERSONS: m = subj == p; x[m] -= x[m].mean(); y[m] -= y[m].mean()
    return dict(MAE=float(a.mean()), w5=float(np.mean(a <= 5)), w10=float(np.mean(a <= 10)), slope=float(np.polyfit(gm, em, 1)[0]), corr=float(np.corrcoef(gm, em)[0, 1]), wslope=float(np.polyfit(y, x, 1)[0]), mean=float(e.mean()), sd=float(e.std()))
LAB = {"default": "default (63 cells, 2-s segments, inverse-power weights)", "p0": "no segment weighting (p=0)", "p0.5": "weight exponent 0.5", "p2": "weight exponent 2 (=default if p=2)", "cell_1": "1 cell (centre bin, 1 channel)", "cell_bin": "centre bin, 7 channels",
       "cell_az4": "4 azimuth channels, 9 bins", "ext_11": "bins -1..+1", "ext_22": "bins -2..+2", "ext_48": "bins -4..+8", "nowhite": "no spectral whitening", "interp": "+ impulse interpolation per cell",
       "canon1": "V1 front end (fixed bin, DACM), same back end    ", "agc": "time-domain 1/sigma normalisation, 10-s windows", "mti": "EMA clutter removal instead of prefix mean", "null_shape": "CONTROL: broad-shape surrogate (lines removed)",
       "null_white": "CONTROL: white noise with the same envelope", "L200_p0": "10-s segments, no weighting (V2-like fusion)"}
out = {}
print(f"{'variant':58s}  MAE   <=5   <=10  slope corr  within-slope | est mean sd")
for k in A.files:
    r = st(A[k]); out[k] = r; print(f"{LAB.get(k,k):58s} {r['MAE']:5.2f} {r['w5']:.3f} {r['w10']:.3f} {r['slope']:5.2f} {r['corr']:5.2f} {r['wslope']:5.2f} | {r['mean']:5.1f} {r['sd']:4.1f}")
print("harmonic weights on the default spectrum (fundamental, 2nd, 3rd):")
for w in ((1, 0, 0), (0, 1, 0), (0, 0, 1), (1, 1, 0), (0, 1, 0.5), (0, 1, 1), (1, 1, 1)):
    r = st(A["default"], w); out["harm_" + str(w)] = r; print(f"  {str(w):56s} {r['MAE']:5.2f} {r['w5']:.3f} {r['w10']:.3f} {r['slope']:5.2f} {r['corr']:5.2f} {r['wslope']:5.2f}")
json.dump(out, open(S + "/nm_abl.json", "w"), indent=1)

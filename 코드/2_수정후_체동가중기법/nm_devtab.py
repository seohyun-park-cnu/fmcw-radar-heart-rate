# -*- coding: utf-8 -*-
"""DEV subjects only: block-protocol table over the configuration grid (flat prior, argmax)."""
import sys
from nm_common import *
Z = np.load(S + "/refs.npz"); subj = Z["subj"]; gd = np.array([blocks(g)[:3] for g in Z["gd"]])
B = np.load(S + "/blk_all.npz"); dev = np.isin(subj, DEV)
HW = [(1, 0, 0), (1, 1, 0), (1, 2, 0), (0, 1, 0), (1, 1, 1), (0, 1, 1), (1, 1, 0.5)]
def est(F, w): return Gb[np.argmax(hsum(F, w), -1)]
def stat(e, g=gd, m=dev):
    a = np.abs(e - g)[m]; em, gm = e[m].mean(1), g[m].mean(1)
    return a.mean(), np.mean(a <= 5), np.mean(a <= 10), np.percentile(a, 90), np.polyfit(gm, em, 1)[0], np.corrcoef(gm, em)[0, 1]
rows = [(k, w) + stat(est(B[k], w)) for k in B.files for w in HW]
print("DEV subjects, device reference, 15-s blocks | const79: MAE %.2f <=5 %.3f" % stat(np.full((270, 3), 79.0))[:2])
print("--- top 25 by MAE"); 
for r in sorted(rows, key=lambda r: r[2])[:25]: print(f"{r[0]:24s} h={r[1]}  MAE {r[2]:5.2f}  <=5 {r[3]:.3f}  <=10 {r[4]:.3f}  p90 {r[5]:5.1f}  slope {r[6]:5.2f} corr {r[7]:5.2f}")
print("--- marginal means of MAE (best over the other factors)")
import collections
for fac, key in (("L", lambda r: r[0].split("_")[0]), ("p", lambda r: r[0].split("_")[1]), ("w", lambda r: r[0].split("_")[2]), ("mode", lambda r: r[0].split("_")[3]), ("harm", lambda r: str(r[1]))):
    dd = collections.defaultdict(list); [dd[key(r)].append(r[2]) for r in rows]
    print(fac, {k: (round(min(v), 2), round(float(np.median(v)), 2)) for k, v in dd.items()})

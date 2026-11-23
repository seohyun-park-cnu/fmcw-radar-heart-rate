# -*- coding: utf-8 -*-
"""Final evaluation of the new estimator on the thesis protocol (15-s blocks).
usage: python nm_final.py <blocks.npz> <out.json>"""
import sys, json, warnings
from nm_common import *
from scipy.stats import wilcoxon, spearmanr
warnings.filterwarnings("ignore")
Z = np.load(S + "/refs.npz"); subj = Z["subj"]; cat = Z["cat"]
gd = np.array([blocks(g)[:3] for g in Z["gd"]]); gp = np.array([blocks(g)[:3] for g in Z["gp"]])
T = np.load(core.REPRO + "/tracks.npz", allow_pickle=True); tm = json.load(open(core.REPRO + "/tracks_meta.json", encoding="utf-8"))
assert [m["nm"] for m in tm] == list(Z["nm"])
paths = {k: np.array([blocks(T[f"{k}_{i}"].astype(float))[:3] for i in range(270)]) for k in ("Raw", "MA", "Sparse")}
route = np.array([m["route"] for m in tm]); gate = np.array([paths[route[i]][i] for i in range(270)])
motion = np.load(S + "/motion_ratio.npy")
B = np.load(sys.argv[1])
HW = [(1, 2, 0), (1, 1, 0.5), (0, 1, 1), (0, 1, 0.5), (0.5, 1, 0.5), (0.5, 1, 1)]
CFG = [(k, w) for k in B.files if k.endswith("_cum") for w in HW]
_e = {}
def est_conf(c):
    if c not in _e:
        s = hsum(B[c[0]], c[1]); k = np.argmax(s, -1); pk = np.take_along_axis(s, k[..., None], -1)[..., 0]
        far = np.abs(Gb[None, None, :] - Gb[k][..., None]) > 8
        _e[c] = (Gb[k], pk / np.where(far, s, -np.inf).max(-1))
    return _e[c]
def select(mask, g=gd): return min(CFG, key=lambda c: np.abs(est_conf(c)[0] - g)[mask].mean())
dev, hold = np.isin(subj, DEV), np.isin(subj, HOLD)
out = {}
def within_slope(e, g, m):
    a, b = e[m].mean(1).copy(), g[m].mean(1).copy(); s = subj[m]
    for p in np.unique(s): a[s == p] -= a[s == p].mean(); b[s == p] -= b[s == p].mean()
    return (np.polyfit(b, a, 1)[0], np.corrcoef(a, b)[0, 1]) if a.std() > 0 else (0.0, 0.0)
def metrics(e, g=gd, m=None):
    m = np.ones(270, bool) if m is None else m
    a = np.abs(e - g)[m]; pc = [np.abs(e - g)[m & (cat == c)].mean() for c in CATS]; em, gm = e[m].mean(1), g[m].mean(1)
    sl, co = (np.polyfit(gm, em, 1)[0], np.corrcoef(gm, em)[0, 1]) if em.std() > 0 else (0.0, 0.0); ws, wc = within_slope(e, g, m)
    return dict(MAE=a.mean(), RMSE=np.sqrt((a ** 2).mean()), Worst=max(pc), CondStd=np.std(pc), w5=np.mean(a <= 5), w10=np.mean(a <= 10), med=np.median(a), p90=np.percentile(a, 90),
                MAPE=np.mean(a / g[m]) * 100, slope=sl, corr=co, wslope=ws, wcorr=wc)
def show(name, e, g=gd, m=None, key=None):
    r = metrics(e, g, m); out[key or name] = {k: float(v) for k, v in r.items()}
    print(f"{name:34s} MAE {r['MAE']:5.2f} RMSE {r['RMSE']:5.2f} Worst {r['Worst']:5.2f} CondStd {r['CondStd']:4.2f} | <=5 {r['w5']:.3f} <=10 {r['w10']:.3f} med {r['med']:4.2f} p90 {r['p90']:5.1f} | slope {r['slope']:4.2f} corr {r['corr']:4.2f} | within-subj slope {r['wslope']:5.2f} corr {r['wcorr']:5.2f}")
print("check canonical gate/Impulse/Bypass/Detrend MAE: %.3f %.3f %.3f %.3f" % tuple(np.abs(x - gd).mean() for x in (gate, paths["Sparse"], paths["Raw"], paths["MA"])))

# ---------- 1) design on DEV -> single confirmatory evaluation on HOLD
c_dev = select(dev); e_dev, cf_dev = est_conf(c_dev)
print("\n=== (1) configuration selected on the 5 design subjects:", c_dev); out["cfg_dev"] = str(c_dev)
for nm, m in (("DESIGN subjects", dev), ("HELD-OUT subjects", hold)):
    print("--", nm, "n rec =", m.sum())
    for k, e in (("Bypass", paths["Raw"]), ("Detrend", paths["MA"]), ("Impulse", paths["Sparse"]), ("Gate (V1 proposed)", gate), ("Constant 79", np.full((270, 3), 79.0)), ("NEW (DEV config)", e_dev)):
        show(k, e, m=m, key=f"{nm[:4]}_{k}")
# ---------- 2) nested leave-one-subject-out
e_loso = np.zeros((270, 3)); cf_loso = np.zeros((270, 3)); const_loso = np.zeros((270, 3)); folds = {}
for p in PERSONS:
    te = subj == p; c = select(~te); folds[p] = str(c); e_loso[te] = est_conf(c)[0][te]; cf_loso[te] = est_conf(c)[1][te]; const_loso[te] = np.median(gd[~te])
print("\n=== (2) nested LOSO (configuration re-selected on the other 9 subjects in every fold)"); [print("  fold", p, c) for p, c in folds.items()]; out["folds"] = folds
ALL = (("Bypass", paths["Raw"]), ("Detrend", paths["MA"]), ("Impulse", paths["Sparse"]), ("Gate (V1 proposed)", gate), ("Constant 79", np.full((270, 3), 79.0)), ("Constant (LOSO median)", const_loso), ("NEW (nested LOSO)", e_loso), ("NEW (DEV config, all 270)", e_dev))
for k, e in ALL: show(k, e, key="all_" + k)
print("-- same, PPG-waveform reference"); [show(k, e, g=gp, key="ppg_" + k) for k, e in ALL if k.startswith(("Gate", "NEW (nested", "Impulse"))]
# ---------- 3) subject-level statistics
rng = np.random.default_rng(0)
def boot(e1, e2, n=5000):
    a1, a2 = np.abs(e1 - gd), np.abs(e2 - gd)
    def st(idx):
        o = []
        for a in (a1, a2):
            pc = [np.mean(np.concatenate([a[(subj == p) & (cat == c)].ravel() for p in idx])) for c in CATS]
            al = np.concatenate([a[subj == p].ravel() for p in idx]); o.append((al.mean(), max(pc), np.std(pc), np.mean(al <= 5)))
        return np.array(o[0]) - np.array(o[1])
    d = np.array([st(rng.choice(PERSONS, 10)) for _ in range(n)]); pt = st(PERSONS)
    return pt, np.percentile(d, [2.5, 97.5], 0)
print("\n=== (3) NEW (nested LOSO) minus baseline, subject-level bootstrap 95% CI (5000 resamples of the 10 subjects)")
for k, e in (("Gate (V1 proposed)", gate), ("Impulse", paths["Sparse"]), ("Constant 79", np.full((270, 3), 79.0))):
    pt, ci = boot(e_loso, e); ds = np.array([np.abs(e_loso - gd)[subj == p].mean() - np.abs(e - gd)[subj == p].mean() for p in PERSONS])
    out["boot_" + k] = dict(point=pt.tolist(), lo=ci[0].tolist(), hi=ci[1].tolist(), subj_better=int((ds < 0).sum()), wilcoxon=float(wilcoxon(ds).pvalue))
    print(f"  vs {k:20s} dMAE {pt[0]:+.2f} [{ci[0][0]:+.2f},{ci[1][0]:+.2f}]  dWorst {pt[1]:+.2f} [{ci[0][1]:+.2f},{ci[1][1]:+.2f}]  dCondStd {pt[2]:+.2f} [{ci[0][2]:+.2f},{ci[1][2]:+.2f}]  d(<=5) {pt[3]:+.3f} [{ci[0][3]:+.3f},{ci[1][3]:+.3f}] | subjects better {int((ds<0).sum())}/10, Wilcoxon p={wilcoxon(ds).pvalue:.4f}")
# ---------- 4) strata
A = lambda e: np.abs(e - gd)
print("\n=== (4) strata (MAE): gate | Impulse | NEW nested LOSO")
print("per subject (ref BPM):"); sub = {}
for p in sorted(PERSONS, key=lambda p: gd[subj == p].mean()):
    m = subj == p; sub[p] = [float(gd[m].mean()), float(A(gate)[m].mean()), float(A(paths['Sparse'])[m].mean()), float(A(e_loso)[m].mean()), float(np.mean(A(e_loso)[m] <= 5)), float(np.mean((e_loso - gd)[m]))]
    print(f"  {p:4s} ref {sub[p][0]:5.1f}: {sub[p][1]:5.2f} | {sub[p][2]:5.2f} | {sub[p][3]:5.2f}   (<=5 {sub[p][4]:.2f}, bias {sub[p][5]:+.1f}) {'DEV' if p in DEV else 'HOLD'}")
out["per_subject"] = sub; cond = {}
print("per condition:")
for c in CATS: m = cat == c; cond[c] = [float(A(x)[m].mean()) for x in (gate, paths["Sparse"], e_loso)]; print(f"  {c:15s} {cond[c][0]:5.2f} | {cond[c][1]:5.2f} | {cond[c][2]:5.2f}")
out["per_condition"] = cond
print("by motion intensity (ratio of impulse-detected samples; thesis Table 5-6 definition):"); mo = {}
q = np.quantile(motion, [1 / 3, 2 / 3]); gm = np.digitize(motion, q)
for t, nm in enumerate(("low", "medium", "high")): m = gm == t; mo[nm] = [int(m.sum())] + [float(A(x)[m].mean()) for x in (paths["Raw"], paths["Sparse"], gate, e_loso)]; print(f"  {nm:7s} n={m.sum():3d}  Bypass {mo[nm][1]:5.2f}  Impulse {mo[nm][2]:5.2f}  gate {mo[nm][3]:5.2f}  NEW {mo[nm][4]:5.2f}")
out["by_motion"] = mo
print("by reference heart-rate tercile (recordings):"); hrm = gd.mean(1); ht = np.argsort(np.argsort(hrm)) * 3 // 270; hh = {}
for t in range(3): m = ht == t; hh[t] = [float(hrm[m].mean())] + [float(A(x)[m].mean()) for x in (gate, e_loso)] + [float(np.mean((x - gd)[m])) for x in (gate, e_loso)]; print(f"  HR {hh[t][0]:5.1f}: gate {hh[t][1]:5.2f}  NEW {hh[t][2]:5.2f} | bias gate {hh[t][3]:+.2f} NEW {hh[t][4]:+.2f}")
out["by_hr"] = hh
print("by block (ends at 20/35/50 s):", [(round(A(gate)[:, b].mean(), 2), round(A(e_loso)[:, b].mean(), 2)) for b in range(3)]); out["by_block"] = [[float(A(gate)[:, b].mean()), float(A(e_loso)[:, b].mean())] for b in range(3)]
big = lambda e: int((A(e).mean(1) > 15).sum()); print("recordings with MAE > 15:", "gate", big(gate), "NEW", big(e_loso)); out["rec_gt15"] = [big(gate), big(e_loso)]
# ---------- 5) controls
print("\n=== (5) controls")
sh = []
for t in range(200):
    k = 1 + t % 9; perm = np.concatenate([((j + k) % 10) * 27 + rng.permutation(27) for j in range(10)]); sh.append(np.abs(e_loso[perm] - gd).mean())
print("  estimates paired with another subject's reference: MAE %.2f (range %.2f-%.2f) vs true pairing %.2f" % (np.mean(sh), np.min(sh), np.max(sh), A(e_loso).mean())); out["shuffle"] = [float(np.mean(sh)), float(np.min(sh)), float(np.max(sh))]
dref = gd[:, 2] - gd[:, 0]; print("  within-recording change (block 3 - block 1): corr(NEW, ref) %.2f, corr(gate, ref) %.2f ; SD of ref change %.1f BPM" % (np.corrcoef(e_loso[:, 2] - e_loso[:, 0], dref)[0, 1], np.corrcoef(gate[:, 2] - gate[:, 0], dref)[0, 1], dref.std()))
print("  distribution of NEW estimates: mean %.1f sd %.1f (reference mean %.1f sd %.1f)" % (e_loso.mean(), e_loso.std(), gd.mean(), gd.std()))
# ---------- 6) confidence -> coverage
print("\n=== (6) reference-free confidence (peak / highest value farther than 8 BPM): keep the most confident blocks")
o = np.argsort(-cf_loso.ravel(), kind="stable"); an, ag = A(e_loso).ravel(), A(gate).ravel(); cov = {}
for c in (100, 90, 80, 70, 60, 50, 40, 30):
    k = o[:int(round(810 * c / 100))]; cov[c] = [float(an[k].mean()), float(np.mean(an[k] <= 5)), float(np.mean(an[k] <= 10)), float(ag[k].mean())]
    print(f"  {c:3d}%  NEW MAE {cov[c][0]:5.2f}  <=5 {cov[c][1]:.3f}  <=10 {cov[c][2]:.3f} | gate on the same blocks {cov[c][3]:5.2f}")
out["coverage"] = cov; print("  Spearman(confidence, error) = %.2f" % spearmanr(cf_loso.ravel(), an).correlation)
np.savez(S + "/nm_final_est.npz", e_loso=e_loso, e_dev=e_dev, cf_loso=cf_loso, gate=gate, gd=gd, gp=gp, subj=subj, cat=cat, Raw=paths["Raw"], MA=paths["MA"], Sparse=paths["Sparse"])
json.dump(out, open(sys.argv[2], "w"), indent=1, ensure_ascii=False)

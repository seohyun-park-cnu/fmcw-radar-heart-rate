from nm_common import *
Z = np.load(S + "/refs.npz"); subj = Z["subj"]; gd = np.array([blocks(g)[:3] for g in Z["gd"]]); dev = np.isin(subj, DEV)
A = np.load(S + "/blk_all.npz"); B = np.load(S + "/blk_dev_hop10.npz")
HW = [(1, 1, 0), (1, 2, 0), (1, 1, 0.5), (1, 1, 1), (0, 1, 1), (0.5, 1, 1), (0, 1, 0.5), (0.5, 1, 0.5)]
def st(F, w):
    e = Gb[np.argmax(hsum(F, w), -1)]; a = np.abs(e - gd)[dev]; return a.mean(), np.mean(a <= 5), np.percentile(a, 90), [round(np.abs(e - gd)[dev][:, b].mean(), 2) for b in range(3)]
print("config  harm | hop L/2: MAE <=5 p90 | hop 0.5 s: MAE <=5 p90 | per-block MAE (hop 0.5s)")
tot = {"half": [], "10": []}
for k in B.files:
    for w in HW:
        a, b = st(A[k.replace("p1.0","p1").replace("p2.0","p2")], w), st(B[k], w); tot["half"].append(a[0]); tot["10"].append(b[0])
        print(f"{k:22s} {str(w):14s} | {a[0]:5.2f} {a[1]:.3f} {a[2]:5.1f} | {b[0]:5.2f} {b[1]:.3f} {b[2]:5.1f} | {b[3]}")
print("mean over configs: hop half %.2f, hop 10 %.2f ; median %.2f %.2f" % (np.mean(tot["half"]), np.mean(tot["10"]), np.median(tot["half"]), np.median(tot["10"])))

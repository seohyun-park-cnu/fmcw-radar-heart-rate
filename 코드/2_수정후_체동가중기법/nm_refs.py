from nm_common import *
radar, hr, ppg, meta = load()
tg = np.arange(5.0, 60.0, 0.25)
gd = np.array([gt_device(hr[i]) for i in range(270)]); gp = np.array([ppg_hr_track(ppg[i], tgrid=tg)[0] for i in range(270)])
np.savez(S + "/refs.npz", gd=gd, gp=gp, subj=np.array([m[0] for m in meta]), cat=np.array([m[1] for m in meta]), nm=np.array([m[3] for m in meta]))
print(gd.shape, gp.shape, np.abs(gd.mean(1) - gp.mean(1)).mean())

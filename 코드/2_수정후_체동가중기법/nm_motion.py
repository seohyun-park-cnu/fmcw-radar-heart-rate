import os, numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
exec(open(HERE + "/diag1.py", encoding="utf-8").read().split("R = {}")[0])
M = np.zeros((270, 3))
for i in range(270):
    ph = jv.dacm_phase_unwrapping(targets(radar[i])[0]); m = p_sp(ph)[1]
    M[i] = [m[T - 400:T].mean() for T in (400, 700, 1000)]
np.save(S + "/motion_ratio.npy", M); print(np.quantile(M, [1/3, 2/3]), M.mean())

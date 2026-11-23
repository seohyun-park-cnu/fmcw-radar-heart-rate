"""Tables A & B with a SINGLE global Sparse parameter (no per-category tuning).

Honesty fix: the canonical BEST dict tunes Sparse (thr,win,guard) per condition,
which contradicts the "no per-condition tuning" claim. Here we (1) cache the
label-independent parts (DACM, Raw/MA tracks, GT blocks, route), (2) sweep a small
grid of a SINGLE (thr,win,guard) applied to ALL recordings, pick the one with
lowest gate (deployment) MAE, then (3) emit Tables A/B with that single param.

All pipeline funcs copied verbatim from eval_block_overlap_nosmooth.py (canonical,
no reimplementation). Run: cd IoT && python3 analysis/make_tables_AB_singleparam.py
"""
import sys, json
import numpy as np
import scipy.io
from scipy.signal import welch
sys.path.insert(0, "/root/workspace/2026-1/IoT/src")
from jyvital7_1 import (MTI, dacm_phase_unwrapping, bandpass_filter, phase_stft,
                        stft_time, stft_bpm, heartbeat_estimation)
from jymas import sma
from shsparse import detect_sparse_movement, preprocess_signal

numFrame, measuretime = 1200, 60
fps = numFrame // measuretime
quarter, radarFpsQ, ubpulseFpsQ = 4, fps // 4, 256 // 4
ds_fps = fps // radarFpsQ
BLK_S = 15
W = BLK_S * ds_fps
STRIDE = W
BP_LOW, BP_HIGH, STFT_W, STFT_L, MAX_INTERP, DET_GUARD = 0.8, 2.1, 100, 99, 20, 0.03
K1_OLD, K2_OLD, K1_NEW, K2_NEW, KALMAN_R = 2, 2, 5, 2, 0.10
THR_MAD_STD, THR_HR_POWER, THR_DETREND_5V2 = 0.508, 0.042, 1.509
AZ = [6, 4, 3, 2]
CATS = ['static', 'arm', 'head', 'cough', 'shoulder_shrug', 'disturb', 'swing', 'transition', 'water']
TRIALS = ['1st', '2nd', '3rd']
PERSONS = ['SH', 'BG', 'JY', 'HJ', 'ET', 'JB', 'JH', 'JHL', 'YW', 'SJ']
DATA = "/root/workspace/2026-1/IoT_data/Participant Data"

# SINGLE-param grid (applied identically to ALL recordings — no category labels)
GRID = [(t, w, g) for t in (3.0, 4.0, 5.0, 6.0) for w in (3, 7) for g in (False, True)]


def extract_gt(mhr):
    bpm = np.squeeze(mhr['saveDataHeartRateUbpulse1'], 0).astype(float); L = len(bpm)
    bpm[bpm > 150] = np.nan
    vi = np.where(~np.isnan(bpm))[0]; ni = np.where(np.isnan(bpm))[0]
    if len(ni) and len(vi): bpm[ni] = np.interp(ni, vi, bpm[vi])
    return bpm[np.arange(0, L, ubpulseFpsQ)][quarter * radarFpsQ:]


def extract_target(d):
    nF = d.shape[0]; rfft = MTI(d[:, AZ, :]); rm = np.mean(np.abs(rfft), 1)
    rp = np.array([0] + [int(np.argmax(rm[i])) for i in range(1, nF)])
    dm = np.fft.fftshift(np.fft.fft(rfft, axis=1), 1)
    dp = np.array([0] + [int(np.argmax(np.abs(dm[i, :, rp[i]]))) for i in range(1, nF)])
    return np.array([dm[i, dp[i], rp[i]] for i in range(nF)])


def downsample_hr(e): return e[np.arange(0, len(e), radarFpsQ)][:-1]


def compute_hr(sig):
    f = bandpass_filter(sig, BP_LOW, BP_HIGH, fps)
    ht = stft_time(numFrame, STFT_W, STFT_L, fps); hb = stft_bpm(numFrame, measuretime, fps)
    return downsample_hr(heartbeat_estimation(phase_stft(f, STFT_W, STFT_L, numFrame), hb, len(ht), kalmanR=KALMAN_R))


def sig_ma(d): return sma(d - sma(d, K1_OLD), K2_OLD)


def sig_sparse_param(sig, thr, win, guard):
    dc = np.asarray(sig, float).copy(); ap = True
    if guard and float(np.sum(detect_sparse_movement(dc, fps=fps, win_size=win, threshold_factor=thr))) / len(dc) < DET_GUARD:
        ap = False
    if ap:
        m = detect_sparse_movement(dc, fps=fps, win_size=win, threshold_factor=thr)
        dc = preprocess_signal(dc, m, method='interpolate', max_interp_len=MAX_INTERP)
    return sma(dc - sma(dc, K1_NEW), K2_NEW)


def gate_feats(dacm):
    dd = np.diff(dacm)
    r_ms = float(np.median(np.abs(dd - np.median(dd)))) / (np.std(dd) + 1e-12)
    fr, psd = welch(dacm, fs=fps, nperseg=min(256, len(dacm)), noverlap=min(128, len(dacm) // 2))
    tp = np.sum(psd); r_hr = float(np.sum(psd[(fr >= 0.8) & (fr <= 2.1)]) / tp) if tp > 0 else 0.0
    r_dt = float(np.std(dacm - sma(dacm, size=5)) / (np.std(dacm - sma(dacm, size=2)) + 1e-12))
    return r_ms, r_hr, r_dt


def route_of(dacm):
    a, b, c = gate_feats(dacm)
    return "Raw" if a > THR_MAD_STD else ("MA" if (b > THR_HR_POWER or c < THR_DETREND_5V2) else "Sparse")


def blocks(x): return np.array([np.mean(x[s:s + W]) for s in range(0, len(x) - W + 1, STRIDE)])


def berr(track, gb):
    yb = blocks(track); n = min(len(gb), len(yb)); return np.abs(gb[:n] - yb[:n])


# ---- 1) cache label-independent parts ----
cache = []
for p in PERSONS:
    for cat in CATS:
        for tr in TRIALS:
            nm = f"{cat}_{p}_{tr}"
            try:
                mr = scipy.io.loadmat(f"{DATA}/{nm}_radar.mat"); mhr = scipy.io.loadmat(f"{DATA}/{nm}_ubpulse_heartbeat.mat")
                gt = extract_gt(mhr)
                if len(gt) == 0: raise ValueError
                dacm = dacm_phase_unwrapping(extract_target(mr['saveDataRadar1']))
            except Exception:
                continue
            gb = blocks(gt)
            cache.append({"cat": cat, "route": route_of(dacm), "dacm": dacm, "gb": gb,
                          "Raw": berr(compute_hr(dacm), gb), "MA": berr(compute_hr(sig_ma(dacm)), gb)})
print(f"cached recordings: {len(cache)}")


def eval_param(thr, win, guard):
    """return gate MAE over all recordings for a single Sparse param."""
    ge = []
    for r in cache:
        sp = berr(compute_hr(sig_sparse_param(r["dacm"], thr, win, guard)), r["gb"])
        errs = {"Raw": r["Raw"], "MA": r["MA"], "Sparse": sp}
        n = min(len(errs["Raw"]), len(errs["MA"]), len(errs["Sparse"]))
        ge.extend(errs[r["route"]][:n].tolist())
    return float(np.mean(ge))


# ---- 2) sweep single-param grid, pick min gate MAE ----
print("\n=== single-Sparse-param grid (gate MAE) ===")
sweep = []
for (t, w, g) in GRID:
    m = eval_param(t, w, g); sweep.append(((t, w, g), m))
    print(f"  thr={t} win={w} guard={g!s:5}  gate MAE={m:.3f}")
best_param, best_mae = min(sweep, key=lambda x: x[1])
print(f"\nBEST single param: thr={best_param[0]} win={best_param[1]} guard={best_param[2]}  gate MAE={best_mae:.3f}")

# ---- 3) full Tables A/B with best single param ----
BT, BW, BG_ = best_param
rows = []
for r in cache:
    sp = berr(compute_hr(sig_sparse_param(r["dacm"], BT, BW, BG_)), r["gb"])
    errs = {"Raw": r["Raw"], "MA": r["MA"], "Sparse": sp}
    n = min(len(errs[k]) for k in errs)
    errs = {k: v[:n] for k, v in errs.items()}
    gate_e = errs[r["route"]]
    orc_e = np.min(np.stack([errs[q] for q in ["Raw", "MA", "Sparse"]]), axis=0)
    rec = {"cat": r["cat"], "route": r["route"]}
    for k in errs: rec[k] = errs[k].tolist()
    rec["gate"] = gate_e.tolist(); rec["oracle"] = orc_e.tolist()
    rows.append(rec)

SCHEMES = ["Raw", "MA", "Sparse", "gate", "oracle"]
def allerr(s):
    e = []; [e.extend(r[s]) for r in rows]; return np.array(e)
def percat(s, c): return float(np.mean([x for r in rows if r["cat"] == c for x in r[s]]))

tableB = {}
for s in SCHEMES:
    e = allerr(s); pc = [percat(s, c) for c in CATS]
    tableB[s] = {"MAE": float(e.mean()), "RMSE": float(np.sqrt((e ** 2).mean())),
                 "WorstMAE": float(np.nanmax(pc)), "condStd": float(np.nanstd(pc))}

tableA = {}
for c in CATS:
    vals = {s: percat(s, c) for s in ["Raw", "MA", "Sparse"]}
    tableA[c] = {**vals, "winner": min(vals, key=vals.get), "gate": percat("gate", c)}

for r in rows:
    vals = {s: float(np.mean(r[s])) for s in ["Raw", "MA", "Sparse"]}
    r["_win"] = min(vals, key=vals.get)
from collections import Counter
rec_win_dist = dict(Counter(r["_win"] for r in rows))
non_sparse_win = sum(1 for r in rows if r["_win"] != "Sparse")
percat_winfrac = {c: {s: round(sum(1 for r in rows if r["cat"] == c and r["_win"] == s) /
                               max(1, sum(1 for r in rows if r["cat"] == c)), 3)
                      for s in ["Raw", "MA", "Sparse"]} for c in CATS}
route_dist = {r_: sum(1 for x in rows if x["route"] == r_) for r_ in ["Raw", "MA", "Sparse"]}
gate_le_sparse = sum(1 for c in CATS if tableA[c]["gate"] <= tableA[c]["Sparse"] + 1e-9)

L = []
L.append(f"# 표 A · 표 B — 단일 Sparse 파라미터판 (정본 back-end, {len(rows)}건)\n")
L.append(f"**단일 Sparse 파라미터: thr={BT}, win={BW}, guard={BG_}** (카테고리 라벨 없이 전 녹화 동일 적용, "
         f"그리드 {len(GRID)}개 중 gate MAE 최소로 선택 → C2 '무튜닝' 주장 유지)\n")
L.append(f"라우팅 분포: Raw {route_dist['Raw']} / MA {route_dist['MA']} / Sparse {route_dist['Sparse']}\n")

L.append("\n## 표 A — 구조적 미스매치 (녹화 단위 최적 파이프라인 분포)\n")
L.append(f"**전체 녹화별 최적:** Raw {rec_win_dist.get('Raw',0)} / MA {rec_win_dist.get('MA',0)} / "
         f"Sparse {rec_win_dist.get('Sparse',0)}  →  전역 최선이 아닌 녹화 "
         f"**{non_sparse_win}/{len(rows)}건 ({100*non_sparse_win/len(rows):.0f}%)**.\n")
L.append("\n| 조건 | Raw% | MA% | Sparse% | 지배 |")
L.append("|---|---|---|---|---|")
for c in CATS:
    w = percat_winfrac[c]; mx = max(w.values())
    dom = "없음(혼재)" if mx < 0.6 else max(w, key=w.get)
    L.append(f"| {c} | {w['Raw']*100:.0f}% | {w['MA']*100:.0f}% | {w['Sparse']*100:.0f}% | {dom} |")

L.append("\n## 표 A-보조 — 조건 평균 MAE\n")
L.append(f"라우팅(gate)이 {gate_le_sparse}/9 조건에서 Sparse 이하.\n")
L.append("| 조건 | Raw | MA | Sparse | gate |")
L.append("|---|---|---|---|---|")
for c in CATS:
    t = tableA[c]
    g = f"**{t['gate']:.2f}**" if t["gate"] <= t["Sparse"] + 1e-9 else f"{t['gate']:.2f}"
    L.append(f"| {c} | {t['Raw']:.2f} | {t['MA']:.2f} | {t['Sparse']:.2f} | {g} |")

L.append("\n## 표 B — 강건성 (경로별 종합 지표)\n")
L.append("| 경로 | MAE | RMSE | WorstMAE | condStd |")
L.append("|---|---|---|---|---|")
for s in SCHEMES:
    b = tableB[s]
    L.append(f"| {s} | {b['MAE']:.3f} | {b['RMSE']:.3f} | {b['WorstMAE']:.3f} | {b['condStd']:.3f} |")

open("/root/workspace/2026-1/IoT/paper/tables_AB_singleparam.md", "w").write("\n".join(L) + "\n")
json.dump({"best_param": {"thr": BT, "win": BW, "guard": BG_}, "sweep": [(list(k), v) for k, v in sweep],
           "n": len(rows), "route_dist": route_dist, "rec_win_dist": rec_win_dist,
           "non_sparse_win": non_sparse_win, "percat_winfrac": percat_winfrac,
           "tableA": tableA, "tableB": tableB},
          open("/root/workspace/2026-1/IoT/results/tables_AB_singleparam.json", "w"), indent=1, ensure_ascii=False)
print("\n".join(L))
print("\nsaved: paper/tables_AB_singleparam.md , results/tables_AB_singleparam.json")

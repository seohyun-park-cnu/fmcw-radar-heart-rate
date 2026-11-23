# 수치 출처 추적 (Data Provenance)

논문에 사용되는 모든 수치가 어떤 데이터, 어떤 코드에서 도출되었는지 대응시킨 문서.
수치 재현 시 이 문서를 참조하여 해당 스크립트를 실행하면 동일 결과를 얻을 수 있다.

**마지막 검증: 2026-08-01** — 3개 독립 스크립트 교차검증 완료, 소수점 6자리 일치 확인.

---

## 공통 프로토콜

| 항목 | 값 | 근거 |
|---|---|---|
| 평가 메트릭 | 15초 비겹침 블록 MAE | `BLK_S=15, STRIDE=BLK_W` |
| GT 처리 | >150 BPM 보간, **스무딩 없음**, 4Hz 다운샘플, 첫 5초 드롭 | `extract_gt()` |
| Sparse 파라미터 | thr=3.0, win=7, guard=False (전역 단일) | 그리드 16개 중 gate MAE 최소 |
| Gate 임계값 | τ₁=0.508, τ₂=0.042, τ₃=1.509 (고정 설계상수) | `gate_feats()` + `route_of()` |
| Kalman R | 0.10 (모든 방법 공통) | `heartbeat_estimation(kalmanR=0.10)` |
| 데이터 | 270건 (10명 × 9조건 × 3회) | `IoT_data/Participant Data/` |

---

## 1. 내부 파이프라인 + Gate + Oracle

### 수치

| 방법 | MAE | RMSE | WorstMAE | condStd |
|---|---|---|---|---|
| Raw | 12.271 | 15.569 | 13.910 | 1.028 |
| MA | 10.413 | 13.034 | 12.135 | 1.185 |
| Sparse | 7.717 | 10.044 | 9.945 | 0.998 |
| **gate** | **7.573** | **9.896** | **8.933** | **0.803** |
| oracle | 4.110 | 5.914 | 4.992 | 0.550 |

라우팅 분포: Raw 71 / MA 41 / Sparse 158

### 정본 코드 및 결과 파일

| 항목 | 파일 경로 |
|---|---|
| **정본 스크립트** | `IoT/analysis/make_tables_AB_singleparam.py` |
| **정본 결과 (JSON)** | `IoT/results/tables_AB_singleparam.json` |
| **정본 결과 (Markdown)** | `IoT/paper/tables_AB_singleparam.md` |

### 재현 명령

```bash
cd /root/workspace/2026-1/IoT
python3 analysis/make_tables_AB_singleparam.py
```

### 핵심 함수 매핑

| 수치 | 코드 내 함수/변수 | 파일 |
|---|---|---|
| GT 추출 (무스무딩) | `extract_gt(mhr)` (42~47행) | `make_tables_AB_singleparam.py` |
| DACM 추출 | `extract_target(d)` → `dacm_phase_unwrapping()` | 동일 + `src/jyvital7_1.py` |
| Raw HR | `compute_hr(dacm)` (61~64행) | `make_tables_AB_singleparam.py` |
| MA HR | `compute_hr(sig_ma(dacm))`, `sig_ma()` (67행) | 동일 |
| Sparse HR | `compute_hr(sig_sparse_param(dacm, 3.0, 7, False))` (70~77행) | 동일 |
| Gate 라우팅 | `route_of(dacm)` → `gate_feats()` (80~91행) | 동일 |
| 15s 블록 MAE | `blocks()` (94행), `berr()` (97~98행) | 동일 |
| WorstMAE | `np.nanmax(pc)` (163행), `pc = [percat(s, c) for c in CATS]` | 동일 |
| condStd | `np.nanstd(pc)` (164행) | 동일 |

### 독립 검증

동일 수치를 `eval/eval_external_block15s.py` 와 `eval/eval_ablation_block15s.py` 에서도 산출.
3개 스크립트 간 MAE 소수점 6자리 일치 확인됨 (2026-08-01).

---

## 2. 외부 베이스라인 8종

### 수치

| 방법 | MAE | RMSE | WorstMAE | condStd | p vs gate |
|---|---|---|---|---|---|
| VMD | 18.825 | 22.303 | 21.367 | 2.795 | 4.1e-91 |
| RLS | 15.012 | 18.004 | 17.993 | 1.480 | 1.3e-56 |
| DWT | 12.390 | 15.693 | 14.020 | 1.050 | 2.1e-44 |
| EWT | 12.292 | 15.635 | 14.033 | 1.001 | 4.5e-40 |
| AdaptNotch | 11.813 | 15.105 | 13.060 | 1.121 | 1.6e-34 |
| SSA | 11.558 | 14.923 | 13.154 | 0.998 | 8.2e-27 |
| EEMD | 10.464 | 13.318 | 13.940 | 1.450 | 1.5e-16 |
| CEEMDAN | 10.365 | 14.442 | 12.821 | 1.531 | 6.4e-10 |

### 정본 코드 및 결과 파일

| 항목 | 파일 경로 |
|---|---|
| **평가 스크립트** | `IoT/eval/eval_external_block15s.py` |
| **결과 (JSON)** | `IoT/results/external_block15s.json` |
| **실행 로그** | `IoT/results/external_block15s_log.txt` |
| **원본 베이스라인 구현** | `IoT/src/external_baselines_v2.py` (함수 정의 참조) |

### 재현 명령

```bash
cd /root/workspace/2026-1/IoT
python3 eval/eval_external_block15s.py
# 소요 시간: ~570초 (270건 × 8 방법)
```

### 핵심 함수 매핑

| 외부 방법 | 구현 함수 | eval_external_block15s.py 내 행 | 원본 참조 (external_baselines_v2.py) |
|---|---|---|---|
| VMD | `run_vmd(dacm)` | 구현 포함 | `run_vmd()` 202~244행 |
| DWT | `run_dwt(dacm)` | 구현 포함 | `run_dwt()` 255~294행 |
| EEMD | `run_eemd(dacm)` | 구현 포함 | `run_eemd()` 297~321행 |
| RLS | `run_rls(dacm)` | 구현 포함 | `run_rls()` 323~352행 |
| AdaptNotch | `run_adaptive_notch(dacm)` | 구현 포함 | `run_adaptive_notch()` 354~381행 |
| SSA | `run_ssa(dacm)` | 구현 포함 | `run_ssa()` 383~447행 |
| CEEMDAN | `run_ceemdan(dacm)` | 구현 포함 | `run_ceemdan()` 449~488행 |
| EWT | `run_ewt(dacm)` | 구현 포함 | `run_ewt()` 490~532행 |

### 공통 백엔드 (모든 외부 방법 동일)

```
signal → bandpass_filter(0.8~2.1 Hz) → phase_stft(W=100, L=99)
       → heartbeat_estimation(kalmanR=0.10) → downsample(→4Hz)
```
구현: `common_backend()` in `eval_external_block15s.py`
의존: `src/jyvital7_1.py` 의 `bandpass_filter`, `phase_stft`, `heartbeat_estimation`

### p-value 산출

Wilcoxon signed-rank test (`scipy.stats.wilcoxon`), gate 블록 오차 vs 각 방법 블록 오차.
코드: `eval_external_block15s.py` 내 "Wilcoxon tests" 블록.

### 주의사항

- `eval_external_block15s.py`는 `external_baselines_v2.py`의 함수를 **복사하여 내장**함 (import 아님).
  이유: GT 추출 함수가 다름 (구 버전은 5초 스무딩, 신 버전은 무스무딩).
- EDACM은 위상 추출 단계 방법이므로 후단 비교에서 제외 (논문에 명시).
- RT(SPARQRealtime)는 배치 전용 스코프이므로 제외.

---

## 3. Ablation

### 수치

| 구성 | MAE | WorstMAE | condStd | Delta MAE |
|---|---|---|---|---|
| A. MA only, R=0.01 | 10.419 | 11.960 | 1.144 | --- |
| B. MA only, R=0.10 | 10.413 | 12.135 | 1.185 | -0.007 |
| C. Sparse only, R=0.10 | 7.717 | 9.945 | 0.998 | -2.696 |
| D. 2-way gate, R=0.10 | 7.604 | 9.123 | 0.867 | -0.113 |
| E. 3-way gate, R=0.10 | 7.573 | 8.933 | 0.803 | -0.031 |
| F. Oracle, R=0.10 | 4.110 | 4.992 | 0.550 | -3.463 |

### 정본 코드 및 결과 파일

| 항목 | 파일 경로 |
|---|---|
| **평가 스크립트** | `IoT/eval/eval_ablation_block15s.py` |
| **결과 (JSON)** | `IoT/results/ablation_block15s.json` |

### 재현 명령

```bash
cd /root/workspace/2026-1/IoT
python3 eval/eval_ablation_block15s.py
```

### 각 구성의 코드 대응

| 구성 | 코드 내 config_name | 신호 처리 | Kalman R |
|---|---|---|---|
| A | `A_MA_R001` | `sig_ma(dacm)` → `compute_hr(..., kalman_r=0.01)` | 0.01 |
| B | `B_MA_R010` | `sig_ma(dacm)` → `compute_hr(..., kalman_r=0.10)` | 0.10 |
| C | `C_Sparse_R010` | `sig_sparse(dacm)` → `compute_hr(..., kalman_r=0.10)` | 0.10 |
| D | `D_2way_R010` | Gate 1만: Raw if R_MS>0.508 else Sparse | 0.10 |
| E | `E_3way_R010` | Gate 1+2: Raw/MA/Sparse 3-way 라우팅 | 0.10 |
| F | `F_Oracle_R010` | `np.min(stack([Raw, MA, Sparse]), axis=0)` 블록 단위 | 0.10 |

### 교차검증

B/C/E/F의 MAE가 정본(`tables_AB_singleparam.json`)의 MA/Sparse/gate/oracle과 소수점 6자리 일치 확인.

---

## 4. LOSO 교차검증

### 수치

| 구성 | MAE |
|---|---|
| in-sample (전체 270건 고정) | 7.573 |
| param-only LOSO (10-fold held-out 평균) | 7.584 |
| always-Sparse (게이트 없음) | 7.717 |
| 전역 파라미터 재발견 비율 | 7/10 folds |

### 정본 코드 및 결과 파일

| 항목 | 파일 경로 |
|---|---|
| **평가 스크립트** | `IoT/analysis/loso_singleparam.py` |
| **결과 (JSON)** | `IoT/results/loso_singleparam.json` |

### 재현 명령

```bash
cd /root/workspace/2026-1/IoT
python3 analysis/loso_singleparam.py
```

### LOSO 프로토콜 상세

1. 10-fold (피험자 1명씩 held-out)
2. 각 fold에서 나머지 9명으로 Sparse 파라미터 (thr, win, guard) 16-point 그리드 탐색
3. gate MAE 최소인 파라미터를 선택하여 held-out 피험자에 적용
4. **Gate 임계값(τ₁, τ₂, τ₃)은 고정** — fold별 재선택하지 않음
5. held-out MAE = 해당 피험자 녹화들의 블록 MAE 평균

---

## 5. 구조적 미스매치 (Table IV용)

### 수치

- 전역 최선(Sparse)이 아닌 녹화: 156/270건 (58%)
- 녹화별 최적 분포: Raw 81 / MA 75 / Sparse 114
- 9개 조건 모두 세 파이프라인 승자 혼재 (어느 조건도 60% 미만 지배)

### 정본 코드 및 결과 파일

| 항목 | 파일 경로 |
|---|---|
| **정본 스크립트** | `IoT/analysis/make_tables_AB_singleparam.py` |
| **결과 (JSON)** | `IoT/results/tables_AB_singleparam.json` |

### JSON 내 대응 키

| 수치 | JSON 키 |
|---|---|
| 녹화별 최적 분포 | `rec_win_dist` → `{"Sparse": 114, "Raw": 81, "MA": 75}` |
| 비-Sparse 최적 건수 | `non_sparse_win` → `156` |
| 조건별 승자 비율 | `percat_winfrac` → 9개 조건 × 3 파이프라인 비율 |
| 조건별 평균 MAE | `tableA` → 조건별 Raw/MA/Sparse/gate MAE |

---

## 6. Table I (피험자별 라우팅 분포) / Table II (조건별 파이프라인 MAE)

### 수치

**Table I** — 라우팅 분포 (gate_route 기준, 총 271/41/158):

| ID | N | Raw | MA | Sparse |
|---|---|---|---|---|
| SH | 27 | 0 | 3 | 24 |
| BG | 27 | 2 | 11 | 14 |
| ... | ... | ... | ... | ... |
| All | 270 | 71 | 41 | 158 |

**Table II** — 조건 그룹별 파이프라인 MAE (15s 블록):

| 그룹 | N | Raw | MA | Sparse |
|---|---|---|---|---|
| HB-dominant | 71 | **8.13** | 13.11 | 8.56 |
| Drift-dominant | 41 | 17.98 | **7.44** | 7.65 |
| Impulse-dominant | 158 | 12.65 | 9.97 | **7.36** |

### 정본 코드

이 수치들은 `make_tables_AB_singleparam.py`의 `route_of()` 라우팅 결과와
`compute_hr()` 블록 MAE를 피험자별/조건그룹별로 집계하여 산출.
별도 저장 파일은 없으며, 위 결과 보고 시 인라인 스크립트로 계산 (2026-08-01).

재현: `make_tables_AB_singleparam.py`의 cache 데이터에서 `route` 및 블록 오차를 피험자/그룹별 집계.

---

## 의존성 트리

```
IoT_data/Participant Data/
  └─ {cat}_{person}_{trial}_radar.mat        (레이더 원시 데이터)
  └─ {cat}_{person}_{trial}_ubpulse_heartbeat.mat  (GT 심박)

IoT/src/
  ├─ jyvital7_1.py    (MTI, DACM, BPF, STFT, Kalman, HR estimation)
  ├─ jymas.py         (sma, wma, ema)
  ├─ shsparse.py      (detect_sparse_movement, preprocess_signal)
  └─ external_baselines_v2.py  (외부 베이스라인 원본 구현, 참조용)

IoT/analysis/
  ├─ make_tables_AB_singleparam.py  ← 정본 (내부 파이프라인)
  └─ loso_singleparam.py            ← 정본 (LOSO)

IoT/eval/
  ├─ eval_external_block15s.py      ← 외부 베이스라인 (신규)
  └─ eval_ablation_block15s.py      ← Ablation (신규)

IoT/results/
  ├─ tables_AB_singleparam.json     ← 정본 결과
  ├─ loso_singleparam.json          ← LOSO 결과
  ├─ external_block15s.json         ← 외부 BL 결과
  └─ ablation_block15s.json         ← Ablation 결과
```

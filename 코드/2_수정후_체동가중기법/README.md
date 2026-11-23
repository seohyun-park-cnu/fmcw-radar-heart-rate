# 수정 후: 체동 가중 다중 셀 고조파 스펙트럼 기법

기존 경로 선택 방식(`../1_수정전_경로선택방식/`)을 대체하는 추정기의 설계·검증 스크립트와 결과.

## 방법 (확정 설정)

1. 중심 거리 빈(교류 전력 최대) 기준 −2..+6 빈 × 7 채널 = 63 셀, 셀별 평균 제거 후 위상 속도 `angle(x[n]·conj(x[n−1]))`
2. 2초 Hann 구간(40 샘플), 0.5초 간격 주기도
3. 구간 가중 = (그 구간의 48–126 BPM 대역 전력)^−2, 셀별 가중 평균
4. 주파수 방향 이동 중앙값(80 BPM 폭)으로 백색화 후 셀 평균
5. 고조파 합 s(f) = W(2f) + 0.5·W(3f), 48–126 BPM에서 최댓값 → 심박수
6. 블록 b의 추정에는 블록 끝 시각까지의 프레임만 사용

## 파일

| 파일 | 역할 |
|---|---|
| `nm_common.py` | 셀 추출, 위상 속도, 구간 주기도, 백색화, 고조파 합 |
| `core.py` | 데이터 캐시 로드, 기준 심박 처리, 블록 평균. 기존 신호처리 모듈을 `../1_수정전_경로선택방식/`에서 불러온다 |
| `load_all.py` | 270건을 하나의 캐시 파일로 묶음 |
| `nm_refs.py`, `nm_motion.py` | 기준 심박(기기 출력, PPG 파형)과 체동 비율 계산 |
| `nm_blocks.py` | 설정 격자별 블록 단위 융합 스펙트럼 |
| `nm_final.py` | 최종 평가(설계/확인 분할, nested LOSO, 기존 방식과 비교) |
| `nm_ablate.py`, `nm_abl_eval.py` | 절제 실험 |
| `nm_inject.py` | 신호 주입 실험 |
| `explore*.py`, `nm_devtab*.py` | 설계용 피험자 5명만 쓴 탐색 스크립트 |
| `diag1.py` | 기존 처리 구조 진단. `nm_motion.py`가 앞부분을 불러 쓴다 |
| `eval_external_block15s_seeded.py` | 외부 베이스라인 8종 재평가(EEMD/CEEMDAN 난수 시드 고정) |
| `baseline_tracks/` | 기존 세 경로의 녹화별 심박 추적(`tracks.npz`)과 생성 스크립트. `nm_final.py`의 비교 대상 |
| `results/` | 결과 파일 |

## 재현 순서

이 폴더에서 실행한다. 중간 파일은 `cache/`에 생기며(약 600 MB, 저장소에는 올리지 않음) 지워도 된다. 다른 위치를 쓰려면 환경변수 `NM_CACHE`를 지정한다.

```bash
python load_all.py cache/all270.npz
python nm_refs.py
python nm_motion.py
python nm_blocks.py cache/blk_final.npz all 10 40,60 1,2 cum
python nm_final.py cache/blk_final.npz cache/nm_final.json      # 출력 = results/result_final.txt
python nm_ablate.py 40 10 2 161 cache/abl.npz
python nm_abl_eval.py 0,1,0.5                                   # 출력 = results/result_ablation.txt
python nm_inject.py 40 10 2 161 0,1,0.5                         # 출력 = results/result_injection.txt
```

`baseline_tracks/tracks.npz`를 다시 만들려면 `python baseline_tracks/run_tracks.py baseline_tracks/tracks.npz`.

필요 패키지: NumPy, SciPy. `eval_external_block15s_seeded.py`는 추가로 vmdpy, PyWavelets, PyEMD(EMD-signal), ewtpy가 필요하다.

## 설계/확인 분할

설계용 피험자: SH, JY, ET, JH, YW. 확인용: BG, HJ, JB, JHL, SJ (새 방법 결과를 보기 전에 고정).
`explore*.py`, `nm_devtab*.py`는 설계용 5명만 사용한다. 절제 실험의 "bins −4..+8", 고조파 (0,1,1) 등 확인 후에 본 결과는 설정에 반영하지 않았다.

## 주요 결과 (15초 블록, `results/result_final.txt`)

| 방법 | MAE (BPM) | ≤5 BPM 비율 |
|---|---:|---:|
| 기존 경로 선택 방식 (전체 270건) | 7.57 | 43.6% |
| 체동 가중 기법, 확인용 5명 | 5.86 | 80.5% |
| 체동 가중 기법, nested LOSO (전체 270건) | 4.82 | 78.6% |

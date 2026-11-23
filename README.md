# fmcw-radar-heart-rate

「체동 환경에서 강건한 FMCW 레이다 기반 비접촉 심박수 추정」 코드 및 데이터

## 구성

| 폴더 | 내용 |
|---|---|
| `데이터/` | 측정 데이터 270건(.mat 810개). 목록과 변수 형상은 `데이터/FOLDER_INFO.md` 참고 |
| `코드/` | 평가 스크립트, 신호처리 모듈, 결과 JSON(`코드/results/`) |

## 데이터

파일명 규칙: `<동작>_<피험자>_<회차>_{radar|ubpulse_heartbeat|ubpulse_ppg}.mat`

- 동작 9종: static, arm, head, cough, shoulder_shrug, disturb, swing, transition, water
- 피험자 10명 × 3회
- `radar`: 레이다 데이터, `ubpulse_heartbeat`: 기준 심박수, `ubpulse_ppg`: PPG 파형

## 코드

| 파일 | 역할 |
|---|---|
| `make_tables_AB_singleparam(...).py` | 정본: 내부 파이프라인(Raw/MA/Sparse), 게이트, Oracle |
| `loso_singleparam(...).py` | Sparse 파라미터 LOSO 검증 |
| `eval_external_block15s(...).py` | 외부 베이스라인 8종 비교 |
| `eval_ablation_block15s(...).py` | Ablation |
| `jyvital7_1(...).py`, `jymas(...).py`, `shsparse(...).py` | 신호처리 모듈 |
| `external_baselines_v2(...).py` | 외부 베이스라인 원본 구현(참조용) |

수치와 스크립트의 대응은 `코드/data_provenance.md`에 정리되어 있다.

## 실행 시 유의사항

- 파일명 뒤 괄호의 한글 설명은 보관용으로 붙인 것이다. 스크립트는 원래 이름(예: `jyvital7_1.py`)으로 모듈을 불러오므로, 실행하려면 괄호 부분을 지운 이름으로 두어야 한다.
- 스크립트의 데이터·모듈 경로는 원래 작업 환경(`/root/workspace/2026-1/...`) 기준으로 적혀 있어, 실행 환경에 맞게 고쳐야 한다.
- 필요 패키지: Python 3, NumPy, SciPy. 외부 베이스라인 스크립트는 추가로 vmdpy, PyWavelets, PyEMD(EMD-signal), ewtpy가 필요하다.

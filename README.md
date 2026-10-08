# flywire-connectome

초파리 커넥톰(BANC v888) 위에서 PN -> KC -> APL 버섯체 회로를 단순 발화율 모형으로 돌려 보고, 실제 배선이 무작위 대조 배선과 기능적으로 다른지 확인하는 개인 탐구입니다. 질환, 환각, 행동에 대한 결론은 아직 내리지 않습니다.

**현재 상황과 방향은 [docs/RESEARCH_STATUS.md](docs/RESEARCH_STATUS.md)에 정리되어 있습니다.**

## 지금까지의 핵심 결과

- Hallem 2006 냄새 반응과 Olsen 2010 PN 변환으로 입력을 정하고, KC 반응 10%로 보정하면 생리 범위의 모형이 됩니다 (16).
- BANC PN->KC 배선에는 우연 이상의 PN 유형 과수렴 구조가 있습니다 (DM1-DM4 등, 17).
- 이 배선은 일반적인 냄새 구분에서 강도 보존 무작위 대조보다 약간 불리합니다 (18).

세부: [docs/HALLEM_COMMUNITY_RESULTS.md](docs/HALLEM_COMMUNITY_RESULTS.md)

## 구조

```
pipelines/   단계별 실험 스크립트 (05~18, 번호 = 진행 순서)
tests/       unittest (계산 검증용, 생물학적 검증 아님)
docs/        단계별 설명, 사전 등록, 결과 문서
qc_reports/  각 단계의 산출물 (manifest, 요약 csv, 그림)
legacy/      철회된 초기 실험 01~04 (기록용)
data/        원자료 위치 (git 제외)
```

## 단계와 문서

| 단계 | 내용 | 문서 |
|---|---|---|
| 05 | 초기 결과(01~04)의 대조 검증, 철회 근거 | [CONNECTOME_NULL_CONTROL](docs/CONNECTOME_NULL_CONTROL.md) |
| 06~07 | 인공 입력 구분, APL 억제, 계수 민감도 | [INPUT_DISCRIMINATION](docs/INPUT_DISCRIMINATION.md), [DISCRIMINATION_ROBUSTNESS](docs/DISCRIMINATION_ROBUSTNESS.md) |
| 08~12 | KC->MBON 학습, 판독법, KC 유형 보존 대조 | [ASSOCIATIVE_READOUT](docs/ASSOCIATIVE_READOUT.md), [SIGNAL_PRESENCE](docs/SIGNAL_PRESENCE.md), [READOUT_COMPARISON](docs/READOUT_COMPARISON.md), [READOUT_CALIBRATION](docs/READOUT_CALIBRATION.md), [SUBTYPE_NULLS](docs/SUBTYPE_NULLS.md) |
| 13~15 | DoOR 냄새 자료 감사, 관측 냄새 부분 회로, 문턱 원인 분리 | [ODOR_INPUT_AUDIT](docs/ODOR_INPUT_AUDIT.md), [OBSERVED_ODOR_GEOMETRY](docs/OBSERVED_ODOR_GEOMETRY.md), [ODOR_THRESHOLD_MECHANISM](docs/ODOR_THRESHOLD_MECHANISM.md) |
| 정리 | 06~15 종합 | [SYNTHESIS_06_15](docs/SYNTHESIS_06_15.md) |
| 16~18 | 보정 모형, PN 과수렴 구조, 구분 기능 | [PREREGISTRATION_16_18](docs/PREREGISTRATION_16_18.md), [HALLEM_COMMUNITY_RESULTS](docs/HALLEM_COMMUNITY_RESULTS.md) |

개념 설명은 [RESEARCH_DIRECTION_EXPLAINED](docs/RESEARCH_DIRECTION_EXPLAINED.md)에 있습니다.

## 데이터

원자료는 저장소에 포함하지 않습니다. 각 스크립트는 실행 시 SHA-256으로 입력을 확인합니다.

- BANC v888: `data/banc_888_meta.feather`, `data/banc_888_edgelist_simple_v2.feather` ([BANC 프로젝트](https://github.com/htem/BANC-project)). 연결의 ID는 `root_888` 입니다.
- DoOR.data 커밋 `db323a49`: `data/door/` 의 CSV 4개와 `data/door/receptors/*.csv` ([ropensci/DoOR.data](https://github.com/ropensci/DoOR.data)). URL과 해시는 `data/door/SOURCE.json`, `qc_reports/hallem_calibration/manifest.json` 에 기록되어 있습니다.

## 실행

```bash
python -m venv .venv && .venv/bin/pip install -r requirements.txt
.venv/bin/python -m unittest discover -s tests -q
# 예: 16~18
OPENBLAS_NUM_THREADS=4 .venv/bin/python pipelines/16_hallem_calibration.py
OPENBLAS_NUM_THREADS=1 .venv/bin/python pipelines/17_pn_community.py --nulls 1000 --workers 10
OPENBLAS_NUM_THREADS=1 .venv/bin/python pipelines/18_community_discrimination.py --nulls 100 --workers 10
```

각 단계의 실행 옵션은 해당 문서의 "재현" 절에 있습니다. 용량이 큰 재생성 산출물(`*.npz`, 08의 전체 metrics.csv)은 git에서 제외했습니다.

## 진행 원칙

새 실험은 가설, 지표, 판정 규칙을 먼저 문서로 고정한 뒤 실행합니다. 결과가 바뀐 경우 이전 판단을 지우지 않고 수정 과정을 문서에 남깁니다.

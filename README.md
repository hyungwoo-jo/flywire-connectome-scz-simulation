# flywire-connectome

초파리 커넥톰(BANC v888) 위에서 PN -> KC -> APL 버섯체 회로를 단순 발화율 모형으로 돌려 보고, 실제 배선이 무작위 대조 배선과 기능적으로 다른지 확인하는 개인 탐구입니다. 질환, 환각, 행동에 대한 결론은 아직 내리지 않습니다.

**현재 상황과 방향은 [docs/RESEARCH_STATUS.md](docs/RESEARCH_STATUS.md), 처음 읽는 분을 위한 쉬운 설명은 [docs/RESEARCH_FLOW_EASY.md](docs/RESEARCH_FLOW_EASY.md)에 있습니다.**

## 지금까지의 핵심 결과

- Hallem 2006 냄새 반응과 Olsen 2010 PN 변환으로 입력을 정하고 KC 반응 10%로 보정하면, 양쪽 반구 모두 생리 범위의 모형이 됩니다 (16, 22).
- BANC PN->KC 배선에는 우연 이상의 사구체 쌍 과수렴 구조가 있고, 양쪽 반구에서 재현됩니다 (DM1-DM4 등, 17, 22).
- 이 배선은 범용 냄새 구분에서 무작위 대조보다 약간 낮습니다. 오른쪽에서만 유의했고 왼쪽은 같은 방향이지만 유의하지 않았습니다 (18~20, 22).
- 과일 냄새 사이의 학습 일반화가 늘어난다는 결과는 왼쪽에서 재현되지 않았습니다 (21, 22).
- APL 억제를 약화해도 냄새 유무 판단의 민감도는 그대로이고 오탐은 아주 조금만 늡니다. 문턱 근처에서는 억제보다 KC 흥분성이 판단을 좌우합니다 (23).
- KC 흥분성을 높이면 오탐이 크게 늘지만 민감도는 그대로입니다. 특정 냄새에 대한 기대(흥분성 증가로 근사)는 그 냄새의 검출을 돕고 작은 냄새 특이적 오탐을 만듭니다 (24).
- 학습만 해도, 잡음으로 생긴 거짓 "있다" 판단의 약 3분의 2가 학습한 냄새의 가치 신호를 띱니다. 입력이 많은 허브 KC가 냄새와 잡음 양쪽에 참여하기 때문이며, KC 입력을 고르게 맞추면 이 누출은 사라지고 구분은 좋아집니다 (25, 26). KC 사이의 보상은 누출을 줄이고, 냄새 구분은 부분 보상에서 가장 좋습니다 (27, 29). 이 누출은 나머지 사구체에 입력을 대입한 전체 회로에서도 유지되고 (28), KC 입력을 충분히 받는 거의 모든 MBON에서 나타납니다 (30).
- 배선 수준의 KC 보상은 hemibrain에서만 약하게 보이고 BANC와 FlyWire에서는 반대 방향입니다. 같은 정의로 비교해도 다르므로 자료 차이입니다 (32, 34, 36).
- 보정, 사구체 쌍 과수렴, 학습된 가치의 누출은 다른 두 개체(FlyWire 두 반구, hemibrain)에서도 재현됩니다. 세 개체, 다섯 반구입니다 (34, 36). KC끼리의 연결을 흥분성이나 억제성으로 넣어도 이 누출은 거의 바뀌지 않습니다 (37). 다만 누출의 상당 부분은 냄새 표상 자체보다 "입력 많은 KC에서 학습이 일어났다"는 점에서 옵니다 (38).

## 구조

```
pipelines/   단계별 실험 스크립트 (05~38, 번호 = 진행 순서, x = 사후/탐색 분석)
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
| 19 | 불리함의 원인 분리 (KC 하위 유형 보존 대조) | [PREREGISTRATION_19](docs/PREREGISTRATION_19.md), [DEFICIT_CAUSE_RESULTS](docs/DEFICIT_CAUSE_RESULTS.md) |
| 20 | 사구체 표지 순열 (구조 자체인가, 수용체 정렬인가) | [PREREGISTRATION_20](docs/PREREGISTRATION_20.md), [LABEL_PERMUTATION_RESULTS](docs/LABEL_PERMUTATION_RESULTS.md) |
| 21 | 학습의 일반화 | [PREREGISTRATION_21](docs/PREREGISTRATION_21.md), [LEARNING_GENERALIZATION_RESULTS](docs/LEARNING_GENERALIZATION_RESULTS.md) |
| 22 | 왼쪽 반구 재현 | [PREREGISTRATION_22](docs/PREREGISTRATION_22.md), [LEFT_REPLICATION_RESULTS](docs/LEFT_REPLICATION_RESULTS.md) |
| 23 | 냄새 유무 검출과 오탐, APL 약화 | [PREREGISTRATION_23](docs/PREREGISTRATION_23.md), [DETECTION_RESULTS](docs/DETECTION_RESULTS.md) |
| 24 | KC 흥분성과 냄새 특이적 기대 | [PREREGISTRATION_24](docs/PREREGISTRATION_24.md), [EXCITABILITY_EXPECTATION_RESULTS](docs/EXCITABILITY_EXPECTATION_RESULTS.md) |
| 25 | 학습된 가치의 잡음 누출 | [PREREGISTRATION_25](docs/PREREGISTRATION_25.md), [LEARNED_VALUE_LEAKAGE_RESULTS](docs/LEARNED_VALUE_LEAKAGE_RESULTS.md) |
| 26 | 허브 KC 기제의 인과 검정 | [PREREGISTRATION_26](docs/PREREGISTRATION_26.md), [HUB_NORMALIZATION_RESULTS](docs/HUB_NORMALIZATION_RESULTS.md) |
| 27 | KC별 문턱 보상의 강도 | [PREREGISTRATION_27](docs/PREREGISTRATION_27.md), [THRESHOLD_COMPENSATION_RESULTS](docs/THRESHOLD_COMPENSATION_RESULTS.md) |
| 28 | 전체 회로 대입으로 견고성 점검 | [PREREGISTRATION_28](docs/PREREGISTRATION_28.md), [FULL_CIRCUIT_IMPUTATION_RESULTS](docs/FULL_CIRCUIT_IMPUTATION_RESULTS.md) |
| 29 | 입력 강도 보상으로 부분 보상 최적 검정 | [PREREGISTRATION_29](docs/PREREGISTRATION_29.md), [INPUT_COMPENSATION_RESULTS](docs/INPUT_COMPENSATION_RESULTS.md) |
| 30 | 모든 MBON에서 누출의 일반성 | [PREREGISTRATION_30](docs/PREREGISTRATION_30.md), [ALL_MBONS_RESULTS](docs/ALL_MBONS_RESULTS.md) |
| 31 | 허브 KC의 정체 | [PREREGISTRATION_31](docs/PREREGISTRATION_31.md), [HUB_IDENTITY_RESULTS](docs/HUB_IDENTITY_RESULTS.md) |
| 32 | 실제 배선 안의 보상 | [PREREGISTRATION_32](docs/PREREGISTRATION_32.md), [WIRING_COMPENSATION_RESULTS](docs/WIRING_COMPENSATION_RESULTS.md) |
| 33 | KC별 APL 억제와 보상 | [PREREGISTRATION_33](docs/PREREGISTRATION_33.md), [APL_COMPENSATION_RESULTS](docs/APL_COMPENSATION_RESULTS.md) |
| 34 | FlyWire(다른 개체) 재현 | [PREREGISTRATION_34](docs/PREREGISTRATION_34.md), [FLYWIRE_REPLICATION_RESULTS](docs/FLYWIRE_REPLICATION_RESULTS.md) |
| 35 | KC 이해: 문헌과 세 커넥톰의 기술 지도 | [KC_PRIMER](docs/KC_PRIMER.md) |
| 36 | hemibrain 재현과 보상 정의 비교 | [PREREGISTRATION_36](docs/PREREGISTRATION_36.md), [HEMIBRAIN_REPLICATION_RESULTS](docs/HEMIBRAIN_REPLICATION_RESULTS.md) |
| 37 | KC->KC 연결을 넣은 모형 | [PREREGISTRATION_37](docs/PREREGISTRATION_37.md), [KC_RECURRENCE_RESULTS](docs/KC_RECURRENCE_RESULTS.md) |
| 38 | 누출의 냄새 특이성 (입력 강도 맞춘 통제) | [PREREGISTRATION_38](docs/PREREGISTRATION_38.md), [MATCHED_CONTROL_RESULTS](docs/MATCHED_CONTROL_RESULTS.md) |

개념 설명은 [RESEARCH_DIRECTION_EXPLAINED](docs/RESEARCH_DIRECTION_EXPLAINED.md)에 있습니다.

## 데이터

원자료는 저장소에 포함하지 않습니다. 각 스크립트는 실행 시 SHA-256으로 입력을 확인합니다.

- BANC v888: `data/banc_888_meta.feather`, `data/banc_888_edgelist_simple_v2.feather` ([BANC 프로젝트](https://github.com/htem/BANC-project)). 연결의 ID는 `root_888` 입니다.
- hemibrain v1.2 압축 연결: `https://storage.googleapis.com/hemibrain/v1.2/exported-traced-adjacencies-v1.2.tar.gz`를 `data/hemibrain/`에 풉니다(35~).
- FlyWire v783 연결: [Zenodo 10676866](https://zenodo.org/records/10676866)의 `proofread_connections_783.feather`를 `data/flywire/`에 둡니다(34). 주석은 `data/flywire_v783_neuron_annotations.tsv`.
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

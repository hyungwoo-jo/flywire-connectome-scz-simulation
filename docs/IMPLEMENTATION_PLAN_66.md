# 66 단계 실행 계획 (실행 담당 모델용)

이 문서는 [PREREGISTRATION_66.md](PREREGISTRATION_66.md)를 코드로 옮겨 실행하는 절차다. 사전 등록이 기준이다. 이 문서와 사전 등록이 다르면 사전 등록을 따르고, 차이를 결과 문서에 적는다.

## 0. 지켜야 할 규칙

- **고치지 말 것:** `docs/PREREGISTRATION_66.md`와 기존 파이프라인(05~65)은 수정하지 않는다. 계산 정의를 바꿔야 하면 결과 문서의 "사전 등록과 다른 점" 절에 이유와 함께 적는다.
- **판정과 해석:** 판정은 사전 등록의 규칙을 기계적으로 적용한다. 숫자가 기대와 달라도 기준을 바꾸지 않는다.
- **문체:** 문서는 한국어로, 건조한 문체로 쓰고 이모지를 쓰지 않는다. 코드 주석은 영어로 짧게, 필요한 곳에만 단다.
- **코드 문체:** 기존 파이프라인처럼 짧은 함수와 한 줄에 여러 문장을 쓰는 문체를 따른다.
- **실행 환경:**
  - 파이썬은 `.venv/bin/python`이다. 테스트는 `.venv/bin/python -m unittest discover -s tests -q`로 돌린다(pytest 없음).
  - 긴 실행에는 `OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1`을 붙여 백그라운드로 돌리고, 로그를 파일로 남긴다.
  - 기다릴 때는 `until ! pgrep -f "6[6]_solvent_input" >/dev/null; do sleep 10; done` 같은 대기 루프를 백그라운드로 쓴다. 전경에서 sleep을 이어 붙이지 않는다.
- **멈출 조건:** Q0(재현 점검)가 실패하면 문서를 고치지 말고 멈춘다. 실패 내용(기대 값, 얻은 값, 차이)을 보고한다.
- **커밋:** 마지막 점검 목록(7절)을 모두 통과했을 때만 한다.

## 1. 만들 파일

| 파일 | 내용 |
|---|---|
| `pipelines/66_solvent_input.py` | 파이프라인 |
| `tests/test_solvent_input.py` | 단위 테스트 (원자료 없이 돌아가야 함) |
| `docs/SOLVENT_INPUT.md` | 결과 문서 |
| `qc_reports/solvent_input/` | 산출물 |

## 2. 기존 코드에서 가져다 쓰는 것

65 모듈을 불러오면 64, 63, 55, 58, 34, cal, 60 lib가 따라 들어온다. 65 파일 맨 위의 불러오기 방식을 그대로 쓴다.

```python
spec=_u.spec_from_file_location('p65',Path(__file__).with_name('65_kc_imaging_chan2026.py'))
p65=_u.module_from_spec(spec);sys.modules['p65']=p65;spec.loader.exec_module(p65)
p64,p63,lib=p65.p64,p65.p63,p65.lib;p55,p58,p34,cal=p63.p55,p63.p58,p63.p34,p63.cal
p25,p24=p55.p25,p55.p24
spec=_u.spec_from_file_location('p43',Path(__file__).with_name('43_leakage_theory.py'));p43=_u.module_from_spec(spec);spec.loader.exec_module(p43)
```

쓰는 함수:

**회로와 문턱**
- `p55.p37.build_all(root)`, `p58.build(root)`: 일곱 반구 회로.
- `p34.make_g(c,w,receptors,delta,sfr,calib)`: 회로별 입력 묶음. x_full, x_cal, x_noise, x_weak, keep, lab, n, w가 들어 있다.
- `p63.thresholds(kind,E,En,c,theta0)`: 네 조건의 문턱. 부르기 전에 반드시 `p63._D['m']=m`을 넣어야 한다.
- `p26.equalize_rows`: flat 조건. 63과 65처럼 `26_hub_normalization.py`에서 불러온다. 57과 58의 flat(p29)과는 다르다.

**반응과 측정**
- `p55.respond(E,c,th)`: KC 반응. th는 `(n_kc,1)` 모양이어야 한다.
- `cal.sample_orn`, `cal.pn_drive`, `cal.calibrate`, `cal.load_hallem`: 입력 생성과 보정.
- `p25.evaluate(m,g)`: 누출. g에 theta가 있어야 한다. `(rows, pp)`를 돌려준다. rows는 냄새마다 leak_learned와 leak_control을 담고, pp에는 noise_FA가 있다.
- `p43.index_terms(w,learn,m)`: `(L_hat, H, base)`를 돌려준다.
- `p65.rho`: NaN 처리가 있는 Spearman. Chan 흉내에서 쓴다.

**실측 자료 처리**
- `p63.ODORS`, `p65.HALLEM`: 냄새 이름 대응. 모형 냄새 번호는 63과 65의 main처럼 `names.reindex(odors.index)`로 찾는다.
- `lib.hemibrain`, `lib.flywire`, `lib.banc`, `lib.malecns`: KC 아형 `cls`. 65의 main을 그대로 따른다.

**Chan 마리 구성:** `qc_reports/kc_imaging_chan2026/per_fly.csv`의 `type`, `n` 열에서 읽는다. 65를 다시 돌리지 않는다.

## 3. 파이프라인 구조

### 상수

```python
SEED0=660000;N_REAL=200;N_REAL_SENS=50;N_SOLV=500;N_BOOT=1000
SCALE_MAIN=1.;SCALES_SENS=(.5,.25)
CAMP=dict(n_rep=6,n_solv=6,n_spont=600,min_rel=3,n_cell=124,n_sub=5)   # step 63/64
CHAN=dict(n_rep=3,n_solv=3,n_spont=280,min_rel=2)                        # step 65
SPEC_MIN=.003;MIN_HEMI=5
```

### 함수 (테스트할 수 있게 순수 함수로 분리)

1. `load_solvent(root, receptors)`
   - SHA-256 대조: `qc_reports/hallem_calibration/manifest.json`의 `['source']['receptor_csv_sha256']`를 읽는다. 각 `data/door/receptors/<r>.csv`의 SHA-256이 같지 않거나 키가 없으면 `ValueError`를 낸다.
   - 값 읽기: `pd.read_csv(f,sep=';',index_col=0)`에서 `InChIKey=='solvent'`인 행의 `Hallem.2004.EN` 값을 읽는다.
   - 확인: 24개가 모두 유한해야 하며, 아니면 `ValueError`를 낸다. 반환은 `np.array` (24,)이고, 순서는 receptors를 따른다.
   - 기대 값(점검용): Or47b 72.576, Or33b 0.5, Or85b 58.333, 평균 약 18.1.

2. `layout(n_odor, n_rep, n_solv, n_spont)`
   - 반환: `lab`(입력 열 번호), `sl`(구간 사전 `odor`, `solv`, `spont`).
   - 열 구성: 0..n_odor-1은 냄새, n_odor는 용매, n_odor+1은 0 입력이다.
   - 시행 순서: 냄새별 n_rep회, 그다음 용매 n_solv회, 자발 n_spont회.

3. `cell_measures(on, n_odor, n_rep, n_solv, min_rel)`
   - on은 KC × 시행의 bool이다.
   - 반환: `B`(냄새마다 n_rep 중 min_rel 이상 켜진 냄새 수), `S`(용매 구간 켜짐 비율), `R`(자발 구간 켜짐 비율).

4. `circuit(name, kind, s, inclusive)`
   - 반환: `(c, g, m, th, delta_eff, receptors, sfr)`.
   - 기존 방식 (`inclusive=False`):
     - 65의 `_run` 앞부분과 같다. 회로를 만들고, flat이면 equalize한다.
     - `p63._D['m']=m`을 넣고 E와 theta0을 구한 뒤, `th=p63.thresholds(...)[:,None]`로 문턱을 정한다.
   - 용매 포함 (`inclusive=True`, C 변형):
     - g를 직접 만든다. 냄새 열에 `s[:,None]`를 더한 `delta_inc=delta+s[:,None]`로 `x_full`(keep 열)과 보정 E(calib 열)를 만든다.
     - `x_weak`는 `.1*delta[:,keep]+s[:,None]`를 `cal.sample_orn`으로 만든다. 시드와 lab은 make_g와 같다.
     - `keep`는 원래 delta 기준(make_g의 것)을 그대로 쓴다. `x_cal`, `x_noise`는 바꾸지 않는다.
     - theta0과 조건별 문턱은 이 E로 다시 정한다.

5. `imaging_sim(job)`: A 부분 작업 하나. job은 `(name, kind, scale, inclusive, n_real)`이다.
   - 난수: `rng=np.random.default_rng(SEED0+1000*i_name+100*i_kind+10*i_scale+(5 if inclusive else 0))`이다.
   - Campbell 흉내:
     - 입력 열 `d=[delta_eff[:,idx_camp], scale*s[:,None], zeros]`로 `layout(6,6,6,600)`을 만든다.
     - 반복마다 `on=p55.respond(m@cal.pn_drive(c,receptors,cal.sample_orn(d,sfr,lab,.5,rng)),c,th)>1e-8`를 구하고 `cell_measures(...,min_rel=3)`를 적용한다.
     - KC 124개 무복원 추출을 5번 해서 `stats.spearmanr(B,S)`, `stats.spearmanr(B,R)`를 잰다. NaN이면 0이다(63 관례).
   - Chan 흉내:
     - 입력 열은 같은 구조로, `idx_chan`과 `layout(6,3,3,280)`을 쓴다. `min_rel=2`다.
     - 마리 (type, n)마다 65처럼 아형 풀에서 추출해 `p65.rho`를 잰다. 마리 평균은 nanmean이다.
   - 기록: 반복마다 4개 상관과 함께 `solv_active`(용매 구간 켜짐 비율), `odor_active`(냄새 구간 켜짐 비율)를 남긴다.
   - 주의: `delta_eff`는 inclusive일 때 냄새 열에 s를 더한 값이다. 용매 열은 inclusive와 관계없이 `scale*s`다.

6. `leak_job(job)`: B 부분 작업 하나. job은 `(name, kind, scale, inclusive)`이다.
   - 용매 시행 입력: `x_solv=cal.pn_drive(c,receptors,cal.sample_orn(scale*s[:,None],sfr,np.zeros(N_SOLV,int),.5,np.random.default_rng(SEED0+1)))`
   - 누출 계산:
     - `rows_sp,pp_sp=p25.evaluate(m,dict(g,theta=th))`는 자발 누출이다.
     - `rows_so,pp_so=p25.evaluate(m,dict(g,theta=th,x_noise=x_solv))`는 용매 누출이다.
   - specific: 냄새마다 `leak_learned-leak_control`이고, 평균과 부트스트랩 95% 구간을 낸다. 부트스트랩은 `default_rng(SEED0+2)`로 냄새를 복원 추출해 1000회 돌린다.
   - 보고율: `pp_so['noise_FA']`.
   - 냄새별 분해:
     - `kt=p24.kc(m,g['x_full'],c,th)[0]`, `ms=p24.kc(m,x_solv,c,th)[0].mean(1)`를 구한다.
     - 냄새 a마다 `H_solvent=p43.index_terms(g['w'],kt[:,a],ms)[1]`이다.
     - `cos_solvent`는 `delta[:,g['keep']][:,a]`와 s의 코사인이다. inclusive여도 원래 delta로 잰다.

7. 판정 함수들 (테스트 대상)
   - `compatible(lo, hi, obs)`: `lo<=obs<=hi`이고, 구간이 NaN이면 False다.
   - `z3(spec, lo)`: 일곱 반구 배열을 받는다.
     - "present"(있음): `sum((spec>SPEC_MIN)&(lo>0))>=MIN_HEMI`
     - "absent"(없음): `sum(spec<=SPEC_MIN)>=MIN_HEMI`
     - 둘 다 아니면 "mixed"(혼합).
     - 둘 다 참일 수는 없다. 7개 중 5개가 서로 겹칠 수 없기 때문이며, 테스트로 확인한다.
   - `interval(draws)`: NaN을 빼고 2.5와 97.5 백분위를 구한다. NaN 비율도 함께 낸다.

8. `main()`
   - 인자: `--part main|sens|inclusive|report`. main은 사전 등록의 주 분석(scale 1.0, inclusive False, N_REAL 200)이다.
   - 공통 준비: 63과 65의 main처럼 회로, 입력, 냄새 번호, cls, 마리 구성을 `p63._D`와 모듈 `_D`에 채운 뒤 `Pool(14)`를 연다(`chunksize=1`). fork 전에 전역을 채워야 한다.
   - main 순서:
     1. B 작업 28개를 먼저 돌린다. 빠르다.
     2. Q0를 점검하고 실패하면 즉시 종료한다(exit code 2).
     3. A 작업 28개를 돌린다.
   - sens: SCALES_SENS × 28로 A(N_REAL_SENS)와 B를 돌린다.
   - inclusive: scale 1.0, 28개로 A(N_REAL_SENS)와 B를 돌린다.
   - 각 part는 `qc_reports/solvent_input/<part>/`에 csv를 쓴다. 상관 표본은 `draws_<part>.npz`로 따로 둔다(git 제외 확인).
   - report: 있는 산출물을 모아 `qc_reports/solvent_input/report.json`을 쓴다.

### Q0 상세

- 자발 누출:
  - main의 B 결과에서 `input=='spont'`의 specific을 기준 값과 비교한다.
  - 기준: 5반구는 `qc_reports/mixed_homeostasis/per_condition.csv`이고, MaleCNS는 `qc_reports/malecns_prediction/per_condition.csv`다. 두 파일 모두 `condition=='original'`과 `condition=='total_homeo'`(par 0.01)다.
  - 차이 절댓값이 1e-6 이하여야 한다.
  - 참고로 저장된 값의 예는 BANC_right original 0.034247, total_homeo 0.002493이다.
- 자발 구간:
  - main의 A 결과에서 Campbell 자발 구간을 64 보고서 `model_pooled_interval`와, Chan 자발 구간을 65 보고서 `model_interval.spont`와 비교한다.
  - 각 경계의 차이가 0.05 이하여야 한다. 단, A는 B 뒤에 돌므로 이 점검은 A가 끝난 뒤에 한다. 실패하면 판정은 "Q0 실패"로 적고 Z 판정은 하지 않는다.
- 1e-6이 맞지 않는 흔한 원인:
  - `p24.SEEDS.update(noise=p25.SEEDS['noise'],signal=p25.SEEDS['signal'])`를 빠뜨렸다.
  - th 모양이 다르다.
  - flat의 정의를 혼동했다. 다만 Q0에는 flat이 없다.

## 4. 산출물 형식

- **`model_imaging.csv`** 열: part, scale, inclusive, dataset, condition, measure, med, lo, hi, nan_frac, solv_active, odor_active.
  - measure 값: campbell_oil, campbell_spont, chan_oil, chan_spont.
- **`pooled_intervals.csv`** 열: part, scale, inclusive, condition, measure, lo, hi, nan_frac, obs, compatible.
- **`leak.csv`** 열: part, scale, inclusive, dataset, condition, input, leak_learned, leak_control, specific, ci_lo, ci_hi, report_rate.
  - input 값: spont, solvent.
- **`per_odor_solvent.csv`** 열: part, scale, inclusive, dataset, condition, odor, specific, H_solvent, cos_solvent.
- **`report.json`** 키:
  - `real`: 네 실측 값과 출처 경로.
  - `solvent_vector`: 24값과 수용체 이름.
  - `Q0`: 항목별 기대, 얻은 값, 통과 여부.
  - `verdicts`: Z1, Z2(조건별 통합 양립 bool), Z3(조건별 present, absent, mixed와 반구별 값).
  - `interpretation_rule`: 해석 규칙 1~4 중 해당 번호.
  - `report_only`: 배율 민감도, 분해 상관, 활동 비율, C.

## 5. 테스트 (`tests/test_solvent_input.py`, 원자료 없이)

1. **`load_solvent`:**
   - 준비: 임시 폴더에 가짜 manifest와 수용체 CSV 두 개(`InChIKey`, `Hallem.2004.EN` 열과 `solvent` 행)를 만든다.
   - 값이 순서대로 나와야 한다.
   - CSV 한 바이트를 바꾸면 `ValueError`가 나야 한다.
   - 값이 NaN이면 `ValueError`가 나야 한다.
2. **`layout`:** `layout(2,3,2,4)`의 lab이 `[0,0,0,1,1,1,2,2,3,3,3,3]`이고, 구간이 맞아야 한다.
3. **`cell_measures`:** 손으로 만든 on 배열에서 B, S, R이 기대와 같아야 한다.
4. **`z3`:**
   - present 예: spec 7개 중 5개 > 0.003이고 lo > 0.
   - absent 예.
   - mixed 예.
   - 무작위 배열 1000개에서 present와 absent가 동시에 참이 아니어야 한다.
5. **`compatible`:** NaN 구간이면 False여야 한다.
6. 불러오기 방식은 65 테스트처럼 `spec_from_file_location`으로 한다. 모듈을 불러올 때 무거운 계산이 돌지 않아야 한다. 계산은 모두 main 안에 둔다.

## 6. 실행 순서

```bash
cd ~/codespace/flywire_connectome
.venv/bin/python -m unittest tests.test_solvent_input -q
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python pipelines/66_solvent_input.py --part main > <scratchpad>/run66_main.log 2>&1   # background
# Q0 통과 확인 후
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python pipelines/66_solvent_input.py --part sens > <scratchpad>/run66_sens.log 2>&1
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python pipelines/66_solvent_input.py --part inclusive > <scratchpad>/run66_inc.log 2>&1
.venv/bin/python pipelines/66_solvent_input.py --part report
.venv/bin/python -m unittest discover -s tests -q
```

- 예상 시간(16코어 기준):
  - main: 약 40~60분. 63과 65의 시행 수를 합친 것과 비슷하다.
  - sens: 약 30분.
  - inclusive: 약 15분.
- 처음에는 반구 하나, 조건 하나, N_REAL 2로 시간을 재고 나서 전체를 돌린다. 이 시험 실행 결과는 보지 않고 버린다(판정 전 수치 노출 방지).

## 7. 결과 문서와 색인

`docs/SOLVENT_INPUT.md`는 65 문서(`KC_IMAGING_CHAN2026.md`)의 절 구성을 따른다.

- **질문**
- **입력 (s 표):** 24개 값 표 또는 상위 5개와 평균.
- **결과:**
  - A: 조건 × 측정 표에 구간과 양립 여부를 적는다.
  - B: 조건 × 반구 표에 용매 specific, 구간, 보고율, 같은 조건의 자발 specific을 적는다.
- **판정 표:** Q0, Z1, Z2, Z3.
- **해석:** 사전 등록의 해석 규칙 번호를 그대로 적용한다. 해석은 그 규칙의 문장을 벗어나지 않는다.
- **보고만:** 배율, 분해, 활동 비율, C.
- **한계:** 사전 등록의 한계와 실행 중 발견한 것.
- **재현:** 실행 명령과 산출물 위치.

색인 갱신:

- **`README.md`:**
  - 표에 66 행을 넣는다. 형식: `| 66 | 용매를 약한 비특이 입력으로 넣은 모형: 실측 정합과 누출 | [PREREGISTRATION_66](docs/PREREGISTRATION_66.md), [SOLVENT_INPUT](docs/SOLVENT_INPUT.md) |`
  - 구조 블록의 범위를 `05~66`으로, "정리" 행을 `01~66`으로 바꾼다.
  - 핵심 결과 마지막 항목에 66 결과를 한두 문장 덧붙인다.
- **`docs/RESEARCH_STATUS.md`:**
  - 흐름 표에 66 행을 넣는다.
  - "지금 알게 된 것"을 갱신한다.
  - "다음 단계"를 해석 규칙의 결과에 맞게 고친다.
  - 문서 지도에 `- 66 계획과 결과: PREREGISTRATION_66.md, SOLVENT_INPUT.md (...)`를 넣는다.
- **`docs/THEORY_NOTE.md`:**
  - 머리 문장의 범위를 01~66으로 바꾼다.
  - 3절 표의 65 행 다음에 66 행을 넣는다.
  - 1절의 "66에서 검정한다"를 결과에 맞게 고친다.

## 8. 커밋 전 점검 목록

- [ ] 전체 테스트가 통과했다.
- [ ] Q0를 통과했다. 실패했다면 커밋하지 않고 보고한다.
- [ ] 결과 문서의 모든 숫자가 `report.json` 및 csv와 일치한다.
- [ ] 판정이 사전 등록 규칙의 기계적 적용이다.
- [ ] 사후 분석에는 "사후" 또는 "보고만"이라고 표시했다.
- [ ] `git status`에 원자료(`data/`)와 큰 npz가 없다.

통과하면 다음을 한다.

```bash
git add pipelines/66_solvent_input.py tests/test_solvent_input.py docs/SOLVENT_INPUT.md qc_reports/solvent_input README.md docs/RESEARCH_STATUS.md docs/THEORY_NOTE.md
git commit -m "Step 66: solvent as weak nonspecific input; joint fit to imaging and leak

Co-Authored-By: <실행 모델 표기>"
git push
herdr-send-telegram --topic 초파리 --file <요약 파일>
```

- 텔레그램 요약은 한국어 10줄 안팎으로 쓴다: 질문, 입력, 판정 4개, 해석 규칙 번호, 한계 한 줄, 커밋 해시.
- `--channel`과 chat id는 쓰지 않는다.

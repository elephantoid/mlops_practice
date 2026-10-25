# MLflow 3.x 실전 — 2.x 튜토리얼이 깨지는 여섯 지점

> ChurnWatch(통신사 이탈 예측) 레포에서 실제로 부딪힌 것만 씁니다.
> 버전·에러·수치는 전부 실행으로 확인했고, 개념 설명은 공식 문서와 교차 검증했습니다.

**검증 시점:** 모든 실행 결과와 코드 인용은 **커밋 `21c43bf`(2026-08-11)** 기준입니다. 이 레포는
2026-09-22에 RiskWatch(credit/fraud 2트랙)로 retarget되어 `churnwatch` 모델과 `churn_probability`
계약은 현재 main에 없습니다. **MLflow 관련 여섯 가지 교훈은 전부 main에서도 그대로 살아 있고**,
현재 위치는 맨 아래 [검증 시점과 현재 위치](#검증-시점과-현재-위치)에 매핑해 두었습니다.

---

## 시리즈 개요 (4부작)

**1편 · MLflow 3.x 실전 — 2.x 튜토리얼이 깨지는 여섯 지점** ← 이 글
`file:` 백엔드 예외, stage 폐기, skops 기본 직렬화, pyfunc 반환값, artifact 절대 경로.
3.x에서 바뀐 지점을 실행으로 확인하고 실패한 순서 그대로 정리합니다.

**2편 · FastAPI 서빙 — 학습/서빙 스큐를 실제로 잡아낸 방법**
snake_case 공개 계약과 CamelCase 학습 컬럼을 pydantic `serialization_alias`로 잇습니다.
컨테이너 경계 너머에서 확률이 소수점 끝자리까지 일치하는지 검증합니다.

**3편 · 관측 가능성 — "그럴듯하지만 틀린 대시보드" 피하기**
기본 히스토그램 버킷이 3~6ms 서비스에서 왜 무용지물인지, 4xx와 5xx를 왜 합치면 안 되는지.
침묵하도록 설계한 로거에 신호를 되돌려 주는 방법.

**4편 · Evidently 0.7 드리프트 감지 — 0.4와의 단절**
`Report.run`이 Snapshot을 반환하고 `as_dict`가 사라진 API 단절, Pushgateway로 배치 지표 내보내기.
그리고 `drift_share = 1.0`이 맞으면서 동시에 쓸모없는 이유.

---

## 왜 이 글인가

온라인 자료 대부분은 2.x 기준입니다. 아래 세 줄을 설치된 3.15.1에서 실행하면 이렇게 됩니다 —
**깨지는 건 하나뿐이고, 나머지 둘은 조용히 통과합니다.** 그게 이 글의 주제입니다.

```python
mlflow.set_tracking_uri("file:./mlruns")                      # MlflowException — 즉시 실패
client.transition_model_version_stage(name, v, "Production")  # FutureWarning — 동작함
mlflow.sklearn.log_model(model, "model")                      # WARNING: `artifact_path` is
                                                              # deprecated. Use `name` instead.
```

검증 환경(`uv.lock` 해석 결과): `mlflow 3.15.1` · `scikit-learn 1.9.0` · `lightgbm 4.7.0` ·
`cloudpickle 3.1.2` · `skops 0.14.0` · Python 3.12.

---

## 1. tracking backend — `file:`은 이제 예외다

**① 공식** 파일시스템 백엔드는 유지보수 모드입니다. `mlflow migrate-filestore`로 무손실 이관하되
**타깃은 SQLite만** 지원합니다(FileStore의 experiment ID가 32비트 정수를 넘김).

**② 실측** 경고가 아니라 `MlflowException`입니다.

```
The filesystem tracking backend (e.g., './mlruns') is in maintenance mode and will not
receive further updates. Please migrate to a database backend (e.g., 'sqlite:///mlflow.db')
... set `MLFLOW_ALLOW_FILE_STORE=true` to opt out of this exception.
```

URI 없이 빈 디렉터리에서 run을 만들면 `mlruns/`가 아니라 **`mlflow.db`가 생깁니다**(기본값
`sqlite:///<cwd>/mlflow.db`). "`mlruns`가 보이면 정상"이라는 2.x의 감각이 무너집니다.

```python
# CWD 상대 경로는 빈 DB를 조용히 새로 만든다 → 레포 루트에 앵커링
DEFAULT_TRACKING_URI = f"sqlite:///{PROJECT_ROOT / 'mlflow.db'}"   # train.py:59
```

**③ 커뮤니티** 공지 [#18534](https://github.com/mlflow/mlflow/issues/18534)에 3.6 경고 → 3.7 sqlite
전환 타임라인과 폐기 근거(트레이스·평가 데이터셋·웹훅 미지원)가 있습니다. `MLFLOW_ALLOW_FILE_STORE`는
rsync로 받은 `mlruns`를 읽기 전용으로 여는 식의 예외용이지, 개발 편의용 우회로가 아닙니다.

---

## 2. Model Registry — stage 폐기, alias로

**① 공식** Model Stage는 **2.9.0부터 폐기**, 이후 메이저에서 제거 예정입니다. alias는 버전을 가리키는
가변 이름이고 stage와 달리 **한 버전에 여러 개**를 붙일 수 있어 A/B와 점진 롤아웃이 쉽습니다.

**② 실측** 옛 API는 **아직 동작합니다.** 예외가 아니라 경고라 더 위험합니다.

```
FutureWarning: ``MlflowClient.transition_model_version_stage`` is deprecated since 2.9.0.
Model registry stages will be removed in a future major release.
```

정작 물린 건 다른 지점입니다. **`register_model()`의 반환 객체는 alias·태그를 붙이기 전 스냅샷**이라
`.tags`가 비어 있습니다. 또 3.x 등록 버전의 `source`는 run 경로가 아니라 **`models:/m-<uuid>`** 였습니다
— 모델이 run artifact에서 독립한 3.0 변경의 결과입니다.

```python
version = mlflow.register_model(f"runs:/{best.run_id}/model", MODEL_NAME)
client.set_registered_model_alias(MODEL_NAME, PRODUCTION_ALIAS, version.version)
return client.get_model_version(MODEL_NAME, version.version)   # train.py:257 — 재조회 필수
```

**③ 커뮤니티** alias 조언은 흔하지만 대개 "champion/challenger로 부르라" 수준입니다. 스냅샷 재조회
문제는 근거를 찾지 못했습니다 — **내 환경에서 관측한 것**으로만 씁니다.

---

## 3. 직렬화 — 기본값이 skops, 서드파티 estimator를 거부한다

**① 공식** `serialization_format` 기본값이 **`'skops'`** 입니다. 근거는 보안 — pickle/cloudpickle은
역직렬화 중 임의 코드를 실행할 수 있습니다. 대신 **신뢰 타입 목록** 제약이 붙습니다
(`skops_trusted_types`로 명시 허용 가능).

**② 실측** LightGBM이 든 sklearn `Pipeline`을 기본값으로 로깅하면 실패합니다.

```
MlflowException: The saved sklearn model references untrusted types. ...
Root error: Untrusted types found in the file:
['collections.OrderedDict', 'lightgbm.basic.Booster',
 'lightgbm.sklearn.LGBMClassifier', 'numpy.dtype']
```

`OrderedDict`와 `numpy.dtype`까지 걸립니다. LightGBM 하나가 아니라 **파이프라인 전체**가 목록을
통과해야 합니다. 신뢰 목록 대신 포맷을 바꿨습니다.

```python
# 신뢰 목록은 조용히 낡는다: LightGBM 내부 타입이 하나 늘면 학습이 아니라
# *로드* 시점에, 즉 서빙 컨테이너가 뜰 때 깨진다. 보안 이득을 포기한 트레이드오프.
SERIALIZATION_FORMAT = "cloudpickle"   # train.py:71
```

**③ 커뮤니티** skops 문서는 `get_untrusted_types()`로 목록을 먼저 뽑으라 안내하고, XGBoost도
`xgboost.core.Booster` 등이 같이 걸립니다. **sklearn 외 부스팅 라이브러리를 쓰면 거의 반드시 만나는
벽**입니다. 단 `lightgbm.Booster`(네이티브)는 대상이 아니고 sklearn 인터페이스만 해당합니다.

---

## 4. pyfunc가 돌려주는 것 — 서빙 계약의 시작점

**① 공식** `pyfunc_predict_fn` 기본값은 **`'predict'`** 입니다. 즉 설정을 안 하면 pyfunc 분류기는
**확률이 아니라 클래스 레이블**을 돌려줍니다.

**② 실측** `predict_proba`로 로깅한 뒤 한 행을 넣은 결과와, 그것이 결정한 API 코드:

```python
# model.predict(row) → numpy.ndarray, shape (1, 2)
#   [[0.37984705 0.62015295]]   ← 1번 열이 이탈 확률
probability = float(request.app.state.model.predict(features)[0][1])   # main.py:173
# 임계값은 아티팩트가 아니라 API에: 결정 경계는 재학습 없이 바꿀 수 있어야 한다
outcome = "churn" if probability >= DECISION_THRESHOLD else "no_churn"
```

**③ 커뮤니티** 확률을 받는 방법으로 가장 흔한 조언은 `PythonModel`을 상속해 `predict`를 오버라이드하는
커스텀 래퍼입니다([#694](https://github.com/mlflow/mlflow/issues/694) 시절 패턴). **sklearn flavor라면
인자 하나로 끝납니다** — 오래된 답변이 상위에 남아 불필요한 래퍼를 쓰기 쉽습니다.

---

## 5. artifact 경로 — DB에 절대 경로가 박힌다

**① 공식** 아티팩트 서빙은 tracking 서버가 대신 해 주지 않습니다. **서버와 모든 클라이언트가 동일하게
접근 가능한 경로**를 쓰라고 안내하며, 로컬 경로는 실용적이지 않다고 명시합니다.

**② 실측** DB를 열면 뜻이 바로 보입니다. **학습 시점의 호스트 절대 경로**가 박혀 있어, 컨테이너에서
깔끔한 위치에 마운트하면 절대 해결되지 않습니다.

```sql
sqlite> select experiment_id, name, artifact_location from experiments;
1|churnwatch|/Users/elefunt/orca/workspaces/mlops_practice/spookfish/mlruns/1
```

```yaml
volumes:
  - ./mlruns:${PWD}/mlruns   # compose:38 — 깔끔한 경로가 아니라 DB가 기억하는 경로
```

파생 문제 둘. API도 아티팩트를 tracking 서버가 아닌 **파일시스템에서 직접 읽고**
(`LocalArtifactRepository`), 등록 모델 로드가 `registered_model_meta`를 **되쓰기** 때문에 `:ro`는
EROFS로 실패합니다. (둘 다 이 레포 커밋에 기록된 관측입니다.)

**③ 커뮤니티** [#15289](https://github.com/mlflow/mlflow/issues/15289)가 같은 벽을 반대편에서
보고합니다 — 상대 경로면 클라이언트가 `/mlruns`로 바꿔 권한 오류, 절대 경로면 저장은 되나 UI에서
안 보임. 2025년 4월 이후 응답 없이 열려 있는 **구조적 결합**입니다.

---

## 6. 선택(selection)과 보고(reporting)의 분리

**① 공식** MLflow가 아니라 실험 설계 문제입니다. scikit-learn 문서는 *"there is still a risk of
overfitting on the test set because the parameters can be tweaked"* 라고 명시합니다.

**② 실측** MLflow 테이블은 이 실수를 **부추깁니다.** 지표가 전부 정렬 가능한 컬럼이니 `test_roc_auc`로
정렬해 1등을 승격하는 게 가장 자연스럽습니다. 그래서 **선택은 5-fold CV AUC, 보고는 run당 한 번만
건드리는 테스트셋**으로 못 박았습니다(`order_by=["metrics.cv_auc_mean DESC"]`). 14개 config 결과:

| 기준 | 승격되는 config | CV AUC | test AUC |
|---|---|---|---|
| CV AUC (실제 채택) | `num_leaves=8, lr=0.1, n=100` | **0.8476** | 0.8435 |
| test AUC (하면 안 되는 것) | `num_leaves=8, lr=0.05, reg_lambda=5.0` | 0.8470 | **0.8448** |

**승격 모델은 테스트 1등이 아닙니다. 그게 정상입니다.** 다만 CV 차이 0.0006은 fold 표준편차 0.0109에
묻히고, LightGBM 1위와 LogisticRegression 최고(0.8462)의 차이 0.0014도 마찬가지 — **승리가 아니라
무승부입니다.** 정규화가 바꾼 건 순위가 아니라 과적합 폭이었습니다: `num_leaves=31`의 train−CV 갭
**0.1194**가 `num_leaves=8`에서 **0.0363**으로 줄었습니다. 그래서 `overfit_gap`을 별도 지표로 남깁니다.

**③ 커뮤니티** MLflow 맥락의 근거는 찾지 못했습니다. 위 수치는 **이 레포에서 관측한 것**이며 데이터셋
하나·시드 하나의 결과이므로 일반화하지 않습니다.

---

# 이러면 안 된다

**증상 → 원인 → 해결 → 미리 알아채기** 순, 끝에 **[일반]** / **[이 프로젝트 한정]** 표시.

### ① `python src/models/train.py` 로 실행

- **증상** `ModuleNotFoundError: No module named 'src'`
- **원인** 파일 직접 실행은 레포 루트가 아니라 `src/models/`를 `sys.path`에 올림
- **해결** `uv run python -m src.models.train`
- **미리** 모듈이 형제 패키지를 import하는 순간 `-m`이 강제됩니다. import 문 하나로 압니다. **[일반]**

### ② tracking URI를 CWD 상대 경로로 둠

- **증상** 레지스트리에 있는데 "모델 없음". **레지스트리 문제처럼** 보입니다
- **원인** `sqlite:///mlflow.db`는 CWD 기준. 다른 디렉터리면 **빈 DB를 조용히 새로 만듭니다**
- **해결** 레포 루트 앵커링
- **미리** 부팅 로그에 `mlflow.get_tracking_uri()`를 찍으면 즉시 보입니다. 실패가 "없음"이 아니라 "빈 것"으로 나타나는 저장소는 전부 같은 함정입니다. **[일반]**

### ③ `transition_model_version_stage`를 그대로 사용

- **증상** 없음. 잘 돌아갑니다. 그게 문제입니다
- **원인** 폐기됐지만 예외가 아니라 `FutureWarning`이라 로그를 안 보면 티가 안 남
- **해결** `set_registered_model_alias` + `models:/name@alias`
- **미리** 초기에 `-W error::FutureWarning`으로 한 번 돌리면 폐기 API가 전부 드러납니다(테스트에 상시 켜는 건 별개 문제 — 서드파티 경고까지 잡습니다). **[일반]**

### ④ 승격 검색을 experiment 전체에 검

- **증상** 주간 재학습이 **과거 최고점을 영원히 못 넘습니다**
- **원인** 옛 run은 *옛날 데이터* 점수인데 무제한 검색은 신규와 한 줄에 세웁니다. 게다가 `search_runs`는 status가 아니라 lifecycle stage로 거르므로 **FAILED run도 반환** — `cv_auc_mean`을 찍고 `log_model` 전에 죽은 run이 영원히 1등이 되어 이후 승격을 전부 깨뜨립니다
- **해결** 이번 sweep의 run id로 한정 + `attributes.status = 'FINISHED'`
- **미리** "이 함수가 **두 번째** 호출될 때"를 물으면 둘 다 드러납니다. 첫 호출에선 안 보입니다. **[일반]**

### ⑤ 아티팩트 마운트를 "깔끔한" 경로로 정리

- **증상** compose에서 `No such artifact: ''`. 호스트에선 같은 코드가 정상
- **원인** `artifact_location`에 **호스트 절대 경로**가 박혀 있음
- **해결** DB가 기억하는 경로에 그대로 마운트
- **미리** `select artifact_location from experiments;` 한 줄. 컨테이너화 **전에** 봤다면 설계 순서가 달랐습니다. **[일반]**

### ⑥ healthcheck가 초록이면 정상이라고 믿음

- **증상** healthcheck는 healthy인데 API의 모든 `/api/2.0` 호출이 **403**
- **원인** MLflow **3.5.0+**의 DNS 리바인딩 보호가 Host 헤더를 검사. compose에서 API는 `mlflow:5000`으로 접속하는데 허용 목록에 없음
- **해결** `MLFLOW_SERVER_ALLOWED_HOSTS: "mlflow:5000,localhost:5000,127.0.0.1:5000"`
- **미리** **`/health`는 이 검사에서 면제**라 초록인 채로 스택이 죽어 있었습니다. 헬스체크는 그 엔드포인트가 지나는 경로만 검증합니다. **[일반]** (면제는 공식 문서에서 확인하지 못했고 **내 환경에서 관측한 것**입니다.)

### ⑦ `:ro` 마운트로 안전하게

- **증상** API 컨테이너가 부팅 중 EROFS로 사망
- **원인** 등록 모델 로드가 `registered_model_meta`를 **되씀**. 로드는 읽기 전용 동작이 아님
- **해결** 해당 마운트에서 `:ro` 제거
- **미리** "읽기"라 불리는 동작이 정말 읽기만 하는지는 해 봐야 압니다. `:ro`를 먼저 붙이는 습관은 이 실패를 **빠르게** 드러냈으니 유지할 값어치가 있습니다. **[일반]**

### ⑧ 컨테이너 빌드 안에서 모델 export

- **증상** 빌드 중 레지스트리를 찾지 못함
- **원인** `mlflow.db`·`mlruns/`는 `.dockerignore` 대상 — 빌드 컨텍스트에 **원천적으로 없음**
- **해결** 호스트에서 먼저 `python -m src.models.export` → 산출물을 `COPY`, 해석된 버전을 `MODEL_VERSION`으로 굽기
- **미리** 로컬 경로 `MODEL_URI`는 버전 조회 수단을 주지 않아, 굽지 않으면 `/health`가 `"unknown"`을 보고합니다 — **어떤 모델이 떠 있는지가 가장 중요한 그 배포에서.**
- **[이 프로젝트 한정]** "MLflow 서버 없이 `docker run` 단독 동작"이라는 인수 조건에서 나온 제약입니다. 레지스트리를 항상 붙일 수 있으면 export 자체가 불필요합니다.

### ⑨ macOS의 `brew install libomp` 해법을 Dockerfile로 이식

- **증상** 컨테이너에서 LightGBM import 실패
- **원인** 이름만 닮은 다른 물건. macOS는 `libomp`, Linux 이미지는 **`libgomp1`** — `python:3.12-slim-bookworm`에 libgomp이 없고 리눅스 휠이 이를 링크
- **해결** Dockerfile에는 `libgomp1`만
- **미리** 호스트 전용 수정을 이미지로 옮길 때는 늘 이름 충돌을 의심합니다. **[일반]**

---

## 검증 시점과 현재 위치

이 글의 인용은 전부 `21c43bf`(2026-08-11) 기준입니다. 레포는 그 뒤 RiskWatch로 retarget되었지만
**여섯 교훈이 의존하는 코드는 전부 main(`45b42a5`)에 남아 있습니다** — 위치만 이동했습니다.

| 인용한 것 | `21c43bf` | main `45b42a5` |
|---|---|---|
| `DEFAULT_TRACKING_URI` (sqlite 앵커링) | `train.py:59` | `train.py:77` |
| `SERIALIZATION_FORMAT = "cloudpickle"` | `train.py:71` | `train.py:111` |
| `pyfunc_predict_fn="predict_proba"` | `train.py:190` | `train.py:257` |
| `order_by=["metrics.cv_auc_mean DESC"]` | `train.py:229` | `train.py:303` |
| `get_model_version()` 재조회 | `train.py:257` | `train.py:424` |
| `predict(features)[0][1]` | `main.py:173` | `main.py:279` |
| `./mlruns:${PWD}/mlruns` | `compose:38` | `compose:38` |
| `MLFLOW_SERVER_ALLOWED_HOSTS` | `compose:25` | `compose:25` |

main에서 달라진 것은 **코드가 아니라 도메인**입니다. 모델명이 `churnwatch` 하나에서
`riskwatch_credit`·`riskwatch_fraud` 둘로 갈라지면서 `MODEL_NAME` 상수가 트랙 파생 `model_name`
함수가 되었고, `/predict`의 반환이 `churn_probability`에서 `risk_probability` + 3값 `decision`으로
바뀌었습니다. 의존성 버전(mlflow 3.15.1, skops 0.14.0, lightgbm 4.7.0, scikit-learn 1.9.0,
cloudpickle 3.1.2)은 **main에서도 동일**하므로 버전·에러 메시지 서술은 그대로 유효합니다.

§6의 성능 수치(CV AUC 0.8476 등)는 Telco 데이터셋 스윕 결과라 현재 트랙에는 대응물이 없습니다.

---

## 참고 링크

### 공식 문서 (교차 검증에 사용)

- [Migrate from File Store](https://mlflow.org/docs/latest/self-hosting/migrate-from-file-store/) — 유지보수 모드, `mlflow migrate-filestore`, SQLite 타깃 제약
- [Backend Stores](https://mlflow.org/docs/latest/self-hosting/architecture/backend-store/)
- [ML Model Registry](https://mlflow.org/docs/latest/ml/model-registry/) — alias 개념, `models:/name@alias`
- [Model Registry Workflows](https://mlflow.org/docs/latest/ml/model-registry/workflow/)
- [MLflow 3 Migration Guide](https://mlflow.org/docs/latest/ml/mlflow-3/) — `artifact_path` → `name`, LoggedModel
- [Breaking Changes in MLflow 3.0](https://mlflow.org/docs/latest/ml/mlflow-3/breaking-changes)
- [mlflow.sklearn API reference](https://mlflow.org/docs/latest/api_reference/python_api/mlflow.sklearn.html) — `serialization_format='skops'` / `pyfunc_predict_fn='predict'` 기본값
- [Pickle-free models](https://mlflow.org/docs/latest/ml/tracking/pickle-free-models/) — skops 전환 근거
- [Protect Your Tracking Server from Network Exposure](https://mlflow.org/docs/latest/self-hosting/security/network/) — 3.5.0+ DNS 리바인딩 보호, `--allowed-hosts`
- [skops — Secure persistence](https://skops.readthedocs.io/en/stable/persistence.html) — `get_untrusted_types()`
- [scikit-learn — Cross-validation](https://scikit-learn.org/stable/modules/cross_validation.html) — 테스트셋 과적합

### 커뮤니티 · 이슈 트래커

- [mlflow#18534 — NOTICE: Filesystem backend deprecation](https://github.com/mlflow/mlflow/issues/18534) — 폐기 타임라인(3.6 경고 / 3.7 sqlite 기본)과 근거
- [mlflow#15289 — Artifacts uri is impossible to set correctly in local mode](https://github.com/mlflow/mlflow/issues/15289) — 로컬 아티팩트 절대 경로 문제, 미해결
- [mlflow#694 — Proposal: add predict_proba to Model API](https://github.com/mlflow/mlflow/issues/694) — 커스텀 pyfunc 래퍼 패턴의 출처
- [mlflow#22095 — DNS rebinding protection rejects requests when Host header includes port](https://github.com/mlflow/mlflow/issues/22095)
- [mlflow#21460 — Model serialization with skops results in various warnings](https://github.com/mlflow/mlflow/issues/21460)
- [Databricks Community — How to get probability score for each prediction from mlflow](https://community.databricks.com/t5/data-engineering/how-to-get-probability-score-for-each-prediction-from-mlflow/td-p/35152)
- [Medium — Building Custom ML Models with MLflow](https://medium.com/@luanhcss/building-custom-ml-models-with-mlflow-1e191d4bce70) — 커스텀 pyfunc 래퍼 예제

---

*코드는 모두 이 레포의 `21c43bf`에서 실제로 동작하던 코드입니다. 버전·에러·수치는 mlflow 3.15.1
환경에서 실행해 확인했습니다.*

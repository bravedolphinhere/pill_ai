# 알약 데이터 라벨 QA 설계 제안

작성일: 2026-11-05. 상태: **검토용 제안, 구현·전체 실행 전**.

이 문서는 전체 QA 시스템의 설계 제안이다. 이후 사용자 승인으로 진행한
내부 YOLO11n 베이스라인의 제한된 QA·데이터 준비 결과는
[YOLO11N_PREFLIGHT.md](YOLO11N_PREFLIGHT.md)에 따로 기록했다.
아래 전체 시스템·수동 검수 UI·공식 제품 대조가 구현 완료됐다는 의미는 아니다.

이번 작업은 기존 노트북·CSV·코드 검토와 규칙·출력 설계에 한정한다.
원본 데이터 재탐색, ZIP 전체 디코딩, OCR·특징 모델 실행, 공식 제품 정보 일괄 수집,
자동 라벨 수정, 학습 데이터 생성·삭제는 하지 않았다.
아래 임계값·표본 수·승인 정책은 사용자 검토 후 확정할 초안이다.

## 1. 확인한 EDA와 재사용 범위

현재 노트북은 `C:/pill_ai/outputs/pill_ai/`에 있다.
CSV와 분석 코드는 각각 `../internal_eda_20261104/`, `../external_eda_20261105/`에 있다.
노트북의 `Path.cwd()/tables` 셀은 이동한 위치에서 그대로 실행하면 경로가 맞지 않을 수 있다.
QA는 노트북 실행 위치에 의존하지 않고 설정 파일의 EDA 결과 루트를 사용한다.

| 관측 항목 | 내부 EDA | 외부 EDA | QA로 가져올 의미 |
|---|---:|---:|---|
| 학습 이미지 / 외부 검증 이미지 | 학습 232장 | 학습 10,509장 / 검증 1,500장 | 공급자 split을 보존, QA 표본과 최종 test 역할은 별도 |
| 라벨 행 / 제품 | 763 / 56 | 45,933 / 116 | 라벨 행 수와 유효 bbox 수를 구분 |
| bbox 결측·형식 문제 | 집계상 0 | 163 | 외부 163은 빈 리스트뿐 아니라 형식·수치 문제를 포함 |
| 화면 밖 bbox | 1 | 61 | 복구 가능한 문제이므로 기본 REVIEW |
| 면적 계산 불일치 | 6 | 458 | 외부 집계 `area_mismatch=623`과 구분 |
| bbox/area 결측 포함 면적 검사 경고 | 6 | 623 | 세부 원인을 합쳐 하나의 오류 수로 발표하지 않음 |
| 같은 위치의 다른 라벨 | 3쌍 | 겹침 후보 71쌍 | 쌍 수이지 이미지 수·확정 오라벨 수가 아님 |
| 파일명 기대 제품의 부분 연결 누락 | 8건 | 29건 | 실제 알약 수·정체의 확정 근거가 아님 |
| 개별 사진에 JSON 연결 없음 | 0 | 12장 | 참고 이미지와 실제 사진을 먼저 구분 |
| JSON 파일명과 참조 이미지명 불일치 | 별도 규칙 추가 필요 | 24행 | 파일명으로 자동 연결·수정하지 않음 |
| `_index.png` 참고 이미지 | 해당 없음 | 4,003장 | 사진 안 제품명·코드 때문에 학습 대상에서 분리 |
| 외부 JSON 파싱 실패 | 해당 없음 | 1개 | 깨진 JSON을 복구 불가능하다고 바로 단정하지 않음 |

근거: 양쪽 `summary.json`, `annotations.csv`, 내부 `overlapping_boxes.csv`,
외부 `overlap_candidates.csv`, `filename_json_stem_conflicts.csv`, `json_errors.csv`.
위 수치는 **기존 EDA 시점의 관측**이며 현재 원본 전수 검사를 다시 수행한 결과가 아니다.
한 행·이미지가 여러 규칙에 걸리므로 각 행의 수를 더해 고유 문제 이미지 수를 만들지 않는다.

### 기존 구현을 그대로 QA 판정기로 사용하면 안 되는 부분

- 외부 `category_id=1`, `categories.name=Drug`는 제품 분류 번호가 아니다.
  `drug_N`, `dl_mapping_code`, `dl_idx`, `item_seq`를 별도 필드로 보존한다.
  내부와 공통인 제품 코드 54개에서 `dl_idx`는 모두 달랐고, 품목코드 충돌은 관측되지 않았다.
  외부 번호 차이는 대부분 1이지만 예외 450행도 있다. 일괄 `+1` 매핑은 금지한다.
- 외부 문서의 `annotation_id=1`, `image_id=1` 반복은 로컬 ID 구조다.
  내부에서 관측된 전역 ID 유일성을 외부에 강제하지 않는다.
- 알약별 JSON을 **사진 단위로 먼저 합친 뒤** 중복·개수·클래스 충돌을 검사한다.
  사진 안 다른 알약의 JSON끼리 클래스가 다르다는 이유로 충돌 처리하지 않는다.
- 외부 실제 사진 디코딩은 133장, 참고 이미지 디코딩은 26장이다.
  나머지 사진에 대해 디코딩·실제 크기·픽셀 기반 규칙 통과를 주장할 수 없다.
- CSV 저장 과정에서 원본 null과 빈 문자열이 같은 빈 칸이 될 수 있다.
  `missing_by_field.csv`는 집계 근거로 쓰되, 개별 행의 결측 상태 복원이 필요하면
  승인 후 **해당 JSON만** 읽어 `raw_value`, `value_state`를 기록한다.
- 외부 `visual_review_notes.csv`의 GB/Airtal/SK는 이전 시각 관측 메모다.
  사용자 승인·공식 제품 정체 확인으로 승격하지 않고 `legacy_visual_observation`으로 가져온다.
  메타데이터가 클래스 안에서 일정하다는 사실은 실제 사진의 클래스가 맞다는 증거가 아니다.

### 재사용할 파일·함수

| 결과·코드 | 재사용 방법 | 새로 필요한 부분 |
|---|---|---|
| 양쪽 `annotations.csv`, 이미지 inventory, manifest | 정규화된 입력·출처·기존 finding을 읽는 어댑터 | 공통 스키마, 명시적 coverage, snapshot 유효성 검사 |
| 내부 `bbox_tables`, `overlap`, 외부 `label_diagnostics` | 좌표·IoU 수식과 기존 결과 재사용 | 실제 이미지 크기 검사 여부와 규칙 실행 여부 분리 |
| `missing_by_field`, `missing_by_class` / `missing_by_product` | 결측 패턴·삭제 영향 설명 | null/빈 값의 개별 행 상태 보존 |
| `filename_missing_candidates`, `images_without_any_label` | 연결 누락 후보 큐 | 원본 사진 전체의 미박스 물체 수동 검사 |
| `overlapping_boxes`, `overlap_candidates` | 기존 쌍 finding 가져오기 | 같은 클래스/다른 클래스, 포함·부분 겹침의 구분 |
| `internal_external_product_mapping`, `code_dlidx_offsets` | 원본 번호 체계 대조 근거 | 검수된 버전별 class map, 공식 정보와 별도 연결 |
| `image_inventory` / `image_decode_sample`, 해시·CRC 표 | 이미 계산한 값 재사용 | 미실행 사진만 추가 처리, CRC는 후보로만 사용 |
| `gallery_manifest`, 기존 figures, 한글·bbox 렌더링 함수 | 이미 만든 비교 이미지·source 연결 활용 | 전체 장면, 선택 bbox 확대, 검수 CSV 저장 UI |

기존 `run_analysis()`/`main()`은 전체 분석을 재실행하므로 QA 입력 로더에서 호출하지 않는다.
캐시 키는 입력 fingerprint + 규칙·설정·코드·모델 버전이다.
입력이 변하지 않고 해당 규칙 결과가 호환될 때만 재사용한다.
현재 snapshot 확인이 부족하면 `cached_unverified`로 표시하고 PASS에 사용하지 않는다.
ZIP member의 CRC·크기는 전체 바이트 SHA256과 구분하고, 읽지 않은 바이트의 해시를 만들지 않는다.

## 2. 판정과 학습 자격을 분리

각 이미지와 실제 어노테이션 레코드에 `auto_status`를 부여한다.

| 상태 | 의미 | 기본 처리 |
|---|---|---|
| PASS | 정의된 필수 자동 검사가 실행되어 통과한 학습 후보 | B 무작위 감사 모집단, 의미상 정확성 보증 아님 |
| REVIEW | 충돌·결측·검사 불가·필수 검사 미실행·복구 가능 오류 | 논리적 격리, 검수 큐 |
| REJECT | 객관적인 오류와 복구 불가능성까지 확인 | 제외 manifest에 기록, 원본은 그대로 보존 |

**기본 설정은 자동 REJECT 비활성화**다. bbox 초과, JSON 파싱 실패, OCR 불일치,
다른 제품처럼 보이는 사진, 일시적인 드라이브 읽기 실패는 REVIEW다.
REJECT에는 `confirmed_error`, `unrecoverable`, 원본 재취득·복구 시도 결과,
판정 근거, 검수자와 시각이 모두 필요하다. 제외 선택 자체만으로 REJECT가 되지 않는다.
정책상 제외한 참고 이미지·중복·보류 데이터는 복구 불가능한 오류와 구분한다.

- `rule_result`: `PASS / FLAG / NOT_APPLICABLE / NOT_RUN / ERROR`.
  필수 규칙의 NOT_RUN/ERROR는 해당 대상의 REVIEW로 집계한다.
- 필수 범위: 문서·미디어 연결, 디코딩/실제 크기, bbox 형식·범위,
  사진 단위 중복·충돌·개수 이상, 제품 번호 체계의 연결, 역할·보호 split 확인.
- 선택적 OCR·외형 모델 규칙은 초기 비활성화 가능하다. 미실행을 통과로 기록하지 않고
  `semantic_status=UNVERIFIED`, `checks_not_run`에 남긴다.
  결과가 생겨 충돌하면 REVIEW다. 의미 검증 필수 정책을 선택하면 정보 부족도 REVIEW다.
- `human_decision`: `PENDING / APPROVE / NEEDS_CORRECTION / EXCLUDE / DEFER`.
  자동 결과는 불변이며 사람의 결정으로 덮어쓰지 않는다.
- `effective_status`: 승인된 사람 결과와 후속 재검증을 반영한 파생 상태.
  APPROVE에는 검사 종류별 확인 범위를 기록하며 제품 정체의 확인으로 자동 해석하지 않는다.
- `semantic_status`: `UNVERIFIED / CONFLICT / CORROBORATED / UNRESOLVED`.
  독립 제품 정보의 어떤 속성까지 일치했는지를 함께 기록한다.
- `train_eligibility`: `CANDIDATE / HOLD / INELIGIBLE / APPROVED`.
  PASS만으로 바로 APPROVED가 되지 않는다. 기본적으로 사람 승인 또는 승인된 B 감사의
  모집단 수용 조건을 만족해야 하며 별도 학습 manifest 발행도 사용자 승인 후 진행한다.

이미지 수준의 REVIEW는 그 사진의 일부 bbox만 떼어 학습하는 것까지 기본 보류한다.
누락 물체·클래스 충돌은 다른 bbox가 정상이어도 배경 정답에 영향을 준다.
어노테이션 자체 상태와 이미지에서 상속된 `context_hold`를 함께 기록한다.
한 어노테이션의 REJECT를 다른 정상 어노테이션의 오류로 복사하지는 않는다.

`_index.png`와 OS 부가 파일은 `asset_role=reference/os_metadata`, `train_eligibility=INELIGIBLE`로 분리한다.
역할 미확정은 REVIEW다. 잘못된 파일이라는 이유로 REJECT를 붙이지 않는다.
내부의 정답 없는 test 이미지 842장은 `labels_expected=false`인 별도 보호 역할로 둔다.
라벨 부재만으로 오류를 만들거나 학습 후보로 편입하지 않는다.

## 3. 자동 검증 규칙 초안

모든 새 임계값은 **초기 후보 탐색값**이다. 사용자 검토·소규모 교정 전 확정 기준이 아니다.
픽셀에서 관측한 특징, 원본이 주장하는 정보, 공식 참고 정보는 각각 다른 테이블에 저장한다.

| ID | 검사와 근거 | 초기 기준·한계 | FLAG 시 |
|---|---|---|---|
| LINK-01 | 미디어·JSON·로컬 image/category 참조 | 이미지 경로 유일성, JSON stem/참조명, source/split 일치 | 이미지와 연관 라벨 REVIEW |
| IO-01 | JSON 파싱·이미지 디코딩·실제 크기 | 실제 읽은 바이트의 해시·decoder 버전 기록. 일시적 클라우드 실패 분리 | REVIEW, 재시도·원본 재취득 검토 |
| BOX-01 | bbox 형식·유한성·양수 | 숫자 4개 `[x,y,w,h]`, bool 제외, w/h>0. 빈 리스트와 잘못된 타입 분리 | 해당 라벨·사진 REVIEW |
| BOX-02 | bbox 실제 화면 범위 | 원본 좌표에서 x/y>=0, x+w<=W, y+h<=H. 부동소수 비교 오차 1e-6px만 허용 | REVIEW, 자동 clamp 금지 |
| BOX-03 | area의 이용 가능성·계산 일치 | area 결측/비수치, bbox 결측, 순수 w×h 불일치를 따로 기록 | REVIEW, bbox 검수 전 area 덮어쓰기 금지 |
| PAIR-01 | 같은 클래스의 중복 bbox | 같은 사진을 합친 뒤 IoU>=0.95 후보. 동일 물체인지 확인 필요 | 쌍과 이미지 REVIEW |
| PAIR-02 | 같은 위치의 서로 다른 클래스 | 같은 사진에서 서로 다른 product key, IoU>=0.95. 기존 3쌍/71쌍 재사용 | 양쪽 라벨·사진 REVIEW |
| PAIR-03 | 비정상 겹침·포함 | IoU>=0.8 또는 작은 bbox의 교집합 비율>=0.9를 초기 후보로 사용 | REVIEW, 실제 중첩·가림 가능 |
| COUNT-01 | 사진 내 bbox 수 이상 | 같은 촬영 조합의 다른 사진과 비교. 기존 scene 표·파일명 기대 코드 함께 표시 | REVIEW, 실제 물체 수 확정 금지 |
| COUNT-02 | 미박스 알약 가능성 | 전체 장면 수동 검사, 선택적 전경 분리/물체 제안 모델의 미할당 영역 | REVIEW, 모델 제안은 정답·확정 개수가 아님 |
| CLASS-01 | 클래스·제품 코드 연결 | source별 class map, drug_N/매핑코드/품목코드/이름 충돌. 로컬 category ID 구분 | REVIEW, 번호 +1·이름 유사도 자동 수정 금지 |
| APP-01 | 같은 코드의 속성 메타데이터 충돌 | 기존 catalog conflict 표 재사용. 결측은 불일치로 바꾸지 않음 | REVIEW, 사진 의미 검증과 별개 |
| APP-02 | 같은 코드의 실제 외형·각인 차이 | bbox crop의 색·모양·OCR·특징 거리 후보, 촬영조건·회전·면·crop 오염 고려 | REVIEW, 예측 제품명은 정답으로 채택 금지 |
| DUP-01 | 동일 바이트/동일 픽셀 사진의 라벨 차이 | SHA256 우선. 픽셀 해시는 명시한 디코딩·orientation 정규화 버전과 함께 기록 | 쌍·군집 REVIEW |
| DUP-02 | 매우 유사한 사진의 라벨 차이 | 64bit pHash 해밍거리<=6을 탐색 초안으로 사용, 재정렬·등록 검증 필요 | REVIEW, 유사 해시만으로 동일 사진 확정 금지 |
| SPLIT-01 | 촬영·중복 군집의 split 충돌 | 원래 split과 최종 역할 보존, 연결 군집·보호 test와 중복 확인 | HOLD, 자동 split 이동 금지 |
| REF-01 | 독립 제품 정보와 관측 특징 대조 | 해당 제품·시점·속성이 제공되는지 먼저 확인. 보이는 면의 관측만 대조 | 충돌 REVIEW, 미제공은 UNVERIFIED |

겹침은 유효 bbox 쌍에서만 계산한다. 제품 번호의 연결이 불확실하면 PAIR-02의 클래스 판정도
`mapping_unresolved`로 기록한다. 비정상적인 bbox에 crop 기반 외형 모델을 적용해
그 결과를 제품 오라벨 근거로 삼지 않는다.

APP-02는 raw crop와 마스크를 보존하고 촬영 조명·포즈 변화와 다른 물체 혼입을 함께 보여 준다.
색 거리·embedding 임계값은 현 단계에서 정하지 않는다. 독립 검수된 교정 표본에서 결정한다.
같은 코드의 다수결·첫 사진·medoid도 검증된 정답처럼 취급하지 않는다.
앞·뒷면 허용 특징은 대조하되 `drug_dir=앞면`을 실제 카탈로그 앞면 보증으로 쓰지 않는다.
OCR의 `not_observed / unreadable / occluded / observed_text`를 구분한다.
안 보이는 뒷면은 결측 관측이며 무각인·불일치가 아니다.

DUP-01/02는 알약별 JSON이 아니라 **합쳐진 사진 전체의 라벨 집합**을 비교한다.
같은 전체 사진에서 bbox를 1:1로 대응시킨 후 코드·좌표·누락 차이를 계산한다.
회전·크기·크롭이 다르면 등록 성공과 대응 신뢰도가 있을 때만 비교하고,
정렬 실패는 해결되지 않은 후보로 남긴다. CRC32+크기는 후보 탐색용이다.
현재 외부의 CRC 후보 0개는 전수 SHA256·근접 중복 검사 완료를 뜻하지 않는다.

## 4. 세 정보의 대조와 독립 제품 정보

| 층 | 저장할 내용 | 해석 |
|---|---|---|
| annotation claim | 원본 코드·품목코드·약 이름·bbox·각인 설명 | 데이터셋의 주장, 수정하지 않음 |
| image observation | 관측 색·모양·각인·분할선·가시성, 관측자/도구·버전·confidence | 사진에서 보이는 것, 제품 정체와 구분 |
| independent reference | 별도로 조회한 품목코드·업체·함량·제형·식별 특징·유효시점·출처 | 일치하는 제품인지 먼저 검증할 대조 정보 |

식약처 [의약품안전나라 제품 검색](https://nedrug.mfds.go.kr/searchDrug)과
[식약처 의약품 낱알식별 정보 서비스](https://www.data.go.kr/data/15057639/openapi.do)를 대조 출처 후보로 한다.
2026-11-05 공식 [서비스 메타데이터](https://www.data.go.kr/catalog/15057639/openapi.json)에서
품목기준코드·품목명·업체·모양·색을 제공하는 서비스임을 확인했다.
**이 프로젝트의 각 제품에 앞뒤 각인·시점별 식별 정보까지 모두 제공된다는 확인은 아직 하지 않았다.**
API 호출·인증키 발급·제품별 일괄 조회는 이번에 실행하지 않았다.

공식 코드와 데이터셋의 K-코드를 동일 식별자로 가정하지 않는다.
우선 `item_seq` 연결을 검토하고 제품명·업체·함량·제형으로 교차 확인한다.
동일 이름 검색의 유일 결과만으로 자동 확정하지 않는다.
`reference_status`는 `NOT_REQUESTED / AVAILABLE / PARTIAL / NOT_FOUND / AMBIGUOUS / DATE_MISMATCH`로 구분한다.
조회 실패·제품 미등재·속성 미제공은 제품 오라벨의 증거가 아니다.
공식 자료가 원본 라벨과 같은 상위 카탈로그를 공유할 수 있으므로 출처 계보도 남긴다.
원본 라벨의 `img_key`를 다시 열었다는 이유만으로 독립 검증이 됐다고 하지 않는다.

`reference_id`, URL/API 버전, 조회시각, 원문 응답 해시·보관 위치, 품목키,
대상 속성의 제공 여부, 식별정보 변경 시점, 제조사·제형·함량 매칭 근거를 기록한다.
사진 촬영 시점이 불명확하거나 과거 식별표시와 현재 자료가 충돌하면 REVIEW다.
세 층의 **관측 가능한 속성**이 일치할 때 CORROBORATED로 기록하되
사진 한 장으로 뒷면·제품 정체 전체를 확정했다고 표현하지 않는다.

## 5. A 의심 사례와 B PASS 감사

### A: 의심 사례

REVIEW 전부와 REJECT 판단 근거를 가능한 한 전수 검수한다.
동일 finding·사진은 큐에서 합치되 관련 어노테이션·쌍의 연결은 보존한다.
현재 검수 못 한 건수·기한·원인을 기록하고, 미완료 건을 승인으로 간주하지 않는다.
원본 연결 실패·bbox 결측·클래스 충돌·미라벨 물체 가능성을 우선한다.

### B: 자동 PASS에서 확률 표본 추출

모집단 snapshot을 고정한 뒤 `source_family × supplied_split × source_product_code`별로 추출한다.
내부·외부, 공급자 train/valid를 보존하고 모든 source archive의 표본 커버리지도 점검한다.
class map 검수가 끝나기 전에는 번호가 비슷하다는 이유로 제품 층을 합치지 않는다.

- **파일럿 초안:** 각 비어 있지 않은 층에서 `min(N, max(10, ceil(0.05*N)))`개 어노테이션을
  단순무작위 비복원 추출한다. 작은 층은 전수, seed42·알고리즘·모집단 해시를 기록한다.
  규모와 검수 시간을 먼저 산출해 승인받고 실행한다.
- 같은 사진이 여러 제품 층에서 뽑히면 사진은 한 번 열되 `drawn_annotation_ids`는 모두 기록한다.
  전체 장면에서 미라벨 물체도 확인한다.
- 선택 어노테이션 외에서 발견한 문제는 `incidental_discovery`로 기록한다.
  이를 확률 표본의 분모·분자에 섞어 오류율을 계산하지 않는다.
- 제품 층별 추정률과 표본 크기를 제공하고, 전체 어노테이션 오류율은 층의 모집단 크기로 가중한다.
  포함 확률, 추출 확률의 정의, 사진 단위 중복 제거 방법을 보존한다.
- **이미지 수준의 누락/불량률**은 별도 source/split별 이미지 또는 촬영 그룹 무작위 표본으로 측정한다.
  여러 bbox를 독립 이미지 표본처럼 세지 않는다.
  같은 촬영 조합·유사 이미지의 상관을 고려해 그룹 단위 집계/구간을 사용한다.
- 최초 추출본은 불변이다. 미검수·판정 불가를 오류 없음으로 계산하지 않고 응답률·보류율을 보고한다.
  규칙 개선 후 원래 B 사례는 회귀 검증용으로 보존하고, 새 B 확인 표본도 추출한다.

`N`, 추출 `n`, 완료 `n`, 판정 불가 수, 확인 오류 수, 가중 오류율, 신뢰구간을 함께 보고한다.
표본 10개에서 오류 0개는 충분한 보증이 아니다.
예를 들어 독립 단위의 비가중 이항 근사에서 0/149이면 단측 95% 상한은
`1 - 0.05**(1/149) ≈ 1.99%`다.
이 수를 같은 사진의 bbox 149개나 모든 클래스 각각의 보증으로 적용하지 않는다.
실제 수용 검정은 층화·그룹 구조와 유한 모집단을 반영해 설계하고 허용 오류율을 승인받는다.

### 개선·확대 검수 트리거 초안

- B에서 클래스 연결 오류나 미박스 알약이 한 건이라도 확인되면 해당 source/제품/촬영 군집의
  학습 승인을 보류하고 원인을 분석한다. 이것이 전체 데이터 오류율 확정은 아니다.
- 원인에 맞춰 규칙을 개선하고 관련 부분을 재검증한다. 영향 범위를 정할 수 없으면 더 넓게 보류한다.
- 기타 오류 허용률은 예시 2%로 두되 미확정이다. 파일럿의 관측률만 낮다고 수용하지 않는다.
  신뢰상한이 승인 기준을 만족하지 못하면 표본을 늘리거나 격리를 유지한다.
- A의 확인율은 후보의 precision 참고값이다. B의 오류율은 PASS에 남은 오류의 추정치다.
  둘을 합쳐 recall을 계산하지 않는다. 전수 정답이 없으면 전체 recall은 미확인으로 기록한다.

## 6. 검수 갤러리·선택·CSV 저장

승인 후 로컬 검수 UI를 구현한다. 1차로 기존 한글·bbox 렌더링과 figures를 재사용하고,
파일 선택·CSV 다운로드가 가능한 로컬 UI로 검수 결과를 저장한다. 이번에는 UI를 구현하지 않는다.

**가로 한 줄:** 전체 장면(모든 bbox) / 선택 bbox 확대 / 같은 제품 또는 충돌 쌍의 비교 참조.
비교 참조에는 `미검증 참조` 표시를 기본으로 붙인다.
다른 이름의 파일도 이미지 uid와 source URI로 선택해 경로 충돌을 방지한다.

화면 필수 정보:

- 이미지 파일명, source family/split/archive, 원본 이미지 경로 또는 ZIP+member.
- 제품 코드, 원본 class ID·로컬 category ID, 약 이름, 연결된 class-map 버전.
- 원본 bbox `[x,y,w,h]`, 실제/선언 이미지 크기, annotation uid, 원본 JSON 경로.
- 의심 오류 유형·규칙 ID·값·임계값·실행/캐시 여부·근거와 한계.
- 원본 주장 / 관측 특징 / 독립 정보의 세 칸. OCR·모델 출력에는 도구 추정 표시.
- 전체 사진에서 추가 알약이 보이는지, 다른 bbox도 문제인지 기록하는 별도 질문.
- A/B 구분은 저장하되 B의 최초 관측 화면에서는 필요시 규칙 점수·정답 힌트를 숨겨 편향을 줄인다.

검수자는 **승인 / 수정 필요 / 제외 / 보류**를 이미지와 어노테이션 범위별로 선택한다.
선택 시 판단 근거·확인한 속성·독립 출처와 `identity_verified` 여부를 기록한다.
수정 제안은 `proposed_bbox`·`proposed_product_code`에만 저장한다.
신규 알약 제안은 원본 annotation uid를 위조하지 않고 `proposal_uid`로 관리한다.
실제 적용 전 두 번째 검토 또는 지정 승인자 확인과 구조 재검증을 요구한다.
제외는 `exclude_reason`과 `exclusion_scope`를 기록하고 원본 삭제와 연결하지 않는다.
보류는 `unresolved_reason`, 추가 자료 필요 조건을 남긴다.

CSV는 UTF-8로 저장하며 검수 이력은 덮어쓰지 않는 이벤트 형태로 내보낸다.
빈/잘못된 uid·중복 event_id·이전 source hash와 다른 입력·잘못된 enum은 거절한다.
서로 충돌하는 검수 결과는 마지막 저장 우선으로 결정하지 않고 `adjudication_required`로 보류한다.
자유 텍스트는 CSV quoting과 스프레드시트 수식 실행 방지를 적용하되 원문 값은 따로 보존한다.
검수 도중 중단해도 미완료 선택을 승인으로 간주하지 않는다.

## 7. 제안 출력 구조

원본 폴더 밖의 설정 가능한 QA 출력 루트 아래에 실행별로 생성한다.
격리는 파일 이동이 아닌 **manifest와 학습 자격의 논리적 격리**다.
아래는 생성 예정 구조이며 이번 작업에서 생성·채우지 않았다.

```text
qa_runs/<run_id>/
  run_manifest.json                 # 코드·입력·설정·도구·참고 정보 버전과 실행 범위
  config_snapshot.yaml             # 승인된 규칙·범위·임계값·seed
  provenance/
    eda_inputs.csv                 # 재사용한 노트북/CSV/코드 해시와 원래 위치
    source_assets.csv              # dataset root ID, 상대 경로, ZIP member, 역할
    access_log.csv                 # 자동 처리/사람 열람/QA 노출/보호 split
  normalized/
    images.csv
    annotations.csv
    class_map.csv                  # 미승인 매핑도 상태를 명시
    metadata_value_states.csv      # 원문 null/빈 문자열/키 부재 구분
  automatic/
    rule_results.csv               # 실행된 규칙과 미실행 사유, 대상별 결과
    findings.csv                   # 규칙×대상×근거별 다중 finding
    image_status.csv
    annotation_status.csv
    pairs.csv                      # 충돌·중복·유사 이미지 쌍
    observations.csv               # OCR/색/모양 등, 정답과 분리
    duplicate_groups.csv
  reference/
    catalog_records.csv
    catalog_links.csv              # 제품 연결 상태·시점·커버리지
    snapshots/                     # 허용되는 공식 응답과 응답 해시
  sampling/
    population.csv                 # A/B 모집단 snapshot, source/class/역할
    sample_manifest.csv            # 추출 unit, 확률·층·seed·목적
  gallery/
    index.html                     # 전체 장면/확대/비교, 원본 좌표 보존
    assets/                        # 렌더링 그림, 원본 재저장본과 구분
  human/
    review_events.csv              # append-only 승인/수정 필요/제외/보류
    correction_proposals.csv       # 미적용 수정 제안, 독립된 버전
    adjudications.csv
    review_state.csv               # 이벤트를 합친 현재 상태
  reports/
    rule_metrics.csv
    audit_metrics.csv
    coverage.csv                   # 미실행·실패·보류·제품 정보 부재도 집계
    summary.md
  manifests/
    quarantine.csv
    excluded_by_policy.csv
    pass_candidates.csv
    qa_exposed.csv
    protected_test_registry.csv
    train_approved.csv             # 별도 승인 뒤에만 발행
```

기존 `data_process/`, `models/` 뼈대는 이동·합치지 않는다.
구현 승인 시 QA 모듈 위치를 별도로 확정한다. 코드·규칙은 Git에서 관리할 수 있지만
원본·갤러리·실제 검수 결과·공식 응답·개인 경로·인증키를 자동 스테이징하지 않는다.

### 핵심 스키마

| 파일 | 최소 필드 |
|---|---|
| `source_assets` | dataset_id, source_root_id, split, role, relative_path, archive, member, size, mtime, fingerprint_type, fingerprint, fingerprint_coverage |
| `images` | image_uid, asset_uid, source_revision, image_sha256/pixel_hash와 계산 상태, filename, declared_W/H, actual_W/H, capture_group_id |
| `annotations` | annotation_uid, source_asset_uid, json_member, record_position, local_annotation_id, image_uid, raw_class_id, product_code, item_seq, raw_name, raw_bbox, value_state |
| `rule_results` | run_id, target_uid/type, rule_id/version, result, evidence_source, threshold, measured_value, coverage, skipped_reason, computed_at |
| `findings` | finding_id, rule_id/version, target_uid, related_uids, reason_code, evidence_json, suggested_status, origin(auto/legacy_observation/human/reference), source_fingerprint |
| `image_status` / `annotation_status` | uid, auto_status, blocking_finding_ids, checks_required/executed/skipped, semantic_status, context_hold, human_decision, effective_status, train_eligibility |
| `observations` | observation_id, image/annotation_uid, feature, observed_value, visible_side/visibility, method, model/version, confidence, crop_hash, created_at |
| `review_events` | event_id, run_id, target_uid/type, reviewer_id, decision, reason, reviewed_fields, finding_ids, observation_ids, reference_ids, identity_verified, source_hash, created_at, supersedes_event_id |
| `correction_proposals` | proposal_uid, target_uid/image_uid, old_values, proposed_values, evidence_ids, proposed_by, second_review_status, applied=false |
| `sample_manifest` | sample_id, population_hash, cohort(A/B), stratum, sampling_unit, selected_uid, drawn_annotation_ids, N, n, inclusion_probability, seed, algorithm_version, purpose |
| `rule_metrics` | rule_id/version, eligible_targets, executed, flagged, skipped, errors, unique_images, annotations, pairs, human_confirmed, human_unresolved |
| `audit_metrics` | source/class/stratum, population_N, drawn_n, reviewed_n, unresolved_n, confirmed_error_n, weighted_rate, confidence_method/bounds, auxiliary_discoveries |

파일명만 사용하지 않고 루트 ID와 상대 경로를 사용한다.
논리 ID와 source revision을 구분하고 경로 변경·파일 갱신 이력을 보존한다.
annotation_uid는 dataset 이름공간·원본 JSON URI·로컬 ID·행 위치로 구성해
같은 로컬 ID를 가진 외부 JSON을 잘못 합치지 않는다.
수정 제안은 원본 uid를 보존한 별도 revision으로 관리하고 신규 제안과 원래 라벨을 구분한다.

## 8. 재현성·평가 데이터 보호

`run_manifest.json`에는 QA 코드 Git commit, dirty 여부와 실제 코드 snapshot SHA256,
기존 EDA 코드·CSV 해시, 입력 manifest, 규칙·설정 해시, Python/패키지·decoder 버전,
OCR/특징 모델의 이름·버전·가중치 해시, seed, 조회 자료 버전, 시작·종료시각과 처리 범위를 기록한다.
실제 code commit이 없으면 임의 버전을 만들지 않고 `uncommitted`와 코드 해시를 남긴다.
규칙 수정은 새 rule version·run으로 남기고 사람의 과거 검수 이력을 보존한다.
전체 원본을 다시 읽지 않고 변경·영향받은 규칙·자산만 재계산할 수 있게 한다.

**QA 갤러리에서 사람이 본 자료는 최종 독립 test와 분리**한다.
현재 EDA·기존 갤러리에서 본 자료도 `qa_exposed`로 등록한다.
자동 픽셀 처리와 사람 열람은 서로 다른 access type으로 기록한다.
최종 test는 추가 검수·규칙 튜닝 전에 보호 registry로 고정하고 일반 QA 큐·갤러리에서 차단한다.
중복·촬영 군집에 보호 test가 있으면 학습/튜닝 편입을 보류한다.
자동 PASS 감사 B도 개발용 QA 자료이며 독립 test 성능을 대체하지 않는다.
공급자 valid를 이미 보고 규칙을 바꿨다면 더 이상 손대지 않은 최종 test로 표현하지 않는다.
최종 test의 별도 품질 검수가 필요하면 모델·규칙 개발자와 분리된 절차로 사전 정의한다.

## 9. 검토 후 진행할 순서

1. 본 설계의 상태·규칙·표본 예산·공식 정보 연결 범위·보호 test 정책을 검토한다.
2. 승인 후 기존 EDA 결과의 어댑터와 스키마 검증부터 구현한다. 기존 분석 전체는 실행하지 않는다.
3. 이미 알려진 문제와 정상 참조, 근거가 불충분한 사례를 포함하는 소규모 교정 묶음으로
   규칙과 UI·CSV 왕복 저장·상태 전이를 검증한다. 이 묶음의 정상 참조도 독립 검증 여부를 명시한다.
4. 규칙·coverage가 타당하면 변경되지 않은 기존 결과를 재사용하고 승인된 추가 검사만 실행한다.
5. A 전수 검수와 B 확률 감사를 수행하고, 확인율·PASS 잔여 오류율·보류율·검사 커버리지를 검토한다.
6. 규칙 개선·새 B 확인 감사와 별도 승인 후에만 학습 승인 manifest를 발행한다.

수용 조건에는 필수 검사 미실행의 PASS 오분류 0건, 원본 쓰기 0건,
잘못된 uid/상태/출처 hash의 CSV 반입 거절, 검수 상태 재개 가능,
현재 EDA seed 사례의 finding 회귀 검증, 보호 test 갤러리 노출 0건을 포함한다.
새 통계·vision 규칙의 성능은 확인된 검수 결과로 평가하며 미확인 정상 참조를 정답으로 삼지 않는다.

## 부록: 이번에 읽은 노트북·분석 코드의 SHA256

경로는 앞의 EDA 위치를 기준으로 한다. 전체 데이터의 현재 해시를 다시 계산한 것은 아니다.

| 파일 | SHA256 |
|---|---|
| `pill_ai/01_internal_data_eda.ipynb` | `34ABDA2CAB82C94C139D906A37DDF4EBED3C17AB0A33D57C51722F2A562BD0EE` |
| `pill_ai/02_external_data_eda.ipynb` | `E0217EDF241F92B6991905C26266B248B27156ADC4BD3830C3AC9EA8E25F76FC` |
| `internal_eda_20261104/analysis.py` | `889C1D785886A52BA834B529AE65B3915999F881C158B75E4634A4F4DF39FCD8` |
| `external_eda_20261105/analyze_external.py` | `95DB90894ECE0E87842E19C1FBF4A8457FFF869B5F913F3154B8ADBC65F65F85` |

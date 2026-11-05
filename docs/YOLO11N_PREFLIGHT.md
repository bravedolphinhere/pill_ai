# YOLO11n 환경·데이터 구조 사전 검사

기준일: 2026-11-05 KST. 제출 마감: 당일 17:00.
상태: **사용자 승인 후 내부 QA·YOLO 변환·20epoch 학습·best.pt 평가 완료**.
실제 결과는 [최종 보고서](FINAL_REPORT.md)에 기록했다.

최신 데이터셋 위치: `C:/pill_ai/datasets/internal_yolo11n_8020_20261105`.
기존 `internal_yolo11n_20261105`의65:35 분할도 덮어쓰지 않고 보존했다.

| 실제 준비 결과 | 이미지 | 라벨 행 | 포함 클래스 |
|---|---:|---:|---:|
| 원본 학습 후보 | 232 | 763 | 56 |
| PASS | 197 | 660 | 56 |
| REVIEW (학습 후보에서 제외) | 35 | 103 | 클래스 전체 손실 없음 |
| train | 158 | 528 | 56 |
| validation | 39 | 132 | 29 |

실제 최신 분할은 **80.20:19.80**이다. 사용자 요청에 따라 기존 그룹을 그대로 고정하고
train의56클래스를 보존하는 조건에서 사진 수의80:20 근접도를 가장 먼저 최적화했다.
197장의20%는39.4장이므로, 가능한 정수 사진 수 중39장이 가장 가깝다.
같은 최단 비율 안에서는 검증 클래스 수를 최대화하고 seed42로 동률을 결정했다.
기존의검증29클래스도 유지했다. 기존73개 PASS 그룹의 멤버십·QA 판정·라벨 내용은 변경하지 않았다.
검증에 없는27클래스의 성능은 이 분할로 평가할 수 없다.
`qa/class_split_support.csv`에 클래스별 표본 수를 기록했다.
SciPy MILP의 비율·검증 클래스 범위·동률 결정 단계 모두 최적해 종료(status0)를 기록했다.
이 최적성은 고정된 현재 그룹·학습 클래스 보존 조건에 한정되며 다른 그룹 기준까지 포함하지 않는다.

## 1. 실제 확인한 환경

| 항목 | 확인 결과 |
|---|---|
| OS | Windows 11, VS Code 작업 공간 |
| GPU | NVIDIA GeForce RTX 5070, 표시 VRAM 12,227MiB |
| NVIDIA driver | 610.88 |
| Python 정상 설치 | 3.14.2 |
| PyTorch / torchvision | 2.14.0+cu130 / 0.29.0+cu130 |
| PyTorch CUDA runtime | 13.0, sm_120 빌드 포함 |
| 실제 GPU 테스트 | 128×128 행렬 곱 통과, torchvision CUDA NMS 통과 |
| Ultralytics | 8.4.193, import 통과 |
| C 드라이브 여유 | 약 506.8GiB (930.4GiB 중, 검사 시점) |
| Python 명령 | 현재 쉘의 `python` 직접 호출은 없음. `uv ... --python 3.14` 사용 |

별도로 존재하는 프로젝트 `.venv`는 Python 3.12.14,
PyTorch 2.14.1+cu126 / torchvision 0.29.1+cu126 / Ultralytics 8.4.182이다.
여기서는 `torch.cuda.is_available()`가 True여도 sm_120 커널이 없어
실제 GPU 연산이 `no kernel image is available for execution on the device`로 실패했다.
따라서 Python 버전만 낮춘다고 해결되지 않는다.
정상 동작한 Python 3.14/cu130을 우선 사용하고, 프로젝트 공통 `.venv`를 조용히 덮어쓰지 않는다.
별도 가상환경을 구성할 경우 Python 3.12 또는 3.14와 **sm_120을 실제 지원하는 PyTorch 빌드**를 조합해
행렬·NMS 연산을 다시 확인해야 한다. 기존 PyTorch/CUDA 환경은 교체하지 않았다.
분할 계산용 SciPy와 평가 JSON용 faster-coco-eval1.8.0을 재현 의존성에 기록했다.

드라이버가 표시하는 CUDA 지원 버전, PyTorch의 CUDA runtime, 개별 GPU 아키텍처 지원은 구분한다.
관련 공식 근거: [PyTorch Blackwell 지원 안내](https://pytorch.org/blog/pytorch-2-7/),
[NVIDIA CUDA 12.8 아키텍처 지원](https://docs.nvidia.com/cuda/archive/12.8.0/cuda-features-archive/index.html).
위 버전 표는 공식 문서의 권장값이 아니라 **이 PC의 실제 설치·실행 결과**다.

## 2. 실제 데이터 위치와 구조

내부 원본: `C:/pill_ai/datasets/internal_raw`.

```text
internal_raw/
  train_images/        # EDA 기준 232장
  train_annotations/   # EDA 기준 알약별 JSON 763개, 56클래스
  test_images/         # EDA 기준 842장, 평가 정답 없음
```

현재 원본 루트에 YOLO의 `images/labels/data.yaml` 구조는 없다.
다른 YOLO26m 실험 폴더의 데이터·분할은 이번 YOLO11n 결과로 재사용·덮어쓰기하지 않았다.
외부 Google Drive 데이터는 이번 베이스라인의 학습 입력에 포함하지 않는다.

JSON 샘플에서 `images`·`annotations`·`categories` 연결과 픽셀 `[x,y,w,h]` 형식을 확인했다.
사진 한 장의 알약이 여러 JSON에 나뉘므로 파일명으로 먼저 사진 단위로 합쳐야 한다.
변환식은 `(x+w/2)/W`, `(y+h/2)/H`, `w/W`, `h/H`이고 class ID는 0부터 연속 매핑한다.
원본 class ID·제품 코드·약 이름은 `class_map.csv`에 보존한다.

현재 EDA 노트북 원본은 `C:/pill_ai/outputs/pill_ai/`에서 찾았다.
CSV는 각각 `C:/pill_ai/outputs/internal_eda_20261104`,
`C:/pill_ai/outputs/external_eda_20261105`에 남아 있다.
GitHub용 사본은 이 저장소의 `notebooks/`에 별도로 준비했다.

## 3. 기존 EDA에서 이미 알려진 위험

- 내부 화면 밖 bbox 1개, area 불일치 6개, 동일 위치의 다른 클래스 라벨 3쌍.
- 파일명 기대 클래스의 JSON 연결 누락 후보 8개. 실제 알약 수를 파일명만으로 확정하지 않는다.
- 과거 검토 목록에 몸통 잘림·빈 배경 박스·클래스 연결 오류·각인 판정 보류도 있다.
  이전 메모는 독립적인 약품 정체 확정이나 사용자 승인으로 간주하지 않는다.
- 카탈로그 null만으로 행을 지우면 특정 클래스 전체가 사라진다.
  CNN 입력에 필요 없는 설명 결측은 사진 제외 사유로 자동 사용하지 않는다.
- SHA256 중복 없음은 유사 이미지 없음의 증거가 아니다.
  작성한 준비 코드는 촬영 조합·정확 해시·pHash 거리 후보를 같은 분할 그룹으로 묶는다.
  pHash 6비트 이내는 보수적 그룹화 초안이며 동일 물체나 오라벨의 확정 기준이 아니다.
- raw test는 정답이 없어 mAP·precision·recall을 계산할 수 없다.
  개발용 validation을 독립적인 최종 test 성능으로 보고하지 않는다.

이후 사용자의 실행 승인에 따라 내부 사진232장과 JSON763개를 검사했다.
모든 원본 지문이 EDA 시점 및 준비 후 검증에서 일치했다.
클래스 충돌·누락·면적 오류와 과거 보류 메모·새 유사 라벨 충돌을 합쳐 고유35장을 REVIEW로 보류했다.
자동 REJECT와 원본 수정은 하지 않았다.

## 4. 준비한 코드와 검증 수준

| 파일 | 목적 | 현재 확인 범위 |
|---|---|---|
| `data_process/prepare_internal_baseline.py` | 원본 지문 대조·구조 QA·REVIEW 격리·유사 그룹 분할·YOLO 변환 | 가상4장 및 실제 내부232장 처리 |
| `data_process/verify_internal_dataset.py` | 원본/복사 지문·전체 라벨 좌표 역변환·분할 누수 대조 | 원본 사진232장·JSON763개, 복사197장·라벨660개 검증 통과 |
| `data_process/group_split.py` | 그룹 고정·train 클래스 보존·사진 비율 우선 정수 최적화 | 작은 그룹의 전수 조합과 최적값 대조, seed42 결과 재현 |
| `data_process/resplit_internal_baseline.py` | 기존 QA·그룹·라벨을 바꾸지 않고 새 폴더로 재분할 | 197장 그룹 소속·라벨 내용·클래스맵 보존 확인 |
| `models/train.py` | YOLO11n 사전학습, 기본20epoch/640/batch8/seed42, OOM 시 배치 축소 | 실제20epoch 완료, OOM 없이 batch8 유지 |
| `models/evaluate.py` | 학습된 best.pt의 validation 지표·plots·예측 예시 | 실제 best.pt 평가 및16장 예측 저장 완료 |
| `notebooks/eda_report_view.py` | 로컬 EDA 표·그림 연결 | GitHub에 원본 이미지 첨부를 넣지 않는 구조 |

가상 데이터 검증에서 확인한 것:

1. 픽셀 bbox의 정규화 좌표와 0부터 연속 class mapping.
2. 원본 파일 해시 보존.
3. bbox 하나가 잘못되면 사진 전체 및 그 안 라벨을 REVIEW로 보류.
4. 촬영·유사 그룹이 train/val에 동시에 들어가지 않음.
5. 기본 학습 호출에서 모델 다운로드·학습·출력 폴더 생성 없음.

실제 데이터 결과는 문서 상단 표와 `qa/dataset_validation.json`에 기록했다.
촬영·정확해시·정의된 pHash 관계와 split group에 대해 train/val 누수0건을 확인했다.
임의의 모든 시각적 유사성까지 누수 없음으로 인증한 것은 아니다.
PASS에서 클래스별1개씩56어노테이션/49사진을 seed42로 감사 표본에 뽑았으며
사람 검수는 아직0건이다. `qa/pass_audit_sample.csv`의 판정은 PENDING이다.
이 작은 표본은 PASS 오류율이 충분히 낮다는 통계적 보증이 아니다.

## 5. 생성된 구조와 학습 계획

```text
<새 YOLO 출력 폴더>/
  images/train/  images/val/
  labels/train/  labels/val/
  data.yaml
  class_map.csv
  split_manifest.csv
  preparation_summary.json
  qa/
    image_inventory.csv   json_inputs.csv   evidence_inputs.csv
    image_status.csv      annotation_status.csv
    findings.csv          similar_pairs.csv   json_errors.csv
```

확인된 오류와 불확실한 사례 모두 REVIEW로 사진 단위 보류했다. 자동 REJECT·라벨 수정은 하지 않았다.
과거 검수 CSV를 입력으로 명시하고 현재 사진·JSON 지문을 대조한다.
최신 분할은 seed42로80:20 근접도를 우선하며 촬영·유사 그룹을 쪼개지 않았다.
현재 결과는158/39이고 train56클래스, val29클래스다. 클래스 전체 손실은 없다.
validation 클래스 범위가 좁다는 한계를 학습 승인 전에 명시한다.

| 학습 항목 | 실제 실행 |
|---|---|
| 모델·초기화 | YOLO11n, yolo11n.pt, pretrained=True |
| epochs / image size | 20 / 640 |
| batch / seed | 8 / 42 |
| Windows dataloader | workers=0, 로컬 데이터, cache=False |
| OOM | 없음, batch8 유지 |
| 시간 | epoch별 train+val 누적98.5073초, 초기 다운로드·설정 제외 |

사용자 승인 후20epoch 학습을 완료했다. 최초 계획 예산5~20분은 실측값으로 갱신했다.
현재 train158장/batch8이면 마지막 배치를 포함해 epoch당20배치, 20epoch에서400 train 배치다.
무작위 초기화 YOLO11n의 가상 입력 batch8/640 forward는 10회 평균 약9.95ms,
최대 할당 메모리 약282MiB였다. 역전파·optimizer·실제 이미지·사전학습 가중치를 쓰지 않은 검사다.
이를 실제 학습 속도나 학습 VRAM으로 해석하지 않는다.
실제 학습 로그에서는 GPU 메모리 약1.5GB가 관측됐다. 최대 메모리를 별도 측정하지 않았다.

학습 결과에는 best.pt, Ultralytics args/results, training_request.json을 저장했다.
별도 평가 결과는 mAP50=0.737463, mAP50-95=0.710427,
Precision=0.817668, Recall=0.640245다. confusion matrix, 예측16장과 클래스별 support도 저장했다.
검증에 포함된29클래스의 개발용 성능이며 나머지27종은 미평가다.
API와 초기 가중치 근거: [YOLO11 공식 문서](https://docs.ultralytics.com/models/yolo11/),
[학습](https://docs.ultralytics.com/modes/train/), [검증](https://docs.ultralytics.com/modes/val/).

## 6. 완료 결과 위치

내부 전체 QA·REVIEW 격리·새 YOLO 폴더 생성은 사용자 승인 후 완료했다.
이후 사용자의 학습 승인에 따라 YOLO11n 20epoch 학습과 best.pt 평가를 완료했다.
실제 출력: `C:/pill_ai/outputs/yolo11n_baseline_20261105_run2`.
`best.pt`, `evaluation/metrics.json`, `report.html`을 저장했다.
제출용 [최종 보고서](FINAL_REPORT.md)와 집계 차트는 저장소의 `docs/results/`에 있다.
원본 이미지·JSON·기존 노트북·다른 실험 결과는 보존했다.

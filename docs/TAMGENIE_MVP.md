# 탐지니(TamGenie) MVP

**TamGenie: 다단계 AI 기반 알약 탐지 및 식별 보조 시스템**.
서비스명은 탐지니, Python 패키지는 `tamgenie`, 실행 파일은 `app.py`다.
기존 저장소 `bravedolphinhere/pill_ai`와 가중치·데이터 폴더 이름은 변경하지 않았다.

## 실제 구현 상태

| 단계 | 현재 상태 |
|---|---|
| 업로드 | JPEG·PNG·WebP,15MB·2,000만 픽셀 상한, EXIF 방향 보정, 메모리 처리 |
| 1차 | 기존 내부56종 YOLO11n best.pt 로드·클래스 맵 검증·여러 박스 추론 |
| 시각화 | 번호·후보명·원시 점수·bbox, 개별 Crop, 표시 이미지 수동 다운로드 |
| 추가 확인 | 낮은 점수·품질 임시 규칙·같은 위치의 다른 클래스 겹침 |
| 2차 | EfficientNet-B0 인터페이스만 정의. 학습·가중치·예측 미구현, 화면에 표시 |
| 3차 | 구분 가능한 검증 속성만 질문, 모름 허용, 출처 없는 후보는 유지 |
| 종료 | 참고 후보·근거 표시. 구분 정보 부족·후보 없음·사용자 중단 시 식별 불가 |

높은 점수나 두 모델의 일치를 정체 확정으로 바꾸지 않는다.
약품 정체·복용 여부를 결정하는 서비스가 아니다.
탐지 박스 수는 실제 알약 수가 아니고 탐지0개도 알약 부재의 보증이 아니다.
원본 이미지·가중치는 수정하지 않고 앱이 업로드를 별도 데이터셋으로 영구 보관하지 않는다.

## 실행

이 PC의 실제 GPU 환경: Python3.14.2 / torch2.14.0+cu130 /
torchvision0.29.0+cu130 / RTX5070. 기존 프로젝트 Python3.12/cu126은
이 GPU의 실제 연산에 실패했으므로 `uv sync`만으로 같은 GPU 환경이 된다고 주장하지 않는다.
기존 GPU 패키지를 교체하지 않고 별도 uv 환경이 시스템 패키지를 상속하게 했다.

프로젝트 루트의 PowerShell에서:

```powershell
# 처음 한 번: 이미 동작하는 Python3.14 GPU 환경을 상속한다.
uv venv --system-site-packages --python 3.14 .venv-tamgenie
$env:VIRTUAL_ENV = "$PWD/.venv-tamgenie"
uv sync --project configs/mvp --active

# 기존 가중치의 실제 경로를 지정한다. 다른 PC에서는 이 값만 바꾼다.
$env:TAMGENIE_WEIGHTS = "C:/pill_ai/outputs/yolo11n_baseline_20261105_run2/best.pt"
uv run --no-project --python .venv-tamgenie/Scripts/python.exe python -m streamlit run app.py
```

주소: `http://127.0.0.1:8501`. 기본 서버는 로컬 PC에만 연다.
`.env`에 경로를 저장할 수도 있으며 `.env`·가중치는 Git에 올리지 않는다.
Mac/Linux의 가상환경 Python 경로는 `.venv-tamgenie/bin/python`이다.
실제 테스트 대상은 위 Windows·RTX5070이며 다른 환경의 성공을 인증하지 않는다.

`requirements.txt`는 MVP 직접 의존성 목록이다.
`configs/mvp/pyproject.toml`과 lock은 UI 전용 환경을 재현한다.
PyTorch·torchvision·Ultralytics는 시스템 설치를 상속하므로 UI lock만으로
GPU 환경까지 재현되는 것은 아니다. 실행 중 자동 패키지 설치는 비활성화했다.

## 검증된 속성과 질문의 한계

`configs/tamgenie/verified_metadata.json`에 공식 제품 설명과 대조한 속성만 담았다.
학습 JSON의 카탈로그를 그대로 검증 속성으로 사용하지 않았다.
현재 출처 대조된 제품은 **56종 중2종**, 모양·색상 일부다.

- [종근당 자누비아50mg](https://www.ckdpharm.com/product/productView.do?prodCode=CKD0000395)
- [유한양행 트윈스타](https://yuhan.co.kr/Products/List/?YPRD_IDX=2215&mode=view)

제품 설명의 근거이며 업로드 사진의 정체를 검증한 것은 아니다.
각인·분할선은 양면을 포괄하는 검증 정보가 없어 현재 질문하지 않는다.
추후 `covers_all_sides=true`와 공식 출처가 있는 정보만 해당 질문에 사용한다.
결측·빈 문자열을 각인 없음이나 분할선 없음으로 바꾸지 않는다.
현재 질문은2종 사이의 차이만 일부 구분하고 나머지54종은 미확보로 남긴다.
따라서 실제로 식별 불가로 종료할 수 있으며 질문20개나 완전한 식별 기능을 주장하지 않는다.

## 점수·모델·사용자 상태

표시 점수0.25, 추가 확인0.60과 품질 규칙은 검증 전 임시 설정이다.
정답 확률이나 보장된 신뢰성 기준이 아니다.
앱은 단일 클래스 predict 경로이므로 기존 validation multi-label 캐시의
threshold 지표를 앱 성능으로 그대로 사용하지 않는다.
56종 best.pt에 외부118종 맵을 연결하면 추론을 거부한다.
공유 모델 추론은 잠금으로 직렬화하고 사진은 전역 캐시에 저장하지 않는다.
새 이미지·설정·가중치·카탈로그 버전이 바뀌면 이전 분석을 재사용하지 않는다.

## 테스트

```powershell
uv run --no-project --python .venv-tamgenie/Scripts/python.exe python -m unittest discover -s tests -p test_tamgenie.py -v
$env:TAMGENIE_TEST_IMAGE = "실제 검증 사진의 경로"
uv run --no-project --python .venv-tamgenie/Scripts/python.exe python -m tests.smoke_streamlit
```

AppTest는 파일 업로드 반환값만 주입하고 그 뒤 실제 앱·기존 모델·GPU 추론·Crop·질문을 실행한다.
브라우저의 파일 선택 창까지 자동화한 테스트는 아니다.
실측 결과는 `docs/TAMGENIE_TEST_REPORT.md`에 기록한다.

## 이후 작업

검수된 Crop 데이터와 그룹 분할을 준비한 뒤 EfficientNet-B0를 별도로 학습·평가한다.
점수 보정과 학습 범위 밖 입력 판별은 아직 검증하지 않았다.
외부118종 정제는 별도 백그라운드 작업이며 현재 앱 모델은 기존56종이다.
16:00 KST부터는 새 기능보다 실행 안정화·README·보고서·제출 점검을 우선한다.

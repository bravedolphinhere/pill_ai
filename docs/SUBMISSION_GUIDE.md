# 제출 자료와 PDF 재생성

2026-11-05 기준, 탐지니 MVP·학습·평가·Recall 분석을 요약한6페이지 PDF를 준비했다.

- 제출 PDF: [tamgenie_final_report.pdf](../output/pdf/tamgenie_final_report.pdf)
- 상세 실험 근거: [베이스라인](FINAL_REPORT.md), [Recall 분석](RECALL_ANALYSIS.md)
- PDF 생성 코드: [build_submission_pdf.py](build_submission_pdf.py)
- 자동 확인 기록: [tamgenie_final_report_checks.json](../output/pdf/tamgenie_final_report_checks.json)

PDF는 원본 사진을 포함하지 않고 집계 표와 벡터 그래프를 사용한다.
원본 데이터·기존 가중치·평가를 보존하고 추가 학습 없이 생성했다.
모든 페이지를 렌더링해 한글·표·차트 배치를 확인했다.
별도의 사람 라벨 검수나 독립 test 평가가 완료됐다는 뜻은 아니다.

## 다시 만들기

폰트 파일 경로는 환경변수로 전달한다. Windows에서는 보통
`$env:WINDIR/Fonts/malgun.ttf`를 사용할 수 있다.
다른 환경에서는 한글을 지원하는 TTF 폰트를 지정한다.

```powershell
$env:PILL_REPORT_FONT = "$env:WINDIR/Fonts/malgun.ttf"
uv run --no-project --python 3.14 --with reportlab --with pymupdf python -m docs.build_submission_pdf --font $env:PILL_REPORT_FONT
```

ReportLab과 PyMuPDF는 개발 의존성에 기록했다.
위 명령은 PDF용 패키지를 격리 실행해 기존 GPU 학습 환경을 교체하지 않는다.
출력은 `output/pdf/`, 시각 검사용 임시 PNG는 `tmp/pdfs/`에 생성한다.
생성 코드는 집계 CSV/JSON의 표본 수, 주요 지표,6페이지 구성과 상대 링크를 확인한다.
다시 생성한 PDF도 페이지별 시각 검수를 거쳐 제출한다.

## 남은 작업

1. 저성능 5종의 REVIEW 사례를 사람이 공식 식별 정보와 대조해 최종 판정한다.
2. 희소 클래스와 학습에 없는 앞·뒷면의 독립 촬영 표본을 확보한다.
3. 승인 후 이미지 샘플링만 바꾼 비교 실험을 수행한다. 아직 실행하지 않았다.
4. 모델 선택·임계값 분석에 사용하지 않은 정답 있는 독립 test로 최종 평가한다.

GitHub에 게시된 노트북은 출력을 지웠으므로 데이터 없이 전 셀 실행되지는 않는다.
재실행에는 원본 데이터와 README에 설명한 로컬 경로 환경변수가 필요하다.

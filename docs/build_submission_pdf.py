"""검증된 집계 결과로 한글 제출용 PDF를 만들고 렌더링·링크 검사를 수행한다."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from pathlib import Path
from xml.sax.saxutils import escape

import pymupdf
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    Flowable,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

from src.config import PROJECT_ROOT

BLUE = colors.HexColor("#1C4670")
TEAL = colors.HexColor("#087F8C")
GRAY = colors.HexColor("#586575")
FOCUS = [30, 44, 20, 36, 25]
REPO = "https://github.com/bravedolphinhere/pill_ai"
BRANCH = "exp/yolo11n-baseline-sohee"


def read_csv(path: Path) -> list[dict]:
    """출력을 지운 노트북 대신 저장된 CSV를 보고서의 수치 근거로 읽는다."""
    with path.open(encoding="utf-8-sig", newline="") as file:
        return list(csv.DictReader(file))


def hash_file(path: Path) -> str:
    """배포할 파일의 내용을 재확인할 지문을 계산한다."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def local_links(root: Path) -> dict:
    """공개 문서의 상대 링크와 그림이 제출 사본 안에 존재하는지 검사한다."""
    checked, missing = [], []
    for relative in ["README.md", "docs/FINAL_REPORT.md", "docs/RECALL_ANALYSIS.md"]:
        document = root / relative
        for target in re.findall(r"!?\[[^\]]*\]\(([^)]+)\)", document.read_text(encoding="utf-8")):
            if "://" in target or target.startswith("#"):
                continue
            target = target.split("#", 1)[0]
            resolved = document.parent / target
            item = {"document": relative, "target": target, "exists": resolved.is_file()}
            checked.append(item)
            if not item["exists"]:
                missing.append(item)
    return {"checked": checked, "missing": missing}


class LineChart(Flowable):
    """PDF 안에서 확대해도 선명하도록 집계 지표를 벡터 선 그래프로 그린다."""

    def __init__(self, rows: list[dict], x_key: str, series: list[tuple], title: str) -> None:
        super().__init__()
        self.rows, self.x_key, self.series, self.title = rows, x_key, series, title
        self.width, self.height = 480, 178

    def draw(self) -> None:
        canvas = self.canv
        canvas.setFont("Korean", 10)
        canvas.setFillColor(BLUE)
        canvas.drawString(0, 161, self.title)
        left, bottom, width, height = 36, 25, 415, 110
        values_x = [float(row[self.x_key]) for row in self.rows]
        minimum, maximum = min(values_x), max(values_x)
        for index in range(6):
            value = index / 5
            canvas.setStrokeColor(colors.HexColor("#DFE6ED"))
            canvas.line(left, bottom + height * value, left + width, bottom + height * value)
            canvas.setFillColor(GRAY)
            canvas.setFont("Korean", 8)
            canvas.drawRightString(left - 7, bottom + height * value - 3, f"{value:.1f}")
        ticks = values_x if len(values_x) <= 5 else [1, 5, 10, 15, 20]
        for value in ticks:
            position = left + width * (value - minimum) / max(maximum - minimum, 1e-8)
            canvas.drawCentredString(position, bottom - 13, f"{value:g}")
        for index, (key, label, color) in enumerate(self.series):
            canvas.setStrokeColor(color)
            canvas.setFillColor(color)
            canvas.setLineWidth(1.7)
            path = canvas.beginPath()
            for step, row in enumerate(self.rows):
                position_x = left + width * (float(row[self.x_key]) - minimum) / (maximum - minimum)
                position_y = bottom + height * float(row[key])
                if step == 0:
                    path.moveTo(position_x, position_y)
                else:
                    path.lineTo(position_x, position_y)
            canvas.drawPath(path)
            canvas.setFont("Korean", 8)
            canvas.drawString(40 + index * 135, 145, label)


def build(root: Path, output: Path, font: Path) -> dict:
    """기존 학습·평가를 바꾸지 않고 요약 PDF와 페이지 PNG를 생성한다."""
    pdfmetrics.registerFont(TTFont("Korean", str(font)))
    pdfmetrics.registerFontFamily("Korean", normal="Korean", bold="Korean", italic="Korean")
    metrics = json.loads(
        (root / "docs/results/yolo11n_20261105/metrics.json").read_text(encoding="utf-8")
    )
    thresholds = read_csv(root / "docs/results/recall_20261105_final/threshold_overall.csv")
    support = read_csv(root / "docs/results/recall_20261105_final/class_support.csv")
    per_class = read_csv(root / "docs/results/yolo11n_20261105/per_class_metrics.csv")
    history = read_csv(root / "docs/results/yolo11n_20261105/training_history.csv")
    assert sum(int(row["validation_instances"]) for row in per_class) == 132
    assert len(history) == 20
    styles = getSampleStyleSheet()
    styles.add(
        ParagraphStyle(
            "BodyKo",
            fontName="Korean",
            fontSize=9.2,
            leading=15,
            spaceAfter=8,
            wordWrap="CJK",
            textColor=colors.HexColor("#253547"),
        )
    )
    styles.add(
        ParagraphStyle(
            "TitleKo",
            parent=styles["BodyKo"],
            fontSize=23,
            leading=33,
            textColor=BLUE,
            spaceAfter=17,
        )
    )
    styles.add(
        ParagraphStyle(
            "HeadingKo",
            parent=styles["BodyKo"],
            fontSize=14,
            leading=22,
            textColor=BLUE,
            spaceBefore=9,
            spaceAfter=10,
        )
    )
    styles.add(ParagraphStyle("SmallKo", parent=styles["BodyKo"], fontSize=8, leading=12))
    styles.add(
        ParagraphStyle("CellKo", parent=styles["BodyKo"], fontSize=8.2, leading=12, spaceAfter=0)
    )
    styles.add(ParagraphStyle("HeaderKo", parent=styles["CellKo"], textColor=colors.white))
    styles.add(ParagraphStyle("CenterKo", parent=styles["CellKo"], alignment=TA_CENTER))
    story = []

    def paragraph(text: str, style: str = "BodyKo") -> None:
        story.append(Paragraph(text, styles[style]))

    def heading(text: str) -> None:
        paragraph(text, "HeadingKo")

    def table(headers: list[str], rows: list[list], widths: list[int]) -> None:
        cells = [[Paragraph(escape(str(value)), styles["HeaderKo"]) for value in headers]]
        cells += [
            [Paragraph(escape(str(value)), styles["CellKo"]) for value in row] for row in rows
        ]
        content = Table(cells, colWidths=widths, repeatRows=1, hAlign="LEFT")
        content.setStyle(
            TableStyle(
                [
                    ("BACKGROUND", (0, 0), (-1, 0), BLUE),
                    (
                        "ROWBACKGROUNDS",
                        (0, 1),
                        (-1, -1),
                        [colors.HexColor("#F0F5F9"), colors.white],
                    ),
                    ("VALIGN", (0, 0), (-1, -1), "TOP"),
                    ("LEFTPADDING", (0, 0), (-1, -1), 8),
                    ("RIGHTPADDING", (0, 0), (-1, -1), 8),
                    ("TOPPADDING", (0, 0), (-1, -1), 7),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 7),
                ]
            )
        )
        story.extend([content, Spacer(1, 11)])

    def link(label: str, relative: str) -> None:
        url = f"{REPO}/blob/{BRANCH}/{relative}"
        paragraph(f'<link href="{url}" color="#087F8C">{escape(label)}</link>', "SmallKo")

    paragraph("탐지니 · TamGenie", "TitleKo")
    paragraph("다단계 AI 기반 알약 탐지 및 식별 보조 시스템", "HeadingKo")
    paragraph("최종 제출용 요약 보고서 | 2026-11-05 | bravedolphinhere / pill_ai", "SmallKo")
    paragraph(
        "기존 내부56종 YOLO11n을 Streamlit MVP에 연결했다. "
        "1~5쪽은 기존 모델 실험,6쪽은 서비스 구현·실행 검사다. "
        "약품 정체나 복용 여부를 확정하는 서비스가 아니다."
    )
    heading("1. 무엇을 했는가")
    paragraph(
        "알약 사진에서 위치와 제품 클래스를 함께 찾는 첫 기준 모델을 만들었다. "
        "기존 내부·외부 EDA를 확인하고, 내부 데이터의 불확실한 라벨을 보류한 뒤 "
        "촬영·유사 이미지 묶음을 유지해 분할했다. 외부 데이터는 이번 학습에 사용하지 않았다."
    )
    table(
        ["단계", "완료 내용"],
        [
            ["데이터 준비", "232사진·763라벨 점검 → PASS197사진·660라벨"],
            ["분할", "Train158사진 / Validation39사진, 80.20:19.80"],
            ["첫 학습", "YOLO11n 사전학습, 20epoch / 640 / batch8 / seed42"],
            ["후속 분석", "기존 가중치 보존, confidence5종 비교 및 저성능5종 분석"],
        ],
        [95, 385],
    )
    heading("2. 베이스라인 결과")
    table(
        ["mAP50", "mAP50-95", "Precision", "Recall"],
        [
            [
                f"{metrics['mAP50']:.4f}",
                f"{metrics['mAP50-95']:.4f}",
                f"{metrics['precision']:.4f}",
                f"{metrics['recall']:.4f}",
            ]
        ],
        [120] * 4,
    )
    paragraph(
        "<b>평가 범위: 검증39사진·132개 정답·29클래스.</b> "
        "모델은56클래스를 학습했지만, 검증에 없는27종의 성능은 평가하지 못했다. "
        "이 값은 개발용 validation 결과이며 독립 test 성능이나 약품 정체의 확정률이 아니다."
    )
    paragraph(
        "위 Precision/Recall은 Ultralytics native 요약 지표다. 뒤의 고정 confidence "
        "micro 지표와는 운영점·집계 방식이 달라 직접 증감 비교하지 않는다.",
        "SmallKo",
    )
    link("프로젝트 저장소와 전체 README", "README.md")
    story.append(PageBreak())

    heading("3. 데이터 품질과 누수 방지")
    table(
        ["구분", "사진", "라벨", "클래스 / 판정"],
        [
            ["원본 학습 후보", 232, 763, "56종"],
            ["PASS", 197, 660, "자동 구조 검사 통과 후보"],
            ["REVIEW", 35, 103, "학습 후보에서 사진 단위 보류"],
            ["Train", 158, 528, "56종"],
            ["Validation", 39, 132, "29종"],
        ],
        [105, 60, 60, 255],
    )
    paragraph(
        "기존 촬영·유사 이미지 그룹73개를 쪼개지 않았다. Train의 모든 클래스를 보존하고 "
        "사진 수의80:20 근접도를 우선해 분할했다. 정의한 촬영·SHA256·pHash 관계에서는 "
        "분할 간 누수0건이었다. 모든 시각적 유사성을 완전히 검출했다는 뜻은 아니다."
    )
    paragraph(
        "원본232사진·763JSON의 지문 보존과 복사197사진, YOLO660라벨의 좌표 역변환을 "
        "확인했다. 자동 REJECT·원본 수정·자동 라벨 교정은 하지 않았다. "
        "PASS 감사 표본56어노테이션/49사진의 사람 검수는 대기 상태다."
    )
    heading("4. 실행 환경과 학습 경향")
    paragraph(
        "RTX5070 약12GB · Python3.14.2 · torch2.14.0+cu130 · torchvision0.29.0+cu130 · "
        "Ultralytics8.4.193. 실제 GPU 학습을 완료했고 OOM 없이 batch8을 유지했다. "
        "epoch별 train+val 누적 시간은98.51초이며 초기 다운로드·설정 시간은 제외했다. "
        "학습 로그에서 약1.5GB를 관측했지만 peak VRAM은 별도 측정하지 않았다."
    )
    story.append(
        LineChart(
            history,
            "epoch",
            [("metrics/mAP50(B)", "mAP50", BLUE), ("metrics/mAP50-95(B)", "mAP50-95", TEAL)],
            "epoch별 native validation mAP",
        )
    )
    paragraph(
        "mAP50은 epoch10=0.5722 →15=0.7028 →20=0.7373으로 상승했다. "
        "추가 학습 가능성은 남지만 단일 seed·작은 검증셋만으로 일반화나 과적합 부재를 "
        "확정할 수 없다. 기본 close_mosaic=10 설정에 따라 epoch11부터 mosaic이 꺼졌다.",
        "SmallKo",
    )
    story.append(PageBreak())

    heading("5. Confidence 임계값 비교")
    paragraph(
        "재학습 없이 동일한 검증 예측 캐시를 사용했다. 같은 클래스이며 IoU≥0.50인 "
        "예측과 GT를 confidence 순서로1:1 연결했다. 같은 GT의 추가 예측은 FP다. "
        "다른 클래스 예측은 정답 탐지로 세지 않는다."
    )
    table(
        ["conf", "TP", "FP", "FN", "Precision", "Recall", "F1"],
        [
            [
                f"{float(row['threshold']):.2f}",
                row["TP"],
                row["FP"],
                row["FN"],
                f"{float(row['precision']):.4f}",
                f"{float(row['recall']):.4f}",
                f"{float(row['f1']):.4f}",
            ]
            for row in thresholds
        ],
        [45, 45, 45, 45, 100, 100, 100],
    )
    paragraph(
        "TP: 맞힌 정답 / FP: 정답과 연결되지 않은 예측 / FN: 놓친 정답. "
        "P=TP/(TP+FP), R=TP/(TP+FN), F1=2TP/(2TP+FP+FN). "
        "전체는56클래스의 카운트를 합친 micro 지표이며, 클래스별0분모는 미정의로 남겼다.",
        "SmallKo",
    )
    story.append(
        LineChart(
            thresholds,
            "threshold",
            [
                ("precision", "Precision", BLUE),
                ("recall", "Recall", TEAL),
                ("f1", "F1", colors.HexColor("#BD6B23")),
            ],
            "고정 임계값별 Precision / Recall / F1",
        )
    )
    paragraph(
        "<b>지정한 다섯 후보 중 conf0.25 유지가 가장 균형이 좋았다.</b> "
        "0.25→0.05에서 TP는12개 증가했지만 FP가310개 늘었다. "
        "Recall은9.09%p 증가하고 Precision은43.77%p 감소했다. "
        "0.20은0.25와 TP가 같고 FP만12개 많았다."
    )
    paragraph(
        "캐시 조건: imgsz640 / batch8 / conf0.001 / NMS IoU0.7 / max_det300 / "
        "multi_label=True. JSON category_id−1로 YOLO ID를 복원했다. "
        "좌표3자리·score5자리 반올림에 따른 경계 차이가 가능하다. "
        "일반 predict() 갤러리의 단일클래스 출력과는 다를 수 있다. "
        "임계값 변경만으로 AP50이 개선됐다고 주장하지 않는다.",
        "SmallKo",
    )
    story.append(PageBreak())

    heading("6. 저성능5종: 위치보다 분류가 문제인가")
    supports = {int(row["class_id"]): row for row in support}
    scores = {int(row["yolo_class_id"]): row for row in per_class}
    short_names = [
        "자누비아50mg",
        "트라젠타듀오2.5/850mg",
        "콜리네이트400mg",
        "트윈스타40/5mg",
        "플라빅스75mg",
    ]
    table(
        ["클래스", "Train 사진/라벨", "Train 그룹", "Val 사진/라벨", "AP50"],
        [
            [
                label,
                supports[key]["train_annotations"],
                supports[key]["train_groups"],
                supports[key]["val_annotations"],
                f"{float(scores[key]['mAP50']) * 100:.2f}%",
            ]
            for key, label in zip(FOCUS, short_names, strict=True)
        ],
        [160, 80, 70, 90, 80],
    )
    paragraph(
        "<b>conf0.05에서 이5종의 GT25개 중24개는 위치가 맞는 다른 클래스 예측이고, "
        "1개만 정답 TP였다.</b> 자누비아·트라젠타듀오·트윈스타는0.05에서도 TP0개다. "
        "단순히 confidence를 낮추는 것만으로 회복되지 않았다."
    )
    table(
        ["conf0.25에서 GT 기준 혼동", "건수", "증거"],
        [
            ["자누비아 → 가바토파", 3, "IoU0.952~0.982"],
            ["플라빅스 → 크레스토", 3, "IoU0.955~0.977"],
            ["트윈스타 → 뉴로메드/아토젯/엑스포지", "2 / 2 / 1", "위치 매칭은 성립"],
            ["트라젠타듀오 → 로수바미브", 1, "위치 매칭은 성립"],
        ],
        [285, 70, 125],
    )
    paragraph(
        "원본 GT를 기준으로 한 모델 진단이다. 실제 약품 정체의 확정이 아니다. "
        "conf0.25에서는25GT가 분류 실패12개·잔여 누락13개였고, 위치 오류 후보는0개였다. "
        "낮은 threshold에서 유형이 바뀌므로 잔여 누락을 물체 자체의 완전 미검출로 단정하지 않는다.",
        "SmallKo",
    )
    heading("7. Train과 Validation에서 보이는 면의 차이")
    paragraph(
        "트라젠타듀오: Train5장은 로고 면, Val3장은 D2/850 문자 면으로 보였다. "
        "플라빅스: Train9장은75 면, Val6장 중3장은1171 면으로 보였다. "
        "이 차이는 회전 증강으로 반대편 각인을 만들 수 없다는 한계를 보여 준다."
    )
    paragraph(
        "자누비아는 train/val 모두112로 보이는 각인과 비슷한 외형을 관찰했다. "
        "학습3장·그룹1개라는 부족은 확인됐지만, 낮은 AP만으로 오라벨이라고 단정할 근거는 없다. "
        "우선5종의58GT를 에이전트가 시각 비교했으며 사람의 최종 검수는 완료되지 않았다."
    )
    story.append(PageBreak())

    heading("8. 검수 상태와 다음 한 변수 실험")
    table(
        ["구분", "현재 상태 / 다음 행동"],
        [
            [
                "검수",
                "58GT 자동 REVIEW / 사람 PENDING. 공식 제품 정보와 이미지의 각인·색·모양·bbox 대조",
            ],
            [
                "데이터 범위",
                "희소 클래스 및 앞·뒷면 부족 확인. 새로운 독립 촬영 확보는 별도 데이터 과제로 관리",
            ],
            ["단일 변수 추천", "Train 이미지 샘플링만 변경해 희소 클래스의 학습 노출 조정"],
            ["고정 조건", "모델·분할·20epoch·imgsz640·batch8·seed42·증강·초기 가중치 유지"],
            ["비교 기준", "동일 IoU0.50·conf0.25의 P/R/F1, 클래스별 AP50, 실패 유형"],
        ],
        [100, 380],
    )
    paragraph(
        "샘플링 변경은 제안 단계이며 추가 학습을 실행하지 않았다. "
        "반복 샘플링으로 새로운 각인 면이 생기지는 않는다. "
        "라벨이 틀린 경우 오류도 반복되므로 검수가 선행돼야 한다. "
        "validation을 train으로 옮기거나 기존 그룹을 쪼개는 방법은 사용하지 않는다."
    )
    heading("9. 남은 한계")
    paragraph(
        "① 검증에 없는27종은 미평가다. ② 검증 표본이 클래스별1~24개로 작다. "
        "③ 모델 선택과 임계값 분석에 같은 validation을 사용했다. "
        "④ 자동 PASS는 라벨 의미의 정확성 보증이 아니다. "
        "⑤ 정답 없는 test842장은 정량 평가에 사용하지 않았다. "
        "최종 성능 주장은 정답이 있고 분석·선택에 사용하지 않은 독립 test로 확인해야 한다."
    )
    heading("10. 제출 자료와 원본 근거")
    link("전체 베이스라인 보고서 · 설정 · 지표 · 혼동행렬", "docs/FINAL_REPORT.md")
    link(
        "Recall 분석 전체 · 계산 기준 · 클래스별 수치 · 독립 정보 대조 한계",
        "docs/RECALL_ANALYSIS.md",
    )
    link("내부 EDA 노트북", "notebooks/01_internal_data_eda.ipynb")
    link("외부 EDA 노트북", "notebooks/02_external_data_eda.ipynb")
    link(
        "클래스별 평가와 NOT_EVALUATED 상태", "docs/results/yolo11n_20261105/per_class_metrics.csv"
    )
    paragraph(
        "가중치 best.pt와 원본 사진·JSON, 예측 사진은 로컬에 보존했다. "
        "GitHub에는 코드·출력을 지운 EDA·집계 CSV·차트를 제공한다. "
        "로컬 검수 갤러리에는25개 사례의 GT·conf0.05·conf0.25 가로 비교와 "
        "58GT contact sheet를 저장했다.",
        "SmallKo",
    )
    paragraph("가중치 SHA256", "SmallKo")
    paragraph(metrics["weights_sha256"][:32] + "<br/>" + metrics["weights_sha256"][32:], "SmallKo")
    paragraph(
        "공식 정보 대조: 자누비아 제품 설명과 트윈스타 성상은 국내 제품 페이지를 확인했다. "
        "나머지 일부 각인·제품 정보는 독립 확인을 완료하지 못해 UNVERIFIED로 유지했다. "
        "OCR·모델 예측을 정답으로 사용하거나 원본 라벨을 자동 수정하지 않았다.",
        "SmallKo",
    )

    story.append(PageBreak())
    heading("11. 탐지니 Streamlit MVP")
    table(
        ["단계", "실제 상태"],
        [
            ["1차 YOLO11n", "기존56종 best.pt 실제 추론, 여러 bbox·모델 점수·개별 Crop 표시"],
            ["추가 확인", "낮은 점수·Crop 품질 우려·같은 위치의 다른 클래스 겹침을 분기"],
            ["2차 EfficientNet-B0", "미구현. 학습 가중치가 없어 명시적으로 건너뜀"],
            ["3차 질문", "구분 가능한 검증 속성이 있을 때만 질문. 모름과 정보 미확보 후보 유지"],
            ["최종 표시", "참고 후보와 근거, 정보 부족·후보 없음·확인 중단 시 식별 불가"],
        ],
        [125, 355],
    )
    paragraph(
        "현재 공식 제품 설명과 대조한 속성은56종 중2종의 모양·색상 일부다. "
        "나머지54종을 임의로 제거하지 않아 질문 후에도 식별 불가일 수 있다. "
        "각인·분할선은 양면의 검증 정보가 부족해 현재 질문하지 않는다. "
        "모델 점수·일치 여부를 실제 정답 확률이나 약품 정체 확정으로 바꾸지 않는다."
    )
    heading("12. 실제 기능 검사")
    table(
        ["검사", "결과"],
        [
            ["단위 테스트", "불확실성·모름·출처 없는 속성·손상 입력 등11개 통과"],
            ["실제 모델 AppTest", "기존 검증 사진에서4박스·Crop·질문4개 렌더링, 앱 예외0건"],
            ["상태·오류", "모름 적용, 손상 이미지 거절, 새 업로드 시 이전 분석 제거"],
            ["HTTP 실행", "로컬 앱 / 와 /_stcore/health 모두200"],
            ["브라우저 육안 검수", "현재 URL을 확인할 수 없는 도구 제한으로 미완료"],
        ],
        [125, 355],
    )
    paragraph(
        "AppTest는 업로드 반환값만 주입한 뒤 실제 앱·기존 가중치·GPU 추론을 사용했다. "
        "브라우저 파일 선택 창의 자동화나 모바일·부하 검사는 아니다. "
        "검출4개는 실제 알약4개나 정답4개라는 뜻이 아니며, "
        "한 사진으로 앱 정확도를 측정하지 않았다.",
        "SmallKo",
    )
    heading("13. 실행과 미완료 작업")
    paragraph(
        "패키지와 식별자는 tamgenie, 진입점은 app.py다. "
        "Streamlit1.66.1과 기존 Python3.14/cu130 환경을 별도 uv 환경으로 재사용했다. "
        "가중치는 TAMGENIE_WEIGHTS로 연결하며 기존 경로와 저장소 이름은 변경하지 않았다."
    )
    link("MVP 실행 명령과 메타데이터 출처·한계", "docs/TAMGENIE_MVP.md")
    link("실제 테스트 상세와 재현 방법", "docs/TAMGENIE_TEST_REPORT.md")
    paragraph(
        "외부118종 정제는 별도 작업이며 이 앱에 새 모델로 반영하지 않았다. "
        "EfficientNet-B0 학습, 완전한 약품 카탈로그, 점수 보정, 학습 범위 밖 입력 판별은 "
        "이후 검증 과제다. 새 학습이나 원본·기존 가중치 변경 없이 MVP를 만들었다.",
        "SmallKo",
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    document = SimpleDocTemplate(
        str(output),
        pagesize=A4,
        leftMargin=57,
        rightMargin=57,
        topMargin=48,
        bottomMargin=45,
        title="TamGenie: 다단계 AI 기반 알약 탐지 및 식별 보조 시스템",
        author="bravedolphinhere",
    )

    def footer(canvas: object, document: object) -> None:
        canvas.saveState()
        canvas.setFont("Korean", 8)
        canvas.setFillColor(GRAY)
        canvas.drawString(57, 25, "TamGenie | 2026-11-05 | 식별 보조 MVP · 개발용 validation")
        canvas.drawRightString(A4[0] - 57, 25, str(document.page))
        canvas.restoreState()

    document.build(story, onFirstPage=footer, onLaterPages=footer)
    preview = root / "tmp/pdfs/submission_preview"
    preview.mkdir(parents=True, exist_ok=True)
    with pymupdf.open(output) as pdf:
        assert len(pdf) == 6, f"예상6페이지와 실제{len(pdf)}페이지가 다릅니다."
        extracted = "\n".join(page.get_text() for page in pdf)
        assert "자누비아" in extracted and "0.7375" in extracted and "0.6886" in extracted
        for index, page in enumerate(pdf):
            page.get_pixmap(matrix=pymupdf.Matrix(1.3, 1.3)).save(preview / f"page-{index + 1}.png")
        result = {
            "pages": len(pdf),
            "sha256": hash_file(output),
            "bytes": output.stat().st_size,
            "pdf_links": sum(len(page.get_links()) for page in pdf),
            "source_metrics_sha256": hash_file(root / "docs/results/yolo11n_20261105/metrics.json"),
            "source_thresholds_sha256": hash_file(
                root / "docs/results/recall_20261105_final/threshold_overall.csv"
            ),
            "builder_sha256": hash_file(Path(__file__)),
            "local_links": local_links(root),
        }
    assert not result["local_links"]["missing"]
    (output.parent / f"{output.stem}_checks.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return result


def main() -> None:
    """폰트 경로를 외부에서 받아 개인 PC 절대 경로를 코드에 고정하지 않는다."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--font", type=Path, required=True)
    parser.add_argument(
        "--output", type=Path, default=PROJECT_ROOT / "output/pdf/tamgenie_final_report.pdf"
    )
    args = parser.parse_args()
    print(json.dumps(build(PROJECT_ROOT, args.output, args.font), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

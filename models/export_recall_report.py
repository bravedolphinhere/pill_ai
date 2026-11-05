"""Recall 분석 CSV의 집계 차트와 제출용 해석 보고서를 만든다."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib import font_manager

from models.analyze_recall import FOCUS, digest
from src.config import PROJECT_ROOT


def markdown_table(frame: pd.DataFrame) -> str:
    """추가 패키지 없이 표를 Markdown으로 기록한다."""

    def cell(value: object) -> str:
        if pd.isna(value):
            return "미정의"
        return f"{value:.4f}" if isinstance(value, float) else str(value)

    return "\n".join(
        [
            "| " + " | ".join(frame.columns) + " |",
            "| " + " | ".join(["---"] * len(frame.columns)) + " |",
            *[
                "| " + " | ".join(map(cell, row)) + " |"
                for row in frame.itertuples(index=False, name=None)
            ],
        ]
    )


def export(analysis: Path, run: Path, destination: Path, refresh: bool = False) -> None:
    """기존 베이스라인은 그대로 두고 집계 결과만 별도 공개 폴더로 내보낸다."""
    if destination.exists():
        if not refresh:
            raise FileExistsError("기존 보고서는 --refresh를 명시해야 갱신할 수 있습니다.")
        previous = json.loads((destination / "export_manifest.json").read_text(encoding="utf-8"))
        assert all(digest(destination / name) == sha for name, sha in previous["artifacts"].items())
    destination.mkdir(parents=True, exist_ok=True)
    overall = pd.read_csv(analysis / "threshold_overall.csv")
    per_class = pd.read_csv(analysis / "threshold_per_class.csv")
    support = pd.read_csv(analysis / "class_support.csv")
    evidence = pd.read_csv(analysis / "gt_outcomes.csv")
    focused = evidence.loc[evidence.class_id.isin(FOCUS)]
    failure_counts = (
        focused.groupby(["threshold", "pill_name", "failure"]).size().reset_index(name="count")
    )
    confusions = (
        focused.loc[focused.failure == "Classification Failure"]
        .groupby(["threshold", "pill_name", "predicted_name"])
        .size()
        .reset_index(name="count")
    )
    confidence = (
        focused.loc[focused.threshold == 0.25]
        .groupby("pill_name")["best_correct_class_confidence_at_iou50"]
        .agg(["min", "median", "max"])
        .reset_index()
    )
    for name in ["threshold_overall.csv", "threshold_per_class.csv", "class_support.csv"]:
        shutil.copy2(analysis / name, destination / name)
    failure_counts.to_csv(destination / "failure_counts.csv", index=False, encoding="utf-8-sig")
    confusions.to_csv(destination / "confusion_pairs.csv", index=False, encoding="utf-8-sig")
    confidence.to_csv(
        destination / "correct_confidence_summary.csv", index=False, encoding="utf-8-sig"
    )
    predictions = pd.read_csv(analysis / "prediction_outcomes.csv")
    score_summary = (
        predictions.loc[predictions.threshold == 0.05]
        .groupby(["class_id", "status"])
        .confidence.agg(["count", "min", "median", "max"])
        .reset_index()
    )
    score_summary.to_csv(
        destination / "prediction_confidence_summary.csv", index=False, encoding="utf-8-sig"
    )
    if any(font.name == "Malgun Gothic" for font in font_manager.fontManager.ttflist):
        plt.rcParams["font.family"] = "Malgun Gothic"
    figure, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
    for metric in ["precision", "recall", "f1"]:
        axes[0].plot(overall.threshold, overall[metric], marker="o", label=metric)
    axes[0].set(xlabel="Confidence", ylabel="micro 지표", ylim=(0, 1))
    axes[0].legend()
    for key in ["TP", "FP", "FN"]:
        axes[1].plot(overall.threshold, overall[key], marker="o", label=key)
    axes[1].set(xlabel="Confidence", ylabel="개수")
    axes[1].legend()
    figure.savefig(destination / "threshold_tradeoff.png", dpi=150)
    plt.close(figure)
    figure, axis = plt.subplots(figsize=(10, 4), constrained_layout=True)
    for index, class_id in enumerate(FOCUS):
        selected = focused.loc[(focused.threshold == 0.25) & (focused.class_id == class_id)]
        axis.scatter(
            selected.best_correct_class_confidence_at_iou50, [index] * len(selected), alpha=0.6
        )
    names = support.set_index("class_id").pill_name
    axis.set_yticks(range(len(FOCUS)), [names[i] for i in FOCUS])
    for threshold in [0.05, 0.25]:
        axis.axvline(threshold, linestyle="--", label=f"conf{threshold}")
    axis.set_xlabel("GT와 IoU≥0.50인 정답 클래스 예측 중 최고 confidence · 없으면0")
    axis.legend()
    figure.savefig(destination / "correct_confidence.png", dpi=150)
    plt.close(figure)
    history = pd.read_csv(run / "train_batch8/results.csv")
    history.loc[history.epoch.isin([10, 15, 20])].to_csv(
        destination / "learning_curve_checkpoints.csv", index=False, encoding="utf-8-sig"
    )
    selected_support = support.set_index("class_id").loc[FOCUS].reset_index()
    counts25 = per_class.loc[(per_class.threshold == 0.25) & per_class.class_id.isin(FOCUS)]
    table = selected_support.merge(counts25[["class_id", "TP", "FP", "FN"]], on="class_id")
    table = table[
        [
            "pill_name",
            "train_annotations",
            "train_groups",
            "val_annotations",
            "val_groups",
            "TP",
            "FP",
            "FN",
        ]
    ].rename(
        columns={
            "pill_name": "클래스",
            "train_annotations": "Train 사진/라벨 수",
            "train_groups": "Train 그룹",
            "val_annotations": "Val 사진/라벨 수",
            "val_groups": "Val 그룹",
        }
    )
    assets = destination.relative_to(PROJECT_ROOT / "docs").as_posix()
    report = f"""# YOLO11n Recall 임계값 실험과 저성능 클래스 원인 분석

작성일:2026-11-05. **추가 학습·모델 교체·라벨 수정 없음.**
기존 best.pt와 기존 평가를 보존하고 낮은 confidence로 저장한 검증 예측을 재사용했다.
로컬 상세 자료: `{analysis.name}`. 기존 베이스라인은 [최종 보고서](FINAL_REPORT.md)를 참고한다.

## 결론

지정한 다섯 값 중 전체 micro F1은 **conf0.25에서0.6886으로 가장 높다**.
0.25→0.05에서 Recall은0.7121→0.8030으로9.09%p 증가하지만
Precision은0.6667→0.2289로43.77%p 감소한다. TP12개 증가에 FP310개가 추가된다.
0.20은0.25와 TP94개로 같고 FP만12개 많으므로 이번 표에서 이점이 없다.
현재 다섯 후보 중0.25를 유지하는 것이 합리적이다. 전체 가능한 임계값의 최적점이라는 뜻은 아니다.
오탐 비용·최소 Precision 요구가 정해지면 그 조건으로 다시 선택해야 한다.

저성능 5종의 검증 GT25개 중 conf0.05에서는 **24개가 위치는 맞는 다른 클래스 예측**이고
나머지1개는 정답 TP다. 단순한 알약 위치 미검출보다
분류 실패와 표본·가시적 특징 범위 부족이 두드러진다.
이 진단은 원본 라벨을 임시 기준으로 한 모델 오류 분석이며 실제 약품 정체의 확정이 아니다.
다음 단일 변수 실험으로는 **클래스 불균형 개선을 위한 train 이미지 샘플링 변경**을 우선 제안한다.
라벨 검수와 가시적 면의 학습 범위 확인이 선행돼야 하며 재학습은 실행하지 않았다.

## 같은 조건으로 계산한 threshold별 지표

{markdown_table(overall)}

![임계값별 성능과 오탐]({assets}/threshold_tradeoff.png)

- 동일 validation39사진·132GT·29개 GT 클래스, 모델 출력은56클래스를 모두 포함했다.
- 기존 `evaluation/validation/predictions.json`: imgsz640, batch8, 검증 conf0.001,
  NMS IoU0.7, max_det300, class-aware NMS, multi_label=True의 사후 NMS 결과다.
- custom JSON category_id는1부터 시작하므로 YOLO class ID=category_id−1로 복원했다.
- 각 threshold 이상 예측만 남긴 뒤 사진별 confidence 내림차순으로 처리했다.
  같은 클래스의 아직 연결되지 않은 GT 중 가장 높은 IoU≥0.50를 일대일 매칭한다.
  같은 GT에 대한 추가 예측은 FP다. 다른 클래스 예측은 GT의 TP가 될 수 없다.
- TP=매칭된 예측 수, FP=나머지 예측 수, FN=연결되지 않은 GT 수.
  P=TP/(TP+FP), R=TP/(TP+FN), F1=2TP/(2TP+FP+FN).
- 전체는 모든56클래스의 카운트를 먼저 합친 micro 지표다.
  클래스별0분모는 빈칸/미정의이며, validation GT가 없는27종 Recall은 평가하지 않는다.
  그27종으로 잘못 예측한 FP는 전체 Precision에 포함한다.
- 기존 보고서의 P/R은 native 요약 운영점 및 클래스 평균이다. 이번 고정 threshold의 micro P/R과
  직접 증감 비교하지 않는다. **동일 조건의0.25 행을 이번 실험의 비교 기준**으로 사용한다.
- 배포용 단일클래스 `predict()` 출력과 검증 multi-label NMS 출력은 다를 수 있다.
  이 표를 기존 예측 갤러리의 conf0.25 성능이라고 부르지 않는다.
- JSON 좌표는 소수3자리·score는5자리 반올림이다. 원시 tensor와 극미한 경계 차이가 가능하다.
  재추론 없이 원래 평가 조건을 보존한 캐시 기반 분석이다.
- AP50은 confidence 전체 순위를 사용하는 지표다.
  임계값만 바꿔 기존 AP50이 개선됐다고 주장하지 않는다.

## 저성능 클래스의 표본과 conf0.25 결과

{markdown_table(table)}

해당5종에서는 사진마다 그 클래스 GT가1개라 사진 수와 라벨 수가 같다.
전체 train 클래스별 라벨은2~104개, 중앙값6.5개다. 자누비아·트윈스타는 특히 적고 그룹도1개다.
그룹은 독립 개체 수를 보장하는 단위가 아니라 기존 촬영·유사 이미지 묶음이다.

## 실패 유형과 실제 혼동

정량 TP를 먼저 확정한 뒤 남은 GT와 예측만 진단한다.
Classification Failure는 클래스 무관 IoU≥0.50 일대일 매칭이 있는 FN,
Localization Failure 후보는 그 이후 남은 같은 클래스 예측과 IoU≥0.10인 FN,
Detection Failure는 나머지 FN이다. 마지막 범주는 **그 threshold에서의 잔여 누락**이며
실제 알약을 전혀 못 보았다는 확정이 아니다. 낮은 threshold에서는 유형이 바뀔 수 있다.
Localization 후보의 정확한 IoU는 개별 CSV에 남겨 매칭 경쟁·중복 GT 가능성도 검토한다.
낮은 IoU의 다른 클래스 예측처럼 두 오류가 함께 있는 경우 원인을 단독 확정할 수 없다.

{markdown_table(failure_counts.loc[failure_counts.threshold.isin([0.05, 0.25])])}

conf0.25의 주요 혼동 상대:

{markdown_table(confusions.loc[confusions.threshold == 0.25].drop(columns="threshold"))}

이때 자누비아→가바토파는 IoU0.952~0.982, 플라빅스→크레스토는0.955~0.977이다.
박스 위치는 충분히 맞는데 클래스가 틀린 사례다. 자누비아→트라젠타듀오라는 예시는
이번0.25 사례에서는 확인되지 않았다.
우선5종에는 두 threshold 모두 Localization Failure 후보가0개다.
이 결과만으로 bbox가 모두 정확하다거나 위치 오류가 전체 데이터에 없다고 보증하지 않는다.

## Confidence 분포

{markdown_table(confidence)}

![정답 클래스 confidence 분포]({assets}/correct_confidence.png)

GT마다 IoU≥0.50인 정답 클래스 캐시 예측 중 최대값의 분포다.
0은 점수 자체가0이라는 뜻이 아니라 conf0.001/NMS/max_det 이후 해당 예측이 없다는 뜻이다.
자누비아 최대0.01178, 트라젠타듀오 최대0.00677, 트윈스타 최대0.01422로
0.05까지 낮춰도 복구되지 않는다. 플라빅스 한 건은0.17174여서0.15에서만 복구된다.
전체 예측의 클래스별 TP/FP confidence 요약은 `prediction_confidence_summary.csv`,
로컬 `prediction_outcomes.csv`에는 threshold별 모든 예측의 좌표·confidence·TP/FP를 기록했다.

## 이미지와 원본 라벨 검토

우선5종의 PASS58GT(train33+val25)를5개 contact sheet로 모두 확인했다.
에이전트의 시각 관찰이며 사람의 최종 승인과 구분한다. 정체는 UNVERIFIED로 남겼다.

- 자누비아: 원형·베이지/황색 계열이며112로 보이는 각인, train/val의 외형이 대체로 유사하다.
  단순히 AP가 낮아서 오라벨이라고 볼 근거는 없다.
- 트라젠타듀오: train5장은 로고 면, val3장은 D2/850 문자 면으로 보인다.
  양면의 가시적 특징 범위 차이가 뚜렷한 원인 후보다.
- 콜리네이트: 타원형 황색 연질캡슐과 HM-CN으로 보이는 인쇄.
  카탈로그 color_class1=갈색과 chart=연한 노란색의 표현도 달라 독립 확인이 필요하다.
- 트윈스타: A1과 로고로 보이는 흰 타원정이며 train/val의 보이는 면은 대체로 같다.
  train2장만으로는 다양성이 매우 부족하다.
- 플라빅스: train9장은75 면, val6장 중3장은1171 면으로 보인다.
  나머지3장은75 면으로 보이며 그 중1건만 낮은 threshold에서 정답 탐지됐다.

GT 박스는 contact sheet에서 대체로 알약 몸통을 포함하는 모습이었다.
육안 확대만으로 픽셀 단위 경계 적합성·보이지 않는 면·제품 정체를 인증할 수는 없다.
`label_review.csv`는58개 모두 자동 REVIEW/사람 PENDING이며 기존 데이터셋 PASS를 덮어쓰지 않았다.
원본 JSON 경로·SHA256·품목기준코드·각인·색·모양을 보존했다.
`gt_outcomes.csv`에는25GT×5threshold의 매칭 결과와 근거를 포함해 전체132GT를 기록했다.
로컬 `review_gallery.html`에서는 동일 GT를 **GT 확대 / conf0.05 / conf0.25** 가로3열로 비교한다.
초록 GT·빨강 예측이며 관련 클래스명·파일명·제품 코드·좌표·원본 JSON 경로를 제공한다.

독립 정보 대조의 확인 범위(2026-11-05 접속):

- [종근당 자누비아50mg 제품 정보](https://www.ckdpharm.com/product/productView.do?prodCode=CKD0000395)는
  밝은 베이지색·원형·112 각인을 설명한다.
  관찰은 이 설명과 양립하지만 사진의 제품 정체 확정은 아니다.
- [유한양행 트윈스타 제품 정보](https://yuhan.co.kr/Products/List/?YPRD_IDX=2215&mode=view)는
  흰색/연한 파란색 타원 이중정과40/5mg 제품을 확인해 준다. 이 사진에서 뒷면 색을 확인한 것은 아니다.
- [사노피 제품 목록](https://www.sanofi.com/ko/south-korea/product-information)에서
  플라빅스75mg·보험코드652101570은 확인했다. 이 목록만으로 각인75/1171을 독립 검증하지 못했다.
- 트라젠타듀오의 국내 제조사 페이지는 접근 제한, 식약처 상세 페이지는 도구 접근 오류가 있었다.
  콜리네이트의 공식 외형·각인 정보는 이번 조회에서 확보하지 못했다.
  해당 두 제품 및 플라빅스 각인은 정확한 국내 품목·제조시점 정보로 추가 확인해야 한다.
  해외 동일 성분 제품 정보를 국내 제품의 확정 근거로 대체하지 않았다.
- 원본 JSON의 각인·색상·모양은 데이터셋 주장이다. 공식 외부 정답으로 취급하지 않았다.
  OCR·분류 모델을 별도로 실행하거나 그 결과로 라벨을 수정하지 않았다.

## 다음 한 변수 실험 추천

**클래스 불균형 개선: train 이미지 샘플링만 변경**하는 비교를 우선 추천한다.
자누비아3개·트윈스타2개·트라젠타듀오5개이고, 위치가 맞는 오분류가 다수인 점이 근거다.
구체적 초안은 클래스별 train 라벨 수 n의 중앙값6.5를 기준으로
클래스 가중치 min(3, max(1, sqrt(6.5/n)))를 만들고,
다중 알약 사진은 포함 클래스의 최대 가중치를 사용해 이미지 단위 복원 추출하는 방식이다.
seed42, epoch20, imgsz640, batch8, 초기 사전학습 가중치와 분할·증강·optimizer 설정은 유지한다.
epoch당 학습 배치 수20도 유지한다. YOLO 기본 shuffle에 이 가중치가 자동 적용된다고 주장하지 않으며,
실험 전 sampler 구현과 실제 클래스별 노출 수를 별도 확인해야 한다.

단, 복제 샘플링으로 새로운 각인 면이 생기지는 않는다.
트라젠타듀오·플라빅스의 보이는 면 차이는
독립 촬영과 양면 데이터 확보로 해소해야 할 별도 데이터 과제다.
검수 없이 표본을 반복하면 오라벨도 증폭하므로 사람 검수 후 실험을 승인해야 한다.
새 데이터를 넣는 실험은 분리하고 기존 validation을 train으로 이동하거나 그룹을 쪼개지 않는다.

다른 후보의 우선순위를 낮춘 이유:

학습 곡선의 mAP50은 epoch10=0.57224, epoch15=0.70281, epoch20=0.73730이었다.
같은 지점의 validation cls loss는2.3617→1.99877→1.80229로 감소했다.
이는 추가 학습 가능성을 남기지만 희소 클래스의 원인이나 일반화 성능을 확정해 주지 않는다.

- Epoch 증가: 전체 mAP50이 epoch15~20에서도 상승해 가능성은 있다. 다만 같은2~5표본 반복으로
  분류·가시적 면 부족을 해결한다고 보장할 수 없다.
  마지막 epoch가 최적이면 수렴 미완료 가능성으로 남긴다.
- Image size 증가: 각인 해상도에는 도움 가능하지만 현재5종은 conf0.05에서 위치 매칭이 이미25/25다.
  주된 확인 증거는 위치보다 제품 클래스 문제다. 해상도 효과를 배제하는 결론은 아니다.
- 증강 변경: 반복 회전·조명 사진이 이미 있고 train에 없는 반대편 각인은 회전으로 만들 수 없다.
  강한 블러·색변환·뒤집기가 식별을 해칠 수 있어 먼저 적용하지 않는다.

비교 기준은 동일 IoU0.50에서 conf0.25 고정 micro P/R/F1, 같은 per-class AP50,
5종 TP/FP/FN과 실패 유형이다. 최소 Precision을 합의한 뒤 조건부 Recall도 확인한다.
validation으로 임계값을 골랐으므로 최종 성능 주장은 별도의 독립 test에 남긴다.

## 결과 파일과 재현

- 공개 집계: `docs/{assets}/` 아래 threshold 전체·클래스별 CSV, 지원 수, 혼동 쌍,
  confidence 요약, 실패 유형, 집계 차트, 학습 곡선 체크포인트.
- 로컬 원본 증거: `analysis_manifest.json`, `gt_outcomes.csv`, `prediction_outcomes.csv`,
  `label_review.csv`, contact sheet5개와 인덱스, 사례 이미지25개, `review_gallery.html`.
- 입력 best.pt·기존 metrics/predictions·학습 곡선·data.yaml·split manifest·class map은 실행 전후
  SHA256이 일치했다. 원본58JSON도 준비 시점의 SHA256과 대조했다.
- 코드: `models/analyze_recall.py`, `models/export_recall_report.py`.
  학습 호출은 없으며 기존 모델·기존 평가·원본 이미지·JSON을 수정하지 않았다.

재실행 시 환경변수에 실제 경로와 **존재하지 않는 새 출력 폴더**를 지정한다.

```powershell
uv run --no-project --python 3.14 python -m models.analyze_recall `
  --run-dir $env:PILL_RUN_DIR --dataset $env:PILL_YOLO_DATA_DIR `
  --raw-root $env:PILL_INTERNAL_DATA_DIR --output-dir $env:PILL_RECALL_DIR
uv run --no-project --python 3.14 python -m models.export_recall_report `
  --analysis $env:PILL_RECALL_DIR --run-dir $env:PILL_RUN_DIR `
  --destination $env:PILL_RECALL_PUBLIC_DIR
```
"""
    (PROJECT_ROOT / "docs/RECALL_ANALYSIS.md").write_text(report, encoding="utf-8")
    # 로컬 갤러리는 사진을 바로 표시하고 공개 보고서에는 원본 사진을 넣지 않는다.
    gallery = analysis / "review_gallery.html"
    page = gallery.read_text(encoding="utf-8")
    contacts = "<h2>Train/Validation 전체 GT 비교 · 에이전트 관찰 / 사람 미승인</h2>" + "".join(
        f'<h3>{names[class_id]}</h3><img src="contact_class{class_id}.jpg">' for class_id in FOCUS
    )
    page = page.split("<h2>Train/Validation 전체 GT 비교", maxsplit=1)[0]
    page = page.removesuffix("</html>") + contacts + "</html>"
    gallery.write_text(page, encoding="utf-8")
    (destination / "export_manifest.json").write_text(
        json.dumps(
            {
                "analysis_manifest_sha256": digest(analysis / "analysis_manifest.json"),
                "export_code_sha256": digest(Path(__file__)),
                "artifacts": {
                    path.name: digest(path)
                    for path in destination.iterdir()
                    if path.is_file() and path.name != "export_manifest.json"
                },
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(str(PROJECT_ROOT / "docs/RECALL_ANALYSIS.md"))


def main() -> None:
    """로컬 분석 결과와 기존 학습 경로를 받아 보고서만 생성한다."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analysis", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument(
        "--refresh",
        action="store_true",
        help="이 도구로 만든 집계 파일의 해시가 같을 때만 보고서를 갱신한다",
    )
    parser.add_argument(
        "--destination", type=Path, default=PROJECT_ROOT / "docs/results/recall_20261105"
    )
    args = parser.parse_args()
    export(
        args.analysis.resolve(), args.run_dir.resolve(), args.destination.resolve(), args.refresh
    )


if __name__ == "__main__":
    main()

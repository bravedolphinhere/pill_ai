"""실제 실행 결과만 모아 제출용 보고서와 공개 가능한 지표·차트를 만든다."""

from __future__ import annotations

import argparse
import hashlib
import html
import importlib.metadata
import json
import shutil
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from matplotlib import font_manager

from src.config import PROJECT_ROOT


def file_hash(path: Path) -> str:
    """보관한 실제 결과와 코드의 SHA256을 기록한다."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def compact_matrix(matrix: np.ndarray, measured: list[int], output: Path) -> None:
    """미평가 클래스 예측을 합쳐 전체 오류 수를 보존한 읽기 쉬운 혼동행렬을 만든다."""
    if any(font.name == "Malgun Gothic" for font in font_manager.fontManager.ttflist):
        plt.rcParams["font.family"] = "Malgun Gothic"
    class_count = matrix.shape[0] - 1
    others = [index for index in range(class_count) if index not in measured]
    blocks = [[index] for index in measured] + [others, [class_count]]
    reduced = np.array(
        [[matrix[np.ix_(rows, columns)].sum() for columns in blocks] for rows in blocks]
    )
    assert reduced.sum() == matrix.sum()
    labels = [str(index) for index in measured] + ["기타 예측", "배경"]
    figure, axis = plt.subplots(figsize=(13, 11), constrained_layout=True)
    displayed = axis.imshow(reduced, cmap="Blues")
    axis.set_xticks(range(len(labels)), labels, rotation=90, fontsize=8)
    axis.set_yticks(range(len(labels)), labels, fontsize=8)
    axis.set_xlabel("정답 클래스 (YOLO ID)")
    axis.set_ylabel("예측 클래스 (YOLO ID)")
    axis.set_title("검증 혼동행렬 · 미평가 클래스의 예측은 기타로 합침 · 개수")
    for row, column in zip(*np.nonzero(reduced), strict=True):
        axis.text(
            column,
            row,
            str(int(reduced[row, column])),
            ha="center",
            va="center",
            fontsize=7,
            color="white" if reduced[row, column] > reduced.max() / 2 else "black",
        )
    figure.colorbar(displayed, ax=axis, shrink=0.75)
    figure.savefig(output, dpi=140)
    plt.close(figure)


def export_report(run: Path, dataset: Path, destination: Path) -> dict:
    """가중치·사진은 로컬에 두고 메트릭·집계 차트만 저장소에 복사한다."""
    run, dataset, destination = run.resolve(), dataset.resolve(), destination.resolve()
    result = json.loads((run / "training_result.json").read_text(encoding="utf-8"))
    metrics = json.loads((run / "evaluation/metrics.json").read_text(encoding="utf-8"))

    history = pd.read_csv(run / "train_batch8/results.csv")
    per_class = pd.read_csv(run / "evaluation/per_class_metrics.csv")
    assert len(history) == 20 and history.epoch.iloc[-1] == 20
    assert result["effective_batch"] == 8
    assert file_hash(run / "best.pt") == metrics["weights_sha256"]
    assert per_class.evaluation_status.eq("MEASURED").sum() == 29
    assert per_class.validation_instances.sum() == 132
    for key in ["mAP50", "mAP50-95", "precision", "recall"]:
        assert 0 <= metrics[key] <= 1
    destination.mkdir(parents=True, exist_ok=True)
    for source, target in [
        (run / "train_batch8/results.csv", "training_history.csv"),
        (run / "train_batch8/results.png", "training_curves.png"),
        (run / "evaluation/per_class_metrics.csv", "per_class_metrics.csv"),
        (run / "evaluation/validation/confusion_matrix.png", "confusion_matrix_full.png"),
        (dataset / "class_map.csv", "class_map.csv"),
    ]:
        shutil.copy2(source, destination / target)
    matrix = json.loads(
        (run / "evaluation/confusion_matrix_values.json").read_text(encoding="utf-8")
    )
    measured = (
        per_class.loc[per_class.evaluation_status.eq("MEASURED"), "yolo_class_id"]
        .astype(int)
        .tolist()
    )
    compact_matrix(
        np.asarray(matrix["matrix"]), measured, destination / "confusion_matrix_compact.png"
    )
    public_metrics = {
        key: value for key, value in metrics.items() if key not in {"weights", "data"}
    }
    public_metrics["metric_implementation"] = "Ultralytics native metrics.box"
    public_metrics["epochs_completed"] = len(history)
    public_metrics["effective_batch"] = result["effective_batch"]
    public_metrics["epoch_train_val_elapsed_seconds"] = float(history.time.iloc[-1])
    (destination / "metrics.json").write_text(
        json.dumps(public_metrics, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    arguments = yaml.safe_load((run / "train_batch8/args.yaml").read_text(encoding="utf-8"))
    arguments["data"] = "${PILL_YOLO_DATA_DIR}/data.yaml"
    arguments["project"] = "${PILL_RUN_DIR}"
    arguments["save_dir"] = "${PILL_RUN_DIR}/train_batch8"
    (destination / "training_config.yaml").write_text(
        yaml.safe_dump(arguments, allow_unicode=True, sort_keys=False), encoding="utf-8"
    )
    versions = {
        name: importlib.metadata.version(name)
        for name in [
            "torch",
            "torchvision",
            "ultralytics",
            "numpy",
            "pandas",
            "Pillow",
            "scipy",
            "faster-coco-eval",
        ]
    }
    provenance = {
        "run_id": run.name,
        "packages": versions,
        "weights_sha256": file_hash(run / "best.pt"),
        "training_request_sha256": file_hash(run / "training_request.json"),
        "class_map_sha256": file_hash(dataset / "class_map.csv"),
        "split_manifest_sha256": file_hash(dataset / "split_manifest.csv"),
        "source_modified": False,
        "code_sha256": {
            name: file_hash(PROJECT_ROOT / name)
            for name in [
                "models/train.py",
                "models/evaluate.py",
                "data_process/group_split.py",
                "data_process/prepare_internal_baseline.py",
                "data_process/verify_internal_dataset.py",
            ]
        },
    }
    (destination / "provenance.json").write_text(
        json.dumps(provenance, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    metric_rows = "\n".join(
        f"| {name} | {metrics[key]:.6f} | {metrics[key] * 100:.2f}% |"
        for name, key in [
            ("mAP50", "mAP50"),
            ("mAP50-95", "mAP50-95"),
            ("Precision", "precision"),
            ("Recall", "recall"),
        ]
    )
    lowest = per_class.loc[per_class.evaluation_status.eq("MEASURED")].sort_values("mAP50").head(5)
    low_rows = "\n".join(
        f"| {row.pill_name} | {row.validation_instances} | {row.mAP50:.4f} |"
        for row in lowest.itertuples()
    )
    assets = destination.relative_to(PROJECT_ROOT / "docs").as_posix()
    report = f"""# 내부 알약 객체 탐지 YOLO11n 베이스라인 최종 보고서

작성일: 2026-11-05. 상태: **사전학습 기반 20epoch 학습 및 best.pt 평가 완료**.

## 결과와 평가 범위

| 지표 | 값 (0~1) | 백분율 표기 |
|---|---:|---:|
{metric_rows}

검증 사진39장·정답132개·29클래스에 대한 개발용 validation 결과다.
학습 모델은56클래스지만 검증에 없는27클래스는 미평가다.
위 값은 전체56종의 독립 test 성능이나 약품 정체의 확정률이 아니다.
기준 지표는 `evaluation/metrics.json`의 Ultralytics native `metrics.box`다.
JSON 내보내기와 함께 출력된 COCO evaluator AP는 구현·집계가 다른 별도 값이므로 혼용하지 않았다.
Precision/Recall은 Ultralytics 요약 지표이며
예측 그림의 기본 conf=0.25와 동일한 운영점을 보장하지 않는다.

## 데이터와 QA

내부 학습 후보232사진·763라벨·56클래스를 사용했다. 외부 데이터는 학습에 넣지 않았다.
구조 오류와 기존 검수 보류를 사진 단위로 합쳐35사진·103라벨을 REVIEW로 제외했다.
PASS197사진·660라벨·56클래스가 남았고, 원본 이미지·JSON은 수정하지 않았다.
원본232사진·763JSON의 지문과 복사197사진, 전체660라벨의 좌표 역변환 검증을 통과했다.

| 분할 | 사진 | 라벨 | 클래스 |
|---|---:|---:|---:|
| train | 158 | 528 | 56 |
| validation | 39 | 132 | 29 |

기존 촬영·유사 이미지 그룹73개를 쪼개지 않고 train의 모든 클래스를 보존한 채
사진 수80:20 근접도를 우선해 정수 최적화했다. 실제 비율은80.20:19.80이다.
정의된 촬영·SHA256·pHash 관계에서 split 누수는0건이었다.
모든 시각적 유사성에 대한 완전한 중복 제거를 뜻하지는 않는다.
PASS 감사 표본56어노테이션/49사진은 사람 검수 대기 상태이며 자동 PASS는 의미상 정확성 보증이 아니다.
정답 없는 test 이미지842장은 이번 정량 평가에 사용하지 않았다.

## 학습 설정과 실행 환경

| 항목 | 실제 실행 |
|---|---|
| 모델 | YOLO11n, yolo11n.pt 사전학습 초기화 |
| epochs / image size / batch / seed | 20 / 640 / 8 / 42 |
| GPU | NVIDIA GeForce RTX 5070 약12GB |
| Python / torch / torchvision | 3.14.2 / {versions["torch"]} / {versions["torchvision"]} |
| Ultralytics | {versions["ultralytics"]} |
| optimizer | auto, 로그에서 AdamW 선택 |
| dataloader / cache | workers0 / cache=False |
| AMP / deterministic | True / True |
| epoch별 train+val 누적 시간 | {history.time.iloc[-1]:.4f}초 (초기 다운로드·설정 시간 제외) |
| GPU 메모리 | 학습 로그에서 약1.5GB 관측, 최대값 별도 측정은 하지 않음 |
| OOM / batch 변경 | 없음 / 8 유지 |

기존 Python3.12/cu126은 GPU 인식 후 실제 연산에 실패했다.
기본 연산과 YOLO forward를 통과한 Python3.14/cu130으로 실제 학습까지 완료했다.
AMP 확인용으로 라이브러리가 내려받은 YOLO26n은 이 실험의 학습 모델이 아니다.
평가 JSON 내보내기에 사용된 faster-coco-eval {versions["faster-coco-eval"]}도 의존성에 기록했다.

## 학습 경향

![학습 곡선]({assets}/training_curves.png)

20epoch 동안 학습·검증 loss가 전반적으로 감소하고 mAP가 상승했다.
epoch11부터 기본 close_mosaic=10 설정에 따라 mosaic이 꺼져 학습 조건 변화가 포함된다.
단일 seed·작은 검증셋의 결과이므로 일반화 성능이나 과적합 부재를 확정하지 않는다.
추가 epoch·하이퍼파라미터 탐색은 이번 승인 범위에서 실행하지 않았다.

## 혼동행렬과 낮은 성능의 클래스

![검증 혼동행렬]({assets}/confusion_matrix_compact.png)

축의 숫자는 YOLO class ID이며 [전체 매핑]({assets}/class_map.csv)에 이름·제품 코드가 있다.
미평가27클래스의 예측을 기타 행으로 합쳐 오류 수를 보존했다. 원본 전체 행렬도 별도 보관했다.
이 행렬은 검증 기본 confidence=0.001, 매칭 IoU=0.45 기준이다.
낮은 confidence의 배경 오탐도 포함하므로
conf=0.25 예측 그림이나 요약 Precision과 직접 비교하지 않는다.

| 검증 라벨 기준 클래스 | 정답 수 | AP50 |
|---|---:|---:|
{low_rows}

클래스마다 표본이1~24개로 작고 편중돼 있다. 낮은 점수의 원인을 사진·오라벨·혼동·학습 부족으로
분리해 확인해야 하며, 모델 예측이나 원본 라벨만으로 실제 약품 정체를 판정하지 않는다.
미평가 클래스의 지표를0점으로 채우지 않고 NOT_EVALUATED로 보존했다.

## 저장 결과와 재현

- 로컬 가중치: `{run.name}/best.pt`, SHA256 `{metrics["weights_sha256"]}`.
- 로컬 평가: `evaluation/metrics.json`, `per_class_metrics.csv`, `validation/`, `predictions/`.
- 예측16사진은 검증 파일명 순 첫16장으로 정했으며 좋은 결과만 골라 보여 주지 않았다.
- GitHub에는 집계 지표·차트·코드·이미지 미포함 EDA를 제공한다.
  원본 사진·예측 사진·가중치는 제외한다.
- [지표 JSON]({assets}/metrics.json), [클래스별 지표]({assets}/per_class_metrics.csv),
  [학습 설정]({assets}/training_config.yaml), [재현 지문]({assets}/provenance.json).
- 실행 방법은 [README](../README.md), 준비 과정은 [사전 검사 기록](YOLO11N_PREFLIGHT.md)을 따른다.

seed와 환경 버전을 기록했지만 다른 하드웨어·패키지 버전에서 비트 단위 재현을 보장하지 않는다.
학습에 쓴 validation으로 best checkpoint를 선택했으므로 독립적인 최종 test가 추가로 필요하다.
"""
    (PROJECT_ROOT / "docs/FINAL_REPORT.md").write_text(report, encoding="utf-8")
    predictions = sorted((run / "evaluation/predictions").glob("*.jpg"))
    tiles = "".join(
        f'<figure><img src="evaluation/predictions/{html.escape(path.name)}">'
        f"<figcaption>{html.escape(path.name)}</figcaption></figure>"
        for path in predictions
    )
    html_metrics = "".join(
        f"<tr><td>{key}</td><td>{metrics[key]:.6f}</td></tr>"
        for key in ["mAP50", "mAP50-95", "precision", "recall"]
    )
    local_html = f"""<!doctype html><html lang="ko"><meta charset="utf-8">
<title>YOLO11n 학습 결과</title>
<style>body{{font-family:'Malgun Gothic',sans-serif;max-width:1500px;margin:30px auto;padding:20px}}
img{{max-width:100%}}.gallery{{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:12px}}
figure{{margin:0}}figcaption{{overflow-wrap:anywhere;font-size:12px}}table{{border-collapse:collapse}}
td,th{{border:1px solid #ccd;padding:8px}}</style>
<h1>YOLO11n 20epoch 결과</h1>
<p>train158장 / validation39장 · 모델56종 중 검증29종 · 독립 test 아님</p>
<table><tr><th>지표</th><th>값</th></tr>{html_metrics}</table>
<p>epoch별 train+val 누적 {history.time.iloc[-1]:.2f}초 · batch8 · seed42</p>
<h2>학습 곡선</h2><img src="train_batch8/results.png">
<h2>Confusion Matrix</h2><img src="evaluation/validation/confusion_matrix_normalized.png">
<h2>검증 예측 예시16장 · 파일명 순</h2><div class="gallery">{tiles}</div></html>"""
    (run / "report.html").write_text(local_html, encoding="utf-8")
    manifest = [
        {
            "file": path.relative_to(destination).as_posix(),
            "sha256": file_hash(path),
            "bytes": path.stat().st_size,
        }
        for path in sorted(destination.iterdir())
        if path.is_file() and path.name != "artifact_manifest.json"
    ]
    (destination / "artifact_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return {
        "metrics": public_metrics,
        "report": str(PROJECT_ROOT / "docs/FINAL_REPORT.md"),
        "local_gallery": str(run / "report.html"),
        "prediction_images": len(predictions),
    }


def main() -> None:
    """기존 실행 결과로 보고서를 만들고 추가 학습은 수행하지 않는다."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument(
        "--destination", type=Path, default=PROJECT_ROOT / "docs/results/yolo11n_20261105"
    )
    args = parser.parse_args()
    print(
        json.dumps(
            export_report(args.run_dir, args.dataset, args.destination),
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

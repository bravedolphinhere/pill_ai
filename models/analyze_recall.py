"""기존 검증 예측을 재사용해 임계값과 저성능 클래스의 실패 양상을 분석한다."""

from __future__ import annotations

import argparse
import hashlib
import html
import importlib.metadata
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import font_manager
from PIL import Image

from src.config import PROJECT_ROOT

FOCUS = [30, 44, 20, 36, 25]
THRESHOLDS = [0.05, 0.10, 0.15, 0.20, 0.25]


def digest(path: Path) -> str:
    """입력과 실행 코드가 바뀌었는지 확인할 지문을 만든다."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def overlap(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    """픽셀 xyxy 박스 사이 IoU를 계산한다."""
    area_first = np.maximum(first[:, 2:] - first[:, :2], 0).prod(1)
    area_second = np.maximum(second[:, 2:] - second[:, :2], 0).prod(1)
    intersection = np.maximum(
        np.minimum(first[:, None, 2:], second[None, :, 2:])
        - np.maximum(first[:, None, :2], second[None, :, :2]),
        0,
    ).prod(2)
    return intersection / np.maximum(area_first[:, None] + area_second - intersection, 1e-12)


def match_boxes(
    truth: list[dict], predictions: list[dict], minimum: float, same_class: bool
) -> list[tuple[int, int, float]]:
    """신뢰도 순 예측을 아직 연결하지 않은 GT와 일대일로 연결한다."""
    if not truth or not predictions:
        return []
    scores = overlap(
        np.array([row["bbox"] for row in truth]),
        np.array([row["bbox"] for row in predictions]),
    )
    used = set()
    matches = []
    for pred_index in sorted(range(len(predictions)), key=lambda i: -predictions[i]["confidence"]):
        candidates = [
            index
            for index in range(len(truth))
            if index not in used
            and scores[index, pred_index] >= minimum
            and (not same_class or truth[index]["class_id"] == predictions[pred_index]["class_id"])
        ]
        if candidates:
            gt_index = max(candidates, key=lambda index: (scores[index, pred_index], -index))
            used.add(gt_index)
            matches.append((gt_index, pred_index, float(scores[gt_index, pred_index])))
    return matches


def self_check() -> None:
    """중복·오분류를 손으로 계산 가능한 작은 입력에서 확인한다."""
    truth = [{"class_id": 0, "bbox": [0, 0, 10, 10]}]
    predictions = [
        {"class_id": 1, "bbox": [0, 0, 10, 10], "confidence": 0.9},
        {"class_id": 0, "bbox": [0, 0, 10, 10], "confidence": 0.8},
        {"class_id": 0, "bbox": [0, 0, 10, 10], "confidence": 0.7},
    ]
    assert match_boxes(truth, predictions, 0.5, True) == [(0, 1, 1.0)]
    assert match_boxes(truth, predictions[:1], 0.5, True) == []
    assert match_boxes(truth, predictions[:1], 0.5, False) == [(0, 0, 1.0)]
    assert match_boxes([], predictions, 0.5, True) == []
    partial = [{"class_id": 0, "bbox": [0, 0, 5, 10], "confidence": 0.5}]
    assert match_boxes(truth, partial, 0.5, True) == [(0, 0, 0.5)]
    shifted = [{"class_id": 0, "bbox": [6, 0, 16, 10], "confidence": 0.5}]
    assert match_boxes(truth, shifted, 0.5, True) == []
    assert len(match_boxes(truth, shifted, 0.1, True)) == 1


def read_data(dataset: Path) -> tuple[pd.DataFrame, pd.DataFrame, dict, dict]:
    """검증된 YOLO 라벨에서 표본 수와 픽셀 GT를 가져온다."""
    mapping = pd.read_csv(dataset / "class_map.csv")
    manifest = pd.read_csv(dataset / "split_manifest.csv")
    truth, counts = {}, []
    for row in manifest.itertuples():
        image_path = dataset / "images" / row.split / row.file_name
        with Image.open(image_path) as image:
            width, height = image.size
        rows = []
        label_path = dataset / "labels" / row.split / Path(row.file_name).with_suffix(".txt")
        for index, line in enumerate(label_path.read_text(encoding="utf-8").splitlines()):
            class_id, center_x, center_y, box_width, box_height = map(float, line.split())
            rows.append(
                {
                    "class_id": int(class_id),
                    "gt_index": index,
                    "bbox": [
                        (center_x - box_width / 2) * width,
                        (center_y - box_height / 2) * height,
                        (center_x + box_width / 2) * width,
                        (center_y + box_height / 2) * height,
                    ],
                }
            )
            counts.append(
                {
                    "class_id": int(class_id),
                    "split": row.split,
                    "file_name": row.file_name,
                    "group": row.split_group,
                }
            )
        if row.split == "val":
            truth[row.file_name] = rows
    return (
        mapping,
        pd.DataFrame(counts),
        truth,
        dict(zip(mapping.yolo_class_id, mapping.pill_name, strict=True)),
    )


def predictions_from_json(path: Path, files: set[str]) -> dict:
    """custom COCO JSON의 1부터 시작하는 category_id를 YOLO ID로 되돌린다."""
    predictions = {name: [] for name in files}
    for index, row in enumerate(json.loads(path.read_text(encoding="utf-8"))):
        assert row["file_name"] in files and 1 <= row["category_id"] <= 56
        left, top, width, height = row["bbox"]
        predictions[row["file_name"]].append(
            {
                "prediction_id": index,
                "class_id": row["category_id"] - 1,
                "confidence": row["score"],
                "bbox": [left, top, left + width, top + height],
            }
        )
    return predictions


def score_threshold(truth: dict, cached: dict, threshold: float, names: dict) -> tuple:
    """클래스 일치 정량 평가와 남은 GT의 실패 진단을 구분한다."""
    counts = {class_id: {"TP": 0, "FP": 0, "FN": 0} for class_id in names}
    evidence, pred_rows = [], []
    for filename, targets in truth.items():
        predictions = [row for row in cached[filename] if row["confidence"] >= threshold]
        matches = match_boxes(targets, predictions, 0.5, True)
        gt_used = {row[0] for row in matches}
        pred_used = {row[1] for row in matches}
        diagnostics = {g: ("TP", p, iou) for g, p, iou in matches}
        for class_agnostic, minimum, failure in [
            (True, 0.5, "Classification Failure"),
            (False, 0.1, "Localization Failure"),
        ]:
            remaining_gt = [i for i in range(len(targets)) if i not in diagnostics]
            remaining_pred = [i for i in range(len(predictions)) if i not in pred_used]
            extra = match_boxes(
                [targets[i] for i in remaining_gt],
                [predictions[i] for i in remaining_pred],
                minimum,
                not class_agnostic,
            )
            for gt_index, pred_index, iou in extra:
                actual_gt, actual_pred = remaining_gt[gt_index], remaining_pred[pred_index]
                diagnostics[actual_gt] = (failure, actual_pred, iou)
                pred_used.add(actual_pred)
        quantitative_pred = {row[1] for row in matches}
        for index, pred in enumerate(predictions):
            status = "TP" if index in quantitative_pred else "FP"
            counts[pred["class_id"]][status] += 1
            pred_rows.append(
                {"threshold": threshold, "file_name": filename, **pred, "status": status}
            )
        for index, target in enumerate(targets):
            status = "TP" if index in gt_used else "FN"
            counts[target["class_id"]]["FN"] += int(status == "FN")
            failure, pred_index, iou = diagnostics.get(index, ("Detection Failure", None, 0.0))
            linked = predictions[pred_index] if pred_index is not None else {}
            same = [
                row
                for row in cached[filename]
                if row["class_id"] == target["class_id"]
                and overlap(np.array([target["bbox"]]), np.array([row["bbox"]]))[0, 0] >= 0.5
            ]
            evidence.append(
                {
                    "threshold": threshold,
                    "file_name": filename,
                    **target,
                    "pill_name": names[target["class_id"]],
                    "status": status,
                    "failure": failure,
                    "matched_iou": iou,
                    "predicted_class": linked.get("class_id"),
                    "predicted_name": names.get(linked.get("class_id"), ""),
                    "confidence": linked.get("confidence"),
                    "prediction_bbox": linked.get("bbox"),
                    "best_correct_class_confidence_at_iou50": max(
                        [row["confidence"] for row in same], default=0.0
                    ),
                    "label_review": "REVIEW" if target["class_id"] in FOCUS else "UNVERIFIED",
                    "human_decision": "PENDING",
                }
            )
    rows = []
    for class_id, values in counts.items():
        rows.append(
            {
                "threshold": threshold,
                "class_id": class_id,
                "pill_name": names[class_id],
                **values,
                **rates(values),
            }
        )
    totals = {key: sum(row[key] for row in rows) for key in ["TP", "FP", "FN"]}
    assert totals["TP"] + totals["FN"] == sum(map(len, truth.values()))
    return {"threshold": threshold, **totals, **rates(totals)}, rows, evidence, pred_rows


def rates(counts: dict) -> dict:
    """분모가 0이면 미정의로 남기고 전체 지표는 카운트의 micro 집계를 쓴다."""
    true, false, missed = (counts[key] for key in ["TP", "FP", "FN"])
    return {
        "precision": true / (true + false) if true + false else None,
        "recall": true / (true + missed) if true + missed else None,
        "f1": 2 * true / (2 * true + false + missed) if 2 * true + false + missed else None,
    }


def draw_box(axis: plt.Axes, bbox: list, color: str, label: str) -> None:
    """GT와 예측 좌표·이름을 다른 색으로 겹쳐 표시한다."""
    left, top, right, bottom = bbox
    axis.add_patch(
        plt.Rectangle(
            (left, top), right - left, bottom - top, fill=False, edgecolor=color, linewidth=2
        )
    )
    axis.text(
        left,
        top,
        label,
        fontsize=7,
        color="black",
        va="bottom",
        bbox={"facecolor": color, "alpha": 0.75, "pad": 1},
    )


def make_gallery(
    output: Path,
    dataset: Path,
    truth: dict,
    cached: dict,
    evidence: pd.DataFrame,
    names: dict,
    metadata: pd.DataFrame,
) -> None:
    """동일 GT의 원본과 임계값별 예측을 가로로 배치해 모든 우선 검수 사례를 보여 준다."""
    (output / "gallery").mkdir()
    sections = []
    focus_rows = evidence.loc[(evidence.threshold == 0.25) & evidence.class_id.isin(FOCUS)]
    for class_id in FOCUS:
        cards = []
        selected = focus_rows.loc[focus_rows.class_id == class_id]
        for row in selected.itertuples():
            image_path = dataset / "images/val" / row.file_name
            with Image.open(image_path) as source:
                image = source.convert("RGB")
            left, top, right, bottom = row.bbox
            margin = max(right - left, bottom - top) * 0.35
            figure, axes = plt.subplots(1, 3, figsize=(15, 5), constrained_layout=True)
            for axis, threshold in zip(axes, [None, 0.05, 0.25], strict=True):
                axis.imshow(image)
                draw_box(axis, row.bbox, "#2ecc71", f"GT {class_id}: {names[class_id]}")
                if threshold is not None:
                    for pred in cached[row.file_name]:
                        iou = overlap(np.array([row.bbox]), np.array([pred["bbox"]]))[0, 0]
                        if pred["confidence"] >= threshold and iou > 0.1:
                            draw_box(
                                axis,
                                pred["bbox"],
                                "#ff6961",
                                f"예측 {pred['class_id']}: {names[pred['class_id']]} "
                                f"{pred['confidence']:.3f}",
                            )
                axis.set_xlim(max(0, left - margin), min(image.width, right + margin))
                axis.set_ylim(min(image.height, bottom + margin), max(0, top - margin))
                axis.set_title("GT 확대" if threshold is None else f"GT + 예측 conf≥{threshold}")
                axis.axis("off")
            figure.suptitle(
                f"{row.pill_name} | conf0.25: {row.failure}\n{row.file_name}", fontsize=10
            )
            target = f"class{class_id}_gt{row.gt_index}_{Path(row.file_name).stem}.jpg"
            figure.savefig(output / "gallery" / target, dpi=120)
            plt.close(figure)
            source = metadata.loc[
                (metadata.file_name == row.file_name) & (metadata.class_id == class_id)
            ].iloc[0]
            cards.append(
                f'<article><img src="gallery/{target}"><p>'
                f"{html.escape(row.file_name)} · GT {row.bbox}<br>"
                f"{html.escape(source.product_code)} · {html.escape(row.failure)}<br>"
                f"JSON: {html.escape(source.source_json)}<br>"
                "자동 REVIEW / 사람 PENDING · 예측은 정체의 근거가 아님</p></article>"
            )
        sections.append(f"<h2>{html.escape(names[class_id])}</h2>" + "".join(cards))
    page = (
        '<!doctype html><html lang="ko"><meta charset="utf-8"><title>Recall 실패 검수</title>'
        "<style>body{font-family:Malgun Gothic,sans-serif;max-width:1600px;margin:auto}"
        "img{width:100%}p{overflow-wrap:anywhere}article{border:1px solid #ddd;margin:20px 0}"
        "</style><h1>저성능 5클래스 검수</h1><p>초록:GT / 빨강:예측. "
        "동일 알약을 GT·conf0.05·conf0.25 순으로 비교. "
        "여러 클래스 예측이 겹칠 수 있는 원래 validation NMS 결과다.</p>"
        + "".join(sections)
        + "</html>"
    )
    (output / "review_gallery.html").write_text(page, encoding="utf-8")


def make_contacts(output: Path, dataset: Path, metadata: pd.DataFrame, names: dict) -> None:
    """저성능 클래스의 학습·검증 GT를 모두 한 화면에서 비교하도록 저장한다."""
    manifest = pd.read_csv(dataset / "split_manifest.csv").set_index("file_name")
    for class_id in FOCUS:
        selected = metadata.loc[(metadata.class_id == class_id) & (metadata.status == "PASS")]
        selected = selected.copy()
        selected["split"] = selected.file_name.map(manifest.split)
        selected = selected.sort_values(["split", "file_name"])
        figure, axes = plt.subplots(
            (len(selected) + 3) // 4,
            4,
            figsize=(14, 3.5 * ((len(selected) + 3) // 4)),
            squeeze=False,
            constrained_layout=True,
        )
        for axis in axes.flat:
            axis.axis("off")
        for index, (axis, row) in enumerate(zip(axes.flat, selected.itertuples(), strict=False)):
            with Image.open(dataset / "images" / row.split / row.file_name) as source:
                image = source.convert("RGB")
            left, top, width, height = json.loads(row.bbox)
            margin = max(width, height) * 0.2
            axis.imshow(image)
            draw_box(axis, [left, top, left + width, top + height], "#2ecc71", "GT")
            axis.set_xlim(max(0, left - margin), min(image.width, left + width + margin))
            axis.set_ylim(min(image.height, top + height + margin), max(0, top - margin))
            axis.set_title(f"{index}: {row.split} · {Path(row.file_name).stem[-18:]}", fontsize=9)
        figure.suptitle(f"{names[class_id]} · 전체 PASS GT · 정체는 미확정", fontsize=14)
        figure.savefig(output / f"contact_class{class_id}.jpg", dpi=120)
        plt.close(figure)
        selected.to_csv(
            output / f"contact_class{class_id}_index.csv", index=False, encoding="utf-8-sig"
        )


def analyze(run: Path, dataset: Path, raw_root: Path, output: Path) -> None:
    """기존 모델과 원본을 수정하지 않고 새 결과 폴더에만 분석을 기록한다."""
    self_check()
    if output.exists():
        raise FileExistsError("기존 결과를 보존하도록 새 출력 폴더를 지정하세요.")
    inputs = [
        run / "best.pt",
        run / "evaluation/metrics.json",
        run / "evaluation/validation/predictions.json",
        run / "train_batch8/results.csv",
        dataset / "data.yaml",
        dataset / "split_manifest.csv",
        dataset / "class_map.csv",
    ]
    before = {str(path): digest(path) for path in inputs}
    assert (
        before[str(run / "best.pt")]
        == json.loads((run / "evaluation/metrics.json").read_text(encoding="utf-8"))[
            "weights_sha256"
        ]
    )
    mapping, counts, truth, names = read_data(dataset)
    for class_id, prefix in zip(
        FOCUS,
        ["자누비아정", "트라젠타듀오정", "콜리네이트", "트윈스타정", "플라빅스정"],
        strict=True,
    ):
        assert names[class_id].startswith(prefix)
    assert len(truth) == 39 and sum(map(len, truth.values())) == 132
    cached = predictions_from_json(inputs[2], set(truth))
    overall, classes, evidence, prediction_rows = [], [], [], []
    for threshold in THRESHOLDS:
        total, per_class, items, preds = score_threshold(truth, cached, threshold, names)
        overall.append(total)
        classes.extend(per_class)
        evidence.extend(items)
        prediction_rows.extend(preds)
    output.mkdir(parents=True)
    (output / "analysis_code_snapshot.py").write_bytes(Path(__file__).read_bytes())
    overall = pd.DataFrame(overall)
    classes, evidence = pd.DataFrame(classes), pd.DataFrame(evidence)
    overall.to_csv(output / "threshold_overall.csv", index=False, encoding="utf-8-sig")
    classes.to_csv(output / "threshold_per_class.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(prediction_rows).to_csv(
        output / "prediction_outcomes.csv", index=False, encoding="utf-8-sig"
    )
    support = []
    for row in mapping.itertuples():
        entry = {
            "class_id": row.yolo_class_id,
            "pill_name": row.pill_name,
            "product_code": row.product_code,
        }
        for split in ["train", "val"]:
            subset = counts.loc[(counts.class_id == row.yolo_class_id) & (counts.split == split)]
            entry.update(
                {
                    f"{split}_annotations": len(subset),
                    f"{split}_images": subset.file_name.nunique(),
                    f"{split}_groups": subset.group.nunique(),
                }
            )
        support.append(entry)
    pd.DataFrame(support).to_csv(output / "class_support.csv", index=False, encoding="utf-8-sig")
    metadata = pd.read_csv(dataset / "qa/annotation_status.csv")
    source_to_yolo = dict(zip(mapping.source_class_id, mapping.yolo_class_id, strict=True))
    metadata["class_id"] = metadata.class_id.map(source_to_yolo)
    source_rows = []
    for row in metadata.loc[
        metadata.class_id.isin(FOCUS) & (metadata.status == "PASS")
    ].itertuples():
        path = raw_root / row.source_json
        assert digest(path) == row.json_sha256
        info = json.loads(path.read_text(encoding="utf-8"))["images"][0]
        source_rows.append(
            {
                "file_name": row.file_name,
                "class_id": row.class_id,
                "source_json": str(path),
                "source_sha256": row.json_sha256,
                **{
                    key: info.get(key)
                    for key in [
                        "item_seq",
                        "print_front",
                        "print_back",
                        "drug_shape",
                        "color_class1",
                        "color_class2",
                        "chart",
                        "drug_dir",
                    ]
                },
                "automatic_status": "REVIEW",
                "human_decision": "PENDING",
                "identity_verification": "UNVERIFIED",
            }
        )
    pd.DataFrame(source_rows).to_csv(output / "label_review.csv", index=False, encoding="utf-8-sig")
    evidence.to_csv(output / "gt_outcomes.csv", index=False, encoding="utf-8-sig")
    if any(font.name == "Malgun Gothic" for font in font_manager.fontManager.ttflist):
        plt.rcParams["font.family"] = "Malgun Gothic"
    make_gallery(output, dataset, truth, cached, evidence, names, metadata)
    make_contacts(output, dataset, metadata, names)
    assert before == {str(path): digest(path) for path in inputs}
    (output / "analysis_manifest.json").write_text(
        json.dumps(
            {
                "inputs_sha256": before,
                "code_sha256": digest(Path(__file__)),
                "packages": {
                    name: importlib.metadata.version(name)
                    for name in ["numpy", "pandas", "Pillow", "matplotlib"]
                },
                "thresholds": THRESHOLDS,
                "iou_match": 0.5,
                "localization_candidate_iou": 0.1,
                "class_id_mapping": "JSON category_id - 1 = YOLO class_id",
                "matching": "confidence descending, best unmatched same-class GT IoU>=0.50",
                "source": "cached native validation JSON, imgsz640/batch8/conf0.001/NMS IoU0.7",
                "multi_label_nms": True,
                "json_rounding": "bbox3 decimals / score5 decimals",
                "micro_aggregation": True,
                "weights_and_evaluation_preserved": True,
                "identity_verification": "UNVERIFIED",
                "trained": False,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(overall.to_string(index=False))
    print(pd.DataFrame(support).query("class_id in @FOCUS").to_string(index=False))
    print(
        evidence.query("threshold == 0.25 and class_id in @FOCUS")
        .groupby(["pill_name", "failure"])
        .size()
        .to_string()
    )


def main() -> None:
    """기존 실행 경로와 새 출력 경로를 명시적으로 받아 분석한다."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "outputs/recall_analysis")
    args = parser.parse_args()
    analyze(
        args.run_dir.resolve(),
        args.dataset.resolve(),
        args.raw_root.resolve(),
        args.output_dir.resolve(),
    )


if __name__ == "__main__":
    main()

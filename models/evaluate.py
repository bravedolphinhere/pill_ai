"""학습 후 best.pt를 검증하고 실제 지표와 예측 예시를 저장한다."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import time
from pathlib import Path

import yaml

from src.config import PROJECT_ROOT, get_device


def evaluate_model(
    weights: Path, data: Path, output: Path, image_size: int = 640, batch: int = 8
) -> dict:
    """검증 split에서 지표를 계산하고 독립 test 성능과 구분한다."""
    if not weights.is_file() or not data.is_file():
        raise FileNotFoundError("기존 학습 가중치와 data.yaml이 필요합니다.")
    if output.exists():
        raise FileExistsError("평가 출력은 새 폴더를 지정하세요.")
    os.environ["WANDB_MODE"] = "disabled"
    os.environ["YOLO_CONFIG_DIR"] = str(output / "ultralytics_config")
    from ultralytics import YOLO

    started = time.perf_counter()
    model = YOLO(str(weights.resolve()))
    metrics = model.val(
        data=str(data.resolve()),
        split="val",
        imgsz=image_size,
        batch=batch,
        device=get_device(),
        workers=0,
        plots=True,
        save_json=True,
        project=str(output.resolve()),
        name="validation",
        exist_ok=False,
    )
    config = yaml.safe_load(data.read_text(encoding="utf-8"))
    root = Path(config.get("path", data.parent))
    if not root.is_absolute():
        root = data.parent / root
    validation_images = root / config["val"]
    supports = {}
    for path in (root / "labels/val").glob("*.txt"):
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                class_id = int(line.split()[0])
                supports[class_id] = supports.get(class_id, 0) + 1
    result = {
        "weights": str(weights.resolve()),
        "data": str(data.resolve()),
        "split": "val",
        "mAP50": float(metrics.box.map50),
        "mAP50-95": float(metrics.box.map),
        "precision": float(metrics.box.mp),
        "recall": float(metrics.box.mr),
        "classes_with_validation_labels": len(supports),
        "model_class_count": len(config["names"]),
        "validation_class_support": supports,
        "evaluation_scope": "개발용 validation 성능. 독립 test 성능이 아님",
        "weights_sha256": hashlib.sha256(weights.read_bytes()).hexdigest(),
        "data_yaml_sha256": hashlib.sha256(data.read_bytes()).hexdigest(),
        "evaluation_seconds": time.perf_counter() - started,
    }
    output.mkdir(parents=True, exist_ok=True)
    measured = {int(class_id): index for index, class_id in enumerate(metrics.box.ap_class_index)}
    per_class = []
    for class_id, name in config["names"].items():
        index = measured.get(int(class_id))
        row = {
            "yolo_class_id": int(class_id),
            "pill_name": name,
            "validation_instances": supports.get(int(class_id), 0),
            "evaluation_status": "MEASURED" if index is not None else "NOT_EVALUATED",
            "precision": None,
            "recall": None,
            "mAP50": None,
            "mAP50-95": None,
        }
        if index is not None:
            row.update(
                precision=float(metrics.box.p[index]),
                recall=float(metrics.box.r[index]),
                mAP50=float(metrics.box.ap50[index]),
                **{"mAP50-95": float(metrics.box.ap[index])},
            )
        per_class.append(row)
    with (output / "per_class_metrics.csv").open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(per_class[0]))
        writer.writeheader()
        writer.writerows(per_class)
    # 검증에 없는 클래스를 0점으로 채우지 않고 미평가 상태로 남긴다.
    if metrics.confusion_matrix is not None:
        (output / "confusion_matrix_values.json").write_text(
            json.dumps(
                {"names": config["names"], "matrix": metrics.confusion_matrix.matrix.tolist()},
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
    (output / "metrics.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    examples = sorted(
        path
        for path in validation_images.iterdir()
        if path.suffix.lower() in {".png", ".jpg", ".jpeg"}
    )[:16]
    if examples:
        model.predict(
            source=[str(path) for path in examples],
            imgsz=image_size,
            device=get_device(),
            save=True,
            save_txt=True,
            save_conf=True,
            project=str(output.resolve()),
            name="predictions",
            exist_ok=False,
        )
    return result


def main() -> None:
    """학습이 끝난 뒤 별도 실행해 실제 지표만 저장한다."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "runs/yolo11n_evaluation")
    arguments = parser.parse_args()
    print(
        json.dumps(
            evaluate_model(arguments.weights, arguments.data, arguments.output_dir),
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

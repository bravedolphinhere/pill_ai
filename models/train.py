"""명시적 실행 확인 후에만 내부 데이터 YOLO11n 베이스라인을 학습한다."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import shutil
from pathlib import Path

from src.config import PROJECT_ROOT, get_device


def train_baseline(
    data: Path,
    output: Path,
    confirm: bool = False,
    epochs: int = 20,
    image_size: int = 640,
    batch: int = 8,
    seed: int = 42,
) -> dict:
    """검증된 데이터셋으로 학습하고 OOM 때만 배치를 줄여 새 시도를 기록한다."""
    request = {
        "model": "yolo11n.pt",
        "pretrained": True,
        "epochs": epochs,
        "imgsz": image_size,
        "batch": batch,
        "seed": seed,
        "data": str(data.resolve()),
        "output": str(output.resolve()),
        "training_started": False,
    }
    if not confirm:
        return request
    if not data.is_file() or not (data.parent / "preparation_summary.json").is_file():
        raise FileNotFoundError(
            "QA 준비 결과와 data.yaml이 필요합니다. 임의 데이터로 실행하지 않습니다."
        )
    validation_path = data.parent / "qa/dataset_validation.json"
    if not validation_path.is_file():
        raise FileNotFoundError("verify_internal_dataset 실행 결과가 필요합니다.")
    validation = json.loads(validation_path.read_text(encoding="utf-8"))
    preparation = json.loads((data.parent / "preparation_summary.json").read_text(encoding="utf-8"))
    if (
        validation.get("copied_images_verified") != preparation.get("pass_images")
        or validation.get("split_group_overlap") != 0
        or validation.get("exact_hash_overlap") != 0
    ):
        raise ValueError("데이터 준비와 검증 결과가 일치하지 않거나 분할 누수가 있습니다.")
    request["preparation_sha256"] = hashlib.sha256(
        (data.parent / "preparation_summary.json").read_bytes()
    ).hexdigest()
    request["validation_sha256"] = hashlib.sha256(validation_path.read_bytes()).hexdigest()
    request["class_map_sha256"] = hashlib.sha256(
        (data.parent / "class_map.csv").read_bytes()
    ).hexdigest()
    request["data_yaml_sha256"] = hashlib.sha256(data.read_bytes()).hexdigest()
    if output.exists():
        raise FileExistsError("기존 학습 결과를 덮어쓰지 않도록 새 출력 경로를 지정하세요.")
    # Ultralytics가 설정 파일을 만들기 전에 출력 폴더를 직접 준비한다.
    output.mkdir(parents=True)
    os.environ["WANDB_MODE"] = "disabled"
    os.environ["YOLO_CONFIG_DIR"] = str(output / "ultralytics_config")
    import torch
    from ultralytics import YOLO

    device = get_device()
    if device == "cpu":
        raise RuntimeError(
            "제출 일정의 GPU 베이스라인입니다. GPU를 사용할 수 없어 학습을 중단합니다."
        )
    # cuda.is_available만으로는 실제 GPU 커널 지원을 보증할 수 없어 작은 연산도 확인한다.
    if device == "cuda":
        check = torch.ones((16, 16), device=device)
        assert (check @ check).mean().item() == 16
    request["device"] = device
    request["packages"] = {
        name: importlib.metadata.version(name) for name in ["torch", "torchvision", "ultralytics"]
    }
    request["training_started"] = True
    (output / "training_request.json").write_text(
        json.dumps(request, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    current_batch = batch
    attempts = []
    while current_batch >= 1:
        model = None
        try:
            model = YOLO("yolo11n.pt")
            model.train(
                data=str(data.resolve()),
                epochs=epochs,
                imgsz=image_size,
                batch=current_batch,
                seed=seed,
                pretrained=True,
                device=device,
                workers=0,
                cache=False,
                deterministic=True,
                plots=True,
                project=str(output.resolve()),
                name=f"train_batch{current_batch}",
                exist_ok=False,
                val=True,
            )
            best = Path(model.trainer.best)
            if not best.is_file():
                raise FileNotFoundError("학습이 끝났지만 best.pt가 생성되지 않았습니다.")
            shutil.copy2(best, output / "best.pt")
            attempts.append({"batch": current_batch, "status": "completed", "best": str(best)})
            actual_batch = int(model.trainer.batch_size)
            result = {
                **request,
                "effective_batch": actual_batch,
                "attempts": attempts,
                "best": str((output / "best.pt").resolve()),
            }
            (output / "training_result.json").write_text(
                json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            return result
        except torch.cuda.OutOfMemoryError as error:
            attempts.append({"batch": current_batch, "status": "oom", "error": str(error)})
            (output / "oom_attempts.json").write_text(
                json.dumps(attempts, indent=2), encoding="utf-8"
            )
            if current_batch == 1:
                raise
            # 사용자 요청 범위의 배치 조정만 수행하고 epoch·모델·분할은 바꾸지 않는다.
            current_batch = max(1, current_batch // 2)
            del model
            torch.cuda.empty_cache()
    raise RuntimeError("학습이 완료되지 않았습니다.")


def main() -> None:
    """기본 실행은 설정 확인만 하며 --confirm-training일 때만 학습한다."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "runs/yolo11n_baseline")
    parser.add_argument("--confirm-training", action="store_true")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    arguments = parser.parse_args()
    print(
        json.dumps(
            train_baseline(
                arguments.data,
                arguments.output_dir,
                arguments.confirm_training,
                arguments.epochs,
                arguments.imgsz,
                arguments.batch,
                arguments.seed,
            ),
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

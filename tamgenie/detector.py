"""기존 YOLO11n 가중치로 추론하며 업로드 이미지·가중치를 수정하지 않는다."""

from __future__ import annotations

import hashlib
import os
import threading
from dataclasses import dataclass
from pathlib import Path

from PIL import Image

from src.config import PROJECT_ROOT, get_device
from tamgenie.question_engine import Product


@dataclass(frozen=True)
class Detection:
    """검출 박스와 원시 모델 점수를 보존한다."""

    index: int
    class_id: int
    score: float
    bbox: tuple[int, int, int, int]


class Detector:
    """공유 모델의 동시 추론을 직렬화하고 클래스 매핑 불일치를 거부한다."""

    def __init__(self, weights: Path, catalog: dict[int, Product]) -> None:
        if not weights.is_file():
            raise FileNotFoundError("기존 best.pt를 TAMGENIE_WEIGHTS로 지정하세요.")
        os.environ.setdefault("WANDB_MODE", "disabled")
        os.environ["YOLO_AUTOINSTALL"] = "false"
        os.environ.setdefault("YOLO_CONFIG_DIR", str(PROJECT_ROOT / "outputs/tamgenie/yolo_config"))
        from ultralytics import YOLO

        self.model = YOLO(str(weights.resolve()))
        names = {int(key): value for key, value in self.model.names.items()}
        if names != {key: product.name for key, product in catalog.items()}:
            raise ValueError(
                "모델 클래스와 class_map이 다릅니다.118종 정제 맵을56종 모델에 연결하지 마세요."
            )
        self.device = get_device()
        self.weights_sha256 = hashlib.sha256(weights.read_bytes()).hexdigest()
        self.lock = threading.Lock()

    def predict(self, image: Image.Image, threshold: float) -> list[Detection]:
        """단일 클래스 predict 경로의 NMS 출력을 반환하고 파일을 저장하지 않는다."""
        with self.lock:
            result = self.model.predict(
                image,
                conf=threshold,
                iou=0.7,
                imgsz=640,
                max_det=30,
                device=self.device,
                save=False,
                verbose=False,
            )[0]
        found = []
        for box, class_id, score in zip(
            result.boxes.xyxy.cpu().tolist(),
            result.boxes.cls.cpu().tolist(),
            result.boxes.conf.cpu().tolist(),
            strict=True,
        ):
            left, top, right, bottom = box
            coordinates = (
                max(0, int(left)),
                max(0, int(top)),
                min(image.width, int(right + 0.5)),
                min(image.height, int(bottom + 0.5)),
            )
            if coordinates[2] > coordinates[0] and coordinates[3] > coordinates[1]:
                found.append(Detection(len(found) + 1, int(class_id), float(score), coordinates))
        return found

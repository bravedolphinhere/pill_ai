"""EfficientNet-B0 단계의 계약만 정의하며 없는 가중치의 예측을 만들지 않는다."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from PIL import Image


@dataclass(frozen=True)
class Classification:
    """서로 다른 모델의 점수를 정답 확률로 합치지 않는다."""

    class_id: int
    score: float


class Classifier(Protocol):
    """추후 검증된 Crop 분류기를 연결하는 인터페이스다."""

    available: bool

    def predict(self, image: Image.Image) -> Classification | None:
        """Crop에 대한 분류 후보를 반환한다."""
        ...


class UnavailableClassifier:
    """학습된 EfficientNet-B0가 없어2차 단계를 명시적으로 건너뛴다."""

    available = False

    def predict(self, image: Image.Image) -> None:
        """임의 초기화 모델이나 가짜 confidence를 반환하지 않는다."""
        return None

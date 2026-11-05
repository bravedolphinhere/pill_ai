"""업로드·품질 검사·불일치 판정과 후보 생성을 연결하며 정체를 확정하지 않는다."""

from __future__ import annotations

import io
import math

import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps

from tamgenie.classifier import Classifier, UnavailableClassifier
from tamgenie.detector import Detection
from tamgenie.question_engine import Product
from tamgenie.settings import Settings


def read_image(data: bytes, settings: Settings) -> Image.Image:
    """형식·용량·픽셀 수를 확인하고 표시와 추론에 같은 방향의 RGB 이미지를 쓴다."""
    if not data or len(data) > settings.max_bytes:
        raise ValueError("이미지는15MB 이내로 선택해주세요.")
    try:
        with Image.open(io.BytesIO(data)) as image:
            if image.format not in {"JPEG", "PNG", "WEBP"}:
                raise ValueError("JPEG·PNG·WebP 이미지를 선택해주세요.")
            if image.width * image.height > settings.max_pixels:
                raise ValueError("이미지는2,000만 픽셀 이내여야 합니다.")
            image.load()
            return ImageOps.exif_transpose(image).convert("RGB")
    except (OSError, Image.DecompressionBombError) as error:
        raise ValueError("이미지를 읽을 수 없습니다. 다른 이미지로 시도해주세요.") from error


def crop_image(image: Image.Image, detection: Detection) -> Image.Image:
    """예측 범위를 조금 넓혀 잘라내며 입력 원본은 덮어쓰지 않는다."""
    left, top, right, bottom = detection.bbox
    padding = int(max(right - left, bottom - top) * 0.1)
    return image.crop(
        (
            max(0, left - padding),
            max(0, top - padding),
            min(image.width, right + padding),
            min(image.height, bottom + padding),
        )
    )


def quality_flags(image: Image.Image) -> list[str]:
    """크기·밝기·흐림을 임시 기준으로 표시하고 정확도 보장으로 사용하지 않는다."""
    gray = np.asarray(image.convert("L").resize((128, 128)), dtype=float)
    flags = []
    if min(image.size) < 48:
        flags.append("Crop이 작음 (임시48px 기준)")
    if gray.mean() < 35 or gray.mean() > 235:
        flags.append("너무 어둡거나 밝음 (임시 기준)")
    laplacian = (
        gray[2:, 1:-1] + gray[:-2, 1:-1] + gray[1:-1, 2:] + gray[1:-1, :-2] - 4 * gray[1:-1, 1:-1]
    )
    if laplacian.var() < 35:
        flags.append("각인·윤곽 정보가 적을 가능성 (임시 기준)")
    return flags


def route(
    detection: Detection,
    crop: Image.Image,
    review_threshold: float,
    classifier: Classifier | None = None,
    flags: list[str] | None = None,
) -> dict:
    """낮은 점수·품질 우려·모델 불일치를 질문 단계로 보내며 정답을 확정하지 않는다."""
    reasons = quality_flags(crop) + (flags or [])
    if detection.score < review_threshold:
        reasons.append("1차 모델 점수가 추가 확인 기준 미만")
    secondary = None
    classifier = classifier or UnavailableClassifier()
    if reasons and classifier.available:
        secondary = classifier.predict(crop)
        if secondary is not None:
            if not math.isfinite(secondary.score) or not 0 <= secondary.score <= 1:
                reasons.append("2차 모델 점수가 유효하지 않음")
            if secondary.class_id != detection.class_id:
                reasons.append("1차·2차 모델의 클래스 불일치")
    return {
        "needs_review": bool(reasons),
        "reasons": reasons,
        "secondary": secondary,
        "secondary_available": classifier.available,
        "status": "추가 확인 필요" if reasons else "1차 모델의 참고 후보",
        "identity_confirmed": False,
    }


def overlap_flags(detections: list[Detection]) -> dict[int, list[str]]:
    """같은 위치의 다른 클래스 예측을 확정된 알약 한 개로 취급하지 않는다."""
    flags = {item.index: [] for item in detections}
    for first_index, first in enumerate(detections):
        for second in detections[first_index + 1 :]:
            if first.class_id == second.class_id:
                continue
            a, b = first.bbox, second.bbox
            intersection = max(0, min(a[2], b[2]) - max(a[0], b[0])) * max(
                0, min(a[3], b[3]) - max(a[1], b[1])
            )
            union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - intersection
            if union and intersection / union >= 0.65:
                flags[first.index].append("같은 위치에 서로 다른 클래스 예측이 겹침")
                flags[second.index].append("같은 위치에 서로 다른 클래스 예측이 겹침")
    return flags


def annotate(
    image: Image.Image, detections: list[Detection], catalog: dict[int, Product], settings: Settings
) -> Image.Image:
    """번호·후보 이름·원시 모델 점수를 표시하며 입력 이미지는 변경하지 않는다."""
    annotated = image.copy()
    draw = ImageDraw.Draw(annotated)
    font = (
        ImageFont.truetype(str(settings.font), max(16, image.width // 45))
        if settings.font
        else ImageFont.load_default()
    )
    for item in detections:
        left, top, right, bottom = item.bbox
        name = catalog[item.class_id].name if settings.font else catalog[item.class_id].code
        label = f"#{item.index} {name} | {item.score:.3f}"
        width = min(image.width, int(draw.textlength(label, font=font)) + 10)
        text_left = max(0, min(left, image.width - width))
        text_top = max(0, top - 35)
        draw.rectangle(item.bbox, outline="#00B7A5", width=4)
        draw.rectangle((text_left, text_top, text_left + width, text_top + 34), fill="#103852")
        draw.text((text_left + 5, text_top + 2), label, font=font, fill="white")
    return annotated

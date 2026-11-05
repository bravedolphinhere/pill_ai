"""출처를 확인한 속성만으로 질문하고 정보가 없는 후보는 남겨 둔다."""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlparse

ALLOWED_FIELDS = {
    "shape": "보이는 모양",
    "color": "보이는 색상",
    "imprint": "보이는 각인",
    "score_line": "분할선",
}
TRUSTED_DOMAINS = {
    "www.ckdpharm.com",
    "ckdpharm.com",
    "yuhan.co.kr",
    "www.yuhan.co.kr",
    "nedrug.mfds.go.kr",
}


@dataclass(frozen=True)
class Product:
    """학습 라벨 이름과 독립적으로 확인한 속성의 출처를 구분한다."""

    class_id: int
    code: str
    name: str
    attributes: dict


@dataclass(frozen=True)
class Question:
    """현재 후보 사이에 실제로 차이가 있는 질문만 나타낸다."""

    field: str
    title: str
    options: tuple[str, ...]
    known_count: int
    unknown_count: int


def load_catalog(class_map: Path, metadata: Path) -> dict[int, Product]:
    """검증 표시·공식 출처·대조 날짜가 모두 있는 속성만 허용한다."""
    payload = json.loads(metadata.read_text(encoding="utf-8"))
    records = payload.get("products", {})
    catalog = {}
    with class_map.open(encoding="utf-8-sig", newline="") as file:
        for row in csv.DictReader(file):
            attributes = {}
            for field, record in records.get(row["product_code"], {}).items():
                source = record.get("source_url", "")
                if (
                    field in ALLOWED_FIELDS
                    and record.get("verified") is True
                    and record.get("checked_at")
                    and record.get("values")
                    and urlparse(source).hostname in TRUSTED_DOMAINS
                    and urlparse(source).scheme == "https"
                    and (
                        field not in {"imprint", "score_line"}
                        or record.get("covers_all_sides") is True
                    )
                    and isinstance(record["values"], list)
                    and all(isinstance(value, str) and value for value in record["values"])
                ):
                    attributes[field] = record
            class_id = int(row["yolo_class_id"])
            if class_id in catalog or any(p.code == row["product_code"] for p in catalog.values()):
                raise ValueError("중복된 클래스 ID 또는 제품 코드가 있습니다.")
            catalog[class_id] = Product(class_id, row["product_code"], row["pill_name"], attributes)
    if set(catalog) != set(range(len(catalog))):
        raise ValueError("클래스 ID는0부터 연속이어야 합니다.")
    return catalog


def filter_candidates(catalog: dict[int, Product], candidates: set[int], answers: dict) -> set[int]:
    """모름은 제외 근거로 삼지 않고 검증된 값과 명시적으로 충돌할 때만 제외한다."""
    remaining = set(candidates)
    for field, answer in answers.items():
        if field not in ALLOWED_FIELDS or answer == "모르겠어요":
            continue
        remaining = {
            class_id
            for class_id in remaining
            if field not in catalog[class_id].attributes
            or answer in catalog[class_id].attributes[field]["values"]
        }
    return remaining


def next_question(
    catalog: dict[int, Product], candidates: set[int], answered: set[str]
) -> Question | None:
    """서로 다른 검증값을 가진 후보가2개 이상일 때만 다음 질문을 고른다."""
    choices = []
    for field, label in ALLOWED_FIELDS.items():
        if field in answered:
            continue
        known = [
            catalog[index].attributes[field]
            for index in sorted(candidates)
            if field in catalog[index].attributes
        ]
        signatures = {tuple(sorted(record["values"])) for record in known}
        if len(signatures) < 2:
            continue
        options = tuple(sorted({value for record in known for value in record["values"]}))
        choices.append(
            Question(
                field,
                f"사진 속 알약의 {label}은 무엇인가요?",
                options,
                len(known),
                len(candidates) - len(known),
            )
        )
    return max(
        choices, key=lambda question: (question.known_count, len(question.options)), default=None
    )

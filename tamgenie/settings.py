"""기존 데이터·가중치 경로를 옮기지 않고 환경변수로 연결한다."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

from src.config import PROJECT_ROOT


@dataclass(frozen=True)
class Settings:
    """개발·운영 경로와 검증 전 임시 설정을 한곳에 둔다."""

    weights: Path
    class_map: Path
    metadata: Path
    font: Path | None
    max_bytes: int = 15 * 1024**2
    max_pixels: int = 20_000_000


def get_settings() -> Settings:
    """개인 경로를 코드에 고정하지 않고 로컬 환경에서 읽는다."""
    load_dotenv(PROJECT_ROOT / ".env", override=False)
    default_font = Path(os.environ.get("WINDIR", "/nonexistent")) / "Fonts/malgun.ttf"
    font = Path(os.environ.get("TAMGENIE_FONT") or str(default_font))
    return Settings(
        weights=Path(os.environ.get("TAMGENIE_WEIGHTS") or str(PROJECT_ROOT / "models/best.pt")),
        class_map=Path(
            os.environ.get("TAMGENIE_CLASS_MAP")
            or str(PROJECT_ROOT / "docs/results/yolo11n_20261105/class_map.csv")
        ),
        metadata=Path(
            os.environ.get("TAMGENIE_METADATA")
            or str(PROJECT_ROOT / "configs/tamgenie/verified_metadata.json")
        ),
        font=font if font.is_file() else None,
    )

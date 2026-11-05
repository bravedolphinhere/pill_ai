"""GitHub 노트북에 원본 사진을 넣지 않고 로컬 EDA 결과를 연결한다."""

import os
from pathlib import Path

import pandas as pd
from IPython.display import Image, display

from src.config import PROJECT_ROOT


def report_root(source: str) -> Path:
    """환경변수로 지정한 로컬 EDA 폴더를 찾고 원본 경로는 하드코딩하지 않는다."""
    if source not in {"internal", "external"}:
        raise ValueError("source는 internal 또는 external이어야 합니다.")
    variable = f"PILL_{source.upper()}_EDA_DIR"
    date = "20261104" if source == "internal" else "20261105"
    return Path(os.getenv(variable, PROJECT_ROOT / "outputs" / f"{source}_eda_{date}")).resolve()


def show_figure(source: str, name: str) -> None:
    """별도 로컬 폴더의 비교 이미지를 보여 주고 파일이 없으면 설정 방법을 안내한다."""
    root = report_root(source) / "figures"
    path = (root / name).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError("그림 경로가 figures 폴더 밖을 가리킵니다.")
    if path.is_file():
        display(Image(filename=str(path)))
    else:
        print(f"PILL_{source.upper()}_EDA_DIR을 로컬 EDA 결과 폴더로 설정하세요: {name}")


def load_table(source: str, name: str) -> pd.DataFrame:
    """생성된 EDA 표를 읽되 원문 null과 빈 문자열 복원이 불가능할 수 있음을 구분한다."""
    root = report_root(source) / "tables"
    path = (root / name).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError("표 경로가 tables 폴더 밖을 가리킵니다.")
    return pd.read_csv(path, low_memory=False)

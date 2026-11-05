from __future__ import annotations

import os
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import wandb
    from ultralytics import YOLO

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def get_device() -> str:
    """사용 가능한 장치를 CUDA, MPS, CPU 순서로 선택한다."""
    import torch

    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def init_wandb(
    model: YOLO | None = None,
    run_name: str | None = None,
    config: dict | None = None,
    **kwargs: Any,
) -> wandb.Run:
    """wandb run을 시작한다. model을 넘기면 그 모델의 학습 로그가 이 run에 기록된다.

    ultralytics 전역 설정(settings wandb=True)은 바꾸지 않고 넘긴 모델에만 callback을 붙이므로,
    이 프로젝트 밖의 YOLO 학습에는 영향이 없다.
    끄고 싶을 때는 WANDB_MODE=disabled, 인터넷 없이 기록만 남길 때는 WANDB_MODE=offline.
    """
    import wandb
    from dotenv import load_dotenv

    # 기록을 시작할 때만 .env를 읽으며 이미 설정된 환경 변수가 우선한다.
    load_dotenv(PROJECT_ROOT / ".env", override=False)
    project = os.getenv("WANDB_PROJECT", "tamgenie")
    entity = os.getenv("WANDB_ENTITY") or None

    if model is not None:
        from ultralytics.utils.callbacks.wb import callbacks as wandb_callbacks

        for event, callback in wandb_callbacks.items():
            model.add_callback(event, callback)
    return wandb.init(
        project=project,
        entity=entity,
        name=run_name,
        config=config,
        dir=PROJECT_ROOT,
        **kwargs,
    )

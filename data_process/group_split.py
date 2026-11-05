"""고정된 사진 그룹과 train 클래스 범위를 보존하며 목표 비율에 가장 가깝게 나눈다."""

from __future__ import annotations

import numpy as np
import pandas as pd
import scipy
from scipy.optimize import Bounds, LinearConstraint, milp


def optimize_group_split(
    images: pd.DataFrame,
    boxes: pd.DataFrame,
    seed: int = 42,
    val_fraction: float = 0.2,
) -> tuple[dict[str, str], dict]:
    """비율 오차 최소화 후 검증 클래스 수를 최대화하고 seed로 동률을 결정한다."""
    if not 0 < val_fraction < 1:
        raise ValueError("검증 비율은 0과 1 사이여야 합니다.")
    eligible = images.loc[images.status.eq("PASS")]
    grouped = eligible.groupby("split_group").file_name.agg(set).to_dict()
    groups = sorted(grouped)
    if len(groups) < 2:
        raise ValueError("서로 다른 분할 그룹이 두 개 이상 필요합니다.")
    rows = boxes.loc[boxes.file_name.isin(eligible.file_name)]
    classes = sorted(rows.class_id.unique())
    group_count, class_count = len(groups), len(classes)
    variable_count = group_count + class_count + 1
    group_sizes = np.array([len(grouped[group]) for group in groups], dtype=float)
    product_groups = np.array(
        [
            [
                int(class_id in set(rows.loc[rows.file_name.isin(grouped[group]), "class_id"]))
                for group in groups
            ]
            for class_id in classes
        ],
        dtype=float,
    )
    total = len(eligible)
    target = val_fraction * total
    constraints, lower, upper = [], [], []

    def constrain(coefficients: np.ndarray, minimum: float, maximum: float) -> None:
        constraints.append(coefficients)
        lower.append(minimum)
        upper.append(maximum)

    count_row = np.zeros(variable_count)
    count_row[:group_count] = group_sizes
    constrain(count_row.copy(), 1, total - 1)
    # 각 제품이 있는 모든 그룹을 val로 보내지 못하게 해 train의 제품 범위를 보존한다.
    for index, present in enumerate(product_groups):
        row = np.zeros(variable_count)
        row[:group_count] = present
        constrain(row.copy(), -np.inf, present.sum() - 1)
        row[group_count + index] = -1
        constrain(row.copy(), 0, np.inf)
        row[group_count + index] = -present.sum()
        constrain(row, -np.inf, 0)
    # d >= |검증 사진 수 - 목표 사진 수|. 그룹 수가 아니라 실제 사진 수를 최적화한다.
    row = count_row.copy()
    row[-1] = -1
    constrain(row, -np.inf, target)
    row = -count_row.copy()
    row[-1] = -1
    constrain(row, -np.inf, -target)
    bounds = Bounds(np.zeros(variable_count), np.r_[np.ones(variable_count - 1), total])
    integrality = np.r_[np.ones(variable_count - 1), 0]
    solver_records = []

    def solve(objective: np.ndarray, phase: str):
        result = milp(
            objective,
            integrality=integrality,
            bounds=bounds,
            constraints=LinearConstraint(np.array(constraints), lower, upper),
            options={"time_limit": 60, "mip_rel_gap": 0.0},
        )
        if result.status != 0:
            raise RuntimeError(f"{phase} 최적해가 확인되지 않았습니다: {result.message}")
        solver_records.append(
            {
                "phase": phase,
                "status": int(result.status),
                "objective": float(result.fun),
                "mip_gap": float(result.mip_gap),
            }
        )
        return result

    objective = np.zeros(variable_count)
    objective[-1] = 1
    primary = solve(objective, "closest_photo_ratio")
    best_distance = abs(float(group_sizes @ np.rint(primary.x[:group_count])) - target)
    row = np.zeros(variable_count)
    row[-1] = 1
    constrain(row, -np.inf, best_distance + 1e-7)
    objective = np.zeros(variable_count)
    objective[group_count : group_count + class_count] = -1
    secondary = solve(objective, "maximum_val_classes_at_closest_ratio")
    best_coverage = int(np.rint(secondary.x[group_count : group_count + class_count]).sum())
    row = np.zeros(variable_count)
    row[group_count : group_count + class_count] = 1
    constrain(row, best_coverage, best_coverage)
    objective = np.zeros(variable_count)
    objective[:group_count] = np.random.default_rng(seed).uniform(0.01, 1, group_count)
    final = solve(objective, "seeded_tie_break")
    validation_groups = {group for index, group in enumerate(groups) if final.x[index] > 0.5}
    assignment = {
        name: "val" if group in validation_groups else "train"
        for group, names in grouped.items()
        for name in names
    }
    train_names = {name for name, split in assignment.items() if split == "train"}
    valid_names = {name for name, split in assignment.items() if split == "val"}
    train_classes = set(rows.loc[rows.file_name.isin(train_names), "class_id"])
    valid_classes = set(rows.loc[rows.file_name.isin(valid_names), "class_id"])
    assert train_classes == set(classes)
    assert abs(len(valid_names) - target) <= best_distance + 1e-6
    assert len(valid_classes) == best_coverage
    return assignment, {
        "train_images": len(train_names),
        "val_images": len(valid_names),
        "train_classes": len(train_classes),
        "val_classes": len(valid_classes),
        "classes_absent_validation": sorted(map(int, train_classes - valid_classes)),
        "group_overlap": 0,
        "seed": seed,
        "algorithm": "MILP lexicographic",
        "scipy_version": scipy.__version__,
        "target_val_fraction": val_fraction,
        "actual_val_fraction": len(valid_names) / total,
        "target_val_images": target,
        "minimum_val_image_count_error": best_distance,
        "fixed_group_count": group_count,
        "priority": [
            "preserve_all_train_classes",
            "closest_photo_ratio",
            "val_class_coverage",
            "seeded_tie_break",
        ],
        "solver_records": solver_records,
    }

"""내부 JSON을 검증하고 REVIEW 사진을 제외한 YOLO 학습 후보를 별도 생성한다."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from PIL import Image

from data_process.group_split import optimize_group_split
from src.config import PROJECT_ROOT


def save_table(rows: list[dict] | pd.DataFrame, path: Path) -> pd.DataFrame:
    """빈 결과도 읽을 수 있게 헤더가 있는 CSV로 저장한다."""
    frame = rows.copy() if isinstance(rows, pd.DataFrame) else pd.DataFrame(rows)
    if not len(frame.columns):
        frame = pd.DataFrame(columns=["status"])
    frame.to_csv(path, index=False, encoding="utf-8-sig")
    return frame


def perceptual_hash(image: Image.Image) -> int:
    """배경을 공유하는 유사 촬영을 같은 분할에 묶기 위한 64비트 DCT 해시를 구한다."""
    pixels = np.asarray(image.convert("L").resize((32, 32)), dtype=float)
    basis = np.cos(np.pi * (np.arange(32) + 0.5)[None, :] * np.arange(8)[:, None] / 32)
    coefficients = (basis @ pixels @ basis.T).ravel()
    threshold = np.median(coefficients[1:])
    return sum(int(value > threshold) << index for index, value in enumerate(coefficients))


def inspect_images(root: Path, eda: Path, output: Path) -> tuple[pd.DataFrame, dict]:
    """학습 사진만 읽어 해시·디코딩·유사도를 확인하고 기존 EDA 지문과 대조한다."""
    old = pd.read_csv(eda / "tables/image_inventory.csv").query("split == 'train'")
    old_hashes = dict(zip(old.file_name, old.sha256, strict=True))
    rows, thumbnails = [], {}
    for path in sorted((root / "train_images").rglob("*")):
        if not path.is_file() or path.suffix.lower() not in {".png", ".jpg", ".jpeg"}:
            continue
        raw = path.read_bytes()
        digest = hashlib.sha256(raw).hexdigest()
        row = {
            "file_name": path.name,
            "source_path": path.relative_to(root).as_posix(),
            "sha256": digest,
            "capture_group": path.stem.split("_")[0],
            "bytes": len(raw),
            "error": "",
            "matches_eda_snapshot": digest == old_hashes.get(path.name),
        }
        try:
            with Image.open(path) as image:
                image.load()
                row.update(width=image.width, height=image.height, phash=perceptual_hash(image))
                thumbnails[path.name] = np.asarray(image.convert("L").resize((64, 64)), dtype=float)
        except Exception as error:
            row["error"] = f"{type(error).__name__}: {error}"
        rows.append(row)
    table = save_table(rows, output / "image_inventory.csv")
    if table.file_name.duplicated().any():
        raise ValueError("이미지 파일명이 유일하지 않습니다. 자동으로 경로를 선택하지 않습니다.")
    return table, thumbnails


def parse_annotations(root: Path, eda: Path, output: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """로컬 ID와 제품 코드를 대조하고 실제 라벨 원문을 수정 없이 정규화한다."""
    old = pd.read_csv(eda / "tables/json_manifest.csv")
    old_hashes = dict(zip(old.path, old.sha256, strict=True))
    rows, errors, inputs = [], [], []
    for path in sorted((root / "train_annotations").rglob("*.json")):
        relative = path.relative_to(root).as_posix()
        raw = path.read_bytes()
        digest = hashlib.sha256(raw).hexdigest()
        inputs.append(
            {
                "source_json": relative,
                "sha256": digest,
                "matches_eda_snapshot": digest == old_hashes.get(relative),
            }
        )
        try:
            payload = json.loads(raw.decode("utf-8-sig"))
            images = {value["id"]: value for value in payload["images"]}
            categories = {value["id"]: value for value in payload["categories"]}
            for position, annotation in enumerate(payload["annotations"]):
                metadata = images.get(annotation.get("image_id"), {})
                category = categories.get(annotation.get("category_id"), {})
                box = annotation.get("bbox")
                numeric = (
                    isinstance(box, list)
                    and len(box) == 4
                    and all(
                        isinstance(value, (int, float))
                        and not isinstance(value, bool)
                        and math.isfinite(value)
                        for value in box
                    )
                )
                class_id = annotation.get("category_id")
                code = metadata.get("drug_N")
                mapped = metadata.get("dl_mapping_code")
                class_ok = (
                    isinstance(class_id, int)
                    and not isinstance(class_id, bool)
                    and class_id >= 0
                    and str(class_id) == str(metadata.get("dl_idx"))
                    and code == f"K-{class_id:06d}"
                    and mapped == code
                )
                row = {
                    "uid": f"{relative}#{position}",
                    "annotation_id": annotation.get("id"),
                    "file_name": metadata.get("file_name"),
                    "class_id": class_id,
                    "product_code": code,
                    "pill_name": metadata.get("dl_name"),
                    "source_json": relative,
                    "json_sha256": digest,
                    "bbox": json.dumps(box),
                    "numeric_bbox": numeric,
                    "area": annotation.get("area"),
                    "declared_width": metadata.get("width"),
                    "declared_height": metadata.get("height"),
                    "class_ok": class_ok
                    and bool(category)
                    and metadata.get("dl_name") == category.get("name"),
                    "image_join_ok": bool(metadata),
                    "ignore": annotation.get("ignore", 0),
                }
                if numeric:
                    row.update(dict(zip(["x", "y", "w", "h"], box, strict=True)))
                rows.append(row)
        except Exception as error:
            errors.append(
                {
                    "source_json": relative,
                    "error": f"{type(error).__name__}: {error}",
                    "expected_image": path.stem + ".png",
                }
            )
    save_table(inputs, output / "json_inputs.csv")
    return pd.DataFrame(rows), save_table(errors, output / "json_errors.csv")


def validate(
    boxes: pd.DataFrame,
    images: pd.DataFrame,
    errors: pd.DataFrame,
    thumbnails: dict,
    eda: Path,
    legacy: Path | None,
    legacy_log: Path | None,
    review_csvs: list[Path],
    output: Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """의심 근거를 누적하고 사진 전체를 REVIEW로 보류하며 원본 라벨은 수정하지 않는다."""
    findings = []

    def flag(name: str, rule: str, evidence: str, uid: str = "") -> None:
        findings.append(
            {
                "file_name": name,
                "uid": uid,
                "rule_id": rule,
                "status": "REVIEW",
                "evidence": evidence,
                "origin": "automatic",
            }
        )

    image_lookup = images.set_index("file_name")
    boxes["bbox_valid"] = False
    for row in boxes.itertuples():
        if row.file_name not in image_lookup.index:
            flag(row.file_name, "IMAGE_LINK", "실제 이미지 경로 없음", row.uid)
            continue
        image = image_lookup.loc[row.file_name]
        valid = (
            row.numeric_bbox
            and row.w > 0
            and row.h > 0
            and row.x >= 0
            and row.y >= 0
            and row.x + row.w <= image.width
            and row.y + row.h <= image.height
        )
        boxes.at[row.Index, "bbox_valid"] = valid
        if not valid:
            flag(row.file_name, "BBOX_INVALID", row.bbox, row.uid)
        if row.declared_width != image.width or row.declared_height != image.height:
            flag(row.file_name, "IMAGE_SIZE", "선언/실제 이미지 크기 불일치", row.uid)
        if not row.class_ok or not row.image_join_ok:
            flag(
                row.file_name, "CLASS_MAPPING", "제품 코드·클래스·이름·참조 연결 검토 필요", row.uid
            )
        if row.ignore:
            flag(
                row.file_name,
                "IGNORE_LABEL",
                "원본 ignore 라벨을 정상 정답으로 사용하지 않음",
                row.uid,
            )
        if row.numeric_bbox and (
            not isinstance(row.area, (int, float))
            or not math.isfinite(row.area)
            or not math.isclose(row.area, row.w * row.h, abs_tol=1e-6)
        ):
            flag(row.file_name, "AREA_MISMATCH", f"기록={row.area}, w*h={row.w * row.h}", row.uid)
    for row in images.itertuples():
        if row.error:
            flag(row.file_name, "IMAGE_DECODE", row.error)
        if not row.matches_eda_snapshot:
            flag(
                row.file_name,
                "SNAPSHOT_CHANGED",
                "기존 EDA와 해시가 달라 의미 검수 결과 재사용 불가",
            )
        group = boxes.loc[boxes.file_name.eq(row.file_name)]
        expected = {int(value) for value in row.capture_group.replace("K-", "").split("-")}
        missing = expected - set(group.class_id)
        if missing:
            flag(
                row.file_name,
                "EXPECTED_CLASS_MISSING",
                f"파일명 기대 클래스 연결 없음: {sorted(missing)}",
            )
        if group.empty:
            flag(row.file_name, "NO_ANNOTATIONS", "알약 사진의 정답 연결 없음")
        for first_index, first in group.iterrows():
            for second_index, second in group.iterrows():
                if second_index <= first_index or not first.numeric_bbox or not second.numeric_bbox:
                    continue
                intersection = max(
                    0, min(first.x + first.w, second.x + second.w) - max(first.x, second.x)
                )
                intersection *= max(
                    0, min(first.y + first.h, second.y + second.h) - max(first.y, second.y)
                )
                union = first.w * first.h + second.w * second.h - intersection
                iou = intersection / union if union > 0 else 0
                if iou >= 0.8:
                    flag(
                        row.file_name,
                        "BBOX_OVERLAP",
                        f"ann {first.annotation_id}/{second.annotation_id}, IoU={iou:.6f}, "
                        f"class={first.class_id}/{second.class_id}",
                    )
    if len(errors):
        for row in errors.itertuples():
            flag(row.expected_image, "JSON_PARSE", row.error)
    # 과거 검수 메모는 정답 인증으로 승격하지 않고 보수적 제외 근거로만 사용한다.
    if legacy is not None:
        notes = pd.read_csv(legacy, keep_default_na=False)
        old_hashes = {}
        if legacy_log is not None:
            logs = pd.read_csv(legacy_log)
            old_hashes = dict(zip(logs.file_name, logs.sha256, strict=True))
        for row in notes.itertuples():
            if row.file_name in image_lookup.index:
                matched = old_hashes.get(row.file_name) == image_lookup.loc[row.file_name, "sha256"]
                flag(row.file_name, "LEGACY_REVIEW", f"{row.issue}; 과거 사진 해시 일치={matched}")
                findings[-1]["origin"] = "legacy_visual_observation"
    for path in review_csvs:
        previous = pd.read_csv(path, keep_default_na=False)
        if "file_name" not in previous:
            raise ValueError(f"추가 검수 표에 file_name이 없습니다: {path.name}")
        for row in previous.itertuples():
            if row.file_name in image_lookup.index:
                flag(row.file_name, "UNRESOLVED_LEGACY_REVIEW", f"검수 보류 표: {path.name}")
                findings[-1]["origin"] = "legacy_visual_observation"
    # 기존 EDA finding을 재사용하되 누락·충돌을 다시 확인한 결과와 함께 출처를 남긴다.
    for name in ["filename_missing_candidates.csv", "overlapping_boxes.csv"]:
        previous = pd.read_csv(eda / "tables" / name)
        if "file_name" in previous:
            for filename in previous.file_name.unique():
                if filename in image_lookup.index:
                    flag(filename, "EDA_REVIEW", f"기존 EDA: {name}")
    parent = {name: name for name in images.file_name}

    def find(name: str) -> str:
        while parent[name] != name:
            parent[name] = parent[parent[name]]
            name = parent[name]
        return name

    def unite(first: str, second: str) -> None:
        parent[find(second)] = find(first)

    for _, group in images.groupby("capture_group"):
        for name in group.file_name.iloc[1:]:
            unite(group.file_name.iloc[0], name)
    pairs = []
    image_records = list(images.itertuples())
    labels = boxes.groupby("file_name").class_id.agg(lambda values: tuple(sorted(values)))
    for index, first in enumerate(image_records):
        for second in image_records[index + 1 :]:
            if first.error or second.error:
                continue
            distance = (int(first.phash) ^ int(second.phash)).bit_count()
            exact = first.sha256 == second.sha256
            if exact or distance <= 6:
                unite(first.file_name, second.file_name)
                mean_difference = float(
                    np.abs(thumbnails[first.file_name] - thumbnails[second.file_name]).mean() / 255
                )
                conflict = labels.get(first.file_name) != labels.get(second.file_name)
                pairs.append(
                    {
                        "first": first.file_name,
                        "second": second.file_name,
                        "phash_distance": distance,
                        "exact_bytes": exact,
                        "thumbnail_mean_difference": mean_difference,
                        "class_set_differs": conflict,
                    }
                )
                if conflict and (exact or distance <= 2 and mean_difference <= 0.01):
                    for name in [first.file_name, second.file_name]:
                        flag(
                            name,
                            "SIMILAR_LABEL_CONFLICT",
                            f"사진 쌍={first.file_name}/{second.file_name}",
                        )
    images["split_group"] = images.file_name.map(find)
    review = {row["file_name"] for row in findings}
    images["status"] = np.where(images.file_name.isin(review), "REVIEW", "PASS")
    boxes["status"] = np.where(boxes.file_name.isin(review), "REVIEW", "PASS")
    save_table(findings, output / "findings.csv")
    save_table(pairs, output / "similar_pairs.csv")
    save_table(images, output / "image_status.csv")
    save_table(boxes, output / "annotation_status.csv")
    return images, boxes


def choose_split(images: pd.DataFrame, boxes: pd.DataFrame, seed: int) -> tuple[dict, dict]:
    """그룹·train 제품 범위를 유지하고 사진 수 80:20 근접도를 먼저 최적화한다."""
    return optimize_group_split(images, boxes, seed=seed, val_fraction=0.2)


def prepare(
    root: Path,
    eda: Path,
    output: Path,
    legacy: Path | None,
    legacy_log: Path | None,
    seed: int = 42,
    review_csvs: list[Path] | None = None,
) -> dict:
    """원본을 바꾸지 않고 검증 기록과 YOLO 형식 학습 후보를 만든다."""
    root, output = root.resolve(), output.resolve()
    if output.exists() or output == root or output.is_relative_to(root):
        raise ValueError("출력은 원본 밖의 새로운 폴더를 지정하세요.")
    output.mkdir(parents=True)
    qa = output / "qa"
    qa.mkdir()
    images, thumbnails = inspect_images(root, eda, qa)
    boxes, errors = parse_annotations(root, eda, qa)
    snapshots = pd.read_csv(qa / "json_inputs.csv")
    if not snapshots.matches_eda_snapshot.all():
        raise ValueError("JSON이 EDA 시점과 달라 기존 판정 근거를 자동 적용하지 않습니다.")
    review_csvs = review_csvs or []
    evidence_paths = [
        eda / "summary.json",
        eda / "tables/json_manifest.csv",
        eda / "tables/image_inventory.csv",
        *review_csvs,
    ]
    evidence_paths.extend(path for path in [legacy, legacy_log] if path is not None)
    save_table(
        [
            {"path": str(path.resolve()), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
            for path in evidence_paths
        ],
        qa / "evidence_inputs.csv",
    )
    images, boxes = validate(
        boxes, images, errors, thumbnails, eda, legacy, legacy_log, review_csvs, qa
    )
    assignment, split_summary = choose_split(images, boxes, seed)
    retained = boxes.loc[boxes.status.eq("PASS")].copy()
    original_classes = set(boxes.class_id)
    class_ids = sorted(retained.class_id.unique())
    class_map = {int(class_id): index for index, class_id in enumerate(class_ids)}
    names = {
        index: str(retained.loc[retained.class_id.eq(class_id), "pill_name"].iloc[0])
        for class_id, index in class_map.items()
    }
    save_table(
        [
            {
                "yolo_class_id": index,
                "source_class_id": class_id,
                "product_code": f"K-{class_id:06d}",
                "pill_name": names[index],
            }
            for class_id, index in class_map.items()
        ],
        output / "class_map.csv",
    )
    manifest = []
    image_lookup = images.set_index("file_name")
    for split in ["train", "val"]:
        (output / "images" / split).mkdir(parents=True)
        (output / "labels" / split).mkdir(parents=True)
    for name, split in sorted(assignment.items()):
        image = image_lookup.loc[name]
        labels = []
        for row in retained.loc[retained.file_name.eq(name)].itertuples():
            normalized = [
                (row.x + row.w / 2) / image.width,
                (row.y + row.h / 2) / image.height,
                row.w / image.width,
                row.h / image.height,
            ]
            assert all(math.isfinite(value) and 0 <= value <= 1 for value in normalized)
            labels.append(
                f"{class_map[int(row.class_id)]} "
                + " ".join(f"{value:.10f}" for value in normalized)
            )
        assert labels
        shutil.copy2(root / image.source_path, output / "images" / split / name)
        (output / "labels" / split / (Path(name).stem + ".txt")).write_text(
            "\n".join(labels) + "\n", encoding="utf-8"
        )
        manifest.append(
            {
                "file_name": name,
                "split": split,
                "sha256": image.sha256,
                "split_group": image.split_group,
                "source_path": image.source_path,
                "bbox_count": len(labels),
                "status": "PASS",
                "semantic_identity": "UNVERIFIED",
            }
        )
    save_table(manifest, output / "split_manifest.csv")
    # data.yaml은 설정된 출력 경로를 사용하며 원본 데이터 경로를 하드코딩하지 않는다.
    config = {"path": str(output), "train": "images/train", "val": "images/val", "names": names}
    (output / "data.yaml").write_text(
        yaml.safe_dump(config, allow_unicode=True, sort_keys=False), encoding="utf-8"
    )
    summary = {
        "source_images": len(images),
        "source_annotations": len(boxes),
        "pass_images": int(images.status.eq("PASS").sum()),
        "review_images": int(images.status.eq("REVIEW").sum()),
        "reject_images": 0,
        "retained_annotations": len(retained),
        "class_count": len(class_ids),
        "classes_lost_by_quarantine": sorted(original_classes - set(class_ids)),
        "split": split_summary,
        "test_images_used": 0,
        "source_modified": False,
        "status_meaning": "정의된 구조 검사 통과 후보, 제품 정체 정확성 보증 아님",
        "code_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    (output / "preparation_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return summary


def main() -> None:
    """입력과 EDA 경로를 명시해 내부 데이터만 준비한다."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=PROJECT_ROOT / "data")
    parser.add_argument("--eda-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--legacy-quarantine", type=Path)
    parser.add_argument("--legacy-review-log", type=Path)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--review-csv", type=Path, action="append", default=[])
    args = parser.parse_args()
    print(
        json.dumps(
            prepare(
                args.data_dir,
                args.eda_dir,
                args.output_dir,
                args.legacy_quarantine,
                args.legacy_review_log,
                args.seed,
                args.review_csv,
            ),
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

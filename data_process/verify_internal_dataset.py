"""생성된 YOLO 데이터셋의 좌표·원본 보존·누수를 독립적으로 대조한다."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml


def digest(path: Path) -> str:
    """실제 읽은 파일의 SHA256을 구한다."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify_dataset(dataset: Path, raw: Path, seed: int = 42) -> dict:
    """실제 파일·전체 라벨과 원본 지문을 대조하고 PASS 감사 표본을 별도 남긴다."""
    dataset, raw = dataset.resolve(), raw.resolve()
    summary = json.loads((dataset / "preparation_summary.json").read_text(encoding="utf-8"))
    manifest = pd.read_csv(dataset / "split_manifest.csv")
    images = pd.read_csv(dataset / "qa/image_status.csv")
    boxes = pd.read_csv(dataset / "qa/annotation_status.csv")
    classes = pd.read_csv(dataset / "class_map.csv")
    config = yaml.safe_load((dataset / "data.yaml").read_text(encoding="utf-8"))
    assert set(classes.yolo_class_id) == set(range(len(classes)))
    assert set(config["names"]) == set(classes.yolo_class_id)
    assert Path(config["path"]).resolve() == dataset
    assert manifest.file_name.is_unique
    assert set(manifest.file_name) == set(images.loc[images.status.eq("PASS"), "file_name"])
    assert not set(manifest.file_name) & set(images.loc[images.status.eq("REVIEW"), "file_name"])
    train = manifest.loc[manifest.split.eq("train")]
    valid = manifest.loc[manifest.split.eq("val")]
    assert not set(train.split_group) & set(valid.split_group)
    assert not set(train.sha256) & set(valid.sha256)
    captures = images.set_index("file_name").capture_group
    assert not set(train.file_name.map(captures)) & set(valid.file_name.map(captures))
    source_hashes = images.set_index("file_name").sha256
    for row in images.itertuples():
        assert digest(raw / row.source_path) == row.sha256, row.file_name
    inputs = pd.read_csv(dataset / "qa/json_inputs.csv")
    for row in inputs.itertuples():
        assert digest(raw / row.source_json) == row.sha256, row.source_json
    class_map = dict(zip(classes.source_class_id, classes.yolo_class_id, strict=True))
    sizes = images.set_index("file_name")[["width", "height"]]
    file_hashes, split_counts = [], []
    label_count = 0
    for row in manifest.itertuples():
        image = dataset / "images" / row.split / row.file_name
        label = dataset / "labels" / row.split / (Path(row.file_name).stem + ".txt")
        assert image.is_file() and label.is_file()
        assert digest(image) == source_hashes[row.file_name]
        lines = [
            line.split() for line in label.read_text(encoding="utf-8").splitlines() if line.strip()
        ]
        original = boxes.loc[boxes.file_name.eq(row.file_name)]
        assert len(lines) == len(original) == row.bbox_count
        width, height = sizes.loc[row.file_name]
        for values, annotation in zip(lines, original.itertuples(), strict=True):
            assert len(values) == 5 and int(values[0]) == class_map[annotation.class_id]
            center_x, center_y, box_width, box_height = map(float, values[1:])
            assert np.isfinite([center_x, center_y, box_width, box_height]).all()
            assert box_width > 0 and box_height > 0
            assert center_x - box_width / 2 >= -1e-9
            assert center_y - box_height / 2 >= -1e-9
            assert center_x + box_width / 2 <= 1 + 1e-9
            assert center_y + box_height / 2 <= 1 + 1e-9
            reconstructed = [
                (center_x - box_width / 2) * width,
                (center_y - box_height / 2) * height,
                box_width * width,
                box_height * height,
            ]
            assert np.allclose(reconstructed, json.loads(annotation.bbox), atol=1e-6, rtol=0)
        label_count += len(lines)
        file_hashes.append(
            {
                "image": image.relative_to(dataset).as_posix(),
                "image_sha256": digest(image),
                "label": label.relative_to(dataset).as_posix(),
                "label_sha256": digest(label),
            }
        )
    assert label_count == summary["retained_annotations"]
    pairs = pd.read_csv(dataset / "qa/similar_pairs.csv")
    assigned = dict(zip(manifest.file_name, manifest.split, strict=True))
    for row in pairs.itertuples():
        if row.first in assigned and row.second in assigned:
            assert assigned[row.first] == assigned[row.second]
    for split in ["train", "val"]:
        selected = set(manifest.loc[manifest.split.eq(split), "file_name"])
        subset = boxes.loc[boxes.file_name.isin(selected)]
        split_counts.append(
            {
                "split": split,
                "images": len(selected),
                "annotations": len(subset),
                "classes": subset.class_id.nunique(),
            }
        )
    pd.DataFrame(file_hashes).to_csv(
        dataset / "qa/generated_file_hashes.csv", index=False, encoding="utf-8-sig"
    )
    pd.DataFrame(split_counts).to_csv(
        dataset / "qa/split_counts.csv", index=False, encoding="utf-8-sig"
    )
    support = boxes.loc[boxes.status.eq("PASS")].merge(
        manifest[["file_name", "split"]], on="file_name"
    )
    table = (
        support.groupby(["class_id", "pill_name", "split"])
        .size()
        .unstack(fill_value=0)
        .reset_index()
    )
    table.to_csv(dataset / "qa/class_split_support.csv", index=False, encoding="utf-8-sig")
    # 감사 표본은 승인으로 취급하지 않고 사용자 검수 대기 상태로 보존한다.
    generator = np.random.default_rng(seed)
    samples = []
    for class_id, group in support.groupby("class_id"):
        row = group.iloc[int(generator.integers(len(group)))]
        samples.append(
            {
                "cohort": "B",
                "source": "internal",
                "source_class_id": int(class_id),
                "uid": row.uid,
                "file_name": row.file_name,
                "split": row.split,
                "population_n": len(group),
                "drawn_n": 1,
                "inclusion_probability": 1 / len(group),
                "seed": seed,
                "human_decision": "PENDING",
                "reviewer": "",
                "note": "",
            }
        )
    pd.DataFrame(samples).to_csv(
        dataset / "qa/pass_audit_sample.csv", index=False, encoding="utf-8-sig"
    )
    findings = pd.read_csv(dataset / "qa/findings.csv")
    metrics = (
        findings.groupby("rule_id")
        .agg(findings=("file_name", "size"), images=("file_name", "nunique"))
        .reset_index()
    )
    metrics.to_csv(dataset / "qa/rule_counts.csv", index=False, encoding="utf-8-sig")
    verification = {
        "source_image_hashes_unchanged": len(images),
        "source_json_hashes_unchanged": len(inputs),
        "copied_images_verified": len(manifest),
        "yolo_labels_roundtrip_verified": label_count,
        "classes": len(classes),
        "capture_group_overlap": 0,
        "split_group_overlap": 0,
        "exact_hash_overlap": 0,
        "phash_rule_pair_overlap": 0,
        "test_images_used": 0,
        "split": split_counts,
        "pass_audit_drawn_annotations": len(samples),
        "pass_audit_unique_images": len({row["file_name"] for row in samples}),
        "pass_audit_human_completed": 0,
        "training_started": False,
        "semantic_identity_verified": False,
        "code_sha256": digest(Path(__file__)),
    }
    (dataset / "qa/dataset_validation.json").write_text(
        json.dumps(verification, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return verification


def main() -> None:
    """생성된 데이터만 검증하고 모델 학습을 호출하지 않는다."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--raw-root", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(verify_dataset(args.dataset, args.raw_root), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

"""기존 QA와 그룹을 고정한 채 데이터셋의 분할만 새 폴더에 생성한다."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path

import pandas as pd
import yaml

from data_process.group_split import optimize_group_split


def resplit_dataset(source: Path, output: Path, seed: int = 42) -> dict:
    """QA 판정·사진·라벨·그룹을 바꾸지 않고 비율을 최우선으로 재분할한다."""
    source, output = source.resolve(), output.resolve()
    if output.exists() or output.is_relative_to(source):
        raise ValueError("기존 데이터셋 밖의 새로운 출력 폴더를 지정하세요.")
    images = pd.read_csv(source / "qa/image_status.csv")
    annotations = pd.read_csv(source / "qa/annotation_status.csv")
    previous_manifest = pd.read_csv(source / "split_manifest.csv")
    previous = json.loads((source / "preparation_summary.json").read_text(encoding="utf-8"))
    assignment, split_summary = optimize_group_split(images, annotations, seed=seed)
    assert set(assignment) == set(previous_manifest.file_name)
    output.mkdir(parents=True)
    # 분할에 의존하는 기존 검증 증명은 복사하지 않고 새 분할을 검증한 후 다시 생성한다.
    derived = {
        "dataset_validation.json",
        "generated_file_hashes.csv",
        "split_counts.csv",
        "class_split_support.csv",
        "pass_audit_sample.csv",
        "rule_counts.csv",
    }
    shutil.copytree(source / "qa", output / "qa", ignore=lambda _, names: set(names) & derived)
    shutil.copy2(source / "class_map.csv", output / "class_map.csv")
    for split in ["train", "val"]:
        (output / "images" / split).mkdir(parents=True)
        (output / "labels" / split).mkdir(parents=True)
    manifest = previous_manifest.copy()
    for row in previous_manifest.itertuples():
        split = assignment[row.file_name]
        image = source / "images" / row.split / row.file_name
        label_name = Path(row.file_name).stem + ".txt"
        label = source / "labels" / row.split / label_name
        assert hashlib.sha256(image.read_bytes()).hexdigest() == row.sha256
        shutil.copy2(image, output / "images" / split / row.file_name)
        shutil.copy2(label, output / "labels" / split / label_name)
    manifest["split"] = manifest.file_name.map(assignment)
    for field in ["file_name", "sha256", "split_group", "source_path", "bbox_count", "status"]:
        assert manifest[field].equals(previous_manifest[field])
    manifest.to_csv(output / "split_manifest.csv", index=False, encoding="utf-8-sig")
    groups = manifest[["file_name", "split_group", "sha256"]].sort_values("file_name")
    groups.to_csv(output / "fixed_group_manifest.csv", index=False, encoding="utf-8-sig")
    config = yaml.safe_load((source / "data.yaml").read_text(encoding="utf-8"))
    config["path"] = str(output)
    (output / "data.yaml").write_text(
        yaml.safe_dump(config, allow_unicode=True, sort_keys=False), encoding="utf-8"
    )
    summary = {
        **previous,
        "split": split_summary,
        "parent_dataset": str(source),
        "parent_split": previous["split"],
        "resplit_code_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "split_solver_code_sha256": hashlib.sha256(
            Path(__file__).with_name("group_split.py").read_bytes()
        ).hexdigest(),
        "fixed_group_manifest_sha256": hashlib.sha256(
            (output / "fixed_group_manifest.csv").read_bytes()
        ).hexdigest(),
        "training_started": False,
    }
    (output / "preparation_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return summary


def main() -> None:
    """기존 준비 폴더에서 분할만 다시 계산하며 학습이나 원본 QA는 재실행하지 않는다."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    print(
        json.dumps(
            resplit_dataset(args.source, args.output, args.seed), ensure_ascii=False, indent=2
        )
    )


if __name__ == "__main__":
    main()

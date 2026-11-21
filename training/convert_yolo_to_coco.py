#!/usr/bin/env python3
"""Convert the project's YOLO labels to COCO JSON for RT-DETRv2.

Only annotations are generated; images are referenced in place, so no image
copies are made.

    .venv-train/bin/python training/convert_yolo_to_coco.py

Output:
    dataset/RDD2022/coco/annotations/instances_{train,val,test}2017.json
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from PIL import Image

REPO = Path(__file__).resolve().parent.parent
DEFAULT_ROOT = REPO / "dataset" / "RDD2022"
DEFAULT_OUT = DEFAULT_ROOT / "coco"
CLASS_NAMES = ["D00", "D10", "D20", "D40"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--splits", nargs="+", default=["train", "val", "test"])
    return parser.parse_args()


def image_size(path: Path) -> tuple[int, int]:
    # PIL reads only the JPEG header here; it does not decode the full image.
    with Image.open(path) as image:
        return image.size


def convert_split(root: Path, split: str) -> tuple[dict, dict]:
    image_dir = root / "images" / split
    label_dir = root / "labels" / split
    if not image_dir.is_dir() or not label_dir.is_dir():
        raise SystemExit(f"missing {image_dir} or {label_dir}")

    images = sorted(image_dir.glob("*.jpg")) + sorted(image_dir.glob("*.png"))
    labels = {path.stem: path for path in label_dir.glob("*.txt")}
    image_stems = {path.stem for path in images}
    if image_stems != set(labels):
        missing = len(image_stems - set(labels))
        extra = len(set(labels) - image_stems)
        raise SystemExit(f"{split}: image/label mismatch missing={missing} extra={extra}")

    coco_images = []
    coco_annotations = []
    per_class = Counter()
    empty = 0
    skipped_boxes = 0
    clamped_boxes = 0
    annotation_id = 1

    for image_id, image_path in enumerate(images, start=1):
        width, height = image_size(image_path)
        coco_images.append({
            "id": image_id,
            "file_name": image_path.name,
            "width": width,
            "height": height,
        })

        lines = [line for line in labels[image_path.stem].read_text().splitlines()
                 if line.strip()]
        if not lines:
            empty += 1

        for line in lines:
            parts = line.split()
            if len(parts) != 5:
                skipped_boxes += 1
                continue
            try:
                class_id = int(parts[0])
                cx, cy, bw, bh = map(float, parts[1:])
            except ValueError:
                skipped_boxes += 1
                continue
            if class_id < 0 or class_id >= len(CLASS_NAMES):
                skipped_boxes += 1
                continue

            # YOLO normalized cx/cy/w/h -> COCO absolute x/y/w/h.
            x1 = (cx - bw / 2.0) * width
            y1 = (cy - bh / 2.0) * height
            x2 = (cx + bw / 2.0) * width
            y2 = (cy + bh / 2.0) * height

            # Normalized YOLO coordinates can overshoot by ~1e-3 px after
            # floating-point conversion; clamp so COCO boxes stay in bounds.
            cx1 = min(max(x1, 0.0), float(width))
            cy1 = min(max(y1, 0.0), float(height))
            cx2 = min(max(x2, 0.0), float(width))
            cy2 = min(max(y2, 0.0), float(height))
            if (cx1, cy1, cx2, cy2) != (x1, y1, x2, y2):
                clamped_boxes += 1
            box_w = cx2 - cx1
            box_h = cy2 - cy1
            if box_w <= 0 or box_h <= 0:
                skipped_boxes += 1
                continue

            per_class[CLASS_NAMES[class_id]] += 1
            coco_annotations.append({
                "id": annotation_id,
                "image_id": image_id,
                "category_id": class_id,
                "bbox": [cx1, cy1, box_w, box_h],
                "area": box_w * box_h,
                "iscrowd": 0,
            })
            annotation_id += 1

    dataset = {
        "info": {
            "description": "RDD2022 Japan 4-class road damage",
            "version": "1.0",
            "year": datetime.now(timezone.utc).year,
        },
        "licenses": [],
        "images": coco_images,
        "annotations": coco_annotations,
        "categories": [
            {"id": index, "name": name, "supercategory": "road_damage"}
            for index, name in enumerate(CLASS_NAMES)
        ],
    }
    stats = {
        "images": len(coco_images),
        "objects": len(coco_annotations),
        "empty": empty,
        "skipped_boxes": skipped_boxes,
        "clamped_boxes": clamped_boxes,
        "per_class": dict(per_class),
    }
    return dataset, stats


def main() -> int:
    args = parse_args()
    out = args.out / "annotations"
    out.mkdir(parents=True, exist_ok=True)

    report = {"generated_at": datetime.now(timezone.utc).isoformat(),
              "root": str(args.root.resolve()),
              "out": str(out.resolve()),
              "classes": CLASS_NAMES,
              "splits": {}}

    for split in args.splits:
        dataset, stats = convert_split(args.root, split)
        path = out / f"instances_{split}2017.json"
        path.write_text(json.dumps(dataset, ensure_ascii=False), encoding="utf-8")
        report["splits"][split] = stats
        print(f"{split:5} images={stats['images']:5d} objects={stats['objects']:6d} "
              f"empty={stats['empty']:4d} skipped={stats['skipped_boxes']} "
              f"clamped={stats['clamped_boxes']} -> {path}")
        print("      " + "  ".join(f"{name}={count}" for name, count in stats["per_class"].items()))

    report_path = args.out / "convert_report.json"
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False),
                           encoding="utf-8")
    print(f"report: {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

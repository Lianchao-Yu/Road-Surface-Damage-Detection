#!/usr/bin/env python3
"""Convert the annotated RDD2022 Japan split from VOC XML to YOLO format.

First model version uses 4 damage classes (D00/D10/D20/D40). The distractor
classes D43/D44/D50 are excluded from labels and counted in the report; images
that only contain distractors become background samples.

Pipeline:
  1. Pair every image with its XML and validate both directions.
  2. Validate image readability, XML size vs real image size, class names,
     and every bounding box (drop invalid, clamp slightly out-of-range).
  3. Detect duplicate image content (MD5) before writing anything.
  4. Build YOLO labels in memory.
  5. Split into train/val/test, stratified by the set of classes present.
  6. Copy images and write labels; the source tree is never modified.

Always run the dry run first:

    .venv-train/bin/python training/convert_voc.py --dry-run

Then the real conversion:

    .venv-train/bin/python training/convert_voc.py --clean
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import shutil
import sys
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from PIL import Image

ALL_CLASSES = ["D00", "D10", "D20", "D40", "D43", "D44", "D50"]
CLASSES: list[str] = []
CLASS_TO_ID: dict[str, int] = {}
SPLITS = ("train", "val", "test")


@dataclass
class Sample:
    stem: str
    image: Path
    width: int
    height: int
    boxes: list[tuple[int, float, float, float, float]] = field(default_factory=list)
    digest: str = ""

    @property
    def classes(self) -> tuple[str, ...]:
        return tuple(sorted({CLASSES[box[0]] for box in self.boxes}))


@dataclass
class Issues:
    missing_xml: list[str] = field(default_factory=list)
    missing_image: list[str] = field(default_factory=list)
    filename_mismatch: list[str] = field(default_factory=list)
    unreadable_image: list[str] = field(default_factory=list)
    size_mismatch: list[str] = field(default_factory=list)
    parse_error: list[str] = field(default_factory=list)
    unknown_class: Counter = field(default_factory=Counter)
    excluded_class: Counter = field(default_factory=Counter)
    invalid_box: Counter = field(default_factory=Counter)
    clipped_box: Counter = field(default_factory=Counter)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("dataset/RDD2022/Japan/train"),
                        help="annotated source split (contains images/ and annotations/xmls/)")
    parser.add_argument("--out", type=Path, default=Path("dataset/RDD2022"),
                        help="output dataset root")
    parser.add_argument("--ratios", type=float, nargs=3, default=(0.8, 0.1, 0.1),
                        metavar=("TRAIN", "VAL", "TEST"))
    parser.add_argument("--classes", type=int, choices=(4, 7), default=4,
                        help="4 = D00/D10/D20/D40 (first version), 7 = also keep D43/D44/D50")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dry-run", action="store_true",
                        help="validate and split, but write nothing")
    parser.add_argument("--skip-duplicate-check", action="store_true",
                        help="skip MD5 duplicate detection (slower dry runs but never written)")
    parser.add_argument("--allow-duplicates", action="store_true",
                        help="continue even if identical images land in different splits")
    parser.add_argument("--clean", action="store_true",
                        help="remove previously generated split files before writing")
    parser.add_argument("--strict", action="store_true",
                        help="abort if any pairing/size/box/class issue is found")
    parser.add_argument("--drop-empty", action="store_true",
                        help="exclude images with zero objects (794 background images)")
    parser.add_argument("--allow-size-mismatch", action="store_true",
                        help="use the real image size when XML <size> disagrees")
    parser.add_argument("--copy-holdout", action="store_true",
                        help="also copy unannotated Japan/test images to holdout/images")
    parser.add_argument("--max-issue-log", type=int, default=20,
                        help="how many examples of each issue to print")
    return parser.parse_args()


def configure_classes(count: int) -> None:
    global CLASSES, CLASS_TO_ID
    CLASSES = ALL_CLASSES[:count]
    CLASS_TO_ID = {name: index for index, name in enumerate(CLASSES)}


def file_digest(path: Path) -> str:
    digest = hashlib.md5()
    with open(path, "rb") as handle:
        while chunk := handle.read(1 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def validate_pairing(root: Path, issues: Issues) -> list[tuple[str, Path, Path]]:
    image_dir = root / "images"
    xml_dir = root / "annotations" / "xmls"
    if not image_dir.is_dir() or not xml_dir.is_dir():
        raise SystemExit(f"missing input directories under {root}")

    images = {path.stem: path for path in image_dir.glob("*.jpg")}
    xmls = {path.stem: path for path in xml_dir.glob("*.xml")}

    issues.missing_xml = sorted(set(images) - set(xmls))
    issues.missing_image = sorted(set(xmls) - set(images))
    return [(stem, images[stem], xmls[stem]) for stem in sorted(set(images) & set(xmls))]


def read_image_size(path: Path, issues: Issues) -> tuple[int, int] | None:
    try:
        with Image.open(path) as image:
            size = image.size
            image.verify()
        return size
    except Exception:
        issues.unreadable_image.append(path.name)
        return None


def parse_sample(stem: str, image_path: Path, xml_path: Path, args: argparse.Namespace,
                 issues: Issues) -> Sample | None:
    try:
        root = ET.parse(xml_path).getroot()
    except ET.ParseError:
        issues.parse_error.append(xml_path.name)
        return None

    xml_name = (root.findtext("filename") or "").strip()
    if xml_name and Path(xml_name).name != image_path.name:
        issues.filename_mismatch.append(f"{xml_path.name}: filename={xml_name}")
        return None

    size = root.find("size")
    if size is None:
        issues.parse_error.append(f"{xml_path.name}: missing <size>")
        return None
    try:
        xml_width = int(float(size.findtext("width")))
        xml_height = int(float(size.findtext("height")))
    except (TypeError, ValueError):
        issues.parse_error.append(f"{xml_path.name}: invalid <size>")
        return None

    real_size = read_image_size(image_path, issues)
    if real_size is None:
        return None
    width, height = real_size
    if (width, height) != (xml_width, xml_height):
        issues.size_mismatch.append(
            f"{stem}: xml={xml_width}x{xml_height} image={width}x{height}")
        if not args.allow_size_mismatch:
            return None

    sample = Sample(stem=stem, image=image_path, width=width, height=height)
    for obj in root.findall("object"):
        name = (obj.findtext("name") or "").strip()
        if name not in ALL_CLASSES:
            issues.unknown_class[name] += 1
            continue
        if name not in CLASS_TO_ID:
            issues.excluded_class[name] += 1
            continue

        box = obj.find("bndbox")
        if box is None:
            issues.invalid_box[name] += 1
            continue
        try:
            x1 = float(box.findtext("xmin"))
            y1 = float(box.findtext("ymin"))
            x2 = float(box.findtext("xmax"))
            y2 = float(box.findtext("ymax"))
        except (TypeError, ValueError):
            issues.invalid_box[name] += 1
            continue

        if not all(map(lambda v: v == v, (x1, y1, x2, y2))):  # NaN check
            issues.invalid_box[name] += 1
            continue
        if x2 <= x1 or y2 <= y1:
            issues.invalid_box[name] += 1
            continue

        cx1, cy1 = max(0.0, x1), max(0.0, y1)
        cx2, cy2 = min(float(width), x2), min(float(height), y2)
        if (cx1, cy1, cx2, cy2) != (x1, y1, x2, y2):
            issues.clipped_box[name] += 1
        if cx2 <= cx1 or cy2 <= cy1:
            issues.invalid_box[name] += 1
            continue

        sample.boxes.append((
            CLASS_TO_ID[name],
            ((cx1 + cx2) / 2.0) / width,
            ((cy1 + cy2) / 2.0) / height,
            (cx2 - cx1) / width,
            (cy2 - cy1) / height,
        ))
    return sample


def split_samples(samples: list[Sample], ratios: tuple[float, float, float],
                  seed: int) -> dict[str, list[Sample]]:
    groups: dict[tuple[str, ...], list[Sample]] = defaultdict(list)
    for sample in samples:
        groups[sample.classes].append(sample)

    rng = random.Random(seed)
    result: dict[str, list[Sample]] = {name: [] for name in SPLITS}
    for _, items in sorted(groups.items(), key=lambda kv: (-len(kv[1]), str(kv[0]))):
        items = list(items)
        rng.shuffle(items)
        n = len(items)
        n_val = int(round(n * ratios[1]))
        n_test = int(round(n * ratios[2]))
        if n >= 3:
            n_val = max(1, n_val)
            n_test = max(1, n_test)
        while n_val + n_test >= n and (n_val or n_test):
            if n_val >= n_test and n_val:
                n_val -= 1
            elif n_test:
                n_test -= 1
            else:
                break
        result["val"].extend(items[:n_val])
        result["test"].extend(items[n_val:n_val + n_test])
        result["train"].extend(items[n_val + n_test:])

    # Make sure every class appears in val/test when it exists in train.
    for cls in CLASSES:
        for split in ("val", "test"):
            if any(cls in sample.classes for sample in result[split]):
                continue
            for index, sample in enumerate(result["train"]):
                if cls in sample.classes:
                    result[split].append(result["train"].pop(index))
                    break

    for name in SPLITS:
        result[name].sort(key=lambda sample: sample.stem)
    return result


def split_stats(samples: list[Sample]) -> dict:
    per_class_objects = Counter()
    per_class_images = Counter()
    for sample in samples:
        for class_id, *_ in sample.boxes:
            per_class_objects[CLASSES[class_id]] += 1
        for cls in sample.classes:
            per_class_images[cls] += 1
    return {
        "images": len(samples),
        "objects": sum(len(sample.boxes) for sample in samples),
        "empty_images": sum(not sample.boxes for sample in samples),
        "per_class_objects": {cls: per_class_objects[cls] for cls in CLASSES},
        "per_class_images": {cls: per_class_images[cls] for cls in CLASSES},
    }


def clean_outputs(out: Path) -> None:
    removed = 0
    for split in SPLITS:
        for directory in (out / "images" / split, out / "labels" / split):
            if not directory.is_dir():
                continue
            for path in directory.iterdir():
                if path.is_file():
                    path.unlink()
                    removed += 1
    data_yaml = out / "data.yaml"
    if data_yaml.exists():
        data_yaml.unlink()
        removed += 1
    print(f"cleaned generated files: {removed}")


def outputs_exist(out: Path) -> bool:
    for split in SPLITS:
        for directory in (out / "images" / split, out / "labels" / split):
            if directory.is_dir() and any(path.is_file() for path in directory.iterdir()):
                return True
    return (out / "data.yaml").exists()


def write_outputs(out: Path, splits: dict[str, list[Sample]], args: argparse.Namespace) -> None:
    for split in SPLITS:
        image_dir = out / "images" / split
        label_dir = out / "labels" / split
        image_dir.mkdir(parents=True, exist_ok=True)
        label_dir.mkdir(parents=True, exist_ok=True)
        for sample in splits[split]:
            shutil.copy2(sample.image, image_dir / sample.image.name)
            lines = [
                f"{class_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}"
                for class_id, cx, cy, w, h in sample.boxes
            ]
            text = "\n".join(lines)
            if text:
                text += "\n"
            (label_dir / f"{sample.stem}.txt").write_text(text, encoding="utf-8")

    data_yaml = out / "data.yaml"
    lines = [
        f"path: {out.resolve()}",
        "train: images/train",
        "val: images/val",
        "test: images/test",
        "names:",
    ]
    lines += [f"  {index}: {name}" for index, name in enumerate(CLASSES)]
    data_yaml.write_text("\n".join(lines) + "\n", encoding="utf-8")

    if args.copy_holdout:
        source = args.root.parent / "test" / "images"
        if not source.is_dir():
            source = args.root / "test" / "images"
        target = out / "holdout" / "images"
        if source.is_dir():
            target.mkdir(parents=True, exist_ok=True)
            copied = 0
            for image in sorted(source.glob("*.jpg")):
                shutil.copy2(image, target / image.name)
                copied += 1
            print(f"holdout images copied: {copied}")


def print_issue_examples(issues: Issues, limit: int) -> None:
    categories = [
        ("missing_xml", issues.missing_xml),
        ("missing_image", issues.missing_image),
        ("filename_mismatch", issues.filename_mismatch),
        ("unreadable_image", issues.unreadable_image),
        ("size_mismatch", issues.size_mismatch),
        ("parse_error", issues.parse_error),
    ]
    for name, items in categories:
        if items:
            print(f"\n{name}: {len(items)}")
            for item in items[:limit]:
                print(f"  - {item}")
    for name, counter in (("unknown_class", issues.unknown_class),
                          ("invalid_box", issues.invalid_box),
                          ("clipped_box", issues.clipped_box),
                          ("excluded_class", issues.excluded_class)):
        if counter:
            label = "excluded by --classes" if name == "excluded_class" else name
            print(f"\n{name} ({label}): {sum(counter.values())} {dict(counter)}")


def main() -> int:
    args = parse_args()
    configure_classes(args.classes)
    if not (0.0 < sum(args.ratios) <= 1.0):
        print("ratios must sum to <= 1.0", file=sys.stderr)
        return 2

    issues = Issues()
    pairs = validate_pairing(args.root, issues)
    samples: list[Sample] = []
    for stem, image_path, xml_path in pairs:
        sample = parse_sample(stem, image_path, xml_path, args, issues)
        if sample is not None and (sample.boxes or not args.drop_empty):
            samples.append(sample)

    if not samples:
        print("no valid samples found", file=sys.stderr)
        return 1

    if not args.skip_duplicate_check:
        for sample in samples:
            sample.digest = file_digest(sample.image)
        by_digest: dict[str, list[str]] = defaultdict(list)
        for sample in samples:
            by_digest[sample.digest].append(sample.stem)
        duplicate_groups = {digest: stems for digest, stems in by_digest.items() if len(stems) > 1}
    else:
        duplicate_groups = {}

    hard_issues = (
        issues.missing_xml or issues.missing_image or issues.filename_mismatch
        or issues.unreadable_image or issues.size_mismatch or issues.parse_error
        or issues.unknown_class or issues.invalid_box
    )
    if args.strict and hard_issues:
        print("strict mode: annotation issues found, aborting", file=sys.stderr)
        print_issue_examples(issues, args.max_issue_log)
        return 1

    splits = split_samples(samples, tuple(args.ratios), args.seed)
    stats = {name: split_stats(items) for name, items in splits.items()}

    digest_splits: dict[str, set[str]] = defaultdict(set)
    for name, items in splits.items():
        for sample in items:
            if sample.digest:
                digest_splits[sample.digest].add(name)
    cross_split = {digest: sorted(names) for digest, names in digest_splits.items()
                   if len(names) > 1}

    print(f"classes          : {len(CLASSES)} ({', '.join(CLASSES)})")
    if issues.excluded_class:
        print(f"excluded objects : {dict(issues.excluded_class)}")
    print(f"duplicate groups : {len(duplicate_groups)}")
    print(f"cross-split dups : {len(cross_split)}")

    print("\n=== input ===")
    print(f"paired image/xml : {len(pairs)}")
    print(f"valid samples    : {len(samples)}")
    print(f"objects kept     : {sum(len(sample.boxes) for sample in samples)}")
    print_issue_examples(issues, args.max_issue_log)

    print("\n=== split ===")
    for name in SPLITS:
        item = stats[name]
        print(f"{name:5} images={item['images']:5d} objects={item['objects']:6d} "
              f"empty={item['empty_images']:3d}")
        print("      images/class: " + "  ".join(
            f"{cls}={item['per_class_images'][cls]}" for cls in CLASSES))

    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "root": str(args.root.resolve()),
        "out": str(args.out.resolve()),
        "seed": args.seed,
        "ratios": list(args.ratios),
        "classes": CLASSES,
        "num_classes": len(CLASSES),
        "excluded_class": dict(issues.excluded_class),
        "duplicates": {
            "groups": len(duplicate_groups),
            "images_in_groups": sum(len(stems) for stems in duplicate_groups.values()),
            "cross_split": len(cross_split),
            "cross_split_examples": [
                {"splits": cross_split[digest], "stems": sorted(
                    sample.stem for sample in samples if sample.digest == digest)[:5]}
                for digest in list(cross_split)[:10]
            ],
        },
        "input": {
            "paired": len(pairs),
            "valid_samples": len(samples),
            "objects": sum(len(sample.boxes) for sample in samples),
            "missing_xml": len(issues.missing_xml),
            "missing_image": len(issues.missing_image),
            "filename_mismatch": len(issues.filename_mismatch),
            "unreadable_image": len(issues.unreadable_image),
            "size_mismatch": len(issues.size_mismatch),
            "parse_error": len(issues.parse_error),
            "unknown_class": dict(issues.unknown_class),
            "invalid_box": dict(issues.invalid_box),
            "clipped_box": dict(issues.clipped_box),
        },
        "splits": stats,
        "issues": {
            "missing_xml": issues.missing_xml,
            "missing_image": issues.missing_image,
            "filename_mismatch": issues.filename_mismatch,
            "unreadable_image": issues.unreadable_image,
            "size_mismatch": issues.size_mismatch,
            "parse_error": issues.parse_error,
        },
    }

    if cross_split and not args.allow_duplicates:
        print("\ncross-split duplicate images detected, aborting:", file=sys.stderr)
        for digest, split_names in list(cross_split.items())[:10]:
            stems = [sample.stem for sample in samples if sample.digest == digest][:5]
            print(f"  {split_names}: {stems}", file=sys.stderr)
        return 1

    if args.dry_run:
        print("\ndry run: nothing written")
        return 0

    if outputs_exist(args.out) and not args.clean:
        print(f"\n{args.out} already contains generated files; rerun with --clean",
              file=sys.stderr)
        return 1
    if args.clean:
        clean_outputs(args.out)

    write_outputs(args.out, splits, args)
    report_path = args.out / "convert_report.json"
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False),
                           encoding="utf-8")
    print(f"\nwrote: {args.out}")
    print(f"report: {report_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

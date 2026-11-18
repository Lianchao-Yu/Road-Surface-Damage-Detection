#!/usr/bin/env python3
"""Export the trained YOLO11n checkpoint to ONNX for TensorRT.

Default export is dynamic-batch, opset 17, simplified:

    .venv-train/bin/python training/export.py

The ONNX file is copied to models/ and a JSON report with hashes and tensor
shapes is written next to it.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

import onnx

REPO = Path(__file__).resolve().parent.parent
DEFAULT_WEIGHTS = REPO / "training" / "runs" / "rdd2022_yolo11n" / "weights" / "best.pt"
DEFAULT_OUTPUT = REPO / "models" / "road_damage_yolo11n.onnx"
EXPECTED_CLASSES = 4


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", type=Path, default=DEFAULT_WEIGHTS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--opset", type=int, default=17)
    parser.add_argument("--batch", type=int, default=1,
                        help="static batch size; ignored by dynamic export")
    parser.add_argument("--static", action="store_true",
                        help="export a fixed batch instead of a dynamic batch axis")
    parser.add_argument("--no-simplify", action="store_true",
                        help="skip onnxslim graph simplification")
    parser.add_argument("--device", default="0")
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def tensor_shape(value_info: onnx.ValueInfoProto) -> list:
    """Return concrete dims, or dimension names for dynamic axes."""
    shape = []
    for dim in value_info.type.tensor_type.shape.dim:
        if dim.HasField("dim_value"):
            shape.append(dim.dim_value)
        else:
            shape.append(dim.dim_param or "?")
    return shape


def inspect(path: Path, expected_opset: int, dynamic: bool) -> dict:
    model = onnx.load(path)
    onnx.checker.check_model(model)

    opset = model.opset_import[0].version if model.opset_import else -1
    inputs = [(item.name, tensor_shape(item)) for item in model.graph.input]
    outputs = [(item.name, tensor_shape(item)) for item in model.graph.output]

    if opset != expected_opset:
        raise SystemExit(f"opset mismatch: got {opset}, expected {expected_opset}")

    if dynamic:
        batch_dim = inputs[0][1][0]
        if not isinstance(batch_dim, str):
            raise SystemExit(f"dynamic export expected a named batch dim, got {batch_dim}")

    channel_dim = outputs[0][1][1]
    if isinstance(channel_dim, int):
        classes = channel_dim - 4
        if classes != EXPECTED_CLASSES:
            print(f"warning: output channels={channel_dim}, implies {classes} classes, "
                  f"expected {EXPECTED_CLASSES}")

    return {"opset": opset, "inputs": inputs, "outputs": outputs,
            "dynamic_batch": dynamic, "size_bytes": path.stat().st_size}


def main() -> int:
    args = parse_args()
    if not args.weights.is_file():
        raise SystemExit(f"weights not found: {args.weights}")
    args.output.parent.mkdir(parents=True, exist_ok=True)

    from ultralytics import YOLO  # late import keeps --help fast

    model = YOLO(str(args.weights))
    exported = Path(model.export(
        format="onnx",
        imgsz=args.imgsz,
        batch=args.batch,
        opset=args.opset,
        simplify=not args.no_simplify,
        dynamic=not args.static,
        device=args.device,
    ))

    # Move instead of copy so the run directory does not keep a duplicate.
    if exported.resolve() != args.output.resolve():
        shutil.move(str(exported), args.output)

    info = inspect(args.output, args.opset, dynamic=not args.static)
    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "weights": str(args.weights.resolve()),
        "weights_sha256": sha256(args.weights),
        "onnx": str(args.output.resolve()),
        "onnx_sha256": sha256(args.output),
        "imgsz": args.imgsz,
        "simplify": not args.no_simplify,
        **info,
    }
    report_path = args.output.with_suffix(".json")
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False),
                           encoding="utf-8")

    print("=== ONNX export ===")
    print(f"weights : {args.weights}")
    print(f"onnx    : {args.output}")
    print(f"size    : {info['size_bytes'] / 1e6:.1f} MB")
    print(f"sha256  : {report['onnx_sha256']}")
    print(f"opset   : {info['opset']}")
    print(f"input   : {info['inputs']}")
    print(f"output  : {info['outputs']}")
    print(f"dynamic : {info['dynamic_batch']}")
    print(f"report  : {report_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

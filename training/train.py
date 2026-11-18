#!/usr/bin/env python3
"""Fine-tune YOLO11n on the 4-class RDD2022 Japan dataset.

Run from anywhere; paths are resolved from the repository root:

    .venv-train/bin/python training/train.py --dry-run
    .venv-train/bin/python training/train.py

Training and validation use 640x640 letterbox (Ultralytics default), which must
match the preprocessing contract implemented on the C++/CUDA side.

Memory notes for the 8GB WSL / 8GB GPU machine:
  * dataloader workers and image caches are the main host-RAM risks;
  * CUDA OOM is retried automatically by Ultralytics (batch halved, up to 3x);
  * this script adds a preflight check and a watchdog that stops gracefully
    between batches when host RAM gets low, so last.pt/best.pt are still saved.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path

import yaml

# These must be set before torch / ultralytics import anything CUDA-related.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("MALLOC_ARENA_MAX", "2")

# ---------------------------------------------------------------------------
# Paths and constants
# ---------------------------------------------------------------------------
REPO = Path(__file__).resolve().parent.parent
DEFAULT_DATA = REPO / "dataset" / "RDD2022" / "data.yaml"
DEFAULT_PROJECT = REPO / "training" / "runs"
EXPECTED_NAMES = {0: "D00", 1: "D10", 2: "D20", 3: "D40"}
SPLITS = ("train", "val", "test")


# ---------------------------------------------------------------------------
# Command line
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="yolo11n.pt", help="pretrained checkpoint")
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=8,
                        help="reduce to 4 if CUDA or host memory is tight")
    parser.add_argument("--device", default="0", help="'0' for GPU, 'cpu' for CPU")
    # Two workers is a deliberate memory/speed trade-off for WSL.
    parser.add_argument("--workers", type=int, default=2,
                        help="dataloader workers; 0-2 is safest for 8GB WSL memory")
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--fraction", type=float, default=1.0,
                        help="fraction of the training split to use (smoke tests)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--project", type=Path, default=DEFAULT_PROJECT)
    parser.add_argument("--name", default="rdd2022_yolo11n")
    # RAM caching 10k images would exceed the 7.6GB WSL limit, so only allow disk.
    parser.add_argument("--cache", choices=("none", "disk"), default="none",
                        help="do not use RAM cache: it can exhaust WSL memory")
    parser.add_argument("--min-start-ram-gb", type=float, default=3.0,
                        help="abort before training if available RAM is below this")
    parser.add_argument("--min-free-ram-gb", type=float, default=2.0,
                        help="stop if available RAM drops below this AND swap is nearly exhausted")
    parser.add_argument("--min-swap-free-gb", type=float, default=0.5,
                        help="only stop on low RAM when free swap is also below this")
    parser.add_argument("--memory-log-every", type=int, default=50,
                        help="write a memory log row every N batches (0 disables)")
    parser.add_argument("--resume", action="store_true", help="resume from last.pt")
    parser.add_argument("--dry-run", action="store_true",
                        help="validate the dataset and print the training config only")
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Dataset validation
# ---------------------------------------------------------------------------
def load_names(cfg: dict) -> dict[int, str]:
    """Accept both YOLO list form and the {id: name} dict form."""
    names = cfg.get("names", {})
    if isinstance(names, list):
        return {index: name for index, name in enumerate(names)}
    return {int(key): value for key, value in names.items()}


def resolve_split(base: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else base / path


def validate_split(base: Path, split: str, value: str, names: dict[int, str]) -> dict:
    """Check one split: directories, image/label pairing, class ids, counts."""
    image_dir = resolve_split(base, value)
    # Generated layout is <root>/images/<split> and <root>/labels/<split>.
    label_dir = image_dir.parent.parent / "labels" / image_dir.name

    if not image_dir.is_dir():
        raise SystemExit(f"missing image directory: {image_dir}")
    if not label_dir.is_dir():
        raise SystemExit(f"missing label directory: {label_dir}")

    images = sorted(image_dir.glob("*.jpg")) + sorted(image_dir.glob("*.png"))
    labels = sorted(label_dir.glob("*.txt"))

    # A missing label file silently turns an image into a background sample,
    # so fail loudly instead.
    image_stems = {path.stem for path in images}
    label_stems = {path.stem for path in labels}
    if image_stems != label_stems:
        raise SystemExit(
            f"{split}: image/label mismatch "
            f"(images without label={len(image_stems - label_stems)}, "
            f"labels without image={len(label_stems - image_stems)})"
        )

    empty_labels = 0
    class_counts = {name: 0 for name in names.values()}
    max_class_id = -1
    for path in labels:
        lines = [line for line in path.read_text().splitlines() if line.strip()]
        if not lines:
            empty_labels += 1
            continue
        for line in lines:
            class_id = int(line.split()[0])
            max_class_id = max(max_class_id, class_id)
            if class_id in names:
                class_counts[names[class_id]] += 1

    if max_class_id >= len(names):
        raise SystemExit(f"{split}: label class id {max_class_id} >= nc {len(names)}")

    return {
        "images": len(images),
        "labels": len(labels),
        "empty_labels": empty_labels,
        "objects": sum(class_counts.values()),
        "per_class": class_counts,
    }


def check_dataset(data_yaml: Path) -> dict:
    """Validate data.yaml and every split before spending time on training."""
    if not data_yaml.is_file():
        raise SystemExit(
            f"data.yaml not found: {data_yaml}\n"
            "run training/convert_voc.py first"
        )

    cfg = yaml.safe_load(data_yaml.read_text(encoding="utf-8")) or {}
    base = Path(cfg.get("path", data_yaml.parent))
    names = load_names(cfg)

    if names != EXPECTED_NAMES:
        print(f"warning: class names are {names}, expected {EXPECTED_NAMES}")
    if not cfg.get("train") or not cfg.get("val"):
        raise SystemExit("data.yaml must define train and val")

    summary = {"data": str(data_yaml), "path": str(base), "names": names, "splits": {}}
    for split in SPLITS:
        value = cfg.get(split)
        if value:
            summary["splits"][split] = validate_split(base, split, value, names)
    return summary


def print_dataset_summary(summary: dict) -> None:
    print(f"data : {summary['data']}")
    print(f"path : {summary['path']}")
    print("names: " + ", ".join(f"{key}={value}" for key, value in summary["names"].items()))
    for split, item in summary["splits"].items():
        print(f"{split:5} images={item['images']:5d} labels={item['labels']:5d} "
              f"empty={item['empty_labels']:4d} objects={item['objects']:6d}")
        print("      " + "  ".join(f"{name}={count}" for name, count in item["per_class"].items()))


# ---------------------------------------------------------------------------
# Memory guards
# ---------------------------------------------------------------------------
def available_ram_gb() -> float:
    """MemAvailable includes reclaimable page cache, so it is a better OOM hint
    than MemFree. Returns NaN when /proc/meminfo is unavailable."""
    try:
        with open("/proc/meminfo", encoding="utf-8") as handle:
            for line in handle:
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1]) / (1024.0 ** 2)
    except OSError:
        pass
    return float("nan")


def gpu_free_mb() -> float | None:
    """Free VRAM in MB, or None when nvidia-smi is unavailable."""
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5, check=True)
        return float(result.stdout.strip().splitlines()[0])
    except Exception:
        return None


def swap_free_gb() -> float:
    """Free swap acts as an OOM buffer: a low MemAvailable is survivable while
    swap is still available, so the watchdog must consider both."""
    try:
        with open("/proc/meminfo", encoding="utf-8") as handle:
            for line in handle:
                if line.startswith("SwapFree:"):
                    return int(line.split()[1]) / (1024.0 ** 2)
    except OSError:
        pass
    return float("nan")


def memory_preflight(args: argparse.Namespace) -> None:
    """Refuse to start when the machine is already low on memory."""
    ram = available_ram_gb()
    swap = swap_free_gb()
    gpu = gpu_free_mb()
    message = f"memory: available RAM {ram:.2f} GB, swap free {swap:.2f} GB"
    if gpu is not None:
        message += f", GPU free {gpu:.0f} MB"
    print(message)

    if math.isfinite(ram) and ram < args.min_start_ram_gb:
        raise SystemExit(
            f"only {ram:.2f} GB RAM available; need >= {args.min_start_ram_gb:.1f} GB\n"
            "close other processes, reduce --workers, or increase WSL swap before training"
        )
    if gpu is not None and gpu < 1500:
        print(f"warning: only {gpu:.0f} MB GPU memory free")


def install_memory_watchdog(model, args: argparse.Namespace) -> None:
    """Stop gracefully between batches when host RAM drops below the margin.

    Ultralytics checks trainer.stop after each batch and again before the next
    epoch, then still runs validation and saves weights. This avoids the OOM
    killer while keeping a usable checkpoint. CUDA OOM is separate: Ultralytics
    halves the batch and retries up to 3 times by itself.
    """
    state = {"batches": 0}

    def write_row(trainer, ram: float, swap: float, gpu: float | None) -> None:
        save_dir = Path(getattr(trainer, "save_dir", "."))
        save_dir.mkdir(parents=True, exist_ok=True)
        path = save_dir / "memory_log.csv"
        is_new = not path.exists()
        with path.open("a", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            if is_new:
                writer.writerow(["time", "epoch", "batch", "ram_available_gb",
                                 "swap_free_gb", "gpu_free_mb", "batch_size"])
            writer.writerow([
                time.strftime("%Y-%m-%d %H:%M:%S"),
                getattr(trainer, "epoch", -1),
                state["batches"],
                f"{ram:.3f}",
                f"{swap:.3f}",
                "" if gpu is None else f"{gpu:.0f}",
                getattr(trainer, "batch_size", ""),
            ])

    def on_batch_end(trainer) -> None:
        state["batches"] += 1
        ram = available_ram_gb()
        swap = swap_free_gb()
        gpu = gpu_free_mb()

        # Only stop when both RAM and swap are nearly gone. With a healthy
        # swap file the kernel can page out instead of invoking the OOM killer.
        low_ram = math.isfinite(ram) and ram < args.min_free_ram_gb
        low_swap = not math.isfinite(swap) or swap < args.min_swap_free_gb
        if low_ram and low_swap:
            print(f"\n[memory watchdog] available RAM {ram:.2f} GB, "
                  f"swap free {swap:.2f} GB; stopping after this batch", flush=True)
            trainer.stop = True

        if args.memory_log_every and (
                state["batches"] % args.memory_log_every == 0 or (low_ram and low_swap)):
            try:
                write_row(trainer, ram, swap, gpu)
            except Exception as exc:  # logging must never kill training
                print(f"[memory watchdog] log failed: {exc}", file=sys.stderr)

    def on_epoch_end(trainer) -> None:
        try:
            write_row(trainer, available_ram_gb(), swap_free_gb(), gpu_free_mb())
        except Exception as exc:
            print(f"[memory watchdog] log failed: {exc}", file=sys.stderr)

    model.add_callback("on_train_batch_end", on_batch_end)
    model.add_callback("on_train_epoch_end", on_epoch_end)


# ---------------------------------------------------------------------------
# Training configuration
# ---------------------------------------------------------------------------
def build_train_args(args: argparse.Namespace) -> dict:
    """Translate CLI options into the Ultralytics train() argument dict."""
    return {
        "data": str(args.data),
        "epochs": args.epochs,
        "imgsz": args.imgsz,
        "batch": args.batch,
        "device": args.device,
        "workers": args.workers,
        "patience": args.patience,
        "fraction": args.fraction,
        "seed": args.seed,
        "project": str(args.project),
        "name": args.name,
        # Keep the run reproducible and GPU memory low.
        "deterministic": True,
        "amp": True,
        "pretrained": True,
        "cache": False if args.cache == "none" else "disk",
        # Disable mosaic for the last 10 epochs; standard YOLO fine-tune recipe.
        "close_mosaic": 10,
        "plots": True,
        "exist_ok": False,
    }


def resume_args(run_dir: Path, train_args: dict) -> dict:
    """Point Ultralytics at last.pt and let it restore optimizer/epoch state."""
    last = run_dir / "weights" / "last.pt"
    if not last.is_file():
        raise SystemExit(f"resume requested but {last} does not exist")
    return {**train_args, "model": str(last), "resume": True, "exist_ok": True}


def print_results(results, run_dir: Path) -> None:
    """Print the summary needed by the training log / report."""
    result_dir = Path(getattr(results, "save_dir", run_dir))
    metrics = getattr(results, "results_dict", {})
    print("\n=== finished ===")
    print(f"run dir : {result_dir}")
    print(f"best.pt : {result_dir / 'weights' / 'best.pt'}")
    for key in ("metrics/precision(B)", "metrics/recall(B)",
                "metrics/mAP50(B)", "metrics/mAP50-95(B)"):
        if key in metrics:
            print(f"{key:24s}: {metrics[key]:.4f}")

    names = getattr(results, "names", {})
    maps = getattr(getattr(results, "box", None), "maps", [])
    for name, value in zip(names.values(), maps):
        print(f"  {name}: mAP50-95={value:.4f}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main() -> int:
    args = parse_args()

    summary = check_dataset(args.data)
    print_dataset_summary(summary)
    memory_preflight(args)

    train_args = build_train_args(args)
    run_dir = args.project / args.name
    if args.resume:
        train_args = resume_args(run_dir, train_args)

    print("\ntraining config:")
    print(json.dumps(train_args, indent=2))

    # Dry run stops here: dataset is validated but PyTorch is not imported.
    if args.dry_run:
        print("\ndry run: dataset validated, training not started")
        return 0

    from ultralytics import YOLO  # late import keeps --dry-run fast

    print(f"\nsafety: batch={args.batch}, workers={args.workers}, "
          f"cache={args.cache}, AMP=on")
    print(f"        stop only if RAM < {args.min_free_ram_gb:.1f} GB "
          f"and swap free < {args.min_swap_free_gb:.1f} GB")

    model_path = train_args.pop("model", args.model)
    model = YOLO(model_path)
    install_memory_watchdog(model, args)
    results = model.train(**train_args)

    # Write our config after the run directory exists; pre-creating it would
    # make Ultralytics choose a new run name (e.g. name-2).
    result_dir = Path(getattr(results, "save_dir", run_dir))
    (result_dir / "train_config.json").write_text(
        json.dumps({"model": str(model_path), "args": train_args, "dataset": summary},
                   indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print_results(results, run_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())

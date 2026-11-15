#!/usr/bin/env python3
"""Cross-check the C++ CPU reference against PyTorch, OpenCV and NumPy.

Run with the training venv (torch + opencv):

    .venv-train/bin/python scripts/verify_preprocess_cpu.py \
        --exe build/cpu/tools/preprocess_case

The script exits non-zero if any comparison exceeds its tolerance.
"""
import argparse
import subprocess
import sys

import cv2
import numpy as np
import torch
import torch.nn.functional as F

DEFAULT_EXE = "build/cpu/tools/preprocess_case"


def resize_coords(src, dst):
    scale = float(src) / float(dst)
    coord = (np.arange(dst, dtype=np.float32) + 0.5) * scale - 0.5
    return np.clip(coord, 0.0, float(src - 1))


def numpy_ref(img, dw, dh, bgr_to_rgb, mean, std):
    """Explicit half-pixel bilinear implementation, independent of torch/OpenCV."""
    h, w, _ = img.shape
    f = img.astype(np.float32) / 255.0
    sx = resize_coords(w, dw)
    sy = resize_coords(h, dh)
    x0 = np.floor(sx).astype(np.int32)
    y0 = np.floor(sy).astype(np.int32)
    x1 = np.minimum(x0 + 1, w - 1)
    y1 = np.minimum(y0 + 1, h - 1)
    fx = (sx - x0)[None, :]
    fy = (sy - y0)[:, None]
    w00 = (1.0 - fx) * (1.0 - fy)
    w10 = fx * (1.0 - fy)
    w01 = (1.0 - fx) * fy
    w11 = fx * fy

    out = np.empty((3, dh, dw), dtype=np.float32)
    for c in range(3):
        src_c = (2 - c) if bgr_to_rgb else c
        plane = f[:, :, src_c]
        value = (
            w00 * plane[np.ix_(y0, x0)]
            + w10 * plane[np.ix_(y0, x1)]
            + w01 * plane[np.ix_(y1, x0)]
            + w11 * plane[np.ix_(y1, x1)]
        )
        out[c] = ((value - mean[c]) / std[c]).astype(np.float32)
    return out.reshape(-1)


def torch_ref(img, dw, dh, bgr_to_rgb, mean, std):
    x = torch.from_numpy(img.astype(np.float32) / 255.0)
    if bgr_to_rgb:
        x = x[..., [2, 1, 0]]
    x = x.permute(2, 0, 1).unsqueeze(0)
    x = F.interpolate(x, size=(dh, dw), mode="bilinear", align_corners=False)
    m = torch.tensor(mean, dtype=torch.float32).view(1, 3, 1, 1)
    s = torch.tensor(std, dtype=torch.float32).view(1, 3, 1, 1)
    return ((x - m) / s).numpy().reshape(-1)


def opencv_ref(img, dw, dh, bgr_to_rgb, mean, std):
    x = img.astype(np.float32) / 255.0
    if bgr_to_rgb:
        x = cv2.cvtColor(x, cv2.COLOR_BGR2RGB)
    x = cv2.resize(x, (dw, dh), interpolation=cv2.INTER_LINEAR)
    x = (x - np.asarray(mean, dtype=np.float32)) / np.asarray(std, dtype=np.float32)
    return np.transpose(x, (2, 0, 1)).reshape(-1)


def run_cpp(exe, img, dw, dh, pad, bgr_to_rgb, mean, std):
    h, w, _ = img.shape
    stride = w * 3 + pad
    buf = np.zeros((h, stride), dtype=np.uint8)
    buf[:, : w * 3] = img.reshape(h, w * 3)
    args = [
        exe, str(w), str(h), str(dw), str(dh), str(stride),
        "1" if bgr_to_rgb else "0",
    ]
    args += [f"{v:.8f}" for v in list(mean) + list(std)]
    proc = subprocess.run(args, input=buf.tobytes(), capture_output=True)
    if proc.returncode != 0:
        raise RuntimeError(proc.stderr.decode(errors="replace"))
    return np.array(proc.stdout.split(), dtype=np.float32)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--exe", default=DEFAULT_EXE)
    parser.add_argument("--cases", type=int, default=40)
    parser.add_argument("--seed", type=int, default=20261115)
    parser.add_argument("--tol-torch", type=float, default=5e-6)
    parser.add_argument("--tol-opencv", type=float, default=1e-5)
    parser.add_argument("--tol-numpy", type=float, default=5e-6)
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    fixed = [
        (1, 1, 1, 1), (1, 1, 4, 4), (2, 2, 1, 1), (3, 5, 7, 10),
        (5, 4, 2, 3), (240, 320, 480, 640), (480, 640, 240, 320),
        (7, 7, 3, 3), (13, 17, 13, 17),
    ]
    cases = list(fixed)
    for _ in range(args.cases):
        cases.append((
            int(rng.integers(1, 64)), int(rng.integers(1, 64)),
            int(rng.integers(1, 96)), int(rng.integers(1, 96)),
        ))

    worst = {"torch": (0.0, None), "opencv": (0.0, None), "numpy": (0.0, None)}
    failures = 0

    for index, (h, w, dh, dw) in enumerate(cases):
        pad = int(rng.integers(0, 17))
        bgr_to_rgb = bool(index % 2 == 0)
        mean = rng.uniform(0.0, 0.5, 3)
        std = rng.uniform(0.3, 1.5, 3)
        img = rng.integers(0, 256, size=(h, w, 3), dtype=np.uint8)

        out = run_cpp(args.exe, img, dw, dh, pad, bgr_to_rgb, mean, std)
        refs = {
            "torch": torch_ref(img, dw, dh, bgr_to_rgb, mean, std),
            "opencv": opencv_ref(img, dw, dh, bgr_to_rgb, mean, std),
            "numpy": numpy_ref(img, dw, dh, bgr_to_rgb, mean, std),
        }
        for name, ref in refs.items():
            if out.size != ref.size:
                print(f"FAIL size {name} case={index} {out.size} != {ref.size}")
                failures += 1
                continue
            diff = float(np.max(np.abs(out - ref)))
            if diff > worst[name][0]:
                worst[name] = (diff, (h, w, dh, dw, pad))
            limit = getattr(args, f"tol_{name}")
            if diff > limit:
                print(f"FAIL {name} case={index} {h}x{w}->{dh}x{dw} diff={diff:.3e} > {limit:.1e}")
                failures += 1

        # Determinism: identical input must produce bit-identical output.
        again = run_cpp(args.exe, img, dw, dh, pad, bgr_to_rgb, mean, std)
        if not np.array_equal(out, again):
            print(f"FAIL determinism case={index}")
            failures += 1

        # Constant-image invariant.
        const = np.full((h, w, 3), 137, dtype=np.uint8)
        out_c = run_cpp(args.exe, const, dw, dh, pad, bgr_to_rgb, mean, std)
        value = 137.0 / 255.0
        expected = np.repeat(
            np.asarray([(value - mean[c]) / std[c] for c in range(3)], dtype=np.float32),
            dh * dw,
        )
        if not np.allclose(out_c, expected, atol=1e-6, rtol=0):
            print(f"FAIL constant invariant case={index}")
            failures += 1

    print(f"\ncases: {len(cases)}")
    for name, (diff, case) in worst.items():
        print(f"worst vs {name:6s}: {diff:.3e}  case={case}")
    print(f"failures: {failures}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())

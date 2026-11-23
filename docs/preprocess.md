# CPU Preprocess Reference (M1)

实现位置：

- 接口与契约：`include/road_damage/preprocess_cpu.hpp`
- 参考实现：`src/cpu/preprocess_cpu.cpp`
- 测试：`tests/test_preprocess_cpu.cpp`、`tests/test_letterbox.cpp`

## 1. 作用

CPU 参考实现是后续 CUDA K1、NPP baseline 和 TensorRT 输入的**数值基准**。它只负责四个操作：

1. Bilinear Resize
2. BGR → RGB
3. Normalize
4. NCHW 排布（FP32）

**不在 M1 范围**：ROI/Tiling、FP16 转换、NMS。Letterbox 由单独的
`preprocess_bgr8_to_nchw_f32_letterbox()` 提供；直接 resize 路径仍保留给
RT-DETR 等使用直接拉伸的模型。

## 2. 坐标规范（最重要）

采用 **half-pixel / align_corners=false**：

```text
scale_x = src_width  / dst_width
scale_y = src_height / dst_height

sx = (dst_x + 0.5) * scale_x - 0.5
sy = (dst_y + 0.5) * scale_y - 0.5
```

**边缘处理**：`sx` / `sy` 先 clamp 到 `[0, src_dim - 1]`，再做双线性插值（border replicate）。这与 PyTorch `F.interpolate(mode="bilinear", align_corners=False)` 一致，也对应 ONNX Resize 的 `half_pixel` + 边缘 clamp 行为。

实现细节：

- 坐标与权重使用 FP32 计算，不做定点化。
- 源像素先除以 255 转为 `[0, 1]`，再插值、再归一化。
- 四邻域权重：`w00=(1-fx)(1-fy)`、`w10=fx(1-fy)`、`w01=(1-fx)fy`、`w11=fx*fy`。
- 当坐标为整数边界时，`x1/y1` 自动回退到自身，避免越界读取。

## 3. 通道与归一化

- 源：BGR8，每像素 3 字节，支持 `stride >= width * 3`（FFmpeg linesize）。
- `bgr_to_rgb = true` 时输出 R、G、B；纯通道交换，不做色彩空间转换。
- 归一化：

```text
v = (pixel / 255 - mean[c]) / std[c]
```

`mean` / `std` 使用 `[0, 1]` 量纲，按**输出 RGB 通道**索引。

## 4. 内存布局

```text
index = c * H * W + y * W + x      // N = 1, C = 3, FP32
```

输出由 `preprocess_bgr8_to_nchw_f32_single()` 或
`preprocess_bgr8_to_nchw_f32_letterbox_single()` 返回 `std::vector<float>`，
大小可用 `tensor_element_count(spec)` 获取。Letterbox 的 `scale`、
`padX`、`padY` 由 `compute_letterbox_meta()` 计算，并交给后处理做坐标逆映射。

## 5. 校验结果

`tests/test_preprocess_cpu.cpp` 覆盖：

| 用例 | 验证点 |
|---|---|
| ElementCount | 输出元素数 |
| IdentityResizeSwapsBgrToRgb | 恒等 resize + 通道交换 |
| BilinearVerticalHalfPixel | 2→4 垂直插值（0/25/75/100） |
| BilinearHorizontalNonIntegerScale | 3→2 非整数比例（0.25 / 1.75） |
| MatchesPyTorchBilinearGolden | 5×4→3×2 与 PyTorch golden 对齐 |
| NormalizationUsesPerChannelRgbOrder | 每通道 mean/std |
| KeepsBgrOrderWhenRequested | 关闭通道交换 |
| StridePaddingIsIgnored | pitch / 行填充 |
| OddSizesKeepEdgePixels | 奇数尺寸 + 边缘 clamp |
| OneByOneImage | 1×1 图像 |
| RejectsMalformedInput | 非法输入抛异常 |

PyTorch 对拍结果：

```text
elements=18  max_abs_diff=1.233e-07
MATCH
```

## 6. 第三方成熟库交叉验证（可复现）

使用三个相互独立的实现做逐元素对比：

| 来源 | 入口 | 说明 |
|---|---|---|
| OpenCV（C++） | `tests/test_preprocess_vs_opencv.cpp` | 进 `ctest`，`cv::resize` + `cv::cvtColor` |
| PyTorch | `scripts/verify_preprocess_cpu.py` | `F.interpolate(..., align_corners=False)` |
| NumPy | `scripts/verify_preprocess_cpu.py` | 显式 half-pixel 实现，不依赖 torch/OpenCV |

### 6.1 C++ / OpenCV

```bash
cmake --preset cpu
cmake --build --preset cpu
ctest --preset cpu
```

覆盖 1×1、3×5→7×10、5×4→2×3、37×29→64×48、64×48→17×23、恒等尺寸。

当前结果：

```text
13/13 tests passed
PreprocessCpuOpenCv.MatchesCvResizeAndColorConversion ... Passed
```

> OpenCV 对 `8U` 的 `INTER_LINEAR` 使用定点实现，因此测试先转成 `CV_32FC3` 再 resize，避免把定点舍入误判为参考实现错误。

### 6.2 Python 差分（PyTorch + OpenCV + NumPy）

```bash
.venv-train/bin/python scripts/verify_preprocess_cpu.py \
  --exe build/cpu/tools/preprocess_case --cases 40
```

脚本会生成固定用例 + 随机尺寸，随机 stride padding、mean/std、`bgr_to_rgb`，并额外检查：

- 同一输入重复调用输出逐位一致（确定性）
- 常量图经过任意 resize 后仍为常量

当前结果：

```text
cases: 49
worst vs torch : 2.682e-06  case=(47, 49, 25, 60, 6)
worst vs opencv: 8.106e-06  case=(62, 39, 58, 74, 1)
worst vs numpy : 4.768e-07  case=(62, 39, 58, 74, 1)
failures: 0
```

### 6.3 内存安全（ASan + UBSan）

```bash
cmake --preset asan
cmake --build --preset asan
ctest --preset asan
```

当前结果：

```text
13/13 tests passed (ASan + UBSan)
```

### 6.4 容差

| 对比 | 容差 |
|---|---|
| 恒等 resize / 常量图 | 逐位相等 |
| 线性斜坡解析解 | `1e-6` |
| NumPy 显式实现 | `5e-6` |
| PyTorch FP32 | `5e-6` |
| OpenCV float32 | `1e-5` |
| OpenCV 8U（定点） | `1/255` 量级 |
| 后续 CUDA FP32 | `1e-6` |
| 后续 CUDA FP16 | `atol=1e-3`、`rtol=1e-3` |

### 6.5 注意事项

- **验证器本身也要验证**：本次检查脚本曾两次写错通道映射（`mean/std` 按源 BGR 而非输出 RGB 索引、线性斜坡漏掉 BGR→RGB）。实现是正确的，错误在脚本。出现差异时先定位是“实现错”还是“检查错”。
- OpenCV 8U 路径是定点实现，与浮点参考的差异不要用 `1e-6` 硬卡。
- PyTorch CPU 双线性可能使用向量化累加，和标量顺序不同，允许 `5e-6` 量级的浮点差异。
- 这套验证只覆盖直接 resize；Letterbox/ROI 接入后需要新增对应等价变换测试。

## 7. 后续 K1 的验收口径

CUDA K1 在相同输入、相同 Spec 下，与 CPU Reference 比较：

- 整数对齐场景：最大绝对误差 `<= 1/255` 量级（FP32 中间计算）。
- 若 K1 使用 FP16 输出：建议 `atol = 1e-3`、`rtol = 1e-3`，并在报告中单独列出。
- 任何 Letterbox / ROI 变体必须先做坐标变换，再与本实现的等价裁剪结果比较。

## 8. 已知取舍

- `PreprocessSpec` 目前只有目标尺寸，没有 ROI；K1 增加 ROI 时建议新增 `RoiSpec`，不要修改现有字段语义。
- 使用 `std::vector` 每次分配，适合参考路径；benchmark / pipeline 应使用可复用 buffer。
- 未做多线程；CPU 参考只追求正确性与可读性。

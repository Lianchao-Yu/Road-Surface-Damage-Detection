# 开发环境说明（Plan B）

- **项目**：Road Surface Damage Detection（多路道路路面病害检测与严重度分析）
- **项目目录**：`/home/jason/road-damage-detection`（由 `gpu-traffic-analytics` 更名）
- **日期**：2026-11-09（2026-11-14 增补 TensorRT）
- **状态**：基础环境完成；**TensorRT 11.4.0 已在 CUDA 13.5 + sm_89 上完成 ONNX→Engine→C++ 推理验证**；Python 训练环境与 C++ 测试依赖已就绪
- **范围**：仅环境搭建与验证，未开始业务代码
- **说明**：新版 README 的推理后端是 TensorRT；ONNX Runtime 验证内容保留为历史记录，不再作为主链路

## 1. 主机与系统

| 项目 | 值 |
|---|---|
| 系统 | Ubuntu 26.04 LTS (Resolute Raccoon), WSL2 |
| 内核 | 6.18.54.1-microsoft-standard-WSL2 |
| CPU | 20 vCPU |
| 内存 | 7.6 GiB（当前可用约 5.6 GiB），Swap 2.0 GiB |
| 磁盘 | 1007 GiB，已用 36 GiB，可用 920 GiB（ext4 /dev/sdd） |
| GPU | NVIDIA GeForce RTX 4060 Laptop GPU，8188 MiB |
| 计算能力 | sm_89（8.9） |
| Windows NVIDIA 驱动 | 621.14（nvidia-smi CUDA UMD 13.5） |
| WSL 内 Linux 驱动 | 未安装，使用 Windows 驱动透传 |

## 2. 软件版本

| 组件 | 版本 | 安装位置 / 来源 |
|---|---|---|
| CUDA Toolkit | 13.5.37（release 13.5, V13.5.37） | `/usr/local/cuda-13.5`，NVIDIA CUDA apt 源 |
| cuDNN | 9.28.1.4（MAJOR 9, MINOR 28, PATCH 1） | `cudnn9-cuda-13-5` apt 包 |
| ONNX Runtime（历史） | 1.31.1（CUDA 13 GPU 版） | `~/opt/onnxruntime-1.31.1-cuda13`，新版 README 不使用 |
| TensorRT | 11.4.0.106（`+cuda13.4` 构建） | 系统安装（`/usr/bin/trtexec`）；`~/opt/tensorrt-11.4/root` 为回退副本 |
| PyTorch / Ultralytics | torch 2.15.0+cu132 / torchvision 0.29.2 / ultralytics 8.4.213 | 项目内 `.venv-train`（Python 3.14） |
| SQLite / gtest / benchmark | 3.46.1 / 1.17.0 / 1.9.1 | apt |
| GCC / G++ | 15.2.0（Ubuntu 15.2.0-16ubuntu1） | 系统 |
| CMake | 4.2.3 | 系统 |
| Ninja | 1.13.2 | `ninja-build` |
| pkg-config | 2.5.1 | 系统 |
| FFmpeg | 8.0.1-3ubuntu2 | 系统 |
| libavformat | 62.3.100 | `libavformat-dev` |
| libavcodec | 62.11.100 | `libavcodec-dev` |
| libavutil | 60.8.100 | `libavutil-dev` |
| libswscale | 9.1.100 | `libswscale-dev` |
| OpenCV | 4.10.0 | `libopencv-dev` |
| nlohmann/json | 头文件 3.11.3（包版本 3.12.0.really.3.12.0.really.3.11.3-3build1） | `nlohmann-json3-dev` |
| Python | 3.14.4 | 系统 |
| Python venv | onnx 1.23.2 / numpy 2.5.4 | 项目内 `.venv`，仅用于 ONNX 模型生成/检查 |

> 未安装 `onnxruntime-gpu` Python 包；新版 README 使用 TensorRT，ORT 仅保留为历史验证。
> TensorRT 为免 sudo 解包安装，路径 `~/opt/tensorrt-11.4/root/usr`，详见 3.4。
> Ubuntu 自带的 CUDA 12.4（`/usr/bin/nvcc` 包装脚本，`/usr/lib/nvidia-cuda-toolkit`）保留未删除，通过 PATH / `CMAKE_CUDA_COMPILER` 显式选择 13.5。

## 3. 安装来源

### 3.1 基础依赖

```bash
sudo apt install -y \
  ninja-build pkg-config ffmpeg \
  libavformat-dev libavcodec-dev libavutil-dev libswscale-dev \
  libopencv-dev nlohmann-json3-dev \
  python3.14-venv python3-pip
```

### 3.2 CUDA / cuDNN

```bash
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2604/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt update
sudo apt install -y cuda-toolkit-13-5 cudnn9-cuda-13-5
```

- 未安装 `cuda` / `cuda-drivers` 元包
- 未在 WSL 内安装 Linux NVIDIA 驱动

### 3.3 ONNX Runtime 1.31.1 CUDA13

```text
URL:    https://github.com/microsoft/onnxruntime/releases/download/v1.31.1/onnxruntime-linux-x64-gpu_cuda13-1.31.1.tgz
SHA256: b2146862e691f7e338f5aff9c7a9a5567c2ca57cf01442a1adb3a13bf62d9095
路径:   /home/jason/opt/onnxruntime-1.31.1-cuda13
```

包含 `onnxruntime_cxx_api.h`、`libonnxruntime.so`、`libonnxruntime_providers_cuda.so`、`libonnxruntime_providers_shared.so`。

### 3.4 TensorRT 11.4（2026-11-14 增补）

NVIDIA CUDA apt 源中最新 TensorRT 为 **11.4.0.106-1+cuda13.4**（没有 CUDA 13.5 构建）。首次 `sudo apt install` 在下载 1.8GB 的 `libnvinfer11` 时异常退出，未产生任何 dpkg 记录；改用免 sudo 的下载+解包方式完成安装：

```bash
mkdir -p ~/opt/tensorrt-11.4/{debs,root}
cd ~/opt/tensorrt-11.4/debs
apt-get download \
  libnvinfer11 libnvinfer-bin libnvinfer-headers-dev libnvinfer-dev \
  libnvonnxparsers11 libnvonnxparsers-dev \
  libnvinfer-plugin11 libnvinfer-plugin-dev libnvinfer-cmake-dev \
  libnvinfer-safe-headers-dev libnvinfer-headers-plugin-dev \
  libnvinfer-headers-python-plugin-dev
for f in *.deb; do dpkg-deb -x "$f" ~/opt/tensorrt-11.4/root; done
```

- 下载约 1.92GB，解包后约 2.6GB
- 头文件：`~/opt/tensorrt-11.4/root/usr/include/x86_64-linux-gnu/`
- 动态库：`~/opt/tensorrt-11.4/root/usr/lib/x86_64-linux-gnu/`
- `trtexec`：`~/opt/tensorrt-11.4/root/usr/bin/trtexec`
- 只提供 `TensorRT-EnterpriseConfig.cmake`，**没有标准 `TensorRTConfig.cmake`**，CMake 需手写 `find_path`/`find_library` 或自带 `FindTensorRT.cmake`

运行时环境变量：

```bash
export TENSORRT_ROOT="$HOME/opt/tensorrt-11.4/root/usr"
export LD_LIBRARY_PATH="$TENSORRT_ROOT/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH"
```

> 若以后要用系统级安装，可直接 `sudo dpkg -i ~/opt/tensorrt-11.4/debs/*.deb && sudo ldconfig`，无需重新下载。

**2026-11-14 后续**：已用上述 deb 完成系统级安装，并用 `sudo apt --fix-broken install` 补齐 `libnvinfer-lean11` / `libnvinfer-vc-plugin11` / `libnvinfer-dispatch11` 及对应 `-dev` 包。当前系统路径为：

- `trtexec`：`/usr/bin/trtexec`
- 头文件：`/usr/include/x86_64-linux-gnu/`
- 动态库：`/usr/lib/x86_64-linux-gnu/libnvinfer.so.11`

`~/opt/tensorrt-11.4/root` 保留为回退副本，日常构建优先使用系统路径。

## 4. 环境变量

已幂等写入 `~/.bashrc`：

```bash
export CUDA_HOME=/usr/local/cuda-13.5
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:$LD_LIBRARY_PATH"
export ONNXRUNTIME_ROOT="$HOME/opt/onnxruntime-1.31.1-cuda13"
export LD_LIBRARY_PATH="$ONNXRUNTIME_ROOT/lib:$LD_LIBRARY_PATH"
export TENSORRT_ROOT="$HOME/opt/tensorrt-11.4/root/usr"
export LD_LIBRARY_PATH="$TENSORRT_ROOT/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH"
```

> TensorRT 已完成系统级安装后，日常构建不需要设置 `TENSORRT_ROOT` / 额外 `LD_LIBRARY_PATH`；上面两行仅用于回退到 `~/opt` 副本时。

新开交互终端后：

```bash
which nvcc          # /usr/local/cuda-13.5/bin/nvcc
nvcc --version      # release 13.5, V13.5.37
```

## 5. Smoke Test 记录

测试临时目录：`/tmp/opencode/planB_smoke/`（未污染项目根目录）

### 5.1 CUDA Smoke Test

```bash
/usr/local/cuda-13.5/bin/nvcc -std=c++17 -arch=sm_89 smoke.cu -o smoke
./smoke
```

结果：

```text
devices=1
GPU 0: NVIDIA GeForce RTX 4060 Laptop GPU sm_89 mem=8.00 GiB
kernel result sample=6.0 all_ok=yes
```

验证点：`cudaGetDeviceCount`、`cudaGetDeviceProperties`、简单 kernel、C++17 标准库（`<vector>`/`<algorithm>`/`<memory>`/`<string>`）与 GCC 15 协同正常。

### 5.2 FFmpeg / OpenCV / nlohmann JSON 链接测试

```bash
g++ -std=c++17 -O2 ffmpeg_opencv_json.cpp \
  $(pkg-config --cflags --libs libavformat libavcodec libavutil libswscale opencv4) \
  -o ffmpeg_opencv_json
./ffmpeg_opencv_json
```

结果：

```text
avformat_version=4064100 (8.0.1-3ubuntu2)
avcodec_version=4066148
OpenCV=4.10.0
json={"project":"road-damage-detection","smoke":true,"streams":4}
```

> 该 Smoke Test 在目录更名前执行；输出中的 `project` 字段已按现项目名记录，环境配置本身不受影响。

### 5.3 ONNX Runtime CUDA EP 测试

生成 tiny Conv 模型（输入 `1x3x32x32`，Conv 3→8，3x3，输出 `1x8x32x32`）：

```bash
.venv/bin/python make_tiny_onnx.py   # -> /tmp/opencode/planB_smoke/tiny.onnx
```

编译与运行：

```bash
g++ -std=c++17 -O2 -I"$ONNXRUNTIME_ROOT/include" ort_cuda_test.cpp \
  -L"$ONNXRUNTIME_ROOT/lib" -lonnxruntime \
  -Wl,-rpath,"$ONNXRUNTIME_ROOT/lib" -o ort_cuda_test

export LD_LIBRARY_PATH="$ONNXRUNTIME_ROOT/lib:$CUDA_HOME/lib64:$LD_LIBRARY_PATH"
./ort_cuda_test
```

结果：

```text
available providers:
  - TensorrtExecutionProvider
  - CUDAExecutionProvider
  - CPUExecutionProvider
input=input output=output
output shape: 1 8 32 32
output[0..4]: -0.272981 -0.158066 -0.263589 -0.450291 -0.479071
ORT CUDA EP run OK
```

- `Ort::GetAvailableProviders()` 包含 `CUDAExecutionProvider`
- Session 创建并成功运行，输出维度正确
- 日志中无 `not assigned` / `fallback` / `failed` / `error` 告警（verbose 日志）
- 动态库依赖检查：`libonnxruntime_providers_cuda.so` 与 `libonnxruntime.so` 均无 `not found`

### 5.4 CMake + Ninja + CUDA + ORT + OpenCV 工具链测试

临时工程：`/tmp/opencode/planB_smoke/cmake_test/`

```bash
cmake -S . -B build -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CUDA_COMPILER=/usr/local/cuda-13.5/bin/nvcc \
  -DCMAKE_CUDA_ARCHITECTURES=89 \
  -DCMAKE_POLICY_VERSION_MINIMUM=3.5 \
  -DCUDAToolkit_ROOT=/usr/local/cuda-13.5 \
  -DONNXRUNTIME_ROOT="$HOME/opt/onnxruntime-1.31.1-cuda13"

cmake --build build -j
./build/planb_smoke
```

结果：

```text
OpenCV=4.10.0
FFmpeg avformat=4064100 avcodec=4066148
ORT providers: TensorrtExecutionProvider CUDAExecutionProvider CPUExecutionProvider
```

### 5.5 TensorRT 11.4 推理测试（2026-11-14）

验证材料保存在 `~/opt/trt_verify/`（**不要放 `/tmp`**，本次 `/tmp/opencode` 在 19:11 被系统清理过一次）：

- `make_tiny.py` → `tiny.onnx`（`1×3×32×32` Conv，opset 17）
- `trt_smoke.cpp` → 完整 C++ 链路：Builder → ONNX Parser → Engine → ExecutionContext → `enqueueV3` → D2H

构建 engine：

```bash
export LD_LIBRARY_PATH="$HOME/opt/tensorrt-11.4/root/usr/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH"
"$HOME/opt/tensorrt-11.4/root/usr/bin/trtexec" \
  --onnx=tiny.onnx --saveEngine=tiny.engine --skipInference
```

结果：

```text
TensorRT version: 11.4.0
Engine generation completed in 0.85 seconds.
&&&& PASSED TensorRT.trtexec [TensorRT v110400]
```

加载并推理：

```text
Throughput: 25236.4 qps
Latency: median = 0.0144 ms
&&&& PASSED TensorRT.trtexec [TensorRT v110400]
```

C++ 编译与运行：

```bash
g++ -std=c++17 -O2 -Wno-deprecated-declarations \
  -I"$HOME/opt/tensorrt-11.4/root/usr/include/x86_64-linux-gnu" \
  -I/usr/local/cuda-13.5/include \
  trt_smoke.cpp \
  -L"$HOME/opt/tensorrt-11.4/root/usr/lib/x86_64-linux-gnu" -lnvinfer -lnvonnxparser \
  -L/usr/local/cuda-13.5/lib64 -lcudart \
  -Wl,-rpath,"$HOME/opt/tensorrt-11.4/root/usr/lib/x86_64-linux-gnu" \
  -Wl,-rpath,/usr/local/cuda-13.5/lib64 \
  -o trt_smoke
./trt_smoke tiny.onnx
```

结果：

```text
GPU: NVIDIA GeForce RTX 4060 Laptop GPU sm_89
TensorRT: 11.4.0
ONNX parsed: tiny.onnx (inputs=1 outputs=1)
engine built: 21.4 KB
output[0..4]: 0.96000 1.39500 1.39500 1.39500 1.39500
TensorRT CUDA smoke OK
```

**结论**：TRT 11.4（官方针对 CUDA 13.4 构建）在 CUDA 13.5 运行时 + 驱动 621.14 上可以正常构建 engine 并推理，`libnvinfer` / `libnvonnxparser` 均无缺失依赖。该组合可用，但属于官方支持矩阵之外的组合，升级任一组件前需重新跑本测试。

### 5.6 Python 训练环境与 C++ 依赖验证（2026-11-14）

```bash
python3 -m venv .venv-train
.venv-train/bin/pip install -U pip
.venv-train/bin/pip install torch torchvision ultralytics
```

结果：

```text
torch 2.15.0+cu132 | torchvision 0.29.2+cu132 | ultralytics 8.4.213
cuda 13.2 | available True
device NVIDIA GeForce RTX 4060 Laptop GPU | capability (8, 9)
matmul ok
```

> PyTorch 自带 CUDA 13.2 运行时，与系统 CUDA 13.5 工具链并存，驱动 621.14 均兼容。

C++ 依赖：

```bash
sudo apt install -y libsqlite3-dev sqlite3 libgtest-dev libbenchmark-dev
```

链接测试结果：

```text
sqlite=3.46.1 link OK
[       OK ] EnvCheck.ArithmeticWorks
[  PASSED  ] 1 test.
BM_Noop  1.82 ns
```

`trtexec` 系统级验证：

```text
TensorRT version: 11.4.0
Engine built in 1.86 sec.
&&&& PASSED TensorRT.trtexec [TensorRT v110400]
```

## 6. 已知限制与 workaround

1. **WSL 内不安装 Linux NVIDIA 驱动**：GPU 由 Windows 驱动透传，`nvidia-smi` 显示 Driver 621.14 / CUDA UMD 13.5。
2. **RTX 4060 Laptop 8 GB 显存**：后续多路 1080p 道路巡检视频 + batch 推理需实测显存上限，不能假设 4 路 1080p/30FPS 必然达标。
3. **WSL 内存约 7.6 GiB**：多路解码 + 队列 + ORT 需控制帧缓冲和队列容量。
4. **CMake 可能优先选中 Ubuntu CUDA 12.4**：`/usr/bin/nvcc` 是包装脚本，指向 `/usr/lib/nvidia-cuda-toolkit`。仅设置 PATH 时 CMake 仍可能选中它，导致 `ptxas fatal: Value 'sm_52' is not defined`（CUDA 13 已移除 sm_52）。必须显式指定：
   ```bash
   -DCMAKE_CUDA_COMPILER=/usr/local/cuda-13.5/bin/nvcc
   ```
5. **CMake 4.2 + OpenCV 4.10 兼容**：本次临时工程使用了 `-DCMAKE_POLICY_VERSION_MINIMUM=3.5`，用于规避旧 `cmake_minimum_required` 策略错误。正式 CMakeLists 若未出现该错误可不加。
6. **Ubuntu apt 源不稳定**：`archive.ubuntu.com` 早期只有约 15 kB/s，曾切换到 `https://mirrors.kernel.org/ubuntu/`；2026-11-14 该镜像**不可达**（IPv4 连接超时、IPv6 无路由），现改为 `https://mirrors.aliyun.com/ubuntu/`（实测约 41.8 MB/s，archive.ubuntu.com 约 7.7 MB/s）。同时写入 `/etc/apt/apt.conf.d/99force-ipv4` 强制 IPv4。原始 sources 备份保留在 `/etc/apt/sources.list.d/`。
7. **CUDA 12.4 与 13.5 共存**：未卸载系统 CUDA 12.4 包；所有构建显式使用 `/usr/local/cuda-13.5`。
8. **Python venv 仅用于模型处理**：`.venv` 中只有 `numpy` / `onnx`；后续训练需另建包含 PyTorch/Ultralytics 的环境。
9. **TensorRT 版本与 CUDA 13.5 不属于官方支持组合**：TRT 11.4 官方包针对 CUDA 13.4 update 1 构建，支持列表为 13.0–13.4。本次实测在 13.5 上可用（trtexec + C++ 均通过），但升级 CUDA/TRT/驱动后必须重跑 §5.5。
10. **CUDA 12.4 头文件会污染编译**：`/usr/include/cuda_runtime_api.h`（CUDA 12.4）把 `cudaGetDeviceProperties` 宏映射为 `_v2`，而 CUDA 13.5 的 `libcudart` 不导出该符号，表现为链接错误 `undefined reference to cudaGetDeviceProperties_v2`。必须在 `-I` 列表**最前面**加 `-I/usr/local/cuda-13.5/include`。CMake 中应使用 `CUDAToolkit_INCLUDE_DIRS` 并保证顺序。
11. **TensorRT 11 API 与 10.x 不同**：`BuilderFlag::kFP16`、`NetworkDefinitionCreationFlag::kEXPLICIT_BATCH` 已移除，`trtexec` 的 `--fp16` 参数也已移除；精度由 strongly-typed 网络与 `--precisionConstraints` 控制。
12. **TensorRT 包没有标准 CMake config**：apt 包只提供 `TensorRT-EnterpriseConfig.cmake`，项目需自带 `FindTensorRT.cmake` 或直接用 `find_path`/`find_library` 指向 `~/opt/tensorrt-11.4/root/usr`。
13. **`/tmp/opencode` 会被清理**：本次临时验证文件曾整体消失；构建产物、模型和报告不要放 `/tmp`，统一放项目目录或 `~/opt`。
14. **Nsight Compute 无权限**：`ncu` 报 `ERR_NVGPUCTRPERM`。WSL2 下需在 **Windows NVIDIA App → System → Advanced → Developer → Manage GPU Performance Counters** 选择允许所有用户，然后重启 WSL。该设置未完成前，W6 的 Nsight 报告无法生成。
15. **OpenCV 无 CUDA 模块**：Ubuntu `libopencv-dev 4.10.0` 的 `cuda*` 模块均不可用（`opencv_version --verbose` 显示 Unavailable）。K1-B3 的成熟 GPU baseline 使用 CUDA 自带 **NPP**；若确需 OpenCV CUDA，需源码编译，成本较高。
16. **尚未就位的数据/工具**：RDD2022 数据集、≥10 分钟道路 MP4、MediaMTX（RTSP 回放）均未准备；这些属于数据与 P1 测试阶段，不阻塞 W1 Kernel Harness。

## 7. 下一步建议

1. **工程骨架**：建立 `CMakeLists.txt` / `CMakePresets.json`，固定：
   - C++17、`CMAKE_CUDA_ARCHITECTURES=89`
   - `CUDAToolkit_ROOT=/usr/local/cuda-13.5`、`CMAKE_CUDA_COMPILER=/usr/local/cuda-13.5/bin/nvcc`
   - `TENSORRT_ROOT=~/opt/tensorrt-11.4/root/usr`，并保证 CUDA 13.5 include 优先于 `/usr/include`
   - 自带 `FindTensorRT.cmake`
2. **W1 CUDA Harness**：合成 Buffer + CPU 参考实现 + CUDA Events 计时，不依赖模型先跑通。
3. **模型线**：安装 PyTorch/Ultralytics，RDD2022 子集 fine-tune 小 YOLO，导出 ONNX，再用 `trtexec` 验证可构建 engine。
4. **视频线**：FFmpeg 单路解码 + PTS，再接入 TensorRT 推理；RTSP 回放放在 MP4 稳定之后。
5. 后续升级驱动/CUDA/TRT 时，保持 CUDA 13 / cuDNN 9 / TRT 11.4 匹配，并同步更新本文档。

## 8. 验收清单

- [x] `nvidia-smi` Driver 621.14（>= 580），CUDA UMD 13.5
- [x] `nvcc --version` 为 13.5（`/usr/local/cuda-13.5/bin/nvcc`）
- [x] CUDA Smoke Test 通过，识别 sm_89
- [x] FFmpeg / OpenCV / nlohmann 头文件与链接测试通过
- [x] `Ort::GetAvailableProviders()` 包含 `CUDAExecutionProvider`（历史）
- [x] tiny ONNX 模型在 CUDA EP 上运行成功（历史）
- [x] 无未记录的 CPU fallback（历史）
- [x] CMake + Ninja + CUDA + ORT + OpenCV 临时工程构建并运行成功（历史）
- [x] **TensorRT 11.4 + CUDA 13.5：ONNX→Engine→C++ 推理通过**
- [x] **`trtexec` 可运行，`libnvinfer` / `libnvonnxparser` 无缺失依赖**
- [x] **TensorRT 已完成系统安装，`dpkg` 无未配置包**
- [x] **PyTorch 2.15(cu132) + Ultralytics 8.4 GPU 可用，matmul 通过**
- [x] **sqlite / gtest / benchmark 编译链接与运行通过**
- [ ] Nsight Compute 权限（需 Windows NVIDIA App 设置）
- [ ] MediaMTX / RDD2022 / 样本视频（数据阶段）
- [x] `docs/environment.md` 已更新

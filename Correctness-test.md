# Project Test

当前阶段只做一件事：**验证 CPU 预处理结果与正确答案一致**。
后期实现 CUDA 后，再增加 **CUDA 与 CPU 对比**。

## 1. CPU 测试

```bash
# 配置 CPU 构建（关闭 CUDA / TensorRT）
cmake --preset cpu

# 编译库、测试和工具
cmake --build --preset cpu

# 运行全部 CPU 测试，预期 13/13 通过
ctest --preset cpu
```

测试内容包括手算 golden（恒等 resize、2→4 插值、3→2 插值、通道顺序、归一化、stride、1×1、非法输入）。

## 2. 与成熟库对比（正确答案）

```bash
# 编译对拍工具：从 stdin 读原始 BGR，输出 NCHW 张量
cmake --build --preset cpu --target preprocess_case

# 生成 40 组随机用例，与 PyTorch / OpenCV / NumPy 逐元素比较
# --exe 指定 C++ 工具路径，--cases 指定随机用例数量
.venv-train/bin/python scripts/verify_preprocess_cpu.py \
  --exe build/cpu/tools/preprocess_case \
  --cases 40
```

通过标准：

```text
failures: 0
```

当前基线：

```text
worst vs torch : 2.682e-06
worst vs opencv: 8.106e-06
worst vs numpy : 4.768e-07
```

## 3. 后期：CUDA 与 CPU 对比

CUDA 预处理实现后，用相同输入和相同 `PreprocessSpec` 跑两版，逐元素比较：

- FP32：最大绝对误差 `<= 1e-6`
- FP16：`atol = 1e-3`、`rtol = 1e-3`

命令待 CUDA 测试加入后启用：

```bash
# 构建 CUDA + TensorRT 版本，只运行 CUDA 预处理相关测试
# ctest --preset cuda-trt -R CudaPreprocess
```

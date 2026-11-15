# CUDA 道路智能巡检推理模块开发 SOP

**项目名称：** CUDA-Accelerated Road Damage Inference Pipeline  
**中文名称：** CUDA 加速的道路病害智能识别与推理处理模块  
**参考企业：** 厦门路桥信息股份有限公司  
**业务方向：** 公路智慧养护 / AI 智能巡检  
**项目性质：** 独立复现的企业实习级工程项目（Prototype / PoC）  
**模拟岗位：** C++ / CUDA GPU Software Development Intern  
**建议周期：** 10 周  
**目标硬件：** NVIDIA RTX 4060 Laptop GPU，按 8GB VRAM 规划  
**开发环境：** Linux / WSL2、C++17、CUDA、TensorRT、PyTorch、FFmpeg、CMake

---

## 1. 项目背景与实际业务需求

### 1.1 企业业务背景

厦门路桥信息的智慧养护业务包含道路基础设施管理、病害智能识别及巡检信息化等相关技术。

本项目选择其中一个明确的子场景：

**道路巡检影像的 GPU 智能识别与推理性能优化。**

模拟企业已有路面病害检测算法，需要将模型部署到 PC GPU 环境，对来自移动巡检设备的视频进行自动处理。

公司现有软件团队负责整体养护平台、业务管理及设备集成；本项目聚焦 GPU 推理处理模块。

不假设能够访问公司原有模型、数据库或内部源代码。

### 1.2 模拟工程需求

巡检设备采集道路影像，PC 端负责识别路面裂缝、网状裂缝及坑洞。

系统需要：

- 使用 CNN 自动识别道路病害。
- 支持持续视频输入，而不仅是单张图片检测。
- 保留足够图像细节，降低细小病害漏检风险。
- 在有限 GPU 显存和计算资源下提高处理效率。
- 输出可供养护人员复核的病害检测记录。
- 能够通过独立测试证明 GPU 优化效果。

### 1.3 实习生承担的具体任务

**主要负责：**

开发并优化 PC 端 C++/CUDA 推理处理模块，包括 GPU 预处理、TensorRT 集成、异步数据处理、显存管理及性能分析。

**辅助负责：**

训练一个可复现的道路病害 CNN，建立推理正确性测试，并输出必要的病害证据记录。

**不负责：**

Go2 运动控制、SLAM 算法、完整养护业务平台、企业数据库集成或生产环境部署。

### 1.4 核心工程目标

项目的技术目标是：

> 在固定检测模型与图像质量要求下，通过自定义 CUDA Kernel、GPU 显存管理和异步执行，降低道路病害视频推理的处理开销，提高有限 GPU 资源下的持续处理能力。

所有 CUDA 优化必须通过实验验证，而不能预先假定存在性能提升。

---

## 2. 项目范围与验收环境

### 2.1 硬件条件

项目使用一台具备 NVIDIA RTX GPU 的 PC 完成全部核心开发。

不要求：

- 实体 Unitree Go2
- 激光雷达
- GNSS
- 机器人仿真环境
- 公司内部服务器
- 企业专有视频数据

### 2.2 输入来源

| 输入 | 用途 | 是否必须实现 |
|---|---|---|
| RDD2022 图像 | CNN 训练与模型验证 | 必须 |
| 自采道路影像 | 独立测试与域适应 | 建议 |
| 本地 MP4 | 视频推理集成测试 | 必须 |
| Synthetic GPU Buffers | CUDA Kernel Benchmark | 必须 |
| 本地 RTSP Replay | 视频接入测试 | 必须 |
| Mock Robot Metadata | 未来机器人接口测试 | 必须 |
| Go2 实体相机 | 后续硬件扩展 | 不需要 |

### 2.3 验收原则

所有核心验收均在 PC 上完成。

测试必须覆盖三个层级：

**Level 1：CUDA Kernel**

无需 CNN、RTSP 或机器人，独立验证自定义 Kernel 的正确性与性能。

**Level 2：GPU Inference Pipeline**

使用固定模型与图像输入，验证 TensorRT、GPU Memory、CUDA Streams 和执行调度。

**Level 3：完整业务 Pipeline**

输入道路视频，输出真实检测结果及证据记录，验证持续处理能力。

不得使用模拟检测结果证明真实模型准确率，也不得使用 Mock Pose 宣称病害空间定位能力。

---

## 3. 系统架构

### 3.1 总体数据流程

**输入层**

MP4 / RTSP / Synthetic Frames / Future Go2 Adapter

↓

**C++ Video Ingestion**

FFmpeg 解码、时间戳、Frame Queue

↓

**Host / Device Memory Management**

Pinned Host Buffer、GPU Buffer Pool

↓

**Custom CUDA Preprocessing**

Resize、ROI/Tiling、Normalization、FP16、Layout Conversion

↓

**TensorRT CNN Inference**

自行训练并导出的道路病害检测模型

↓

**GPU Detection Postprocessing**

候选框过滤、坐标转换、必要时执行 GPU Compaction

↓

**C++ Inspection Event Adapter**

记录类别、置信度、时间戳、证据与模型版本

↓

**SQLite / JSON / Evidence Storage**

### 3.2 核心模块划分

| 模块 | 推荐技术 | 主要工作 |
|---|---|---|
| VideoSource | C++ / FFmpeg | 视频接收与解码 |
| FrameQueue | C++17 | 有界缓存及背压 |
| CudaPreprocessor | CUDA C++ | 自定义融合 Kernel |
| TrtInferenceEngine | TensorRT C++ | CNN GPU 推理 |
| GpuBufferPool | CUDA / RAII | 显存分配与复用 |
| GpuPipeline | CUDA Streams / Events | 异步执行和依赖管理 |
| DetectionProcessor | CUDA / C++ | 检测结果处理 |
| InspectionRecorder | C++ / SQLite | 病害证据记录 |
| BenchmarkRunner | C++ / CUDA Events | 正确性和性能测试 |

本项目最重要的自主研发模块为：

1. `CudaPreprocessor`
2. `GpuBufferPool`
3. `GpuPipeline`
4. `BenchmarkRunner`

---

## 4. CNN 模型准备与训练 SOP

### 4.1 训练任务

使用 RDD2022 数据集建立四类道路病害检测模型：

- D00：纵向裂缝
- D10：横向裂缝
- D20：网状裂缝
- D40：坑洞

第一版采用小型 YOLO 系列模型。

使用公开预训练权重进行 Fine-tuning，不要求从随机初始化训练全部网络参数。

### 4.2 数据处理

执行以下步骤：

1. 下载 RDD2022 中具有有效标注的图像。
2. 验证 XML 标签与图片文件的一致性。
3. 转换为模型所需的数据格式。
4. 统计各类别样本数量与原始分辨率。
5. 划分 Train、Validation、Test。
6. 保存 Dataset Manifest 和类别映射。
7. 对自采视频按采集路段或时间段划分数据。

避免将同一视频的相邻帧随机分配到训练集与测试集。

### 4.3 训练实验

| 实验 | 内容 |
|---|---|
| T1 | RDD2022 Baseline Fine-tuning |
| T2 | 不同输入分辨率的准确率对比 |
| T3 | 独立地区/来源数据测试 |
| T4 | 自采样本及困难负样本验证 |

训练阶段必须输出：

- Model Checkpoint
- ONNX Model
- Precision / Recall
- AP50 / AP50-95
- Per-Class Metrics
- Evaluation Report
- Model Version 与文件哈希

### 4.4 模型冻结

一旦获得可运行的模型，就冻结权重、类别映射、预处理规范和检测阈值。

之后的 CUDA 优化实验不得同时改变模型配置。

这是为了确保性能差异确实来自 GPU 系统优化。

**本阶段完成标准：**

模型可在 PyTorch 和 TensorRT 中运行，并在预定义数值容差下取得一致的检测结果。

训练任务不得阻塞 CUDA Kernel 的独立开发；前期允许使用固定的测试 Tensor 或现成兼容模型构建推理测试夹具。

---

## 5. CUDA Kernel Development SOP

这是项目的主要研发任务之一。

### 5.1 Kernel K1：Fused GPU Preprocessing

**优先级：P0**

#### 功能

将道路图像直接转换为 TensorRT 所需的 GPU 输入 Tensor。

融合操作包括：

- ROI / Tile Extraction
- Bilinear Resize
- BGR-to-RGB
- Normalization
- FP16 Conversion
- HWC-to-NCHW

输出格式为：

`B × 3 × 640 × 640, FP16`

#### 优化研究

重点研究：

- Thread-to-Pixel Mapping
- Coalesced Global Memory Access
- Memory Alignment
- Register Usage
- Occupancy
- Kernel Fusion
- Input/Output Tensor Layout

#### 对照实验

| Baseline | 内容 |
|---|---|
| K1-B0 | CPU OpenCV Preprocessing |
| K1-B1 | 分离的 CUDA Kernels |
| K1-B2 | Fused CUDA Kernel |
| K1-B3 | 可获得的成熟 GPU 预处理实现 |

对于 K1-B3，可参考与当前操作语义一致的 NPP 或 OpenCV CUDA 实现。

比较时保持相同的裁剪、插值、归一化和 Tensor Layout 规则。

#### 性能指标

- Kernel Latency
- Preprocessing Throughput
- Global Memory Bandwidth
- Register Usage
- Achieved Occupancy
- End-to-End Preprocessing Cost

**验收：**

所有边界输入测试通过，GPU 输出与参考实现一致，并完成至少一次 Nsight Compute Profiling。

### 5.2 Kernel K2：GPU Detection Processing

**优先级：P1，依据 Profiling 决定是否深入**

目标是减少 CNN 推理结果的 CPU 处理和不必要的数据回传。

候选操作：

- Confidence Filtering
- Bounding Box Decode
- Class Selection
- Tile Coordinate Remapping
- GPU Candidate Compaction

对照 CPU 后处理、已有 GPU 后处理和自定义 CUDA 实现。

如果模型导出时已经集成高效 NMS，不应为了展示 CUDA 而重复实现相同操作。

重点比较完整候选数据回传与 GPU 过滤后紧凑数据回传的成本。

### 5.3 Kernel 正确性测试

至少覆盖：

- 1×1、极小图像及奇数尺寸
- 720p、1080p、4K 缓冲区
- 不连续行 Pitch
- ROI 超界、边界 Padding
- Tile Batch 1/2/4/8
- 空检测结果
- 不同有效候选比例
- FP16 数值误差

使用 CPU 参考输出、单元测试及适用的 Compute Sanitizer 工具检查。

**交付物：**

`cuda_kernel_correctness.md`

`cuda_kernel_benchmark.csv`

`nsight_compute_report.md`

---

## 6. GPU Systems Engineering SOP

这是第二个主要研发任务。

### 6.1 GPU Buffer Pool

**优先级：P0**

实现：

- DeviceBuffer RAII
- PinnedHostBuffer RAII
- Fixed-Capacity GPU Buffer Pool
- Buffer Capacity Checking
- Stream-Safe Buffer Reuse

重点解决异步执行中 Buffer 被提前覆盖或释放的问题。

比较以下配置：

| 方案 | 内容 |
|---|---|
| M0 | 每次处理动态分配 |
| M1 | 预分配并复用 GPU Buffer |
| M2 | 可选的 Stream-Ordered Allocator |

M0 用于测量分配成本，不能作为唯一的工程基准。最终还应与合理的预分配实现比较。

### 6.2 CUDA Asynchronous Pipeline

**优先级：P0**

建立以下迭代版本：

**P0：Sequential**

H2D → Preprocess → TensorRT → Postprocess → D2H

**P1：Pinned Host Memory**

引入支持异步传输的 Host Buffer。

**P2：Double Buffering**

使用两个独立 Buffer Slot。

**P3：CUDA Streams & Events**

在正确数据依赖下尝试传输与 GPU 计算重叠。

**P4：Optional Multi-Context Inference**

只有当单推理上下文未充分利用 GPU 且 Profiling 支持时才尝试。

### 6.3 Pipeline Slot 状态管理

每个 Slot 具有独立 Buffer 和完成事件。

状态：

`FREE → TRANSFER → PREPROCESS → INFERENCE → POSTPROCESS → COMPLETE → FREE`

必须保证：

- GPU 执行期间 Buffer 不被覆盖。
- 释放 Buffer 前确认 GPU 工作完成。
- 推理失败时能够清理资源。
- Shutdown 可以处理未完成的任务。
- Input Queue 有界，不会无限增长。

### 6.4 CUDA Graphs

**优先级：P2，可选**

只有在 CPU Kernel Launch 或 TensorRT Enqueue 开销明显时，才进行 CUDA Graphs 实验。

必须分别报告 Graph Capture Setup Cost 和稳定 Replay Cost。

**交付物：**

`gpu_memory_analysis.md`

`async_pipeline_profile.md`

`pipeline_benchmark.csv`

---

## 7. 无 Go2 的完整工程测试 SOP

### 7.1 Test A：Synthetic CUDA Workload

使用合成图像 Buffer 测试 Kernel。

不依赖 CNN、FFmpeg 或机器人。

用于：

- Kernel Correctness
- Block Size Tuning
- Memory Analysis
- Batch Scaling
- Stress Testing

### 7.2 Test B：Offline Video Processing

使用公开或自行录制的道路 MP4。

流程：

MP4 → FFmpeg → CUDA → TensorRT → Detections → Evidence

验收：

- 视频能够完整处理。
- 帧 ID 与时间戳正确。
- 检测结果能够关联原始图片。
- Buffer 占用稳定。
- 输出病害记录格式正确。

### 7.3 Test C：Local RTSP Replay

在本地运行 MediaMTX，并使用 FFmpeg 发布 MP4 视频。

测试内容：

- RTSP Connection
- Continuous Frame Processing
- Network Delay / Jitter Injection
- Stream Interruption
- Recovery
- Queue Overflow
- Dropped Frames

这只能证明 PC 端 RTSP 流处理能力，不能证明真实 Go2 的视频接口和无线网络表现。

### 7.4 Test D：Mock Robot Interface

预留两个接口：

`IFrameSource`

`IPoseProvider`

`IFrameSource` 支持 MP4、RTSP 和测试数据输入。

`IPoseProvider` 为可选接口。

Mock 测试提供固定 Source ID、时间戳及虚拟位姿，验证接口和元数据处理。

没有可信定位数据时，实际病害位置字段必须为 `null`。

### 7.5 Test E：Continuous Stress Test

在 RTX PC 上连续运行至少 30 分钟。

测试：

- Memory Leak
- GPU Buffer Reuse
- Input Queue Stability
- Frame Dropping
- CPU/GPU Utilization
- Thermal Throttling
- Clean Shutdown

---

## 8. 企业业务验收与量化指标

所有性能测试使用固定数据、模型、输入尺寸及检测阈值。

### 8.1 CUDA 性能指标

| 指标 | 验证的问题 |
|---|---|
| Kernel Latency | 自定义 Kernel 是否更高效 |
| Fused Preprocessing Speedup | Kernel Fusion 的收益 |
| H2D/D2H Time | 数据传输是否为瓶颈 |
| GPU Memory Peak | 是否满足 RTX 显存限制 |
| End-to-End FPS | 系统持续处理能力 |
| P50/P95 Latency | 处理延迟及长尾问题 |
| Stream Overlap | 异步执行是否真正生效 |
| CPU Utilization | 是否降低主机端处理负担 |

### 8.2 业务性能指标

| 指标 | 业务意义 |
|---|---|
| Per-Class Recall | 可见病害是否被检出 |
| AP50 / AP50-95 | 模型检测质量 |
| Video-Hours / Processing-Hour | 录像处理效率 |
| False Positive Events | 人工复核负担 |
| Duplicate Event Rate | 重复病害记录数量 |
| Evidence Completeness | 证据记录完整性 |
| Frame Drop Rate | 视频负载下的数据损失 |

事件级指标需要一组独立人工标注的视频，不能仅使用 RDD2022 的静态图片代替。

### 8.3 初始验收负载

以下为独立项目的**测试目标**，不是路桥信息的实际生产 SLA：

| 测试 | 初始配置 |
|---|---|
| 单路输入 | 1080p / 30 FPS |
| CNN 输入 | 640×640 |
| CUDA Tile Batch | 1、2、4、8 |
| 推理精度 | FP32、FP16 |
| 多路压力测试 | 2、4 路回放 |
| 稳定运行 | ≥30 分钟 |
| 视频处理 | 至少一段 ≥10 分钟的完整录像 |
| 高分辨率测试 | 真实高清影像；合成 Buffer 仅用于性能压力测试 |

### 8.4 性能实验版本

| 版本 | 内容 | 目的 |
|---|---|---|
| V0 | C++ + TensorRT 标准实现 | 工程 Baseline |
| V1 | V0 + Fused CUDA Preprocessing | 测试 Kernel 优化 |
| V2 | V1 + Reusable GPU Buffers | 测试显存管理 |
| V3 | V2 + Async Pipeline | 测试系统吞吐 |
| V4 | 可选 GPU Postprocessing | 测试后处理瓶颈 |

必须报告端到端结果。

不能仅凭 Kernel 加速 3×，就宣称整体视频处理也加速 3×。

如果成熟库的性能优于自定义 Kernel，应当保留负面实验结果并解释差异。

---

## 9. 十周实习级开发计划

| 周次 | 任务 | 核心交付物 |
|---|---|---|
| W1 | 环境搭建、需求与 Benchmark 设计 | Requirements、CUDA Harness |
| W2 | CNN Fine-tuning、ONNX 导出 | Model、Evaluation |
| W3 | C++ TensorRT Baseline | 可运行推理程序 |
| W4 | 视频输入与 GPU 分阶段 Profiling | Bottleneck Report |
| W5 | Fused CUDA Preprocessing 开发 | 自定义 Kernel |
| W6 | Kernel Tuning 与正确性测试 | Nsight Compute Report |
| W7 | GPU Buffer Pool | 显存管理模块 |
| W8 | CUDA Streams、Events、Double Buffer | Async Pipeline |
| W9 | MP4/RTSP、Evidence、稳定性测试 | Integration Tests |
| W10 | 完整 Benchmark、文档与演示 | Final Performance Report |

### 优先级控制

**必须完成（P0）：**

- 训练或微调可运行 CNN
- C++/TensorRT Inference
- Fused CUDA Preprocessing
- GPU Memory Pool
- Async GPU Pipeline
- 可复现性能报告
- 完整视频输入与证据输出

**按实测瓶颈选择（P1）：**

- GPU Detection Compaction
- 高分辨率 Tiled Inference
- 更复杂的 Batch Scheduling
- RTSP 故障恢复强化

**不作为实习阶段交付（P2）：**

- Go2 实机驱动
- SLAM / Navigation
- CUDA Graphs 深入优化
- NVDEC GPU-Resident Pipeline
- 多机器人协同

若 P0 工作出现时间压力，优先完成正确性、Kernel Profiling 和异步 GPU Pipeline，而不是增加额外业务功能。

---

## 10. 代码交付结构

`cuda-road-inspection/`

- `training/`：数据处理、模型训练、评估和 ONNX 导出
- `cuda/`：Fused Preprocessing、可选 Detection Processing
- `include/`：CUDA Buffer、Stream、Event、TensorRT 接口
- `src/gpu_runtime/`：GPU Pipeline 与资源管理
- `src/video/`：MP4、RTSP、Synthetic Input
- `src/inference/`：TensorRT C++ 集成
- `src/inspection/`：证据和检测结果
- `tests/`：Kernel Correctness、Memory、Integration
- `benchmarks/`：Microbenchmark、Pipeline、End-to-End
- `reports/`：Nsight、性能报告、实验结论
- `configs/`：模型、输入及运行配置
- `CMakeLists.txt`
- `README.md`

不将公开数据集的大体积原始文件、企业专有数据或机器特定的 TensorRT Engine 直接提交到 GitHub。

---

## 11. 最终交付物

实习级项目完成后，至少应当展示以下成果。

### Deliverable 1：CUDA Kernel

自定义融合图像预处理 Kernel，附带数值正确性测试、边界测试和性能对比。

### Deliverable 2：C++/TensorRT 推理模块

能够在 Linux PC 上加载自己训练的 CNN，接收道路影像并执行 GPU 推理。

### Deliverable 3：GPU Runtime

包括 Buffer Pool、CUDA Streams、Events 和异步执行资源管理。

### Deliverable 4：完整视频演示

使用 MP4 或本地 RTSP 回放处理道路巡检视频，输出真实病害候选及证据记录。

### Deliverable 5：量化性能报告

包含：

- CUDA Kernel Speedup
- Preprocessing Latency
- End-to-End FPS
- P95 Latency
- Peak GPU Memory
- GPU Timeline
- Model Accuracy
- 稳定性测试结果

### Deliverable 6：技术说明

完整记录开发环境、设计取舍、已发现瓶颈、优化结果和未解决问题。

---

## 12. 项目完成标准

只有满足以下条件，才视为完成。

**工程正确性：**

CUDA Kernel、GPU Buffer 生命周期、TensorRT 推理和完整视频处理均通过测试。

**GPU 技术深度：**

有自主实现的 CUDA Kernel、GPU Memory Management 和异步执行模块，并能解释实际 GPU 执行瓶颈。

**性能量化：**

至少完成两项独立 CUDA 优化实验，以及一项端到端系统性能对比。

所有提升数据以 RTX PC 实测为准。

**业务完整性：**

系统能够从道路影像生成带时间戳、来源、类别、置信度、证据和模型版本的检测记录。

**硬件独立性：**

所有核心功能均能够在没有 Go2 的情况下运行和验收。

---

## 13. 与原实习经历的关系

本项目参考厦门路桥信息公开披露的智慧养护和路面病害识别业务，并使用此前 Go2、ROS、SLAM、RTSP 相关经历作为技术背景。

它不声称取得公司内部代码、企业模型或真实业务数据。

2025 年实习期间真实完成的机器人与视频集成工作，应保留在原始实习经历中。

此后独立实现的 CNN、CUDA 和 TensorRT 模块，应作为独立项目展示。

**最终项目定位：**

> 面向道路智慧养护业务的 C++/CUDA GPU 推理优化原型，重点验证自定义 CUDA Kernel、异步显存管理和高性能视频推理在消费级 NVIDIA RTX 平台上的工程可行性。

**最终验收结论应回答：**

在固定模型、检测质量和硬件条件下，自主开发的 CUDA Pipeline 是否使道路巡检影像处理更加高效、稳定，并具备可复现的性能证据？

这就是项目最重要的交付结果。
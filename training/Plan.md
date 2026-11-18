## CNN 道路病害检测模型训练方案

**目标：** 基于 RDD2022 数据集微调 YOLO11n，训练一个能够识别道路裂缝和坑洞的 CNN 模型，最终部署到 C++/CUDA/TensorRT 推理系统。

| 阶段 | 主要任务 | 产出 |
|---|---|---|
| 1. 数据准备 | 下载 RDD2022，转换 YOLO 标注格式，划分 Train / Val / Test | 训练数据集 |
| 2. 模型选择 | 加载预训练 YOLO11n 模型 | `yolo11n.pt` |
| 3. 模型训练 | 使用 PyTorch/Ultralytics 进行 Fine-tuning，输入尺寸 640×640 | `best.pt` |
| 4. 模型评估 | 使用独立测试集计算 Precision、Recall、mAP | 模型评估报告 |
| 5. 模型导出 | 导出 ONNX，并构建 TensorRT Engine | `best.onnx` / `.engine` |
| 6. CUDA 集成 | 使用 C++/CUDA 实现 Letterbox、Normalization、CHW 转换并调用 TensorRT | GPU 推理模块 |

### 关键配置

- **检测类别（第一版）：** D00、D10、D20、D40，共 **4 类道路病害**。D43/D44/D50 暂不参与训练，作为干扰样本被过滤；后续版本可扩展为 7 类，并单独报告干扰类指标。
- **训练方式：** 预训练模型 Fine-tuning，不从零训练 CNN。
- **图像预处理：** 保持宽高比，验证和部署统一使用 640×640 Letterbox。
- **GPU：** RTX 4060 Laptop 8GB，初始 Batch Size 8，显存不足则调整为 4。
- **训练轮数：** 初始 50 Epochs，根据验证结果和 Early Stopping 调整。

### 验收标准

1. 模型能够识别四类道路病害（D00/D10/D20/D40），并提供可复现的测试指标。
2. PyTorch 与 TensorRT 推理结果在规定容差内一致。
3. CUDA 预处理输出与 Python Letterbox 参考结果一致。

**最终交付：** 一个可用于 CUDA 性能优化实验的道路病害检测模型。模型训练只是准备真实推理负载，项目研发重点仍是 CUDA Kernel Optimization 和 GPU Systems Engineering。


### 原理

#### **--- STEP1: Train ---**
1. CNN学习 `dataset/images`的视觉特征
2. CNN读取 `dataset/label` 的正确标注
3. 计算预测结果与真实标注的误差（Loss）。
4. 根据 Loss 反向传播，更新 CNN 参数。
5. 生成多个**Weights**, 存入/val

#### **--- STEP2: Validation ---**
训练过程中会产生多个不同版本的模型参数（Weights），Val帮助我们选择表现最好的那一版

#### **--- STEP3: Test ---**
使用/test里的全新数据集进行测试 最后导出ONNX给TensorRT

### Dataset目录结构
```text
RDD2022/
├── images/
│   ├── train/    ← 训练用图片
│   ├── val/      ← 验证用图片
│   └── test/     ← 测试用图片
│
└── labels/
    ├── train/    ← 训练图片的正确答案
    ├── val/      ← 验证图片的正确答案
    └── test/     ← 测试图片的正确答案
```
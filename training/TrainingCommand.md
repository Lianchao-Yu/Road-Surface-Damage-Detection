# 训练命令

## 1. 检查数据（不训练）

```bash
.venv-train/bin/python training/train.py --dry-run
```

## 2. 正式训练

```bash
.venv-train/bin/python training/train.py
```

默认配置：

```text
模型      yolo11n.pt
数据      dataset/RDD2022/data.yaml
输入      640
epochs    50
batch     8
workers   2
```

## 3. 低内存模式

```bash
.venv-train/bin/python training/train.py --batch 4 --workers 0
```

## 4. 继续训练

```bash
.venv-train/bin/python training/train.py --resume --name rdd2022_yolo11n
```

## 5. 查看进度和内存

```bash
tail -f training/runs/rdd2022_yolo11n/results.csv
tail -f training/runs/rdd2022_yolo11n/memory_log.csv
```

## 6. 训练输出

```text
training/runs/rdd2022_yolo11n/
├── weights/best.pt
├── weights/last.pt
├── results.csv
├── memory_log.csv
└── plots/
```

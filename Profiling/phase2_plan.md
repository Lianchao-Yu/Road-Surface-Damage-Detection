Phase 2B:

- 使用RT-DETR再走一次测试
- 测试下高分辨率视频，重新分析下性能(Nsight Systems， GPU Profiling、异步执行验证)
- 迁移部署至-Nvidia Orin NX


| 测试分辨率| Sync | Async 2-slots |Async 4-slots |
|:---:|:---|:---|:---|
| 720p|	已测试	|已测试	| 已测试 |
|1080p|	待补测	|待测	|待测|
|1440p|	待测|	待测|	待测|
|4K	|待测	|待测	|待测|
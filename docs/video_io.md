# FFmpeg 视频输入基础结构（rd_io）

实现位置：

- 接口：`include/road_damage/io/video_source.hpp`
- 软件解码实现：`src/io/ffmpeg_video_source.cpp`
- 调试工具：`tools/video_probe.cpp`
- 构建目标：`rd_io`（需要 pkg-config 找到 FFmpeg）

## 1. 当前范围

```text
libavformat  → MP4 / RTSP 解封装
libavcodec   → H.264/H.265 软件解码
libswscale   → YUV420P / YUVJ420P / 10-bit 等 → BGR24
```

输出统一的 `VideoFrame`：

```cpp
struct VideoFrame {
    const std::uint8_t* data;  // BGR24
    int width;
    int height;
    int stride;                // 可能大于 width * 3
    std::int64_t pts_us;
    std::uint64_t frame_id;
};
```

`data` 指向 source 内部复用的 BGR buffer，只在下次 `read()` 或 `close()` 前有效。需要跨帧保留时必须复制。

## 2. 接口

```cpp
rd::io::FFmpegVideoSource source;
if (!source.open("video.mp4")) {
    std::cerr << source.last_error() << "\n";
}

rd::io::VideoFrame frame;
while (true) {
    const auto status = source.read(frame);
    if (status == rd::io::ReadStatus::kEndOfStream) break;
    if (status == rd::io::ReadStatus::kError) break;
    // frame.data / width / height / stride / pts_us / frame_id
}
source.close();
```

`IFrameSource` 保留为抽象接口，后续 RTSP 专用 source 或 NVDEC source 可以替换，而不改上层推理循环。

## 3. 已处理的细节

- 使用 `best_effort_timestamp`，转换为微秒 `pts_us`
- `frame_id` 从 0 单调递增
- `av_image_alloc(..., 32)` 生成 32 字节对齐的 BGR24 行，`stride` 可直接传给 GPU preprocess
- 支持 YUVJ420P/YUVJ422P 等废弃全范围格式，显式转换为现代格式并设置 range
- 根据 `color_space` 选择 BT.601 / BT.709 / BT.2020 系数
- send/receive API 正确处理 `EAGAIN` 和 EOF flush
- 解码线程数交给 libavcodec 自动选择（`thread_count = 0`）
- RTSP 默认 `rtsp_transport=tcp`

## 4. 当前限制

- 只做软件解码；NVDEC 是后续 P2 实验
- RTSP 暂无超时、重连、抖动缓冲
- 没有音频处理；非视频包直接丢弃
- 色彩转换是 CPU `sws_scale`，会产生一次 CPU 写 BGR 和后续 H2D
- 解码器输出 BGR24，暂不支持硬件帧（AV_PIX_FMT_CUDA）直接进入 GPU

## 5. 验证

```bash
cmake --preset cpu
cmake --build --preset cpu

# 用真实 MP4 验证解码、PTS、stride、EOF
./build/cpu/tools/video_probe /path/to/video.mp4 10
```

可选集成测试：

```bash
RD_TEST_VIDEO=/path/to/video.mp4 ./build/cpu/tests/rd_tests \
  --gtest_filter='VideoSource.*'
```

不设置 `RD_TEST_VIDEO` 时，只运行打开失败、空 URL、未打开读取等基础用例。

## 6. 下一步

```text
FFmpegVideoSource
→ yolo_video 同步循环
→ 复用 GPU buffer / stream
→ 分阶段 timing
→ Buffer Pool + 异步 Pipeline
→ RTSP 超时与重连
```

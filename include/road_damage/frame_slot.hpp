#include <cstddef>
#include <cstdint>
#include <opencv2/core/mat.hpp>
#include <cuda_runtime.h>

struct FrameSlot
{
    int id = 0;
    int frame_idx = -1;
    int width = 0;
    int height = 0;
    size_t src_stride = 0; // For the synchronized step length, directly reuse what you just wrote

    // 1. CPU Pinned Memory (页锁定内存)
    uint8_t *h_bgr = nullptr;  // 解码后的原图存在这里
    float *h_output = nullptr; // GPU 推理结果拷回这里 (大小很小，几百KB)

    // 2. 异步同步事件
    cudaEvent_t gpu_done_event = nullptr;

    // 辅助函数：把 Pinned Memory 零拷贝包装成 cv::Mat 给 OpenCV 画框用
    cv::Mat wrapAsMat() const
    {
        return cv::Mat(height, width, CV_8UC3, h_bgr, src_stride);
    }
};


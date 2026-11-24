#include "road_damage/preprocess_GPU.hpp"
#include <iostream>
/*

支持 src_stride（行跨度，适配 FFmpeg / OpenCV ROI）

只需在 Kernel 参数中增加一个 int src_stride_bytes（默认等于 origWidth * 3），
把原图寻址从 (y0 * meta.origWidth + x0) * 3 改为 y0 * src_stride_bytes + x0 * 3。

改动只有一行，却能直接打通 FFmpeg 硬件解码帧（AVFrame->linesize[0] 往往为了 32/64 字节内存对齐而大于 width * 3）和 OpenCV 裁剪子图，
避免额外做一次 CPU 内存拷贝来消除 padding。

打通 mean / inv_std 与直接 Resize 模式（兼容 RT-DETR）

你在 Kernel 末尾加上 (value_c[0] - mean.x) * inv_std.x（YOLO 传 mean={0,0,0}, inv_std={1,1,1}），即可同时覆盖 YOLO 和 RT-DETR。

对于 RT-DETR 的直接 Resize（无 Letterbox 灰边），
完全不需要重写新 Kernel——只需在 Host 端构造 LetterboxMeta 时传入 padX = 0, padY = 0，并让 resized_w = inputWidth, resized_h = inputHeight，你现有的 Kernel 就能无缝兼容。

使用锁页内存（cudaHostAlloc / Pinned Memory）加速 H2D

在接入 NVDEC 硬解之前，如果输入帧在 CPU 内存，
使用 cudaHostAlloc 分配锁页内存存放待上传的 BGR 帧，通常能把 PCIe H2D 带宽再提升 1.5× ~ 2×，
并支持 cudaMemcpyAsync 与后续算子重叠执行。
*/
namespace fused_k1
{
    //============= This kernal is designed for YOLO ===========
    //stride: 由于需要进行内存对齐， 图像行之间存在填充字节(padding byte)
    //因此: stride = width * byte + padding byte (内存排布：[1, 2, 3, .... width-1, padding, ...])
    __global__ void preprocess_Kernal(
        const uint8_t *d_src,
        float *d_dst,
        LetterboxMeta meta,
        uint8_t pad_value,
        int src_stride)
    {
        int x = blockIdx.x * blockDim.x + threadIdx.x;
        int y = blockIdx.y * blockDim.y + threadIdx.y;
        int input_w = meta.inputWidth;
        int input_h = meta.inputHeight;

        if (x >= input_w || y >= input_h)
            return; // check the thread boundary

        const int resized_w = max(1, __float2int_rn(meta.origWidth * meta.scale));
        const int resized_h = max(1, __float2int_rn(meta.origHeight * meta.scale));
        const int pad_x = static_cast<int>(meta.padX);
        const int pad_y = static_cast<int>(meta.padY);

        // Use the actual resized dimensions, not meta.scale directly. The
        // rounding of resized_w/resized_h can make the effective x/y scales
        // slightly different from the uniform meta.scale (e.g. 101x67).
        const float scale_x = static_cast<float>(meta.origWidth) / static_cast<float>(resized_w);
        const float scale_y = static_cast<float>(meta.origHeight) / static_cast<float>(resized_h);

        // YOLO contract: packed BGR8, pad value 114, mean=0, std=1.
        // The current kernel does not apply per-channel mean/std and does
        // not support a source row stride (FFmpeg linesize).
        float value_c[3];
        if (x < pad_x || x >= resized_w + pad_x ||
            y < pad_y || y >= resized_h + pad_y)
        {
            float pad_norm = pad_value / 255.0f;
            value_c[0] = pad_norm;
            value_c[1] = pad_norm;
            value_c[2] = pad_norm;
        }
        else
        {
            float origin_x = (x - pad_x + 0.5f) * scale_x - 0.5f;
            float origin_y = (y - pad_y + 0.5f) * scale_y - 0.5f;
            origin_x = fminf(fmaxf(origin_x, 0.0f), static_cast<float>(meta.origWidth - 1)); // clamp the edge
            origin_y = fminf(fmaxf(origin_y, 0.0f), static_cast<float>(meta.origHeight - 1));
            int x0 = __float2int_rd(origin_x);
            int y0 = __float2int_rd(origin_y);
            int x1 = min(x0 + 1, meta.origWidth - 1);
            int y1 = min(y0 + 1, meta.origHeight - 1);

            float fx = origin_x - static_cast<float>(x0);
            float fy = origin_y - static_cast<float>(y0);
            float w00 = (1.0f - fx) * (1.0f - fy); // left-up
            float w01 = fx * (1.0f - fy);          // right-up
            float w10 = (1.0f - fx) * fy;          // left-down
            float w11 = fx * fy;                   // right-down

            // bgr8 to NCHW
            float Norm = 1.0f / 255.0f;
            const size_t offset_y0 = y0 * src_stride;
            const size_t offset_y1 = y1 * src_stride;
            const size_t offset_x0 = x0 * 3;
            const size_t offset_x1 = x1 * 3;
            for (int c = 0; c < 3; c++)
            {
                value_c[2 - c] = 
                    d_src[offset_y0 + offset_x0 + c] * w00 * Norm +
                    d_src[offset_y0 + offset_x1 + c] * w01 * Norm +
                    d_src[offset_y1 + offset_x0 + c] * w10 * Norm +
                    d_src[offset_y1 + offset_x1 + c] * w11 * Norm;
            }
        }
        //output
        const int colorPlane = input_w * input_h;
        d_dst[0 * colorPlane + y * input_w + x] = value_c[0];
        d_dst[1 * colorPlane + y * input_w + x] = value_c[1];
        d_dst[2 * colorPlane + y * input_w + x] = value_c[2];
    }

    bool launchFusedPreprocess(
        const uint8_t *d_src,
        float *d_dst,
        LetterboxMeta meta,
        cudaStream_t stream)
    {
        // 1. Basic parameters and null pointer check
        if (d_src == nullptr || d_dst == nullptr)
        {
            return false;
        }
        if (meta.origWidth <= 0 || meta.origHeight <= 0 ||
            meta.inputWidth <= 0 || meta.inputHeight <= 0 ||
            !(meta.scale > 0.0f) ||
            meta.padX < 0.0f || meta.padY < 0.0f)
        {
            return false;
        }

        // 2. Launch kernal
        const dim3 block(16, 16);
        const dim3 grid((meta.inputWidth + block.x - 1) / block.x,
                        (meta.inputHeight + block.y - 1) / block.y);
        const uint8_t pad_value = 114;
        preprocess_Kernal<<<grid, block, 0, stream>>>(
            d_src,
            d_dst,
            meta,
            pad_value);

        // 3. Check whether the Kernel startup parameters or configuration are legal
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess)
        {
            std::cout << "Fail to launch preprocess kernal!\n";
            return false;
        }
        return true;
    }
}

#include <gtest/gtest.h>

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <vector>

#include "road_damage/preprocess_GPU.hpp"
#include "road_damage/preprocess_cpu.hpp"

#ifdef RD_WITH_OPENCV
#include "road_damage/preprocess_cv.hpp"
#endif

namespace {

struct TestImage {
    std::vector<std::uint8_t> pixels;
    rd::ImageBgr8 view;
};

// Padding bytes are filled with garbage so a kernel that ignores the row
// stride produces visibly wrong pixels.
TestImage make_image(int width, int height, int pad_bytes) {
    const int stride = width * 3 + pad_bytes;
    TestImage result;
    result.pixels.assign(static_cast<std::size_t>(stride) * height, 0xCD);
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            for (int c = 0; c < 3; ++c) {
                result.pixels[static_cast<std::size_t>(y) * stride + x * 3 + c] =
                    static_cast<std::uint8_t>((y * 131 + x * 37 + c * 59 + 11) % 256);
            }
        }
    }
    result.view = rd::ImageBgr8{result.pixels.data(), width, height, stride};
    return result;
}

rd::PreprocessSpec make_spec(int width, int height) {
    rd::PreprocessSpec spec;
    spec.dst_width = width;
    spec.dst_height = height;
    spec.bgr_to_rgb = true;
    return spec;
}

float max_abs_diff(const std::vector<float>& a, const std::vector<float>& b) {
    EXPECT_EQ(a.size(), b.size());
    float diff = 0.0f;
    for (std::size_t i = 0; i < a.size(); ++i) {
        diff = std::max(diff, std::fabs(a[i] - b[i]));
    }
    return diff;
}

struct DeviceBuffer {
    void* ptr = nullptr;
    ~DeviceBuffer() {
        if (ptr != nullptr) {
            cudaFree(ptr);
        }
    }
};

struct DeviceStream {
    cudaStream_t stream = nullptr;
    ~DeviceStream() {
        if (stream != nullptr) {
            cudaStreamDestroy(stream);
        }
    }
};

struct Case {
    int width;
    int height;
    int pad_bytes;
};

}  // namespace

TEST(PreprocessGpu, MatchesCpuReferenceAndHonorsStride) {
    int devices = 0;
    ASSERT_EQ(cudaGetDeviceCount(&devices), cudaSuccess);
    ASSERT_GT(devices, 0);

    const Case cases[] = {
        {600, 600, 0}, {600, 600, 64}, {1280, 720, 128}, {101, 67, 32}, {2, 4, 16},
    };

    for (const Case& c : cases) {
        TestImage image = make_image(c.width, c.height, c.pad_bytes);
        const rd::PreprocessSpec spec = make_spec(640, 640);
        const LetterboxMeta meta = rd::compute_letterbox_meta(image.view, spec);

        const std::vector<float> reference =
            rd::preprocess_bgr8_to_nchw_f32_letterbox_single(image.view, spec, meta, 114.0f);

        DeviceBuffer d_src;
        DeviceBuffer d_dst;
        DeviceStream stream;
        const std::size_t src_bytes = image.pixels.size();
        const std::size_t dst_bytes = reference.size() * sizeof(float);
        ASSERT_EQ(cudaMalloc(&d_src.ptr, src_bytes), cudaSuccess);
        ASSERT_EQ(cudaMalloc(&d_dst.ptr, dst_bytes), cudaSuccess);
        ASSERT_EQ(cudaStreamCreateWithFlags(&stream.stream, cudaStreamNonBlocking), cudaSuccess);
        ASSERT_EQ(cudaMemcpyAsync(d_src.ptr, image.pixels.data(), src_bytes,
                                  cudaMemcpyHostToDevice, stream.stream),
                  cudaSuccess);

        ASSERT_TRUE(fused_k1::launchFusedPreprocess(
            static_cast<const std::uint8_t*>(d_src.ptr), static_cast<float*>(d_dst.ptr), meta,
            stream.stream, static_cast<std::size_t>(image.view.stride)));
        ASSERT_EQ(cudaStreamSynchronize(stream.stream), cudaSuccess);

        std::vector<float> gpu(reference.size(), 0.0f);
        ASSERT_EQ(cudaMemcpyAsync(gpu.data(), d_dst.ptr, dst_bytes,
                                  cudaMemcpyDeviceToHost, stream.stream),
                  cudaSuccess);
        ASSERT_EQ(cudaStreamSynchronize(stream.stream), cudaSuccess);

        EXPECT_LE(max_abs_diff(reference, gpu), 1e-5f)
            << "GPU case " << c.width << "x" << c.height << " stride padding "
            << c.pad_bytes;

#ifdef RD_WITH_OPENCV
        const std::vector<float> cv_float = rd::preprocess_bgr8_to_nchw_f32_letterbox_cv_single(
            image.view, spec, meta, 114.0f, rd::CvInterpolationSource::kFloat32);
        const std::vector<float> cv_u8 = rd::preprocess_bgr8_to_nchw_f32_letterbox_cv_single(
            image.view, spec, meta, 114.0f, rd::CvInterpolationSource::kUint8);

        EXPECT_LE(max_abs_diff(cv_float, gpu), 1e-5f)
            << "CV float vs GPU case " << c.width << "x" << c.height;
        EXPECT_LE(max_abs_diff(cv_u8, gpu), 1.0f / 255.0f + 1e-5f)
            << "CV u8 vs GPU case " << c.width << "x" << c.height;
#endif
    }
}

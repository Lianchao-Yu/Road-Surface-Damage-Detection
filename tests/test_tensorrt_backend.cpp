#include <gtest/gtest.h>

#include <cuda_runtime_api.h>

#include <cmath>
#include <filesystem>
#include <string>
#include <unordered_map>
#include <vector>

#include "road_damage/tensorrt_backend.hpp"

namespace {

std::string engine_path() {
    return std::string(RD_SOURCE_DIR) + "/models/best.engine";
}

}  // namespace

TEST(TensorRTBackend, DynamicBatch) {
    if (!std::filesystem::exists(engine_path())) {
        GTEST_SKIP() << "missing engine: " << engine_path();
    }

    int device_count = 0;
    if (cudaGetDeviceCount(&device_count) != cudaSuccess || device_count == 0) {
        GTEST_SKIP() << "no CUDA device available";
    }

    trt::TensorRTBackend backend;
    ASSERT_TRUE(backend.TensorRTLoader(engine_path()));

    constexpr int kChannels = 3;
    constexpr int kSize = 640;
    constexpr int kOutputChannels = 8;
    constexpr int kAnchors = 8400;
    constexpr int kMaxBatch = 8;

    std::vector<float> output(static_cast<std::size_t>(kMaxBatch) * kOutputChannels * kAnchors,
                              0.0f);

    for (int batch : {1, 2, 4, 8}) {
        std::vector<float> input(static_cast<std::size_t>(batch) * kChannels * kSize * kSize,
                                 0.5f);

        trt::TensorView in_view;
        in_view.data = input.data();
        in_view.dims.nbDims = 4;
        in_view.dims.d[0] = batch;
        in_view.dims.d[1] = kChannels;
        in_view.dims.d[2] = kSize;
        in_view.dims.d[3] = kSize;
        in_view.dtype = nvinfer1::DataType::kFLOAT;
        in_view.memType = trt::MemoryType::kCPU;

        trt::TensorView out_view;
        out_view.data = output.data();
        out_view.dtype = nvinfer1::DataType::kFLOAT;
        out_view.memType = trt::MemoryType::kCPU;

        std::unordered_map<std::string, trt::TensorView> inputs;
        std::unordered_map<std::string, trt::TensorView> outputs;
        inputs.emplace("images", in_view);
        outputs.emplace("output0", out_view);

        ASSERT_TRUE(backend.TensorInfer(inputs, outputs)) << "batch=" << batch;

        const trt::TensorView& result = outputs.at("output0");
        ASSERT_EQ(result.dims.nbDims, 3);
        EXPECT_EQ(result.dims.d[0], batch);
        EXPECT_EQ(result.dims.d[1], kOutputChannels);
        EXPECT_EQ(result.dims.d[2], kAnchors);
        // TensorInfer returns its internal host buffer through result.data;
        // the caller-provided output vector is not written by this API.
        ASSERT_NE(result.data, nullptr);
        const float* result_data = static_cast<const float*>(result.data);
        EXPECT_TRUE(std::isfinite(result_data[0]));
        EXPECT_NE(result_data[0], 0.0f);
    }
}

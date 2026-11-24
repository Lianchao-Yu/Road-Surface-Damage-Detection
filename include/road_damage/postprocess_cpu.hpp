#pragma once
#include <cstddef>
#include <cstdint>
#include <vector>
#include <unordered_map>
#include "road_damage/tensorrt_backend.hpp"
#include "road_damage/shared_data.hpp"
namespace postprocessing
{

    std::vector<Detection> PostProcessing_yolo
    (
        std::unordered_map<std::string, trt::TensorView> OutputView,
        const LetterboxMeta &letterbox, 
        float confThreshold = 0.25f,
        float iouThreshold = 0.45f
    );

    std::vector<Detection> PostProcessing_RTDETR(
        std::unordered_map<std::string, trt::TensorView> OutputView,
        const LetterboxMeta &letterbox,
        float confThreshold = 0.25f);

    /// BGR8 host image. `stride` is the number of bytes between rows and may be
    /// larger than width * 3 (FFmpeg linesize). Pixels are 3 bytes: B, G, R.
    struct ImageBgr8
    {
        const std::uint8_t *data = nullptr;
        int width = 0;
        int height = 0;
        int stride = 0;
    };

    /// Per-channel normalization. `mean` and `std` are expressed in [0, 1] units
    /// and are applied after dividing the 8-bit pixel by 255.
    struct Normalization
    {
        float mean[3] = {0.0f, 0.0f, 0.0f};
        float std[3] = {1.0f, 1.0f, 1.0f};
    };

    struct PreprocessSpec
    {
        int dst_width = 640;
        int dst_height = 640;
        bool bgr_to_rgb = true;
        Normalization norm{};
    };
};

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

    
    std::vector<Detection> PostProcessing_RTDETR
    (
        std::unordered_map<std::string, trt::TensorView> OutputView,
        const LetterboxMeta &letterbox, 
        float confThreshold = 0.25f
    );
     
};
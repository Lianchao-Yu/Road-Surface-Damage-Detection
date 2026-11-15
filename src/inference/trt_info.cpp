#include "road_damage/trt_info.hpp"

#include <NvInfer.h>

#include <string>

namespace rd {

std::string tensorrt_version() {
    return std::to_string(NV_TENSORRT_MAJOR) + "." + std::to_string(NV_TENSORRT_MINOR) + "." +
           std::to_string(NV_TENSORRT_PATCH);
}

}  // namespace rd

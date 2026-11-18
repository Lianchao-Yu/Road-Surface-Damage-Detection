#include "road_damage/cuda_info.hpp"

#include <cuda_runtime_api.h>

#include <cstdio>

namespace rd {

bool cuda_available() {
    int count = 0;
    return cudaGetDeviceCount(&count) == cudaSuccess && count > 0;
}

std::string cuda_device_name() {
    int device = 0;
    cudaDeviceProp prop{};
    if (cudaGetDevice(&device) != cudaSuccess) {
        return "unknown";
    }
    if (cudaGetDeviceProperties(&prop, device) != cudaSuccess) {
        return "unknown";
    }
    return prop.name;
}

std::string cuda_runtime_version() {
    int version = 0;
    if (cudaRuntimeGetVersion(&version) != cudaSuccess) {
        return "unknown";
    }
    char buffer[32];
    std::snprintf(buffer, sizeof(buffer), "%d.%d", version / 1000, (version % 1000) / 10);
    return buffer;
}

}  // namespace rd

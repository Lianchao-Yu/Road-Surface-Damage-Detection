#pragma once

#include <cstdint>
#include <cuda_runtime.h>

#include "road_damage/shared_data.hpp"

namespace fused_k1
{
bool launchFusedPreprocess(
    const uint8_t* d_src,
    float* d_dst,
    LetterboxMeta meta,
    cudaStream_t stream,
    size_t src_stride
);
}

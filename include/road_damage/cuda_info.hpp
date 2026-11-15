#pragma once

#include <string>

namespace rd {

/// True when at least one CUDA device is visible to the process.
bool cuda_available();

/// Name of the current CUDA device, or "unknown".
std::string cuda_device_name();

/// CUDA runtime version as "major.minor", or "unknown".
std::string cuda_runtime_version();

}  // namespace rd

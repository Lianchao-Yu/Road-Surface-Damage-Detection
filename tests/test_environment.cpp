#include <gtest/gtest.h>

#include <string>

#include "road_damage/version.hpp"

#ifdef RD_WITH_CUDA
#include "road_damage/cuda_info.hpp"
#endif

#ifdef RD_WITH_TENSORRT
#include "road_damage/trt_info.hpp"
#endif

TEST(Environment, VersionIsSet) {
    const std::string version = rd::version();
    EXPECT_FALSE(version.empty());
    EXPECT_NE(version.find('.'), std::string::npos);
}

#ifdef RD_WITH_CUDA
TEST(Environment, CudaRuntimeVisible) {
    if (!rd::cuda_available()) {
        GTEST_SKIP() << "no CUDA device available";
    }
    EXPECT_NE(rd::cuda_runtime_version(), "unknown");
    EXPECT_NE(rd::cuda_device_name(), "unknown");
}
#endif

#ifdef RD_WITH_TENSORRT
TEST(Environment, TensorRtVersionParsed) {
    const std::string version = rd::tensorrt_version();
    EXPECT_FALSE(version.empty());
    EXPECT_NE(version.find('.'), std::string::npos);
}
#endif

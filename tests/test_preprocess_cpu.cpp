#include <gtest/gtest.h>

#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <vector>

#include "road_damage/preprocess.hpp"

namespace {

constexpr float kInv255 = 1.0f / 255.0f;

rd::PreprocessSpec make_spec(int width, int height) {
    rd::PreprocessSpec spec;
    spec.dst_width = width;
    spec.dst_height = height;
    spec.bgr_to_rgb = true;
    spec.norm.mean[0] = 0.0f;
    spec.norm.mean[1] = 0.0f;
    spec.norm.mean[2] = 0.0f;
    spec.norm.std[0] = 1.0f;
    spec.norm.std[1] = 1.0f;
    spec.norm.std[2] = 1.0f;
    return spec;
}

}  // namespace

TEST(PreprocessCpu, ElementCount) {
    const rd::PreprocessSpec spec = make_spec(640, 480);
    EXPECT_EQ(rd::tensor_element_count(spec), 3u * 640u * 480u);
}

TEST(PreprocessCpu, IdentityResizeSwapsBgrToRgb) {
    // 2x2 BGR image, row-major.
    const std::vector<std::uint8_t> pixels = {
        10, 20, 30, 40, 50, 60,
        70, 80, 90, 100, 110, 120,
    };
    const rd::ImageBgr8 image{pixels.data(), 2, 2, 6};
    const rd::PreprocessSpec spec = make_spec(2, 2);

    const std::vector<float> out = rd::preprocess_bgr8_to_nchw_f32(image, spec);
    ASSERT_EQ(out.size(), 12u);

    // Planes: R (source channel 2), G (1), B (0).
    const float expected_r[4] = {30, 60, 90, 120};
    const float expected_g[4] = {20, 50, 80, 110};
    const float expected_b[4] = {10, 40, 70, 100};
    for (int i = 0; i < 4; ++i) {
        EXPECT_NEAR(out[static_cast<std::size_t>(i)], expected_r[i] * kInv255, 1e-6f);
        EXPECT_NEAR(out[4u + static_cast<std::size_t>(i)], expected_g[i] * kInv255, 1e-6f);
        EXPECT_NEAR(out[8u + static_cast<std::size_t>(i)], expected_b[i] * kInv255, 1e-6f);
    }
}

TEST(PreprocessCpu, BilinearVerticalHalfPixel) {
    // Rows: 0 then 100. 2 -> 4 vertical upscale gives 0, 25, 75, 100.
    const std::vector<std::uint8_t> pixels = {
        0, 0, 0, 0, 0, 0,
        100, 100, 100, 100, 100, 100,
    };
    const rd::ImageBgr8 image{pixels.data(), 2, 2, 6};
    const rd::PreprocessSpec spec = make_spec(2, 4);

    const std::vector<float> out = rd::preprocess_bgr8_to_nchw_f32(image, spec);
    ASSERT_EQ(out.size(), 24u);

    const float expected[4] = {0.0f, 25.0f, 75.0f, 100.0f};
    for (int y = 0; y < 4; ++y) {
        EXPECT_NEAR(out[static_cast<std::size_t>(y) * 2u], expected[y] * kInv255, 1e-6f);
    }
}

TEST(PreprocessCpu, BilinearHorizontalNonIntegerScale) {
    // 3 -> 2 horizontal resize: source coords 0.25 and 1.75.
    const std::vector<std::uint8_t> pixels = {
        0, 0, 0, 100, 100, 100, 200, 200, 200,
    };
    const rd::ImageBgr8 image{pixels.data(), 3, 1, 9};
    const rd::PreprocessSpec spec = make_spec(2, 1);

    const std::vector<float> out = rd::preprocess_bgr8_to_nchw_f32(image, spec);
    ASSERT_EQ(out.size(), 6u);
    EXPECT_NEAR(out[0], 25.0f * kInv255, 1e-6f);
    EXPECT_NEAR(out[1], 175.0f * kInv255, 1e-6f);
}

TEST(PreprocessCpu, NormalizationUsesPerChannelRgbOrder) {
    const std::vector<std::uint8_t> pixels = {30, 20, 10};  // B, G, R
    const rd::ImageBgr8 image{pixels.data(), 1, 1, 3};
    rd::PreprocessSpec spec = make_spec(1, 1);
    spec.norm.mean[0] = 0.1f;
    spec.norm.mean[1] = 0.2f;
    spec.norm.mean[2] = 0.3f;
    spec.norm.std[0] = 0.5f;
    spec.norm.std[1] = 0.5f;
    spec.norm.std[2] = 0.5f;

    const std::vector<float> out = rd::preprocess_bgr8_to_nchw_f32(image, spec);
    ASSERT_EQ(out.size(), 3u);
    EXPECT_NEAR(out[0], (10.0f * kInv255 - 0.1f) / 0.5f, 1e-6f);
    EXPECT_NEAR(out[1], (20.0f * kInv255 - 0.2f) / 0.5f, 1e-6f);
    EXPECT_NEAR(out[2], (30.0f * kInv255 - 0.3f) / 0.5f, 1e-6f);
}

TEST(PreprocessCpu, MatchesPyTorchBilinearGolden) {
    // 5x4 BGR -> 3x2 RGB, bilinear align_corners=False, then per-channel
    // normalization. Expected values were produced with:
    //   torch.nn.functional.interpolate(x, size=(2, 3), mode="bilinear",
    //                                   align_corners=False)
    // and are kept here so the convention stays locked without a torch
    // dependency in the C++ test suite.
    constexpr int kWidth = 5;
    constexpr int kHeight = 4;
    std::vector<std::uint8_t> pixels(static_cast<std::size_t>(kWidth) * kHeight * 3u);
    for (std::size_t i = 0; i < pixels.size(); ++i) {
        pixels[i] = static_cast<std::uint8_t>((i * 7u + 3u) % 256u);
    }

    rd::PreprocessSpec spec = make_spec(3, 2);
    spec.norm.mean[0] = 0.1f;
    spec.norm.mean[1] = 0.2f;
    spec.norm.mean[2] = 0.3f;
    spec.norm.std[0] = 0.5f;
    spec.norm.std[1] = 0.5f;
    spec.norm.std[2] = 0.5f;

    const rd::ImageBgr8 image{pixels.data(), kWidth, kHeight, kWidth * 3};
    const std::vector<float> out = rd::preprocess_bgr8_to_nchw_f32(image, spec);

    const float expected[18] = {
        0.40000004f, 0.67450982f, 0.94901961f, 1.04313731f, 0.31372553f, 0.58823532f,
        0.14509806f, 0.41960785f, 0.69411767f, 0.78823531f, 0.05882353f, 0.33333334f,
        -0.10980394f, 0.16470587f, 0.43921566f, 0.53333342f, 0.80784309f, 0.07843137f,
    };
    ASSERT_EQ(out.size(), 18u);
    for (std::size_t i = 0; i < out.size(); ++i) {
        EXPECT_NEAR(out[i], expected[i], 1e-6f);
    }
}

TEST(PreprocessCpu, KeepsBgrOrderWhenRequested) {
    const std::vector<std::uint8_t> pixels = {30, 20, 10};
    const rd::ImageBgr8 image{pixels.data(), 1, 1, 3};
    rd::PreprocessSpec spec = make_spec(1, 1);
    spec.bgr_to_rgb = false;
    spec.norm.mean[0] = 0.1f;
    spec.norm.mean[1] = 0.2f;
    spec.norm.mean[2] = 0.3f;
    spec.norm.std[0] = 0.5f;
    spec.norm.std[1] = 0.5f;
    spec.norm.std[2] = 0.5f;

    const std::vector<float> out = rd::preprocess_bgr8_to_nchw_f32(image, spec);
    ASSERT_EQ(out.size(), 3u);
    EXPECT_NEAR(out[0], (30.0f * kInv255 - 0.1f) / 0.5f, 1e-6f);
    EXPECT_NEAR(out[1], (20.0f * kInv255 - 0.2f) / 0.5f, 1e-6f);
    EXPECT_NEAR(out[2], (10.0f * kInv255 - 0.3f) / 0.5f, 1e-6f);
}

TEST(PreprocessCpu, StridePaddingIsIgnored) {
    const std::vector<std::uint8_t> packed = {
        10, 20, 30, 40, 50, 60,
        70, 80, 90, 100, 110, 120,
    };
    // 2 px * 3 bytes = 6 bytes of data, padded to 8 bytes per row.
    std::vector<std::uint8_t> padded(16, 255);
    const std::uint8_t rows[2][6] = {
        {10, 20, 30, 40, 50, 60},
        {70, 80, 90, 100, 110, 120},
    };
    for (int y = 0; y < 2; ++y) {
        for (int i = 0; i < 6; ++i) {
            padded[static_cast<std::size_t>(y) * 8u + static_cast<std::size_t>(i)] = rows[y][i];
        }
    }

    const rd::PreprocessSpec spec = make_spec(2, 2);
    const std::vector<float> packed_out =
        rd::preprocess_bgr8_to_nchw_f32(rd::ImageBgr8{packed.data(), 2, 2, 6}, spec);
    const std::vector<float> padded_out =
        rd::preprocess_bgr8_to_nchw_f32(rd::ImageBgr8{padded.data(), 2, 2, 8}, spec);

    ASSERT_EQ(packed_out.size(), padded_out.size());
    for (std::size_t i = 0; i < packed_out.size(); ++i) {
        EXPECT_FLOAT_EQ(packed_out[i], padded_out[i]);
    }
}

TEST(PreprocessCpu, OddSizesKeepEdgePixels) {
    constexpr int kWidth = 3;
    constexpr int kHeight = 5;
    std::vector<std::uint8_t> pixels(static_cast<std::size_t>(kWidth) * kHeight * 3u);
    for (int y = 0; y < kHeight; ++y) {
        for (int x = 0; x < kWidth; ++x) {
            for (int c = 0; c < 3; ++c) {
                pixels[(static_cast<std::size_t>(y) * kWidth + static_cast<std::size_t>(x)) * 3u +
                       static_cast<std::size_t>(c)] =
                    static_cast<std::uint8_t>(y * 10 + x * 5 + c);
            }
        }
    }

    const rd::ImageBgr8 image{pixels.data(), kWidth, kHeight, kWidth * 3};
    const rd::PreprocessSpec spec = make_spec(7, 10);
    const std::vector<float> out = rd::preprocess_bgr8_to_nchw_f32(image, spec);

    ASSERT_EQ(out.size(), 3u * 7u * 10u);
    for (float value : out) {
        EXPECT_TRUE(std::isfinite(value));
    }

    // Half-pixel mapping clamps the first and last destination pixels to the
    // source corners when upscaling (scale < 1).
    const std::size_t plane = 7u * 10u;
    const float top_left_r = static_cast<float>(pixels[2]) * kInv255;
    const float bottom_right_r =
        static_cast<float>(pixels[(static_cast<std::size_t>(kHeight - 1) * kWidth +
                                   static_cast<std::size_t>(kWidth - 1)) *
                                      3u + 2u]) *
        kInv255;
    EXPECT_NEAR(out[0], top_left_r, 1e-6f);
    EXPECT_NEAR(out[plane - 1], bottom_right_r, 1e-6f);
}

TEST(PreprocessCpu, OneByOneImage) {
    const std::vector<std::uint8_t> pixels = {7, 8, 9};
    const rd::ImageBgr8 image{pixels.data(), 1, 1, 3};
    const rd::PreprocessSpec spec = make_spec(1, 1);
    const std::vector<float> out = rd::preprocess_bgr8_to_nchw_f32(image, spec);
    ASSERT_EQ(out.size(), 3u);
    EXPECT_NEAR(out[0], 9.0f * kInv255, 1e-6f);
    EXPECT_NEAR(out[1], 8.0f * kInv255, 1e-6f);
    EXPECT_NEAR(out[2], 7.0f * kInv255, 1e-6f);
}

TEST(PreprocessCpu, RejectsMalformedInput) {
    const std::vector<std::uint8_t> pixels = {0, 0, 0};
    EXPECT_THROW(rd::preprocess_bgr8_to_nchw_f32(rd::ImageBgr8{nullptr, 1, 1, 3}, make_spec(1, 1)),
                 std::invalid_argument);
    EXPECT_THROW(rd::preprocess_bgr8_to_nchw_f32(rd::ImageBgr8{pixels.data(), 1, 1, 2}, make_spec(1, 1)),
                 std::invalid_argument);
    EXPECT_THROW(rd::preprocess_bgr8_to_nchw_f32(rd::ImageBgr8{pixels.data(), 0, 1, 3}, make_spec(1, 1)),
                 std::invalid_argument);

    rd::PreprocessSpec zero_dst = make_spec(0, 1);
    EXPECT_THROW(rd::preprocess_bgr8_to_nchw_f32(rd::ImageBgr8{pixels.data(), 1, 1, 3}, zero_dst),
                 std::invalid_argument);

    rd::PreprocessSpec zero_std = make_spec(1, 1);
    zero_std.norm.std[1] = 0.0f;
    EXPECT_THROW(rd::preprocess_bgr8_to_nchw_f32(rd::ImageBgr8{pixels.data(), 1, 1, 3}, zero_std),
                 std::invalid_argument);
}

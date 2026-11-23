#include <gtest/gtest.h>

#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <vector>

#include "road_damage/preprocess_cpu.hpp"

namespace {

constexpr float kInv255 = 1.0f / 255.0f;

rd::PreprocessSpec make_spec(int width, int height) {
    rd::PreprocessSpec spec;
    spec.dst_width = width;
    spec.dst_height = height;
    spec.bgr_to_rgb = true;
    spec.norm.mean[0] = spec.norm.mean[1] = spec.norm.mean[2] = 0.0f;
    spec.norm.std[0] = spec.norm.std[1] = spec.norm.std[2] = 1.0f;
    return spec;
}

}  // namespace

TEST(Letterbox, ComputesScaleAndPadding) {
    const std::vector<std::uint8_t> pixels(1280 * 720 * 3, 0);
    const rd::ImageBgr8 image{pixels.data(), 1280, 720, 1280 * 3};
    const rd::PreprocessSpec spec = make_spec(640, 640);

    const LetterboxMeta meta = rd::compute_letterbox_meta(image, spec);
    EXPECT_FLOAT_EQ(meta.scale, 0.5f);
    EXPECT_EQ(meta.inputWidth, 640);
    EXPECT_EQ(meta.inputHeight, 640);
    EXPECT_EQ(meta.origWidth, 1280);
    EXPECT_EQ(meta.origHeight, 720);
    EXPECT_FLOAT_EQ(meta.padX, 0.0f);
    EXPECT_FLOAT_EQ(meta.padY, 140.0f);
}

TEST(Letterbox, PadsTopAndBottomWithNormalizedValue) {
    // 4x2 source -> 4x4 canvas: scale=1, 4x2 content, one padded row top/bottom.
    const std::vector<std::uint8_t> pixels = {
        10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120,
        130, 140, 150, 160, 170, 180, 190, 200, 210, 220, 230, 240,
    };
    const rd::ImageBgr8 image{pixels.data(), 4, 2, 12};
    const rd::PreprocessSpec spec = make_spec(4, 4);

    const LetterboxMeta meta = rd::compute_letterbox_meta(image, spec);
    EXPECT_FLOAT_EQ(meta.scale, 1.0f);
    EXPECT_FLOAT_EQ(meta.padY, 1.0f);
    EXPECT_FLOAT_EQ(meta.padX, 0.0f);

    const std::vector<float> out =
        rd::preprocess_bgr8_to_nchw_f32_letterbox_single(image, spec, meta, 114.0f);
    ASSERT_EQ(out.size(), 3u * 4u * 4u);

    const float pad = 114.0f * kInv255;
    const std::size_t plane = 16;
    for (int x = 0; x < 4; ++x) {
        EXPECT_NEAR(out[0 * plane + 0 * 4 + x], pad, 1e-6f);   // top R
        EXPECT_NEAR(out[0 * plane + 3 * 4 + x], pad, 1e-6f);   // bottom R
        EXPECT_NEAR(out[2 * plane + 0 * 4 + x], pad, 1e-6f);   // top B
    }

    // Content row 1 is source row 0; output R uses source channel 2.
    EXPECT_NEAR(out[0 * plane + 1 * 4 + 0], 30.0f * kInv255, 1e-6f);
    EXPECT_NEAR(out[1 * plane + 1 * 4 + 0], 20.0f * kInv255, 1e-6f);
    EXPECT_NEAR(out[2 * plane + 1 * 4 + 0], 10.0f * kInv255, 1e-6f);
}

TEST(Letterbox, PadsLeftAndRightWhenWidthIsLimiting) {
    // 2x4 source -> 4x4 canvas: scale=1, 2x4 content, one padded column each side.
    const std::vector<std::uint8_t> pixels = {
        0, 10, 20, 30, 40, 50,
        60, 70, 80, 90, 100, 110,
        120, 130, 140, 150, 160, 170,
        180, 190, 200, 210, 220, 230,
    };
    const rd::ImageBgr8 image{pixels.data(), 2, 4, 6};
    const rd::PreprocessSpec spec = make_spec(4, 4);
    const LetterboxMeta meta = rd::compute_letterbox_meta(image, spec);

    EXPECT_FLOAT_EQ(meta.scale, 1.0f);
    EXPECT_FLOAT_EQ(meta.padX, 1.0f);
    EXPECT_FLOAT_EQ(meta.padY, 0.0f);

    const std::vector<float> out =
        rd::preprocess_bgr8_to_nchw_f32_letterbox_single(image, spec, meta, 114.0f);
    const std::size_t plane = 16;
    const float pad = 114.0f * kInv255;
    for (int y = 0; y < 4; ++y) {
        EXPECT_NEAR(out[0 * plane + y * 4 + 0], pad, 1e-6f);
        EXPECT_NEAR(out[0 * plane + y * 4 + 3], pad, 1e-6f);
    }
    EXPECT_NEAR(out[0 * plane + 0 * 4 + 1], 20.0f * kInv255, 1e-6f);
}

TEST(Letterbox, RejectsMismatchedMetadata) {
    const std::vector<std::uint8_t> pixels(4 * 2 * 3, 0);
    const rd::ImageBgr8 image{pixels.data(), 4, 2, 12};
    const rd::PreprocessSpec spec = make_spec(4, 4);

    LetterboxMeta meta = rd::compute_letterbox_meta(image, spec);
    meta.origWidth = 3;
    EXPECT_THROW(rd::preprocess_bgr8_to_nchw_f32_letterbox_single(image, spec, meta, 114.0f),
                 std::invalid_argument);
}

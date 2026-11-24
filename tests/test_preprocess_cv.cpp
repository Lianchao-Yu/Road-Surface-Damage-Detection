#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <vector>

#include "road_damage/preprocess_cpu.hpp"
#include "road_damage/preprocess_cv.hpp"

namespace {

rd::PreprocessSpec make_spec(int width, int height) {
    rd::PreprocessSpec spec;
    spec.dst_width = width;
    spec.dst_height = height;
    spec.bgr_to_rgb = true;
    return spec;
}

struct TestImage {
    std::vector<std::uint8_t> pixels;
    rd::ImageBgr8 view;
};

TestImage make_image(int width, int height, int pad_bytes, std::uint8_t padding = 0xCD) {
    const int stride = width * 3 + pad_bytes;
    TestImage result;
    result.pixels.assign(static_cast<std::size_t>(stride) * height, padding);
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            for (int c = 0; c < 3; ++c) {
                result.pixels[static_cast<std::size_t>(y) * stride + x * 3 + c] =
                    static_cast<std::uint8_t>((y * 131 + x * 37 + c * 59 + 11) % 256);
            }
        }
    }
    result.view = rd::ImageBgr8{result.pixels.data(), width, height, stride};
    return result;
}

float max_abs_diff(const std::vector<float>& a, const std::vector<float>& b) {
    EXPECT_EQ(a.size(), b.size());
    float diff = 0.0f;
    for (std::size_t i = 0; i < a.size(); ++i) {
        diff = std::max(diff, std::fabs(a[i] - b[i]));
    }
    return diff;
}

struct SizeCase {
    int src_w;
    int src_h;
    int dst_w;
    int dst_h;
};

}  // namespace

TEST(PreprocessCv, FloatResizeMatchesHandwrittenReference) {
    const SizeCase cases[] = {
        {1, 1, 1, 1},   {3, 5, 7, 10},  {5, 4, 2, 3},
        {37, 29, 64, 48}, {64, 48, 17, 23}, {101, 67, 320, 320},
    };

    for (const SizeCase& c : cases) {
        TestImage image = make_image(c.src_w, c.src_h, 0);
        const rd::PreprocessSpec spec = make_spec(c.dst_w, c.dst_h);
        const LetterboxMeta meta = rd::compute_letterbox_meta(image.view, spec);

        const std::vector<float> reference =
            rd::preprocess_bgr8_to_nchw_f32_letterbox_single(image.view, spec, meta, 114.0f);
        const std::vector<float> cv = rd::preprocess_bgr8_to_nchw_f32_letterbox_cv_single(
            image.view, spec, meta, 114.0f, rd::CvInterpolationSource::kFloat32);

        EXPECT_LE(max_abs_diff(reference, cv), 1e-5f)
            << "case " << c.src_w << "x" << c.src_h << " -> " << c.dst_w << "x" << c.dst_h;
    }
}

TEST(PreprocessCv, FloatResizeHonorsSourceStride) {
    // Padding bytes are filled with garbage, so a kernel that ignores the
    // row stride would read the wrong pixels.
    TestImage image = make_image(37, 29, 19);
    const rd::PreprocessSpec spec = make_spec(64, 48);
    const LetterboxMeta meta = rd::compute_letterbox_meta(image.view, spec);

    const std::vector<float> reference =
        rd::preprocess_bgr8_to_nchw_f32_letterbox_single(image.view, spec, meta, 114.0f);
    const std::vector<float> cv = rd::preprocess_bgr8_to_nchw_f32_letterbox_cv_single(
        image.view, spec, meta, 114.0f, rd::CvInterpolationSource::kFloat32);

    EXPECT_LE(max_abs_diff(reference, cv), 1e-5f);
}

TEST(PreprocessCv, Uint8ResizeStaysWithinOneLsb) {
    const SizeCase cases[] = {
        {3, 5, 7, 10}, {5, 4, 2, 3}, {37, 29, 64, 48}, {101, 67, 160, 160},
    };

    // OpenCV's 8-bit INTER_LINEAR uses fixed-point arithmetic, so allow one
    // 8-bit quantization step (1/255) plus a small margin.
    constexpr float kTolerance = 1.0f / 255.0f + 1e-5f;
    for (const SizeCase& c : cases) {
        TestImage image = make_image(c.src_w, c.src_h, 0);
        const rd::PreprocessSpec spec = make_spec(c.dst_w, c.dst_h);
        const LetterboxMeta meta = rd::compute_letterbox_meta(image.view, spec);

        const std::vector<float> reference =
            rd::preprocess_bgr8_to_nchw_f32_letterbox_single(image.view, spec, meta, 114.0f);
        const std::vector<float> cv = rd::preprocess_bgr8_to_nchw_f32_letterbox_cv_single(
            image.view, spec, meta, 114.0f, rd::CvInterpolationSource::kUint8);

        EXPECT_LE(max_abs_diff(reference, cv), kTolerance)
            << "case " << c.src_w << "x" << c.src_h << " -> " << c.dst_w << "x" << c.dst_h;
    }
}

TEST(PreprocessCv, AppliesPaddingAndPerChannelNormalization) {
    // 2x1 image placed in the center of a 4x4 letterbox canvas:
    // resized 4x2, so padY = 1 and padX = 0.
    const std::vector<std::uint8_t> pixels = {30, 20, 10, 30, 20, 10};  // B, G, R
    const rd::ImageBgr8 image{pixels.data(), 2, 1, 6};
    rd::PreprocessSpec spec = make_spec(4, 4);
    spec.norm.mean[0] = 0.1f;
    spec.norm.mean[1] = 0.2f;
    spec.norm.mean[2] = 0.3f;
    spec.norm.std[0] = 0.5f;
    spec.norm.std[1] = 0.5f;
    spec.norm.std[2] = 0.5f;
    const LetterboxMeta meta = rd::compute_letterbox_meta(image, spec);
    ASSERT_EQ(meta.padX, 0.0f);
    ASSERT_EQ(meta.padY, 1.0f);

    const std::vector<float> reference =
        rd::preprocess_bgr8_to_nchw_f32_letterbox_single(image, spec, meta, 114.0f);
    const std::vector<float> cv = rd::preprocess_bgr8_to_nchw_f32_letterbox_cv_single(
        image, spec, meta, 114.0f, rd::CvInterpolationSource::kFloat32);

    EXPECT_LE(max_abs_diff(reference, cv), 1e-5f);

    // First content row is y=1; pixel (0,1) is RGB (10, 20, 30) normalized
    // with the per-channel params.
    const std::size_t content = static_cast<std::size_t>(1) * 4u;
    const float pad_norm = 114.0f / 255.0f;
    EXPECT_NEAR(cv[content], (10.0f / 255.0f - 0.1f) / 0.5f, 1e-5f);
    EXPECT_NEAR(cv[16u + content], (20.0f / 255.0f - 0.2f) / 0.5f, 1e-5f);
    EXPECT_NEAR(cv[32u + content], (30.0f / 255.0f - 0.3f) / 0.5f, 1e-5f);
    EXPECT_NEAR(cv[0], (pad_norm - 0.1f) / 0.5f, 1e-5f);
}

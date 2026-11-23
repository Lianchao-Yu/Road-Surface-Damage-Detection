#include <gtest/gtest.h>

#include <cmath>
#include <string>
#include <unordered_map>
#include <vector>

#include "road_damage/postprocess_cpu.hpp"

namespace {

constexpr int kClasses = 4;
constexpr int kChannels = 4 + kClasses;

trt::TensorView make_view(std::vector<float>& data, int anchors) {
    trt::TensorView view;
    view.data = data.data();
    view.dims.nbDims = 3;
    view.dims.d[0] = 1;
    view.dims.d[1] = kChannels;
    view.dims.d[2] = anchors;
    view.dtype = nvinfer1::DataType::kFLOAT;
    view.memType = trt::MemoryType::kCPU;
    return view;
}

std::unordered_map<std::string, trt::TensorView> make_output(
    std::vector<float>& data, int anchors) {
    std::unordered_map<std::string, trt::TensorView> output;
    output.emplace("output0", make_view(data, anchors));
    return output;
}

void set_box(std::vector<float>& data, int anchors, int anchor, float cx, float cy,
             float w, float h, int cls, float score) {
    data[0 * anchors + anchor] = cx;
    data[1 * anchors + anchor] = cy;
    data[2 * anchors + anchor] = w;
    data[3 * anchors + anchor] = h;
    data[(4 + cls) * anchors + anchor] = score;
}

LetterboxMeta make_letterbox(int orig_w, int orig_h, float scale, float pad_x, float pad_y) {
    LetterboxMeta meta;
    meta.origWidth = orig_w;
    meta.origHeight = orig_h;
    meta.inputWidth = 640;
    meta.inputHeight = 640;
    meta.scale = scale;
    meta.padX = pad_x;
    meta.padY = pad_y;
    return meta;
}

}  // namespace

TEST(PostprocessYolo, FiltersAndSuppressesSameClass) {
    constexpr int kAnchors = 4;
    std::vector<float> data(static_cast<std::size_t>(kChannels) * kAnchors, 0.0f);
    set_box(data, kAnchors, 0, 100.0f, 100.0f, 20.0f, 20.0f, 0, 0.90f);
    set_box(data, kAnchors, 1, 100.0f, 100.0f, 20.0f, 20.0f, 0, 0.80f);  // duplicate
    set_box(data, kAnchors, 2, 300.0f, 300.0f, 20.0f, 20.0f, 1, 0.70f);
    set_box(data, kAnchors, 3, 500.0f, 500.0f, 20.0f, 20.0f, 2, 0.10f);  // low score

    const auto output = make_output(data, kAnchors);
    const LetterboxMeta meta = make_letterbox(640, 640, 1.0f, 0.0f, 0.0f);
    const std::vector<Detection> result =
        postprocessing::PostProcessing_yolo(output, meta, 0.25f, 0.45f);

    ASSERT_EQ(result.size(), 2u);
    EXPECT_EQ(result[0].classID, 0);
    EXPECT_FLOAT_EQ(result[0].score, 0.90f);
    EXPECT_FLOAT_EQ(result[0].x1, 90.0f);
    EXPECT_FLOAT_EQ(result[0].y1, 90.0f);
    EXPECT_FLOAT_EQ(result[0].x2, 110.0f);
    EXPECT_FLOAT_EQ(result[0].y2, 110.0f);

    EXPECT_EQ(result[1].classID, 1);
    EXPECT_FLOAT_EQ(result[1].score, 0.70f);
}

TEST(PostprocessYolo, RemovesLetterboxPaddingBeforeScaling) {
    constexpr int kAnchors = 1;
    std::vector<float> data(static_cast<std::size_t>(kChannels) * kAnchors, 0.0f);
    // Box centered in the 640 canvas. Source is 600x300, so scale=640/600,
    // content height=320 and padY=160.
    set_box(data, kAnchors, 0, 320.0f, 320.0f, 64.0f, 64.0f, 0, 0.90f);

    const auto output = make_output(data, kAnchors);
    const LetterboxMeta meta =
        make_letterbox(600, 300, 640.0f / 600.0f, 0.0f, 160.0f);
    const std::vector<Detection> result =
        postprocessing::PostProcessing_yolo(output, meta, 0.25f, 0.45f);

    ASSERT_EQ(result.size(), 1u);
    EXPECT_NEAR(result[0].x1, 270.0f, 1e-4f);
    EXPECT_NEAR(result[0].y1, 120.0f, 1e-4f);
    EXPECT_NEAR(result[0].x2, 330.0f, 1e-4f);
    EXPECT_NEAR(result[0].y2, 180.0f, 1e-4f);
}

TEST(PostprocessYolo, ReturnsEmptyForEmptyOrInvalidOutput) {
    const LetterboxMeta meta = make_letterbox(640, 640, 1.0f, 0.0f, 0.0f);
    EXPECT_TRUE(postprocessing::PostProcessing_yolo({}, meta).empty());

    std::vector<float> data(static_cast<std::size_t>(kChannels), 0.0f);
    auto output = make_output(data, 1);
    output["output0"].memType = trt::MemoryType::kGPU;
    EXPECT_TRUE(postprocessing::PostProcessing_yolo(output, meta).empty());
}

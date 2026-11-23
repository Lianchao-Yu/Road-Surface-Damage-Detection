#include <gtest/gtest.h>
#include <opencv2/opencv.hpp>

#include <cstdint>
#include <vector>

#include "road_damage/preprocess_cpu.hpp"

namespace {

cv::Mat make_bgr(int width, int height, int seed) {
    cv::Mat image(height, width, CV_8UC3);
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            auto& pixel = image.at<cv::Vec3b>(y, x);
            for (int c = 0; c < 3; ++c) {
                pixel[c] = static_cast<std::uint8_t>((x * 7 + y * 13 + c * 31 + seed * 17) % 256);
            }
        }
    }
    return image;
}

struct ResizeCase {
    int width;
    int height;
    int dst_width;
    int dst_height;
};

}  // namespace

TEST(PreprocessCpuOpenCv, MatchesCvResizeAndColorConversion) {
    const std::vector<ResizeCase> cases = {
        {1, 1, 1, 1},     {3, 5, 7, 10},    {5, 4, 2, 3},
        {37, 29, 64, 48}, {64, 48, 17, 23}, {13, 17, 13, 17},
    };

    rd::PreprocessSpec spec;
    spec.bgr_to_rgb = true;
    spec.norm.mean[0] = 0.1f;
    spec.norm.mean[1] = 0.2f;
    spec.norm.mean[2] = 0.3f;
    spec.norm.std[0] = 0.5f;
    spec.norm.std[1] = 0.8f;
    spec.norm.std[2] = 1.1f;

    for (const ResizeCase& test : cases) {
        const cv::Mat bgr = make_bgr(test.width, test.height, test.width + test.height);

        cv::Mat bgr_float;
        bgr.convertTo(bgr_float, CV_32FC3, 1.0 / 255.0);

        cv::Mat resized;
        cv::resize(bgr_float, resized, cv::Size(test.dst_width, test.dst_height), 0.0, 0.0,
                   cv::INTER_LINEAR);

        cv::Mat rgb;
        cv::cvtColor(resized, rgb, cv::COLOR_BGR2RGB);

        spec.dst_width = test.dst_width;
        spec.dst_height = test.dst_height;
        const rd::ImageBgr8 view{bgr.data, bgr.cols, bgr.rows, static_cast<int>(bgr.step)};
        const std::vector<float> actual = rd::preprocess_bgr8_to_nchw_f32_single(view, spec);
        const std::size_t plane =
            static_cast<std::size_t>(test.dst_width) * static_cast<std::size_t>(test.dst_height);

        ASSERT_EQ(actual.size(), 3u * plane);
        for (int y = 0; y < test.dst_height; ++y) {
            for (int x = 0; x < test.dst_width; ++x) {
                const cv::Vec3f& expected = rgb.at<cv::Vec3f>(y, x);
                const std::size_t index =
                    static_cast<std::size_t>(y) * static_cast<std::size_t>(test.dst_width) +
                    static_cast<std::size_t>(x);
                for (int c = 0; c < 3; ++c) {
                    const float normalized = (expected[c] - spec.norm.mean[c]) / spec.norm.std[c];
                    EXPECT_NEAR(actual[static_cast<std::size_t>(c) * plane + index], normalized,
                                1e-5f)
                        << "case " << test.width << "x" << test.height << " -> "
                        << test.dst_width << "x" << test.dst_height << " at (" << x << "," << y
                        << ") channel " << c;
                }
            }
        }
    }
}

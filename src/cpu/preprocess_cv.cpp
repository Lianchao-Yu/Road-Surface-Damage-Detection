#include "road_damage/preprocess_cv.hpp"

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <string>

namespace rd {
namespace {

constexpr float kInv255 = 1.0f / 255.0f;

// Keep the same validation rules as the hand-written CPU reference so both
// implementations reject exactly the same malformed requests.
void validate_request(const ImageBgr8& src, const PreprocessSpec& spec,
                      const LetterboxMeta& meta, const float* dst,
                      std::size_t dst_size) {
    if (src.data == nullptr) {
        throw std::invalid_argument("preprocess_cv: src.data is null");
    }
    if (src.width <= 0 || src.height <= 0) {
        throw std::invalid_argument("preprocess_cv: source dimensions must be positive");
    }
    if (src.stride < src.width * 3) {
        throw std::invalid_argument("preprocess_cv: stride must be at least width * 3");
    }
    if (spec.dst_width <= 0 || spec.dst_height <= 0) {
        throw std::invalid_argument("preprocess_cv: destination dimensions must be positive");
    }
    for (int c = 0; c < 3; ++c) {
        if (!(spec.norm.std[c] > 0.0f)) {
            throw std::invalid_argument("preprocess_cv: std must be positive (channel " +
                                        std::to_string(c) + ")");
        }
    }
    if (dst == nullptr) {
        throw std::invalid_argument("preprocess_cv: dst pointer is null");
    }
    if (dst_size < tensor_element_count(spec)) {
        throw std::invalid_argument(
            "preprocess_cv: dst buffer is smaller than required tensor elements");
    }
    if (meta.origWidth != src.width || meta.origHeight != src.height ||
        meta.inputWidth != spec.dst_width || meta.inputHeight != spec.dst_height ||
        !(meta.scale > 0.0f)) {
        throw std::invalid_argument(
            "preprocess_cv: letterbox metadata does not match input specs");
    }
}

}  // namespace

void preprocess_bgr8_to_nchw_f32_letterbox_cv(
    const ImageBgr8& src, const PreprocessSpec& spec, const LetterboxMeta& meta,
    float pad_value, float* dst, std::size_t dst_size,
    CvInterpolationSource source) {
    validate_request(src, spec, meta, dst, dst_size);

    const int dst_w = spec.dst_width;
    const int dst_h = spec.dst_height;
    const int resized_w =
        std::max(1, static_cast<int>(std::lround(static_cast<float>(src.width) * meta.scale)));
    const int resized_h =
        std::max(1, static_cast<int>(std::lround(static_cast<float>(src.height) * meta.scale)));
    const int pad_x = static_cast<int>(meta.padX);
    const int pad_y = static_cast<int>(meta.padY);
    if (resized_w > dst_w || resized_h > dst_h || pad_x < 0 || pad_y < 0) {
        throw std::invalid_argument("preprocess_cv: invalid letterbox geometry");
    }

    // Wrap the caller's BGR buffer without copying. The row step is taken from
    // ImageBgr8::stride, so FFmpeg linesize / OpenCV ROI padding works here
    // exactly like it does in the hand-written reference and the CUDA kernel.
    cv::Mat bgr(src.height, src.width, CV_8UC3, const_cast<std::uint8_t*>(src.data),
                static_cast<std::size_t>(src.stride));

    // Resize first, then select the output channel order. A BGR->RGB swap is a
    // pure channel permutation, so reading channel (2 - c) from the BGR result
    // is equivalent to cv::cvtColor(..., COLOR_BGR2RGB) but saves one pass
    // over the resized image.
    std::vector<cv::Mat> channels;
    bool channels_are_float01 = false;
    if (source == CvInterpolationSource::kFloat32) {
        // Match the reference contract: interpolate in FP32. OpenCV's 8-bit
        // resize uses fixed-point arithmetic, which would add up to ~1 LSB of
        // error before normalization.
        cv::Mat bgr_f32;
        bgr.convertTo(bgr_f32, CV_32FC3, static_cast<double>(kInv255));
        cv::Mat resized_f32;
        cv::resize(bgr_f32, resized_f32, cv::Size(resized_w, resized_h), 0.0, 0.0,
                   cv::INTER_LINEAR);
        cv::split(resized_f32, channels);
        channels_are_float01 = true;
    } else {
        // Fastest path: let OpenCV resize the original 8-bit image.
        cv::Mat resized_u8;
        cv::resize(bgr, resized_u8, cv::Size(resized_w, resized_h), 0.0, 0.0,
                   cv::INTER_LINEAR);
        cv::split(resized_u8, channels);
    }

    const std::size_t plane =
        static_cast<std::size_t>(dst_h) * static_cast<std::size_t>(dst_w);
    const float *mean = spec.norm.mean;
    const float *stddev = spec.norm.std;
    const float pad_normalized = pad_value * kInv255;

    for (int c = 0; c < 3; ++c) {
        // Write directly into the NCHW destination plane. setTo() fills the
        // letterbox border; convertTo() then writes the resized content into
        // the centered ROI.
        cv::Mat plane_view(dst_h, dst_w, CV_32FC1,
                           dst + static_cast<std::size_t>(c) * plane,
                           static_cast<std::size_t>(dst_w) * sizeof(float));
        plane_view.setTo((pad_normalized - mean[c]) / stddev[c]);

        const int src_channel = spec.bgr_to_rgb ? (2 - c) : c;
        cv::Mat roi = plane_view(cv::Rect(pad_x, pad_y, resized_w, resized_h));
        if (channels_are_float01) {
            // Channel is already in [0, 1]: (v - mean) / std.
            channels[static_cast<std::size_t>(src_channel)].convertTo(
                roi, CV_32FC1, static_cast<double>(1.0f / stddev[c]),
                static_cast<double>(-mean[c] / stddev[c]));
        } else {
            // Channel is still 0..255: v * 1/(255*std) - mean/std.
            channels[static_cast<std::size_t>(src_channel)].convertTo(
                roi, CV_32FC1, static_cast<double>(kInv255 / stddev[c]),
                static_cast<double>(-mean[c] / stddev[c]));
        }
    }
}

std::vector<float> preprocess_bgr8_to_nchw_f32_letterbox_cv_single(
    const ImageBgr8& src, const PreprocessSpec& spec, const LetterboxMeta& meta,
    float pad_value, CvInterpolationSource source) {
    std::vector<float> output(tensor_element_count(spec), 0.0f);
    preprocess_bgr8_to_nchw_f32_letterbox_cv(src, spec, meta, pad_value, output.data(),
                                             output.size(), source);
    return output;
}

}  // namespace rd

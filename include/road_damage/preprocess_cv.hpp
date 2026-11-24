#pragma once

#include <cstddef>
#include <vector>

#include "road_damage/preprocess_cpu.hpp"

namespace rd {

/// Interpolation source used by the OpenCV letterbox reference.
///
/// kFloat32: convert the BGR8 image to FP32 first and call cv::resize with
///           CV_32F. This keeps the same pixel contract as the hand-written
///           CPU reference and the CUDA kernel (half-pixel bilinear).
/// kUint8:   call cv::resize directly on CV_8UC3. OpenCV uses a fixed-point
///           implementation for 8-bit images, which is faster but introduces
///           up to ~1 LSB of interpolation error before normalization.
enum class CvInterpolationSource {
    kFloat32,
    kUint8,
};

// ---------------------------------------------------------------------------
// OpenCV-based letterbox preprocessing (YOLO contract).
//
// This is an independent CPU implementation used for comparison against the
// hand-written reference in preprocess_cpu.cpp and the CUDA K1 kernel. It
// produces the same output tensor layout:
//
//   BGR8 (+ optional stride)
//   -> aspect-preserving resize
//   -> center padding with pad_value
//   -> BGR/RGB channel selection
//   -> per-channel (v / 255 - mean) / std
//   -> NCHW FP32 [1, 3, dst_height, dst_width]
//
// The caller must pass the same LetterboxMeta that was produced by
// compute_letterbox_meta(); it is used for resized dimensions and padding.
// ---------------------------------------------------------------------------
void preprocess_bgr8_to_nchw_f32_letterbox_cv(
    const ImageBgr8& src, const PreprocessSpec& spec, const LetterboxMeta& meta,
    float pad_value, float* dst, std::size_t dst_size,
    CvInterpolationSource source = CvInterpolationSource::kFloat32);

std::vector<float> preprocess_bgr8_to_nchw_f32_letterbox_cv_single(
    const ImageBgr8& src, const PreprocessSpec& spec, const LetterboxMeta& meta,
    float pad_value = 114.0f,
    CvInterpolationSource source = CvInterpolationSource::kFloat32);

}  // namespace rd

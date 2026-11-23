#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

#include "road_damage/shared_data.hpp"

namespace rd {

/// BGR8 host image. `stride` is the number of bytes between rows and may be
/// larger than width * 3 (FFmpeg linesize). Pixels are 3 bytes: B, G, R.
struct ImageBgr8 {
    const std::uint8_t* data = nullptr;
    int width = 0;
    int height = 0;
    int stride = 0;
};

/// Per-channel normalization. `mean` and `std` are expressed in [0, 1] units
/// and are applied after dividing the 8-bit pixel by 255.
struct Normalization {
    float mean[3] = {0.0f, 0.0f, 0.0f};
    float std[3] = {1.0f, 1.0f, 1.0f};
};


struct PreprocessSpec {
    int dst_width = 640;
    int dst_height = 640;
    bool bgr_to_rgb = true;
    Normalization norm{};
};

/// Number of float elements produced by `spec` (3 * H * W).
std::size_t tensor_element_count(const PreprocessSpec& spec);

/// Compute the letterbox scale and padding for `src` -> `spec` destination.
/// The image is resized to (new_w, new_h) with scale = min(dst/src), then
/// centered on a dst canvas with `pad_value` pixels.
LetterboxMeta compute_letterbox_meta(const ImageBgr8& src, const PreprocessSpec& spec);

/// Letterbox preprocessing: resize with preserved aspect ratio, pad the
/// remaining area with `pad_value` (default 114, matching Ultralytics), then
/// apply BGR->RGB, normalization and NCHW layout. `meta` must be the result
/// of compute_letterbox_meta for the same src/spec.
void preprocess_bgr8_to_nchw_f32_letterbox(const ImageBgr8& src,
                                           const PreprocessSpec& spec,
                                           const LetterboxMeta& meta,
                                           float pad_value,
                                           float* dst,
                                           std::size_t dst_size);

/// Convenience wrapper for a single letterboxed image.
std::vector<float> preprocess_bgr8_to_nchw_f32_letterbox_single(
    const ImageBgr8& src, const PreprocessSpec& spec,
    const LetterboxMeta& meta, float pad_value = 114.0f);

/// Run the reference preprocessing pipeline.
/// Throws std::invalid_argument on malformed input (null data, non-positive
/// dimensions, stride smaller than width * 3, zero std).
std::vector<float> preprocess_bgr8_to_nchw_f32_single(const ImageBgr8& src, const PreprocessSpec& spec);
void preprocess_bgr8_to_nchw_f32(const ImageBgr8& src, const PreprocessSpec& spec, float* dst,  std::size_t dst_size);

}  // namespace rd

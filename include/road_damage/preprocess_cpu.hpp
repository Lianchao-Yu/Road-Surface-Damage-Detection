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

// ---------------------------------------------------------------------------
// Direct resize path: RT-DETRv2 / models trained with a plain 640x640 resize.
// No padding is added; the image is stretched to spec.dst_*.
// ---------------------------------------------------------------------------
std::vector<float> preprocess_bgr8_to_nchw_f32_direct_single(const ImageBgr8& src,
                                                             const PreprocessSpec& spec);
void preprocess_bgr8_to_nchw_f32_direct(const ImageBgr8& src, const PreprocessSpec& spec,
                                        float* dst, std::size_t dst_size);

// ---------------------------------------------------------------------------
// Letterbox path: YOLO / models trained with aspect-preserving resize + padding.
// Call order:
//   1. meta = compute_letterbox_meta(src, spec)
//   2. tensor = preprocess_bgr8_to_nchw_f32_letterbox_single(src, spec, meta, 114.0f)
//   3. run inference
//   4. PostProcessing_yolo(outputs, meta, ...)   // uses the same meta
// ---------------------------------------------------------------------------
LetterboxMeta compute_letterbox_meta(const ImageBgr8& src, const PreprocessSpec& spec);

void preprocess_bgr8_to_nchw_f32_letterbox(const ImageBgr8& src,
                                           const PreprocessSpec& spec,
                                           const LetterboxMeta& meta,
                                           float pad_value,
                                           float* dst,
                                           std::size_t dst_size);

std::vector<float> preprocess_bgr8_to_nchw_f32_letterbox_single(
    const ImageBgr8& src, const PreprocessSpec& spec,
    const LetterboxMeta& meta, float pad_value = 114.0f);

}  // namespace rd

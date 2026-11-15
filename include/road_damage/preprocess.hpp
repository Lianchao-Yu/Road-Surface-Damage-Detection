#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

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

/// Reference preprocessing contract (CPU implementation is the golden path).
///
/// 1. Bilinear resize with align_corners = false (half-pixel mapping):
///        scale_x = src_width  / dst_width
///        scale_y = src_height / dst_height
///        sx = (dst_x + 0.5f) * scale_x - 0.5f
///        sy = (dst_y + 0.5f) * scale_y - 0.5f
///    Source coordinates are clamped to [0, src-1] on both axes, i.e.
///    border replicate. This matches PyTorch `interpolate(..., mode="bilinear",
///    align_corners=False)` and ONNX Resize `coordinate_transformation_mode=
///    "half_pixel"` with edge clamping.
///
/// 2. Channel order: BGR -> RGB (pure channel swap, no color-space conversion)
///    when `bgr_to_rgb` is true.
///
/// 3. Normalization: v = (pixel / 255 - mean[c]) / std[c], with `c` the output
///    (RGB) channel index.
///
/// 4. Layout: NCHW float32 with N = 1, index = c * H * W + y * W + x.
///
/// ROI/tiling, letterbox padding and FP16 conversion are intentionally out of
/// scope for the reference path; they are added by the CUDA preprocessing
/// stage and must be validated against this function for the overlapping
/// operations.
struct PreprocessSpec {
    int dst_width = 640;
    int dst_height = 640;
    bool bgr_to_rgb = true;
    Normalization norm{};
};

/// Number of float elements produced by `spec` (3 * H * W).
std::size_t tensor_element_count(const PreprocessSpec& spec);

/// Run the reference preprocessing pipeline.
/// Throws std::invalid_argument on malformed input (null data, non-positive
/// dimensions, stride smaller than width * 3, zero std).
std::vector<float> preprocess_bgr8_to_nchw_f32(const ImageBgr8& src, const PreprocessSpec& spec);

}  // namespace rd

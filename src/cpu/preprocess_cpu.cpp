#include "road_damage/preprocess.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <string>
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
namespace rd
{
    namespace
    {

        constexpr float kInv255 = 1.0f / 255.0f;

        inline float sample(const ImageBgr8 &image, int x, int y, int channel)
        {
            const std::size_t offset =
                static_cast<std::size_t>(y) * static_cast<std::size_t>(image.stride) +
                static_cast<std::size_t>(x) * 3U + static_cast<std::size_t>(channel);
            return static_cast<float>(image.data[offset]) * kInv255;
        }

        inline float clamp_coordinate(float value, int limit)
        {
            return std::min(std::max(value, 0.0f), static_cast<float>(limit - 1));
        }

        void validate(const ImageBgr8 &src, const PreprocessSpec &spec)
        {
            // Validate input parameters
            if (src.data == nullptr)
            {
                throw std::invalid_argument("preprocess: src.data is null");
            }

            // Validate dimensions and stride
            if (src.width <= 0 || src.height <= 0)
            {
                throw std::invalid_argument("preprocess: source dimensions must be positive");
            }

            // Validate stride
            if (src.stride < src.width * 3)
            {
                throw std::invalid_argument("preprocess: stride must be at least width * 3");
            }

            // Validate destination dimensions and normalization parameters
            if (spec.dst_width <= 0 || spec.dst_height <= 0)
            {
                throw std::invalid_argument("preprocess: destination dimensions must be positive");
            }

            // Validate normalization parameters
            for (int c = 0; c < 3; ++c)
            {
                if (!(spec.norm.std[c] > 0.0f))
                {
                    throw std::invalid_argument("preprocess: std must be positive (channel " +
                                                std::to_string(c) + ")");
                }
            }
        }

    } // namespace

    std::size_t tensor_element_count(const PreprocessSpec &spec)
    {
        if (spec.dst_width <= 0 || spec.dst_height <= 0)
        {
            return 0;
        }
        return static_cast<std::size_t>(3) * static_cast<std::size_t>(spec.dst_height) *
               static_cast<std::size_t>(spec.dst_width);
    }

    void preprocess_bgr8_to_nchw_f32(const ImageBgr8 &src, const PreprocessSpec &spec, float *dst, 
                                                                            std::size_t dst_size)
    {
        validate(src, spec);
        if (dst == nullptr)
        {
            throw std::invalid_argument("preprocess: dst pointer is null");
        }

        const std::size_t required = tensor_element_count(spec);
        if (dst_size < required)
        {
            throw std::invalid_argument("preprocess: dst buffer is smaller than required tensor elements");
        }
        const int dst_w = spec.dst_width;
        const int dst_h = spec.dst_height;
        const float scale_x = static_cast<float>(src.width) / static_cast<float>(dst_w);
        const float scale_y = static_cast<float>(src.height) / static_cast<float>(dst_h);
        const float *mean = spec.norm.mean;
        const float *stddev = spec.norm.std;

        // std::vector<float> output(tensor_element_count(spec), 0.0f);
        const std::size_t plane = static_cast<std::size_t>(dst_h) * static_cast<std::size_t>(dst_w);

        for (int dy = 0; dy < dst_h; ++dy)
        {
            const float sy = clamp_coordinate((static_cast<float>(dy) + 0.5f) * scale_y - 0.5f,
                                              src.height);
            const int y0 = static_cast<int>(std::floor(sy)); // down pixel index
            const int y1 = std::min(y0 + 1, src.height - 1); // up pixel index
            const float fy = sy - static_cast<float>(y0);    // Distance from down pixel

            for (int dx = 0; dx < dst_w; ++dx)
            {
                const float sx = clamp_coordinate((static_cast<float>(dx) + 0.5f) * scale_x - 0.5f,
                                                  src.width);
                const int x0 = static_cast<int>(std::floor(sx)); // Left pixel index
                const int x1 = std::min(x0 + 1, src.width - 1);  // Right pixel index
                const float fx = sx - static_cast<float>(x0);    // Distance from left pixel

                const float w00 = (1.0f - fx) * (1.0f - fy); // Weight for top-left pixel
                const float w10 = fx * (1.0f - fy);          // Weight for top-right pixel
                const float w01 = (1.0f - fx) * fy;          // Weight for bottom-left pixel
                const float w11 = fx * fy;                   // Weight for bottom-right pixel

                const std::size_t dst_index = static_cast<std::size_t>(dy) * static_cast<std::size_t>(dst_w) +
                                              static_cast<std::size_t>(dx); // Convert Index to 1D output array

                for (int c = 0; c < 3; ++c)
                {
                    const int src_c = spec.bgr_to_rgb ? (2 - c) : c;       // Convert channel index if BGR to RGB conversion is needed
                    const float value = w00 * sample(src, x0, y0, src_c) + // Bilinear interpolation
                                        w10 * sample(src, x1, y0, src_c) +
                                        w01 * sample(src, x0, y1, src_c) +
                                        w11 * sample(src, x1, y1, src_c);
                    // output[static_cast<std::size_t>(c) * plane + dst_index] = //offset for channel c
                    //     (value - mean[c]) / stddev[c];//Z-score normalization

                    // ======== write the result into dst pointer ===============
                    dst[static_cast<std::size_t>(c) * plane + dst_index] =
                        (value - mean[c]) / stddev[c]; // Z-score normalization
                }
            }
        }
    }

    // Single batch, single input. 
    std::vector<float> preprocess_bgr8_to_nchw_f32_single(const ImageBgr8 &src, const PreprocessSpec &spec)
    {
        size_t tensor_size = tensor_element_count(spec);
        std::vector<float> output(tensor_size, 0.0f);
        preprocess_bgr8_to_nchw_f32(src, spec, output.data(), tensor_size);
        return output;
    }

} // namespace rd

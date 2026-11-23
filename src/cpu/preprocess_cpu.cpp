#include "road_damage/preprocess_cpu.hpp"

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
/// ROI/tiling and FP16 conversion are intentionally out of scope for the
/// reference path; letterbox padding is provided by the separate
/// preprocess_bgr8_to_nchw_f32_letterbox() entry point and must be validated
/// against the direct-resize path for the overlapping operations.
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

    LetterboxMeta compute_letterbox_meta(const ImageBgr8 &src, const PreprocessSpec &spec)
    {
        validate(src, spec);

        LetterboxMeta meta;
        meta.origWidth = src.width;
        meta.origHeight = src.height;
        meta.inputWidth = spec.dst_width;
        meta.inputHeight = spec.dst_height;

        // Keep the aspect ratio: the image is first scaled to fit inside the
        // destination canvas, then the remaining area is padded.
        meta.scale = std::min(static_cast<float>(spec.dst_width) / static_cast<float>(src.width),
                              static_cast<float>(spec.dst_height) / static_cast<float>(src.height));

        const int resized_w = std::max(1, static_cast<int>(std::round(src.width * meta.scale)));
        const int resized_h = std::max(1, static_cast<int>(std::round(src.height * meta.scale)));

        // Ultralytics centers the image. Integer padding keeps the tensor
        // dimensions and the inverse mapping deterministic.
        meta.padX = static_cast<float>((spec.dst_width - resized_w) / 2);
        meta.padY = static_cast<float>((spec.dst_height - resized_h) / 2);
        return meta;
    }

    void preprocess_bgr8_to_nchw_f32_letterbox(const ImageBgr8 &src, const PreprocessSpec &spec,
                                               const LetterboxMeta &meta, float pad_value,
                                               float *dst, std::size_t dst_size)
    {
        validate(src, spec);
        if (dst == nullptr)
        {
            throw std::invalid_argument("preprocess: dst pointer is null");
        }

        // The metadata must describe this exact image/spec pair; otherwise
        // the padding and the postprocess inverse mapping would be wrong.
        if (meta.origWidth != src.width || meta.origHeight != src.height ||
            meta.inputWidth != spec.dst_width || meta.inputHeight != spec.dst_height ||
            !(meta.scale > 0.0f))
        {
            throw std::invalid_argument("preprocess: letterbox metadata does not match input specs");
        }

        const std::size_t required = tensor_element_count(spec);
        if (dst_size < required)
        {
            throw std::invalid_argument("preprocess: dst buffer is smaller than required tensor elements");
        }

        const int dst_w = spec.dst_width;
        const int dst_h = spec.dst_height;
        const int resized_w = std::max(1, static_cast<int>(std::round(src.width * meta.scale)));
        const int resized_h = std::max(1, static_cast<int>(std::round(src.height * meta.scale)));
        const int pad_x = static_cast<int>(meta.padX);
        const int pad_y = static_cast<int>(meta.padY);
        if (resized_w > dst_w || resized_h > dst_h || pad_x < 0 || pad_y < 0)
        {
            throw std::invalid_argument("preprocess: invalid letterbox geometry");
        }

        const std::size_t plane = static_cast<std::size_t>(dst_h) * static_cast<std::size_t>(dst_w);
        const float pad_normalized = pad_value * kInv255;
        const float *mean = spec.norm.mean;
        const float *stddev = spec.norm.std;

        // Start with a canvas filled by the normalized padding value. Only the
        // centered resized region is overwritten below.
        for (int c = 0; c < 3; ++c)
        {
            const float value = (pad_normalized - mean[c]) / stddev[c];
            std::fill_n(dst + static_cast<std::size_t>(c) * plane, plane, value);
        }

        // Bilinear mapping is computed against the resized (not padded) image,
        // matching cv2.resize(src, (resized_w, resized_h)) followed by padding.
        const float scale_x = static_cast<float>(src.width) / static_cast<float>(resized_w);
        const float scale_y = static_cast<float>(src.height) / static_cast<float>(resized_h);

        for (int dy = 0; dy < resized_h; ++dy)
        {
            const float sy = clamp_coordinate((static_cast<float>(dy) + 0.5f) * scale_y - 0.5f,
                                              src.height);
            const int y0 = static_cast<int>(std::floor(sy));
            const int y1 = std::min(y0 + 1, src.height - 1);
            const float fy = sy - static_cast<float>(y0);

            for (int dx = 0; dx < resized_w; ++dx)
            {
                const float sx = clamp_coordinate((static_cast<float>(dx) + 0.5f) * scale_x - 0.5f,
                                                  src.width);
                const int x0 = static_cast<int>(std::floor(sx));
                const int x1 = std::min(x0 + 1, src.width - 1);
                const float fx = sx - static_cast<float>(x0);

                const float w00 = (1.0f - fx) * (1.0f - fy);
                const float w10 = fx * (1.0f - fy);
                const float w01 = (1.0f - fx) * fy;
                const float w11 = fx * fy;

                // Destination coordinates include the letterbox offset.
                const std::size_t dst_index =
                    static_cast<std::size_t>(dy + pad_y) * static_cast<std::size_t>(dst_w) +
                    static_cast<std::size_t>(dx + pad_x);

                for (int c = 0; c < 3; ++c)
                {
                    const int src_c = spec.bgr_to_rgb ? (2 - c) : c;
                    const float value = w00 * sample(src, x0, y0, src_c) +
                                        w10 * sample(src, x1, y0, src_c) +
                                        w01 * sample(src, x0, y1, src_c) +
                                        w11 * sample(src, x1, y1, src_c);
                    dst[static_cast<std::size_t>(c) * plane + dst_index] =
                        (value - mean[c]) / stddev[c];
                }
            }
        }
    }

    std::vector<float> preprocess_bgr8_to_nchw_f32_letterbox_single(
        const ImageBgr8 &src, const PreprocessSpec &spec, const LetterboxMeta &meta,
        float pad_value)
    {
        std::vector<float> output(tensor_element_count(spec), 0.0f);
        preprocess_bgr8_to_nchw_f32_letterbox(src, spec, meta, pad_value, output.data(),
                                              output.size());
        return output;
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

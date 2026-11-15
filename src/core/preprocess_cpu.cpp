#include "road_damage/preprocess.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <string>

namespace rd {
namespace {

constexpr float kInv255 = 1.0f / 255.0f;

inline float sample(const ImageBgr8& image, int x, int y, int channel) {
    const std::size_t offset =
        static_cast<std::size_t>(y) * static_cast<std::size_t>(image.stride) +
        static_cast<std::size_t>(x) * 3U + static_cast<std::size_t>(channel);
    return static_cast<float>(image.data[offset]) * kInv255;
}

inline float clamp_coordinate(float value, int limit) {
    return std::min(std::max(value, 0.0f), static_cast<float>(limit - 1));
}

void validate(const ImageBgr8& src, const PreprocessSpec& spec) {
    // Validate input parameters
    if (src.data == nullptr) {
        throw std::invalid_argument("preprocess: src.data is null");
    }

    // Validate dimensions and stride
    if (src.width <= 0 || src.height <= 0) {
        throw std::invalid_argument("preprocess: source dimensions must be positive");
    }

    // Validate stride
    if (src.stride < src.width * 3) {
        throw std::invalid_argument("preprocess: stride must be at least width * 3");
    }

    // Validate destination dimensions and normalization parameters
    if (spec.dst_width <= 0 || spec.dst_height <= 0) {
        throw std::invalid_argument("preprocess: destination dimensions must be positive");
    }

    // Validate normalization parameters
    for (int c = 0; c < 3; ++c) {
        if (!(spec.norm.std[c] > 0.0f)) {
            throw std::invalid_argument("preprocess: std must be positive (channel " +
                                        std::to_string(c) + ")");
        }
    }
}

}  // namespace

std::size_t tensor_element_count(const PreprocessSpec& spec) {
    if (spec.dst_width <= 0 || spec.dst_height <= 0) {
        return 0;
    }
    return static_cast<std::size_t>(3) * static_cast<std::size_t>(spec.dst_height) *
           static_cast<std::size_t>(spec.dst_width);
}

std::vector<float> preprocess_bgr8_to_nchw_f32(const ImageBgr8& src, const PreprocessSpec& spec) {
    validate(src, spec);

    const int dst_w = spec.dst_width;
    const int dst_h = spec.dst_height;
    const float scale_x = static_cast<float>(src.width) / static_cast<float>(dst_w);
    const float scale_y = static_cast<float>(src.height) / static_cast<float>(dst_h);
    const float* mean = spec.norm.mean;
    const float* stddev = spec.norm.std;

    std::vector<float> output(tensor_element_count(spec), 0.0f);
    const std::size_t plane = static_cast<std::size_t>(dst_h) * static_cast<std::size_t>(dst_w);

    for (int dy = 0; dy < dst_h; ++dy) {
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

            for (int c = 0; c < 3; ++c) {
                const int src_c = spec.bgr_to_rgb ? (2 - c) : c;    // Convert channel index if BGR to RGB conversion is needed
                const float value = w00 * sample(src, x0, y0, src_c) + // Bilinear interpolation
                                    w10 * sample(src, x1, y0, src_c) +  
                                    w01 * sample(src, x0, y1, src_c) +
                                    w11 * sample(src, x1, y1, src_c);
                output[static_cast<std::size_t>(c) * plane + dst_index] = //offset for channel c
                    (value - mean[c]) / stddev[c];//Z-score normalization
            }
        }
    }

    return output;
}

}  // namespace rd

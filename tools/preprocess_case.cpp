// Reads raw BGR rows from stdin and prints the reference tensor, one float per
// line. Used by scripts/verify_preprocess_cpu.py.
//
// usage:
//   preprocess_case W H DW DH STRIDE BGR2RGB mean0 mean1 mean2 std0 std1 std2
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <vector>

#include "road_damage/preprocess_cpu.hpp"

int main(int argc, char** argv) {
    if (argc != 13) {
        std::fprintf(stderr,
                     "usage: %s W H DW DH STRIDE BGR2RGB mean0 mean1 mean2 std0 std1 std2\n",
                     argv[0]);
        return 2;
    }

    const int width = std::atoi(argv[1]);
    const int height = std::atoi(argv[2]);

    rd::PreprocessSpec spec;
    spec.dst_width = std::atoi(argv[3]);
    spec.dst_height = std::atoi(argv[4]);
    const int stride = std::atoi(argv[5]);
    spec.bgr_to_rgb = std::atoi(argv[6]) != 0;
    for (int c = 0; c < 3; ++c) {
        spec.norm.mean[c] = std::strtof(argv[7 + c], nullptr);
        spec.norm.std[c] = std::strtof(argv[10 + c], nullptr);
    }

    std::vector<std::uint8_t> buffer(static_cast<std::size_t>(height) *
                                     static_cast<std::size_t>(stride));
    std::cin.read(reinterpret_cast<char*>(buffer.data()),
                  static_cast<std::streamsize>(buffer.size()));
    if (!std::cin) {
        std::fprintf(stderr, "failed to read %zu bytes from stdin\n", buffer.size());
        return 1;
    }

    const rd::ImageBgr8 image{buffer.data(), width, height, stride};
    const std::vector<float> out = rd::preprocess_bgr8_to_nchw_f32_direct_single(image, spec);
    for (float value : out) {
        std::printf("%.9g\n", value);
    }
    return 0;
}

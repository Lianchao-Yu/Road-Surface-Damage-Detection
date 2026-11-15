#include <iostream>
#include <string>

#include "road_damage/version.hpp"

#ifdef RD_WITH_CUDA
#include "road_damage/cuda_info.hpp"
#endif

#ifdef RD_WITH_TENSORRT
#include "road_damage/trt_info.hpp"
#endif

namespace {

void print_usage(const char* argv0) {
    std::cout << "Usage: " << argv0 << " [--version] [--help]\n"
              << "\n"
              << "  --version  print the road_damage version\n"
              << "  --help     print this message\n"
              << "\n"
              << "Without arguments the program prints the build/runtime summary.\n";
}

}  // namespace

int main(int argc, char** argv) {
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--version") {
            std::cout << "road_damage " << rd::version() << "\n";
            return 0;
        }
        if (arg == "--help" || arg == "-h") {
            print_usage(argv[0]);
            return 0;
        }
        std::cerr << "unknown argument: " << arg << "\n";
        print_usage(argv[0]);
        return 2;
    }

    std::cout << "road_damage " << rd::version() << "\n";

#ifdef RD_WITH_CUDA
    if (rd::cuda_available()) {
        std::cout << "CUDA runtime : " << rd::cuda_runtime_version() << "\n";
        std::cout << "CUDA device  : " << rd::cuda_device_name() << "\n";
    } else {
        std::cout << "CUDA runtime : built, no device available\n";
    }
#else
    std::cout << "CUDA runtime : not built\n";
#endif

#ifdef RD_WITH_TENSORRT
    std::cout << "TensorRT     : " << rd::tensorrt_version() << "\n";
#else
    std::cout << "TensorRT     : not built\n";
#endif

    std::cout << "skeleton OK\n";
    return 0;
}

#include <cstdlib>
#include <iostream>
#include <string>

#include "road_damage/io/video_source.hpp"

namespace {

void usage(const char* argv0) {
    std::cout << "usage: " << argv0 << " URL_OR_PATH [MAX_FRAMES]\n"
              << "\n"
              << "Decodes the video with the FFmpeg software path and prints the first\n"
              << "MAX_FRAMES frames (default 5). Useful for checking timestamps, stride\n"
              << "and pixel dimensions before wiring the source into the inference loop.\n";
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 2) {
        usage(argv[0]);
        return 2;
    }

    const std::string url = argv[1];
    const int max_frames = (argc > 2) ? std::atoi(argv[2]) : 5;
    if (max_frames < 1) {
        std::cerr << "MAX_FRAMES must be positive\n";
        return 2;
    }

    rd::io::FFmpegVideoSource source;
    if (!source.open(url)) {
        std::cerr << "open failed: " << source.last_error() << "\n";
        return 1;
    }

    const rd::io::VideoInfo& info = source.info();
    std::cout << "format     : " << info.format << "\n";
    std::cout << "codec      : " << info.codec << "\n";
    std::cout << "resolution : " << info.width << "x" << info.height << "\n";
    std::cout << "fps        : " << info.fps << "\n";
    std::cout << "duration   : " << info.duration_us << " us\n";

    rd::io::VideoFrame frame;
    int decoded = 0;
    while (decoded < max_frames) {
        const rd::io::ReadStatus status = source.read(frame);
        if (status == rd::io::ReadStatus::kEndOfStream) {
            std::cout << "end of stream after " << decoded << " frame(s)\n";
            break;
        }
        if (status == rd::io::ReadStatus::kError) {
            std::cerr << "read failed: " << source.last_error() << "\n";
            return 1;
        }

        std::cout << "frame " << frame.frame_id << ": " << frame.width << "x"
                  << frame.height << " stride=" << frame.stride << " pts_us="
                  << frame.pts_us << "\n";
        ++decoded;
    }

    source.close();
    return 0;
}

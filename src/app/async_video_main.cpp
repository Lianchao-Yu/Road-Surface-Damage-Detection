#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <string>

#include "road_damage/async_pipeline.hpp"

namespace {

struct Options {
    rd::pipeline::AsyncPipelineConfig config;
    int max_frames = 0;
    int warmup = 5;
};

void usage(const char* argv0) {
    std::cout << "usage: " << argv0
              << " --video PATH [--engine PATH] [--out PATH] [--conf 0.25] [--iou 0.45]\n"
              << "       [--slots N] [--max-frames N] [--warmup N] [--no-write] [--display]\n";
}

bool parse(int argc, char** argv, Options& options) {
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        auto need_value = [&](const char* name) -> std::string {
            if (i + 1 >= argc) {
                std::cerr << "missing value for " << name << "\n";
                std::exit(2);
            }
            return argv[++i];
        };

        if (arg == "--video") {
            options.config.video_path = need_value("--video");
        } else if (arg == "--engine") {
            options.config.engine_path = need_value("--engine");
        } else if (arg == "--out") {
            options.config.output_path = need_value("--out");
        } else if (arg == "--conf") {
            options.config.conf_threshold = std::stof(need_value("--conf"));
        } else if (arg == "--iou") {
            options.config.iou_threshold = std::stof(need_value("--iou"));
        } else if (arg == "--slots") {
            options.config.slot_count = std::stoi(need_value("--slots"));
        } else if (arg == "--max-frames") {
            options.max_frames = std::stoi(need_value("--max-frames"));
            options.config.max_frames = options.max_frames;
        } else if (arg == "--warmup") {
            options.warmup = std::stoi(need_value("--warmup"));
            options.config.warmup_frames = options.warmup;
        } else if (arg == "--no-write") {
            options.config.write_video = false;
        } else if (arg == "--display") {
            options.config.display = true;
        } else if (arg == "--help" || arg == "-h") {
            usage(argv[0]);
            std::exit(0);
        } else {
            std::cerr << "unknown argument: " << arg << "\n";
            usage(argv[0]);
            return false;
        }
    }

    if (options.config.video_path.empty()) {
        std::cerr << "--video is required\n";
        usage(argv[0]);
        return false;
    }
    if (options.config.write_video && options.config.output_path.empty()) {
        const std::filesystem::path video(options.config.video_path);
        options.config.output_path =
            "results/video/" + video.stem().string() + "_async_pred.mp4";
    }
    return true;
}

}  // namespace

int main(int argc, char** argv) {
    Options options;
    if (!parse(argc, argv, options)) {
        return 2;
    }

    rd::pipeline::AsyncPipeline pipeline(options.config);
    if (!pipeline.init()) {
        std::cerr << "init failed: " << pipeline.last_err() << "\n";
        return 1;
    }

    const auto start = std::chrono::steady_clock::now();
    const bool ok = pipeline.run();
    const auto end = std::chrono::steady_clock::now();
    const double wall_s =
        std::chrono::duration<double>(end - start).count();

    const rd::pipeline::AsyncPipelineStats stats = pipeline.stats();
    std::cout << "video      : " << options.config.video_path << "\n";
    std::cout << "engine     : " << options.config.engine_path << "\n";
    std::cout << "slots      : " << options.config.slot_count << "\n";
    std::cout << "frames     : decoded=" << stats.decoded_frames
              << " inferred=" << stats.inferred_frames
              << " encoded=" << stats.encoded_frames
              << " detections=" << stats.total_detections << "\n";
    std::cout << "pipeline   : p50=" << stats.end_to_end_ms_p50
              << " ms, p95=" << stats.end_to_end_ms_p95 << " ms\n";
    std::cout << "stages     : decode p50=" << stats.decode_ms_p50
              << " ms, infer p50=" << stats.infer_ms_p50
              << " ms, encode p50=" << stats.encode_ms_p50 << " ms\n";
    if (wall_s > 0.0) {
        std::cout << "wall-clock : " << (stats.encoded_frames / wall_s)
                  << " fps (all frames, after init)\n";
    }
    if (options.config.write_video) {
        std::cout << "output     : " << options.config.output_path << "\n";
    }

    if (!ok) {
        std::cerr << "pipeline failed: " << pipeline.last_err() << "\n";
        return 1;
    }
    return 0;
}

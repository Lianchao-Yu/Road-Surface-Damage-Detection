#include <opencv2/opencv.hpp>

#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <string>
#include <unordered_map>
#include <vector>

#include "road_damage/postprocess_cpu.hpp"
#include "road_damage/preprocess_cpu.hpp"
#include "road_damage/tensorrt_backend.hpp"

namespace {

const char* kClassNames[] = {"D00", "D10", "D20", "D40"};
const cv::Scalar kColors[] = {
    {0, 0, 255}, {0, 200, 255}, {0, 255, 0}, {255, 128, 0},
};

void usage(const char* argv0) {
    std::cout << "usage: " << argv0
              << " --image PATH [--engine PATH] [--out PATH] [--conf 0.25] [--iou 0.45]\n";
}

}  // namespace

int main(int argc, char** argv) {
    std::string image_path;
    std::string engine_path = "models/best.engine";
    std::string out_path;
    float conf = 0.25f;
    float iou = 0.45f;

    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        auto need_value = [&](const char* name) -> std::string {
            if (i + 1 >= argc) {
                std::cerr << "missing value for " << name << "\n";
                std::exit(2);
            }
            return argv[++i];
        };
        if (arg == "--image") {
            image_path = need_value("--image");
        } else if (arg == "--engine") {
            engine_path = need_value("--engine");
        } else if (arg == "--out") {
            out_path = need_value("--out");
        } else if (arg == "--conf") {
            conf = std::stof(need_value("--conf"));
        } else if (arg == "--iou") {
            iou = std::stof(need_value("--iou"));
        } else if (arg == "--help" || arg == "-h") {
            usage(argv[0]);
            return 0;
        } else {
            std::cerr << "unknown argument: " << arg << "\n";
            usage(argv[0]);
            return 2;
        }
    }

    if (image_path.empty()) {
        usage(argv[0]);
        return 2;
    }

    cv::Mat bgr = cv::imread(image_path, cv::IMREAD_COLOR);
    if (bgr.empty()) {
        std::cerr << "failed to read image: " << image_path << "\n";
        return 1;
    }

    rd::PreprocessSpec spec;
    spec.dst_width = 640;
    spec.dst_height = 640;
    spec.bgr_to_rgb = true;
    spec.norm.mean[0] = spec.norm.mean[1] = spec.norm.mean[2] = 0.0f;
    spec.norm.std[0] = spec.norm.std[1] = spec.norm.std[2] = 1.0f;

    const rd::ImageBgr8 view{bgr.data, bgr.cols, bgr.rows, static_cast<int>(bgr.step)};
    const LetterboxMeta meta = rd::compute_letterbox_meta(view, spec);

    const auto t0 = std::chrono::steady_clock::now();
    const std::vector<float> input =
        rd::preprocess_bgr8_to_nchw_f32_letterbox_single(view, spec, meta, 114.0f);
    const auto t1 = std::chrono::steady_clock::now();

    trt::TensorRTBackend backend;
    if (!backend.TensorRTLoader(engine_path)) {
        std::cerr << "failed to load engine: " << engine_path << "\n";
        return 1;
    }

    trt::TensorView in_view;
    in_view.data = const_cast<float*>(input.data());
    in_view.dims.nbDims = 4;
    in_view.dims.d[0] = 1;
    in_view.dims.d[1] = 3;
    in_view.dims.d[2] = 640;
    in_view.dims.d[3] = 640;
    in_view.dtype = nvinfer1::DataType::kFLOAT;
    in_view.memType = trt::MemoryType::kCPU;

    trt::TensorView out_view;
    out_view.dtype = nvinfer1::DataType::kFLOAT;
    out_view.memType = trt::MemoryType::kCPU;

    std::unordered_map<std::string, trt::TensorView> inputs;
    std::unordered_map<std::string, trt::TensorView> outputs;
    inputs.emplace("images", in_view);
    outputs.emplace("output0", out_view);

    // Warm up TensorRT once so the reported latency does not include the
    // lazy kernel/context initialization of the first enqueue.
    if (!backend.TensorInfer(inputs, outputs)) {
        std::cerr << "TensorRT warmup failed\n";
        return 1;
    }

    const auto t2 = std::chrono::steady_clock::now();
    if (!backend.TensorInfer(inputs, outputs)) {
        std::cerr << "TensorRT inference failed\n";
        return 1;
    }
    const auto t3 = std::chrono::steady_clock::now();

    const std::vector<Detection> detections =
        postprocessing::PostProcessing_yolo(outputs, meta, conf, iou);
    const auto t4 = std::chrono::steady_clock::now();

    auto ms = [](auto a, auto b) {
        return std::chrono::duration<double, std::milli>(b - a).count();
    };

    std::cout << "image    : " << image_path << " (" << bgr.cols << "x" << bgr.rows << ")\n";
    std::cout << "letterbox: scale=" << meta.scale << " pad=(" << meta.padX << ","
              << meta.padY << ")\n";
    std::cout << "timing   : preprocess=" << ms(t0, t1) << " ms, infer=" << ms(t2, t3)
              << " ms, postprocess=" << ms(t3, t4) << " ms\n";
    std::cout << "detections: " << detections.size() << "\n";
    for (const Detection& d : detections) {
        std::cout << "  " << kClassNames[d.classID] << " score=" << d.score << " box=["
                  << d.x1 << ", " << d.y1 << ", " << d.x2 << ", " << d.y2 << "]\n";
    }

    cv::Mat canvas = bgr.clone();
    for (const Detection& d : detections) {
        const cv::Scalar color = kColors[d.classID % 4];
        cv::rectangle(canvas, cv::Point(static_cast<int>(d.x1), static_cast<int>(d.y1)),
                      cv::Point(static_cast<int>(d.x2), static_cast<int>(d.y2)), color, 2);
        cv::putText(canvas,
                    std::string(kClassNames[d.classID]) + " " + std::to_string(d.score).substr(0, 4),
                    cv::Point(static_cast<int>(d.x1), std::max(14, static_cast<int>(d.y1) - 5)),
                    cv::FONT_HERSHEY_SIMPLEX, 0.5, color, 1);
    }

    if (out_path.empty()) {
        const std::filesystem::path image(image_path);
        out_path = "results/demo/" + image.stem().string() + "_pred.jpg";
    }
    std::filesystem::create_directories(std::filesystem::path(out_path).parent_path());
    if (!cv::imwrite(out_path, canvas)) {
        std::cerr << "failed to write: " << out_path << "\n";
        return 1;
    }
    std::cout << "visual  : " << out_path << "\n";
    return 0;
}

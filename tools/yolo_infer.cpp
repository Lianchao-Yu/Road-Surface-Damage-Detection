#include <cuda_runtime.h>
#include <opencv2/opencv.hpp>

#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <string>
#include <unordered_map>
#include <vector>

#include "road_damage/postprocess_cpu.hpp"
#include "road_damage/preprocess_GPU.hpp"
#include "road_damage/preprocess_cpu.hpp"
#include "road_damage/preprocess_cv.hpp"
#include "road_damage/tensorrt_backend.hpp"

namespace {

const char* kClassNames[] = {"D00", "D10", "D20", "D40"};
const cv::Scalar kColors[] = {
    {0, 0, 255}, {0, 200, 255}, {0, 255, 0}, {255, 128, 0},
};

// Selects which letterbox implementation feeds TensorRT. The hand-written CPU
// path stays the default fallback; CV and GPU are opt-in alternatives.
enum class PreprocessMode {
    kCpu,
    kCvFloat,
    kCvU8,
    kGpu,
};

const char* preprocess_name(PreprocessMode mode) {
    switch (mode) {
        case PreprocessMode::kCpu:
            return "CPU hand-written";
        case PreprocessMode::kCvFloat:
            return "OpenCV (float resize)";
        case PreprocessMode::kCvU8:
            return "OpenCV (u8 resize)";
        case PreprocessMode::kGpu:
            return "GPU";
    }
    return "unknown";
}

const char* preprocess_suffix(PreprocessMode mode) {
    switch (mode) {
        case PreprocessMode::kCpu:
            return "_cpu_pred.jpg";
        case PreprocessMode::kCvFloat:
            return "_cv_float_pred.jpg";
        case PreprocessMode::kCvU8:
            return "_cv_u8_pred.jpg";
        case PreprocessMode::kGpu:
            return "_gpu_pred.jpg";
    }
    return "_pred.jpg";
}

// Small RAII helpers so the demo remains correct on every early-return path.
struct CudaBuffer {
    void* ptr = nullptr;
    ~CudaBuffer() {
        if (ptr != nullptr) {
            cudaFree(ptr);
        }
    }
};

struct CudaStream {
    cudaStream_t stream = nullptr;
    ~CudaStream() {
        if (stream != nullptr) {
            cudaStreamDestroy(stream);
        }
    }
};

void usage(const char* argv0) {
    std::cout << "usage: " << argv0
              << " --image PATH [--engine PATH] [--out PATH] [--conf 0.25] [--iou 0.45]"
              << " [--gpu-preprocess | --cv-preprocess [float|u8]]\n"
              << "\n"
              << "  --gpu-preprocess        run letterbox on GPU\n"
              << "  --cv-preprocess [MODE]  run letterbox with OpenCV on CPU; MODE is\n"
              << "                          'float' (default, FP32 resize) or 'u8' (fast 8-bit)\n";
}

double elapsed_ms(std::chrono::steady_clock::time_point a,
                  std::chrono::steady_clock::time_point b) {
    return std::chrono::duration<double, std::milli>(b - a).count();
}

}  // namespace

int main(int argc, char** argv) {
    std::string image_path;
    std::string engine_path = "models/best.engine";
    std::string out_path;
    float conf = 0.25f;
    float iou = 0.45f;
    PreprocessMode preprocess_mode = PreprocessMode::kCpu;
    bool preprocess_selected = false;
    auto select_preprocess = [&](PreprocessMode mode, const char* flag) {
        if (preprocess_selected) {
            std::cerr << flag << " conflicts with another preprocess selection\n";
            std::exit(2);
        }
        preprocess_selected = true;
        preprocess_mode = mode;
    };

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
        } else if (arg == "--gpu-preprocess") {
            select_preprocess(PreprocessMode::kGpu, "--gpu-preprocess");
        } else if (arg == "--cv-preprocess") {
            // Optional mode value; default to the contract-matching FP32 path.
            PreprocessMode mode = PreprocessMode::kCvFloat;
            if (i + 1 < argc) {
                const std::string value = argv[i + 1];
                if (value == "float") {
                    mode = PreprocessMode::kCvFloat;
                    ++i;
                } else if (value == "u8") {
                    mode = PreprocessMode::kCvU8;
                    ++i;
                }
            }
            select_preprocess(mode, "--cv-preprocess");
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

    const auto t_read0 = std::chrono::steady_clock::now();
    cv::Mat bgr = cv::imread(image_path, cv::IMREAD_COLOR);
    const auto t_read1 = std::chrono::steady_clock::now();
    if (bgr.empty()) {
        std::cerr << "failed to read image: " << image_path << "\n";
        return 1;
    }
    // cv::imread returns a packed BGR Mat. All preprocess implementations
    // (hand-written CPU, OpenCV CPU and CUDA) honor ImageBgr8::stride, so no
    // defensive clone is needed here.
    rd::PreprocessSpec spec;
    spec.dst_width = 640;
    spec.dst_height = 640;
    spec.bgr_to_rgb = true;
    spec.norm.mean[0] = spec.norm.mean[1] = spec.norm.mean[2] = 0.0f;
    spec.norm.std[0] = spec.norm.std[1] = spec.norm.std[2] = 1.0f;

    const rd::ImageBgr8 view{bgr.data, bgr.cols, bgr.rows, static_cast<int>(bgr.step)};
    const LetterboxMeta meta = rd::compute_letterbox_meta(view, spec);

    // Device buffers are declared before the backend so the backend context
    // (which temporarily references the input pointer) is destroyed first.
    CudaBuffer d_src;
    CudaBuffer d_dst;
    CudaStream preprocess_stream;

    trt::TensorRTBackend backend;
    if (!backend.TensorRTLoader(engine_path)) {
        std::cerr << "failed to load engine: " << engine_path << "\n";
        return 1;
    }

    trt::TensorView in_view;
    trt::TensorView out_view;
    std::vector<float> cpu_input;

    double preprocess_ms = 0.0;
    if (preprocess_mode == PreprocessMode::kGpu) {
        // Row stride is honored by the kernel and every preprocessing backend,
        // so upload exactly rows*stride bytes.
        const std::size_t src_bytes = static_cast<std::size_t>(bgr.rows) * bgr.step;
        const std::size_t dst_bytes = 3u * 640u * 640u * sizeof(float);
        if (cudaMalloc(&d_src.ptr, src_bytes) != cudaSuccess ||
            cudaMalloc(&d_dst.ptr, dst_bytes) != cudaSuccess) {
            std::cerr << "failed to allocate GPU preprocess buffers\n";
            return 1;
        }
        if (cudaStreamCreateWithFlags(&preprocess_stream.stream, cudaStreamNonBlocking) !=
            cudaSuccess) {
            std::cerr << "failed to create GPU preprocess stream\n";
            return 1;
        }

        const auto t0 = std::chrono::steady_clock::now();
        // H2D: copied from the decoded BGR frame. This is pageable memory for
        // now; the pinned-memory stage will remove the extra staging cost.
        if (cudaMemcpyAsync(d_src.ptr, bgr.data, src_bytes, cudaMemcpyHostToDevice,
                            preprocess_stream.stream) != cudaSuccess) {
            std::cerr << "GPU preprocess H2D failed\n";
            return 1;
        }
        // Pass the real host row stride; the kernel also works with FFmpeg
        // linesize > width*3 without an extra repack.
        if (!fused_k1::launchFusedPreprocess(static_cast<const uint8_t*>(d_src.ptr),
                                             static_cast<float*>(d_dst.ptr), meta,
                                             preprocess_stream.stream,
                                             static_cast<std::size_t>(bgr.step))) {
            std::cerr << "GPU letterbox kernel failed\n";
            return 1;
        }
        // The backend owns a different CUDA stream, so wait until the
        // preprocessed device tensor is complete before TensorRT consumes it.
        if (cudaStreamSynchronize(preprocess_stream.stream) != cudaSuccess) {
            std::cerr << "GPU preprocess stream sync failed\n";
            return 1;
        }
        const auto t1 = std::chrono::steady_clock::now();
        preprocess_ms = elapsed_ms(t0, t1);

        in_view.data = d_dst.ptr;
        in_view.memType = trt::MemoryType::kGPU;
    } else if (preprocess_mode == PreprocessMode::kCpu) {
        // Hand-written reference path: default fallback when neither GPU nor
        // OpenCV preprocessing is requested.
        const auto t0 = std::chrono::steady_clock::now();
        cpu_input =
            rd::preprocess_bgr8_to_nchw_f32_letterbox_single(view, spec, meta, 114.0f);
        const auto t1 = std::chrono::steady_clock::now();
        preprocess_ms = elapsed_ms(t0, t1);

        in_view.data = cpu_input.data();
        in_view.memType = trt::MemoryType::kCPU;
    } else {
        // OpenCV CPU path. 'float' matches the reference pixel contract; 'u8'
        // uses OpenCV's faster fixed-point 8-bit resize.
        const auto t0 = std::chrono::steady_clock::now();
        const rd::CvInterpolationSource source =
            preprocess_mode == PreprocessMode::kCvU8 ? rd::CvInterpolationSource::kUint8
                                                    : rd::CvInterpolationSource::kFloat32;
        cpu_input = rd::preprocess_bgr8_to_nchw_f32_letterbox_cv_single(
            view, spec, meta, 114.0f, source);
        const auto t1 = std::chrono::steady_clock::now();
        preprocess_ms = elapsed_ms(t0, t1);

        in_view.data = cpu_input.data();
        in_view.memType = trt::MemoryType::kCPU;
    }

    in_view.dims.nbDims = 4;
    in_view.dims.d[0] = 1;
    in_view.dims.d[1] = 3;
    in_view.dims.d[2] = 640;
    in_view.dims.d[3] = 640;
    in_view.dtype = nvinfer1::DataType::kFLOAT;

    out_view.dtype = nvinfer1::DataType::kFLOAT;
    out_view.memType = trt::MemoryType::kCPU;

    std::unordered_map<std::string, trt::TensorView> inputs;
    std::unordered_map<std::string, trt::TensorView> outputs;
    inputs.emplace("images", in_view);
    outputs.emplace("output0", out_view);

    // Warm up TensorRT once so the reported latency excludes lazy kernel and
    // context initialization.
    if (!backend.TensorInfer(inputs, outputs)) {
        std::cerr << "TensorRT warmup failed\n";
        return 1;
    }

    const auto t_infer0 = std::chrono::steady_clock::now();
    if (!backend.TensorInfer(inputs, outputs)) {
        std::cerr << "TensorRT inference failed\n";
        return 1;
    }
    const auto t_infer1 = std::chrono::steady_clock::now();

    const std::vector<Detection> detections =
        postprocessing::PostProcessing_yolo(outputs, meta, conf, iou);
    const auto t_post1 = std::chrono::steady_clock::now();

    std::cout << "image      : " << image_path << " (" << bgr.cols << "x" << bgr.rows
              << ")\n";
    std::cout << "preprocess : " << preprocess_name(preprocess_mode) << "\n";
    std::cout << "letterbox  : scale=" << meta.scale << " pad=(" << meta.padX << ","
              << meta.padY << ")\n";
    std::cout << "timing     : read=" << elapsed_ms(t_read0, t_read1)
              << " ms, preprocess=" << preprocess_ms
              << " ms, infer=" << elapsed_ms(t_infer0, t_infer1)
              << " ms, postprocess=" << elapsed_ms(t_infer1, t_post1) << " ms\n";
    std::cout << "detections : " << detections.size() << "\n";
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
                    std::string(kClassNames[d.classID]) + " " +
                        std::to_string(d.score).substr(0, 4),
                    cv::Point(static_cast<int>(d.x1),
                              std::max(14, static_cast<int>(d.y1) - 5)),
                    cv::FONT_HERSHEY_SIMPLEX, 0.5, color, 1);
    }

    if (out_path.empty()) {
        const std::filesystem::path image(image_path);
        out_path = "results/demo/" + image.stem().string() + preprocess_suffix(preprocess_mode);
    }
    std::filesystem::create_directories(std::filesystem::path(out_path).parent_path());
    if (!cv::imwrite(out_path, canvas)) {
        std::cerr << "failed to write: " << out_path << "\n";
        return 1;
    }
    std::cout << "visual     : " << out_path << "\n";
    return 0;
}

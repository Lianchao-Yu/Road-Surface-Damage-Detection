#include <cuda_runtime.h>
#include <opencv2/opencv.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <iterator>
#include <string>
#include <unordered_map>
#include <vector>

#include "road_damage/io/video_source.hpp"
#include "road_damage/postprocess_cpu.hpp"
#include "road_damage/preprocess_GPU.hpp"
#include "road_damage/preprocess_cpu.hpp"
#include "road_damage/tensorrt_backend.hpp"

namespace {

const char* kClassNames[] = {"D00", "D10", "D20", "D40"};
const cv::Scalar kColors[] = {
    {0, 0, 255}, {0, 200, 255}, {0, 255, 0}, {255, 128, 0},
};

constexpr int kInputWidth = 640;
constexpr int kInputHeight = 640;

struct Options {
    std::string video_path;
    std::string engine_path = "models/best.engine";
    std::string out_path;
    float conf = 0.25f;
    float iou = 0.45f;
    int max_frames = 0;  // 0 = all frames
    int warmup_frames = 5;
    int print_detections = 0;  // print first N frames' detections
    bool write_video = true;
    bool display = false;
};

struct DeviceBuffer {
    void* ptr = nullptr;
    std::size_t capacity = 0;

    ~DeviceBuffer() {
        if (ptr != nullptr) {
            cudaFree(ptr);
        }
    }

    bool ensure(std::size_t bytes) {
        if (bytes <= capacity) {
            return true;
        }
        if (ptr != nullptr) {
            cudaFree(ptr);
            ptr = nullptr;
            capacity = 0;
        }
        if (cudaMalloc(&ptr, bytes) != cudaSuccess) {
            ptr = nullptr;
            return false;
        }
        capacity = bytes;
        return true;
    }
};

struct StageSamples {
    std::vector<double> decode;
    std::vector<double> transfer_preprocess;
    std::vector<double> infer;
    std::vector<double> postprocess;
    std::vector<double> draw;
    std::vector<double> encode;
    std::vector<double> total;
};

double elapsed_ms(std::chrono::steady_clock::time_point a,
                  std::chrono::steady_clock::time_point b) {
    return std::chrono::duration<double, std::milli>(b - a).count();
}

double median(const std::vector<double>& values) {
    if (values.empty()) {
        return 0.0;
    }
    std::vector<double> sorted = values;
    std::sort(sorted.begin(), sorted.end());
    return sorted[sorted.size() / 2];
}

double percentile95(const std::vector<double>& values) {
    if (values.empty()) {
        return 0.0;
    }
    std::vector<double> sorted = values;
    std::sort(sorted.begin(), sorted.end());
    const auto index = static_cast<std::size_t>(
        std::llround(0.95 * static_cast<double>(sorted.size() - 1)));
    return sorted[std::min(index, sorted.size() - 1)];
}

void print_stage(const char* name, const std::vector<double>& values) {
    if (values.empty()) {
        std::cout << "  " << name << ": n/a\n";
        return;
    }
    double sum = 0.0;
    for (double value : values) {
        sum += value;
    }
    std::cout << "  " << name << ": median=" << median(values)
              << " ms, mean=" << (sum / static_cast<double>(values.size()))
              << " ms, p95=" << percentile95(values) << " ms\n";
}

void usage(const char* argv0) {
    std::cout << "usage: " << argv0
              << " --video PATH [--engine PATH] [--out PATH] [--conf 0.25] [--iou 0.45]\n"
              << "       [--max-frames N] [--warmup N] [--no-write] [--display]\n"
              << "       [--print-detections N]\n"
              << "\n"
              << "  --video PATH    local MP4/RTSP input\n"
              << "  --engine PATH   TensorRT engine (default models/best.engine)\n"
              << "  --out PATH      annotated output video; default results/video/<name>_pred.mp4\n"
              << "  --max-frames N  stop after N total frames (0 = all)\n"
              << "  --warmup N      frames excluded from timing statistics (default 5)\n"
              << "  --no-write      run inference without encoding an output video\n"
              << "  --display       show frames with cv::imshow (requires a display)\n"
              << "  --print-detections N  print detections for the first N frames\n";
}

bool parse_options(int argc, char** argv, Options& options) {
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
            options.video_path = need_value("--video");
        } else if (arg == "--engine") {
            options.engine_path = need_value("--engine");
        } else if (arg == "--out") {
            options.out_path = need_value("--out");
        } else if (arg == "--conf") {
            options.conf = std::stof(need_value("--conf"));
        } else if (arg == "--iou") {
            options.iou = std::stof(need_value("--iou"));
        } else if (arg == "--max-frames") {
            options.max_frames = std::stoi(need_value("--max-frames"));
        } else if (arg == "--warmup") {
            options.warmup_frames = std::stoi(need_value("--warmup"));
        } else if (arg == "--print-detections") {
            options.print_detections = std::stoi(need_value("--print-detections"));
        } else if (arg == "--no-write") {
            options.write_video = false;
        } else if (arg == "--display") {
            options.display = true;
        } else if (arg == "--help" || arg == "-h") {
            usage(argv[0]);
            std::exit(0);
        } else {
            std::cerr << "unknown argument: " << arg << "\n";
            usage(argv[0]);
            return false;
        }
    }

    if (options.video_path.empty()) {
        std::cerr << "--video is required\n";
        usage(argv[0]);
        return false;
    }
    if (options.max_frames < 0 || options.warmup_frames < 0) {
        std::cerr << "--max-frames and --warmup must be non-negative\n";
        return false;
    }
    if (options.print_detections < 0) {
        std::cerr << "--print-detections must be non-negative\n";
        return false;
    }
    return true;
}

bool open_writer(cv::VideoWriter& writer, const std::string& path, double fps,
                 const cv::Size& size) {
    const int fourcc_mp4v = cv::VideoWriter::fourcc('m', 'p', '4', 'v');
    if (writer.open(path, fourcc_mp4v, fps, size)) {
        return true;
    }

    // Some OpenCV builds expose only the H.264 encoder through FFmpeg.
    const int fourcc_avc1 = cv::VideoWriter::fourcc('a', 'v', 'c', '1');
    return writer.open(path, fourcc_avc1, fps, size);
}

void draw_frame(const rd::io::VideoFrame& frame,
                const std::vector<Detection>& detections,
                const std::vector<cv::Scalar>& colors, bool& canvas_ready,
                cv::Mat& canvas) {
    // Copy the decoded frame because VideoFrame::data is reused by the source
    // on the next read() call. The destination Mat is reused to avoid a fresh
    // allocation on every frame.
    const cv::Mat view(frame.height, frame.width, CV_8UC3,
                       const_cast<std::uint8_t*>(frame.data), frame.stride);
    if (!canvas_ready || canvas.cols != frame.width || canvas.rows != frame.height) {
        canvas.create(frame.height, frame.width, CV_8UC3);
        canvas_ready = true;
    }
    view.copyTo(canvas);

    for (const Detection& d : detections) {
        const cv::Scalar color = colors[static_cast<std::size_t>(d.classID) %
                                        static_cast<int>(colors.size())];
        cv::rectangle(canvas, cv::Point(static_cast<int>(d.x1), static_cast<int>(d.y1)),
                      cv::Point(static_cast<int>(d.x2), static_cast<int>(d.y2)), color, 2);
        const std::string label = std::string(kClassNames[d.classID % 4]) + " " +
                                  std::to_string(d.score).substr(0, 4);
        cv::putText(canvas, label,
                    cv::Point(static_cast<int>(d.x1),
                              std::max(14, static_cast<int>(d.y1) - 5)),
                    cv::FONT_HERSHEY_SIMPLEX, 0.5, color, 1);
    }

    const std::string info = "frame " + std::to_string(frame.frame_id) + "  pts " +
                             std::to_string(frame.pts_us) + " us  det " +
                             std::to_string(detections.size());
    cv::putText(canvas, info, cv::Point(10, 24), cv::FONT_HERSHEY_SIMPLEX, 0.6,
                cv::Scalar(0, 0, 0), 3);
    cv::putText(canvas, info, cv::Point(10, 24), cv::FONT_HERSHEY_SIMPLEX, 0.6,
                cv::Scalar(255, 255, 255), 1);
}

}  // namespace

int main(int argc, char** argv) {
    Options options;
    if (!parse_options(argc, argv, options)) {
        return 2;
    }

    if (cudaSetDevice(0) != cudaSuccess) {
        std::cerr << "cudaSetDevice(0) failed\n";
        return 1;
    }

    rd::io::FFmpegVideoSource source;
    if (!source.open(options.video_path)) {
        std::cerr << "failed to open video: " << source.last_error() << "\n";
        return 1;
    }

    const rd::io::VideoInfo& video = source.info();
    std::cout << "video      : " << options.video_path << " (" << video.width << "x"
              << video.height << ", " << video.codec << ", " << video.fps << " fps, "
              << (video.duration_us / 1000) << " ms)\n";

    trt::TensorRTBackend backend;
    if (!backend.TensorRTLoader(options.engine_path)) {
        std::cerr << "failed to load engine: " << options.engine_path << "\n";
        return 1;
    }

    void* input_buffer = backend.getInputBuffer("images");
    cudaStream_t stream = backend.getStream();
    if (input_buffer == nullptr || stream == nullptr) {
        std::cerr << "engine is missing input tensor 'images' or CUDA stream\n";
        return 1;
    }

    // The GPU preprocess writes 3*640*640 floats directly into the engine's
    // preallocated input buffer; only the raw BGR frame needs a staging buffer.
    rd::PreprocessSpec spec;
    spec.dst_width = kInputWidth;
    spec.dst_height = kInputHeight;
    spec.bgr_to_rgb = true;
    spec.norm.mean[0] = spec.norm.mean[1] = spec.norm.mean[2] = 0.0f;
    spec.norm.std[0] = spec.norm.std[1] = spec.norm.std[2] = 1.0f;

    LetterboxMeta meta;
    bool have_meta = false;

    DeviceBuffer device_source;
    cv::VideoWriter writer;
    bool writer_ready = false;
    bool writer_enabled = options.write_video;
    std::string out_path = options.out_path;
    if (writer_enabled && out_path.empty()) {
        const std::filesystem::path video_path(options.video_path);
        out_path = "results/video/" + video_path.stem().string() + "_pred.mp4";
    }

    trt::TensorView in_view;
    in_view.dtype = nvinfer1::DataType::kFLOAT;
    in_view.memType = trt::MemoryType::kGPU;
    in_view.dims.nbDims = 4;
    in_view.dims.d[0] = 1;
    in_view.dims.d[1] = 3;
    in_view.dims.d[2] = kInputHeight;
    in_view.dims.d[3] = kInputWidth;
    in_view.data = input_buffer;

    trt::TensorView out_view;
    out_view.dtype = nvinfer1::DataType::kFLOAT;
    out_view.memType = trt::MemoryType::kCPU;

    std::unordered_map<std::string, trt::TensorView> inputs;
    std::unordered_map<std::string, trt::TensorView> outputs;
    inputs.emplace("images", in_view);
    outputs.emplace("output0", out_view);

    cudaEvent_t event_start = nullptr;
    cudaEvent_t event_preprocess_done = nullptr;
    cudaEvent_t event_infer_done = nullptr;
    if (cudaEventCreate(&event_start) != cudaSuccess ||
        cudaEventCreate(&event_preprocess_done) != cudaSuccess ||
        cudaEventCreate(&event_infer_done) != cudaSuccess) {
        std::cerr << "failed to create CUDA events\n";
        return 1;
    }

    StageSamples samples;
    std::vector<cv::Scalar> colors(std::begin(kColors), std::end(kColors));
    cv::Mat canvas;
    bool canvas_ready = false;
    double total_detections = 0.0;
    int total_frames = 0;
    int measured_frames = 0;
    bool display_enabled = options.display;
    bool stop_requested = false;
    bool failed = false;

    for (;;) {
        if (options.max_frames > 0 && total_frames >= options.max_frames) {
            break;
        }

        const auto t0 = std::chrono::steady_clock::now();
        rd::io::VideoFrame frame;
        const rd::io::ReadStatus status = source.read(frame);
        const auto t1 = std::chrono::steady_clock::now();
        if (status == rd::io::ReadStatus::kEndOfStream) {
            break;
        }
        if (status == rd::io::ReadStatus::kError) {
            std::cerr << "video read failed: " << source.last_error() << "\n";
            failed = true;
            stop_requested = true;
            break;
        }

        if (!have_meta) {
            const rd::ImageBgr8 view{frame.data, frame.width, frame.height, frame.stride};
            meta = rd::compute_letterbox_meta(view, spec);
            have_meta = true;

            if (writer_enabled) {
                const std::filesystem::path parent =
                    std::filesystem::path(out_path).parent_path();
                if (!parent.empty()) {
                    std::filesystem::create_directories(parent);
                }
                const double fps = (video.fps > 0.0) ? video.fps : 30.0;
                writer_ready =
                    open_writer(writer, out_path, fps, cv::Size(frame.width, frame.height));
                if (!writer_ready) {
                    std::cerr << "warning: failed to open output video " << out_path
                              << "; continuing without writing\n";
                    writer_enabled = false;
                }
            }
        }

        if (frame.width != meta.origWidth || frame.height != meta.origHeight) {
            std::cerr << "video resolution changed mid-stream; this demo expects a "
                         "constant resolution\n";
            failed = true;
            stop_requested = true;
            break;
        }

        // H2D must upload the full row stride, not just width * 3 bytes.
        const std::size_t source_bytes =
            static_cast<std::size_t>(frame.stride) * static_cast<std::size_t>(frame.height);
        if (!device_source.ensure(source_bytes)) {
            std::cerr << "failed to allocate GPU source buffer\n";
            failed = true;
            stop_requested = true;
            break;
        }

        if (cudaEventRecord(event_start, stream) != cudaSuccess ||
            cudaMemcpyAsync(device_source.ptr, frame.data, source_bytes,
                            cudaMemcpyHostToDevice, stream) != cudaSuccess) {
            std::cerr << "H2D copy failed\n";
            failed = true;
            stop_requested = true;
            break;
        }

        // Reuse the backend's execution stream and preallocated input tensor.
        // This preserves H2D -> preprocess -> TensorRT ordering without any
        // cross-stream event/synchronization.
        if (!fused_k1::launchFusedPreprocess(
                static_cast<const std::uint8_t*>(device_source.ptr),
                static_cast<float*>(input_buffer), meta, stream,
                static_cast<std::size_t>(frame.stride))) {
            std::cerr << "GPU preprocess launch failed\n";
            failed = true;
            stop_requested = true;
            break;
        }
        if (cudaEventRecord(event_preprocess_done, stream) != cudaSuccess) {
            std::cerr << "failed to record preprocess event\n";
            failed = true;
            stop_requested = true;
            break;
        }

        if (!backend.TensorInfer(inputs, outputs)) {
            std::cerr << "TensorRT inference failed\n";
            failed = true;
            stop_requested = true;
            break;
        }
        const auto t2 = std::chrono::steady_clock::now();
        if (cudaEventRecord(event_infer_done, stream) != cudaSuccess) {
            std::cerr << "failed to record inference event\n";
            failed = true;
            stop_requested = true;
            break;
        }

        std::vector<Detection> detections =
            postprocessing::PostProcessing_yolo(outputs, meta, options.conf, options.iou);
        const auto t3 = std::chrono::steady_clock::now();

        draw_frame(frame, detections, colors, canvas_ready, canvas);
        const auto t4 = std::chrono::steady_clock::now();

        bool wrote_frame = false;
        if (writer_enabled && writer_ready) {
            writer.write(canvas);
            wrote_frame = true;
        }
        const auto t5 = std::chrono::steady_clock::now();

        if (display_enabled) {
            try {
                cv::imshow("road_damage_video", canvas);
                const int key = cv::waitKey(1);
                if (key == 27 || key == 'q') {
                    stop_requested = true;
                }
            } catch (const cv::Exception& e) {
                std::cerr << "display disabled: " << e.what() << "\n";
                display_enabled = false;
            }
        }

        // Read CUDA event timings after TensorInfer() has synchronized the
        // stream. Events are on the same stream, so no extra sync is needed.
        float transfer_preprocess_ms = 0.0f;
        float infer_ms = 0.0f;
        if (cudaEventElapsedTime(&transfer_preprocess_ms, event_start,
                                 event_preprocess_done) != cudaSuccess ||
            cudaEventElapsedTime(&infer_ms, event_preprocess_done, event_infer_done) !=
                cudaSuccess) {
            std::cerr << "failed to read CUDA event timings\n";
            failed = true;
            stop_requested = true;
            break;
        }

        total_detections += static_cast<double>(detections.size());
        if (total_frames >= options.warmup_frames) {
            const auto t_end = wrote_frame ? t5 : t4;
            samples.decode.push_back(elapsed_ms(t0, t1));
            samples.transfer_preprocess.push_back(
                static_cast<double>(transfer_preprocess_ms));
            samples.infer.push_back(static_cast<double>(infer_ms));
            samples.postprocess.push_back(elapsed_ms(t2, t3));
            samples.draw.push_back(elapsed_ms(t3, t4));
            if (wrote_frame) {
                samples.encode.push_back(elapsed_ms(t4, t5));
            }
            samples.total.push_back(elapsed_ms(t0, t_end));
            ++measured_frames;
        }

        if (total_frames < options.print_detections) {
            std::cout << "frame " << frame.frame_id << " pts_us=" << frame.pts_us
                      << " detections=" << detections.size() << "\n";
            for (const Detection& d : detections) {
                std::cout << "  " << kClassNames[d.classID % 4] << " score=" << d.score
                          << " box=[" << d.x1 << ", " << d.y1 << ", " << d.x2 << ", "
                          << d.y2 << "]\n";
            }
        }
        ++total_frames;

        if (stop_requested) {
            break;
        }
    }

    source.close();
    if (writer_ready) {
        writer.release();
    }
    cudaEventDestroy(event_start);
    cudaEventDestroy(event_preprocess_done);
    cudaEventDestroy(event_infer_done);

    if (failed) {
        return 1;
    }

    std::cout << "engine     : " << options.engine_path << "\n";
    std::cout << "frames     : " << total_frames << " total, " << measured_frames
              << " measured, " << options.warmup_frames << " warmup\n";
    std::cout << "detections : " << total_detections;
    if (total_frames > 0) {
        std::cout << " total, " << (total_detections / total_frames) << " per frame";
    }
    std::cout << "\n";
    std::cout << "timing     :\n";
    print_stage("decode", samples.decode);
    print_stage("H2D+preprocess", samples.transfer_preprocess);
    print_stage("infer", samples.infer);
    print_stage("postprocess", samples.postprocess);
    print_stage("draw", samples.draw);
    print_stage("encode", samples.encode);
    print_stage("total/frame", samples.total);

    if (!samples.total.empty()) {
        double sum = 0.0;
        for (double value : samples.total) {
            sum += value;
        }
        const double fps = 1000.0 * static_cast<double>(samples.total.size()) / sum;
        std::cout << "throughput : " << fps << " fps (measured frames)\n";
    }
    if (writer_enabled && writer_ready) {
        std::cout << "output     : " << out_path << "\n";
    } else if (options.write_video) {
        std::cout << "output     : not written\n";
    }
    return 0;
}

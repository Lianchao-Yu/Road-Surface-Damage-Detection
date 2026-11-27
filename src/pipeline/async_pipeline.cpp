#include "road_damage/async_pipeline.hpp"

#include "road_damage/frame_queue.hpp"
#include "road_damage/frame_slot.hpp"
#include "road_damage/io/video_source.hpp"
#include "road_damage/postprocess_cpu.hpp"
#include "road_damage/preprocess_GPU.hpp"
#include "road_damage/shared_data.hpp"
#include "road_damage/tensorrt_backend.hpp"

#include <cuda_runtime.h>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <unordered_map>
#include <utility>
#include <vector>
/*
A FrameSlot holds the per-frame buffers and metadata. A fixed pool is seeded
into free_queue. Each stage takes a slot from one queue and passes it to the
next shared queue, so only one worker uses a slot at a time.

FrameQueue wraps std::queue with a mutex and condition variables. Push/pop are
thread-safe, and producers block when the queue is full.

decode:      free_queue -> fill h_bgr -> infer_queue
GPU worker:  H2D -> preprocess -> infer -> postprocess -> encode_queue
encode:      write output -> free_queue

TensorRT input/output buffers are backend-owned and reused.
*/



namespace rd::pipeline {
namespace {

const char* kClassNames[] = {"D00", "D10", "D20", "D40"};
const cv::Scalar kColors[] = {
    {0, 0, 255}, {0, 200, 255}, {0, 255, 0}, {255, 128, 0},
};

using Clock = std::chrono::steady_clock;

double elapsed_ms(Clock::time_point start, Clock::time_point end) {
    return std::chrono::duration<double, std::milli>(end - start).count();
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

std::size_t align_stride(int width) {
    const std::size_t bytes = static_cast<std::size_t>(width) * 3U;
    return (bytes + 63U) & ~static_cast<std::size_t>(63U);
}

bool open_writer(cv::VideoWriter& writer, const std::string& path, double fps,
                 const cv::Size& size) {
    const int fourcc_mp4v = cv::VideoWriter::fourcc('m', 'p', '4', 'v');
    if (writer.open(path, fourcc_mp4v, fps, size)) {
        return true;
    }
    const int fourcc_avc1 = cv::VideoWriter::fourcc('a', 'v', 'c', '1');
    return writer.open(path, fourcc_avc1, fps, size);
}

void draw_detections(cv::Mat& canvas, const std::vector<Detection>& detections,
                     std::uint64_t frame_id) {
    for (const Detection& d : detections) {
        if (d.classID < 0 || d.classID >= 4) {
            continue;
        }
        const cv::Scalar color = kColors[d.classID];
        cv::rectangle(canvas, cv::Point(static_cast<int>(d.x1), static_cast<int>(d.y1)),
                      cv::Point(static_cast<int>(d.x2), static_cast<int>(d.y2)), color, 2);
        const std::string label =
            std::string(kClassNames[d.classID]) + " " + std::to_string(d.score).substr(0, 4);
        cv::putText(canvas, label,
                    cv::Point(static_cast<int>(d.x1),
                              std::max(14, static_cast<int>(d.y1) - 5)),
                    cv::FONT_HERSHEY_SIMPLEX, 0.5, color, 1);
    }

    const std::string info =
        "frame " + std::to_string(frame_id) + "  det " + std::to_string(detections.size());
    cv::putText(canvas, info, cv::Point(10, 24), cv::FONT_HERSHEY_SIMPLEX, 0.6,
                cv::Scalar(0, 0, 0), 3);
    cv::putText(canvas, info, cv::Point(10, 24), cv::FONT_HERSHEY_SIMPLEX, 0.6,
                cv::Scalar(255, 255, 255), 1);
}

}  // namespace

struct AsyncPipeline::Impl
{
    explicit Impl(AsyncPipelineConfig config)
        : cfg(std::move(config)),
          free_queue(static_cast<std::size_t>(std::max(1, cfg.slot_count))),
          infer_queue(static_cast<std::size_t>(std::max(1, cfg.decode_queue_capacity))),
          encode_queue(static_cast<std::size_t>(std::max(1, cfg.encode_queue_capacity)))
    {
    }

    ~Impl() { stop(); }

    AsyncPipelineConfig cfg;

    std::unique_ptr<io::FFmpegVideoSource> source;
    trt::TensorRTBackend backend;
    cv::VideoWriter writer;

    std::vector<std::unique_ptr<FrameSlot>> slot_storage;
    FrameQueue<FrameSlot*> free_queue;
    FrameQueue<FrameSlot*> infer_queue;
    FrameQueue<FrameSlot*> encode_queue;

    std::thread decode_thread;
    std::thread infer_thread;
    std::thread encode_thread;

    int src_width = 0;
    int src_height = 0;
    double src_fps = 30.0;
    int model_width = 640;
    int model_height = 640;

    std::string input_name;
    std::string output_name;
    nvinfer1::Dims input_dims{};

    cudaStream_t stream = nullptr;
    void* input_buffer = nullptr;

    LetterboxMeta letterbox;
    std::unordered_map<std::string, trt::TensorView> inputs;
    std::unordered_map<std::string, trt::TensorView> outputs;

    std::size_t host_stride_capacity = 0;
    std::size_t frame_capacity = 0;
    bool display_enabled = false;

    std::atomic<bool> stop_requested{false};
    std::atomic<bool> running{false};
    std::atomic<bool> failed{false};

    bool initialized = false;
    bool cleaned = false;

    mutable std::mutex state_mutex;
    std::string last_error;

    mutable std::mutex stats_mutex;
    AsyncPipelineStats stats;
    std::vector<double> e2e_ms;
    std::vector<double> decode_ms;
    std::vector<double> infer_ms;
    std::vector<double> encode_ms;

    bool init() {
        if (initialized) {
            set_error("pipeline is already initialized");
            return false;
        }
        if (cleaned) {
            set_error("pipeline has already been shut down");
            return false;
        }

        stop_requested = false;
        failed = false;
        last_error.clear();

        source = std::make_unique<io::FFmpegVideoSource>();
        if (!source->open(cfg.video_path)) {
            set_error("failed to open video: " + source->last_error());
            cleanup();
            return false;
        }

        const io::VideoInfo& info = source->info();
        src_width = info.width;
        src_height = info.height;
        src_fps = (info.fps > 0.0) ? info.fps : 30.0;
        if (src_width <= 0 || src_height <= 0) {
            set_error("video has invalid dimensions");
            cleanup();
            return false;
        }

        if (!backend.TensorRTLoader(cfg.engine_path)) {
            set_error("failed to load TensorRT engine: " + cfg.engine_path);
            cleanup();
            return false;
        }

        const auto& input_names = backend.getInputNames();
        const auto& output_names = backend.getOutputNames();
        if (input_names.empty() || output_names.empty()) {
            set_error("engine has no input/output tensors");
            cleanup();
            return false;
        }

        input_name = input_names.front();
        output_name = output_names.front();
        input_dims = backend.getTensorShape(input_name);
        if (input_dims.nbDims == 4) {
            if (input_dims.d[0] < 0) {
                input_dims.d[0] = 1;
            }
            if (input_dims.d[2] > 0) {
                model_height = input_dims.d[2];
            }
            if (input_dims.d[3] > 0) {
                model_width = input_dims.d[3];
            }
        }

        stream = backend.getStream();
        input_buffer = backend.getInputBuffer(input_name);
        if (stream == nullptr || input_buffer == nullptr) {
            set_error("TensorRT stream or input buffer is null");
            cleanup();
            return false;
        }

        letterbox = computeLetterboxMeta(src_width, src_height, model_width, model_height);
        host_stride_capacity = align_stride(src_width);
        frame_capacity = host_stride_capacity * static_cast<std::size_t>(src_height);

        if (!allocate_slots()) {
            cleanup();
            return false;
        }

        if (cfg.write_video) {
            if (cfg.output_path.empty()) {
                set_error("write_video is true but output_path is empty");
                cleanup();
                return false;
            }
            const std::filesystem::path parent =
                std::filesystem::path(cfg.output_path).parent_path();
            if (!parent.empty()) {
                std::filesystem::create_directories(parent);
            }
            if (!open_writer(writer, cfg.output_path, src_fps,
                             cv::Size(src_width, src_height))) {
                set_error("failed to open output video: " + cfg.output_path);
                cleanup();
                return false;
            }
        }

        trt::TensorView in_view;
        in_view.data = input_buffer;
        in_view.dims = input_dims;
        in_view.dtype = nvinfer1::DataType::kFLOAT;
        in_view.memType = trt::MemoryType::kGPU;
        inputs.emplace(input_name, in_view);

        trt::TensorView out_view;
        out_view.dtype = nvinfer1::DataType::kFLOAT;
        out_view.memType = trt::MemoryType::kCPU;
        outputs.emplace(output_name, out_view);

        for (auto& slot : slot_storage) {
            if (!free_queue.push(slot.get())) {
                set_error("failed to seed free slot queue");
                cleanup();
                return false;
            }
        }

        display_enabled = cfg.display;
        running = true;
        initialized = true;

        decode_thread = std::thread(&Impl::decodeWorker, this);
        infer_thread = std::thread(&Impl::inferWorker, this);
        encode_thread = std::thread(&Impl::encodeWorker, this);
        return true;
    }

    bool run() {
        if (!initialized) {
            if (!init()) {
                return false;
            }
        }

        if (decode_thread.joinable()) {
            decode_thread.join();
        }
        if (infer_thread.joinable()) {
            infer_thread.join();
        }
        if (encode_thread.joinable()) {
            encode_thread.join();
        }

        running = false;
        cleanup();
        return !failed;
    }

    void stop() {
        if (cleaned) {
            return;
        }

        stop_requested = true;
        free_queue.close();
        infer_queue.close();
        encode_queue.close();

        if (decode_thread.joinable()) {
            decode_thread.join();
        }
        if (infer_thread.joinable()) {
            infer_thread.join();
        }
        if (encode_thread.joinable()) {
            encode_thread.join();
        }

        running = false;
        cleanup();
    }

    AsyncPipelineStats snapshot_stats() const {
        std::lock_guard<std::mutex> lock(stats_mutex);
        AsyncPipelineStats snapshot = stats;
        snapshot.end_to_end_ms_p50 = median(e2e_ms);
        snapshot.end_to_end_ms_p95 = percentile95(e2e_ms);
        snapshot.decode_ms_p50 = median(decode_ms);
        snapshot.decode_ms_p95 = percentile95(decode_ms);
        snapshot.infer_ms_p50 = median(infer_ms);
        snapshot.infer_ms_p95 = percentile95(infer_ms);
        snapshot.encode_ms_p50 = median(encode_ms);
        snapshot.encode_ms_p95 = percentile95(encode_ms);
        return snapshot;
    }

    std::string error_message() const {
        std::lock_guard<std::mutex> lock(state_mutex);
        return last_error;
    }

private:
    void set_error(std::string message) {
        std::lock_guard<std::mutex> lock(state_mutex);
        if (last_error.empty()) {
            last_error = std::move(message);
        }
    }

    bool fail(std::string message) {
        failed = true;
        set_error(std::move(message));
        stop_requested = true;
        free_queue.close();
        infer_queue.close();
        encode_queue.close();
        return false;
    }

    bool allocate_slots() {
        slot_storage.reserve(static_cast<std::size_t>(std::max(1, cfg.slot_count)));
        for (int i = 0; i < std::max(1, cfg.slot_count); ++i) {
            auto slot = std::make_unique<FrameSlot>();
            slot->id = i;
            slot->width = src_width;
            slot->height = src_height;
            slot->h_bgr_capacity = frame_capacity;
            slot->d_bgr_capacity = frame_capacity;

            if (cudaMallocHost(reinterpret_cast<void**>(&slot->h_bgr), frame_capacity) !=
                cudaSuccess) {
                set_error("cudaMallocHost failed for FrameSlot");
                return false;
            }
            if (cudaMalloc(reinterpret_cast<void**>(&slot->d_bgr), frame_capacity) !=
                cudaSuccess) {
                cudaFreeHost(slot->h_bgr);
                slot->h_bgr = nullptr;
                set_error("cudaMalloc failed for FrameSlot");
                return false;
            }
            slot_storage.push_back(std::move(slot));
        }
        return true;
    }

    void release_slots() {
        for (auto& slot : slot_storage) {
            if (slot->h_bgr != nullptr) {
                cudaFreeHost(slot->h_bgr);
                slot->h_bgr = nullptr;
            }
            if (slot->d_bgr != nullptr) {
                cudaFree(slot->d_bgr);
                slot->d_bgr = nullptr;
            }
        }
        slot_storage.clear();
    }

    void cleanup() {
        if (cleaned) {
            return;
        }

        if (source) {
            source->close();
            source.reset();
        }
        if (writer.isOpened()) {
            writer.release();
        }
        release_slots();
        backend.resetResources();
        inputs.clear();
        outputs.clear();
        cleaned = true;
    }

    void decodeWorker() {
        int frames_read = 0;
        while (!stop_requested) {
            if (cfg.max_frames > 0 && frames_read >= cfg.max_frames) {
                break;
            }

            FrameSlot* slot = nullptr;
            if (!free_queue.pop(slot)) {
                break;
            }

            const auto start = Clock::now();
            io::VideoFrame frame;
            const io::ReadStatus status = source->read(frame);
            if (status == io::ReadStatus::kEndOfStream) {
                free_queue.push(slot);
                break;
            }
            if (status == io::ReadStatus::kError) {
                free_queue.push(slot);
                fail("video decode failed: " + source->last_error());
                break;
            }

            const std::size_t bytes =
                static_cast<std::size_t>(frame.stride) * static_cast<std::size_t>(frame.height);
            if (frame.width != src_width || frame.height != src_height ||
                bytes > slot->h_bgr_capacity) {
                free_queue.push(slot);
                fail("frame dimensions/stride changed or exceed slot capacity");
                break;
            }

            std::memcpy(slot->h_bgr, frame.data, bytes);
            slot->frame_id = frame.frame_id;
            slot->pts_us = frame.pts_us;
            slot->src_stride = static_cast<std::size_t>(frame.stride);
            slot->start_time = start;
            slot->decode_ms = elapsed_ms(start, Clock::now());
            slot->infer_ms = 0.0;
            slot->encode_ms = 0.0;
            slot->e2e_ms = 0.0;

            {
                std::lock_guard<std::mutex> lock(stats_mutex);
                ++stats.decoded_frames;
            }

            if (!infer_queue.push(slot)) {
                free_queue.push(slot);
                break;
            }
            ++frames_read;
        }

        infer_queue.close();
    }

    void inferWorker() {
        while (!stop_requested) {
            FrameSlot* slot = nullptr;
            if (!infer_queue.pop(slot)) {
                break;
            }

            const auto start = Clock::now();
            const std::size_t bytes =
                slot->src_stride * static_cast<std::size_t>(slot->height);

            const cudaError_t copy_status =
                cudaMemcpyAsync(slot->d_bgr, slot->h_bgr, bytes, cudaMemcpyHostToDevice,
                                stream);
            if (copy_status != cudaSuccess) {
                fail(std::string("H2D failed: ") + cudaGetErrorString(copy_status));
                break;
            }

            if (!fused_k1::launchFusedPreprocess(
                    static_cast<const std::uint8_t*>(slot->d_bgr),
                    static_cast<float*>(input_buffer), letterbox, stream, slot->src_stride)) {
                fail("GPU preprocess launch failed");
                break;
            }

            if (!backend.TensorInfer(inputs, outputs)) {
                fail("TensorRT inference failed");
                break;
            }

            slot->infer_ms = elapsed_ms(start, Clock::now());

            std::vector<Detection> detections =
                postprocessing::PostProcessing_yolo(outputs, letterbox, cfg.conf_threshold,
                                                    cfg.iou_threshold);

            cv::Mat canvas = slot->wrapAsMat();
            draw_detections(canvas, detections, slot->frame_id);

            {
                std::lock_guard<std::mutex> lock(stats_mutex);
                ++stats.inferred_frames;
                stats.total_detections += detections.size();
            }

            if (!encode_queue.push(slot)) {
                free_queue.push(slot);
                break;
            }
        }

        encode_queue.close();
    }

    void encodeWorker() {
        while (!stop_requested) {
            FrameSlot* slot = nullptr;
            if (!encode_queue.pop(slot)) {
                break;
            }

            const auto start = Clock::now();
            if (writer.isOpened()) {
                writer.write(slot->wrapAsMat());
            }

            if (display_enabled) {
                try {
                    cv::imshow("road_damage_async", slot->wrapAsMat());
                    const int key = cv::waitKey(1);
                    if (key == 27 || key == 'q') {
                        stop_requested = true;
                    }
                } catch (const cv::Exception& e) {
                    std::fprintf(stderr, "display disabled: %s\n", e.what());
                    display_enabled = false;
                }
            }

            const auto end = Clock::now();
            slot->encode_ms = elapsed_ms(start, end);
            slot->e2e_ms = elapsed_ms(slot->start_time, end);

            {
                std::lock_guard<std::mutex> lock(stats_mutex);
                const bool measured =
                    stats.encoded_frames >= static_cast<std::uint64_t>(cfg.warmup_frames);
                if (measured) {
                    e2e_ms.push_back(slot->e2e_ms);
                    decode_ms.push_back(slot->decode_ms);
                    infer_ms.push_back(slot->infer_ms);
                    encode_ms.push_back(slot->encode_ms);
                }
                ++stats.encoded_frames;
            }

            if (!free_queue.push(slot)) {
                break;
            }
        }

        free_queue.close();
    }
};

AsyncPipeline::AsyncPipeline(AsyncPipelineConfig config)
    : impl_(std::make_unique<Impl>(std::move(config))) {}

AsyncPipeline::~AsyncPipeline() {
    if (impl_) {
        impl_->stop();
    }
}

bool AsyncPipeline::init() {
    return impl_ ? impl_->init() : false;
}

bool AsyncPipeline::run() {
    return impl_ ? impl_->run() : false;
}

void AsyncPipeline::stop() {
    if (impl_) {
        impl_->stop();
    }
}

AsyncPipelineStats AsyncPipeline::stats() const {
    return impl_ ? impl_->snapshot_stats() : AsyncPipelineStats{};
}

std::string AsyncPipeline::last_err() const {
    return impl_ ? impl_->error_message() : std::string("pipeline is null");
}

}  // namespace rd::pipeline

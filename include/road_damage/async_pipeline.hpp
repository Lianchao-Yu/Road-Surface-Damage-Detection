#pragma once
#include <cstdint>
#include <memory>
#include <string>

namespace rd::pipeline
{
    struct AsyncPipelineConfig
    {
        std::string video_path;
        std::string engine_path = "models/best.engine";
        std::string output_path;

        float conf_threshold = 0.25f;
        float iou_threshold = 0.45f;

        int slot_count = 4;
        int decode_queue_capacity = 4;
        int encode_queue_capacity = 4;
        int warmup_frames = 5;
        int max_frames = 0; // 0 = all
        bool write_video = true;
        bool display = false;
    };

    struct AsyncPipelineStats
    {
        std::uint64_t decoded_frames = 0;
        std::uint64_t inferred_frames = 0;
        std::uint64_t encoded_frames = 0;
        std::uint64_t dropped_frames = 0;
        std::uint64_t total_detections = 0;
        double end_to_end_ms_p50 = 0.0;
        double end_to_end_ms_p95 = 0.0;
        double decode_ms_p50 = 0.0;
        double decode_ms_p95 = 0.0;
        double infer_ms_p50 = 0.0;
        double infer_ms_p95 = 0.0;
        double encode_ms_p50 = 0.0;
        double encode_ms_p95 = 0.0;
    };
    class AsyncPipeline
    {
    public:
        explicit AsyncPipeline(AsyncPipelineConfig config);
        ~AsyncPipeline();
        AsyncPipeline(const AsyncPipeline &) = delete;
        AsyncPipeline &operator=(const AsyncPipeline &) = delete;
        // Pipeline owns threads and queues; moving it is unsafe.
        AsyncPipeline(AsyncPipeline &&) = delete;
        AsyncPipeline &operator=(AsyncPipeline &&) = delete;

        bool init();
        void stop();
        bool run();

        AsyncPipelineStats stats() const;
        std::string last_err() const;

    private:
        struct Impl;
        std::unique_ptr<Impl> impl_;
    };
}

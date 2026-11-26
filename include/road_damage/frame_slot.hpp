#pragma once

#include <chrono>
#include <cstddef>
#include <cstdint>

#include <opencv2/core.hpp>
#include <cuda_runtime.h>

/// Per-frame context passed between the decoder, GPU worker and encoder.
///
/// The slot owns the pinned host frame and the device staging buffer. TensorRT
/// input/output buffers are shared backend resources and intentionally not
/// duplicated here for the phase-1 pipeline (one inference worker).
struct FrameSlot
{
    int id = 0;
    std::uint64_t frame_id = 0;
    std::int64_t pts_us = 0;

    int width = 0;
    int height = 0;
    std::size_t src_stride = 0;

    // Pinned host BGR24 frame, written by the decoder.
    std::uint8_t* h_bgr = nullptr;
    std::size_t h_bgr_capacity = 0;

    // Device staging buffer for H2D / GPU preprocess input.
    std::uint8_t* d_bgr = nullptr;
    std::size_t d_bgr_capacity = 0;

    // Stage timings in milliseconds.
    double decode_ms = 0.0;
    double infer_ms = 0.0;
    double encode_ms = 0.0;
    double e2e_ms = 0.0;
    std::chrono::steady_clock::time_point start_time{};

    cv::Mat wrapAsMat() const
    {
        return cv::Mat(height, width, CV_8UC3, h_bgr, src_stride);
    }
};

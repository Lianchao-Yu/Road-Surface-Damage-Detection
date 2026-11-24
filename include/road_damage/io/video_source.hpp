#pragma once

#include <cstdint>
#include <memory>
#include <string>

namespace rd::io {

/// Result of a single frame read.
enum class ReadStatus {
    kOk,           ///< A frame was written to the output.
    kEndOfStream,  ///< No more frames; the decoder has been drained.
    kError,        ///< Decoding failed; see last_error().
};

/// One decoded frame in packed BGR24 (8 bit per channel).
///
/// `data` points into memory owned by the source and stays valid until the
/// next call to read() or close(). Callers that need to keep the frame must
/// copy it. `stride` is the number of bytes between rows and may be larger
/// than width * 3 because of alignment. It can be passed directly to the GPU
/// preprocess kernel, which supports a source row stride.
struct VideoFrame {
    const std::uint8_t* data = nullptr;
    int width = 0;
    int height = 0;
    int stride = 0;
    std::int64_t pts_us = 0;   ///< Presentation timestamp in microseconds.
    std::uint64_t frame_id = 0;  ///< Monotonic output frame counter.
};

/// Static information about the opened stream.
struct VideoInfo {
    int width = 0;
    int height = 0;
    double fps = 0.0;
    std::int64_t duration_us = 0;
    std::string codec;
    std::string format;
};

/// Abstract frame source.
///
/// The first implementation is FFmpegVideoSource (software decode). Keeping
/// the interface separate leaves room for RTSP-specific sources or a future
/// NVDEC implementation without changing the inference pipeline.
class IFrameSource {
public:
    virtual ~IFrameSource() = default;

    /// Opens a local file or network URL. Replaces any previously open stream.
    virtual bool open(const std::string& url) = 0;

    /// Reads the next decoded frame. At EOF returns kEndOfStream repeatedly.
    virtual ReadStatus read(VideoFrame& frame) = 0;

    /// Releases all decoder resources. Safe to call multiple times.
    virtual void close() = 0;

    /// Metadata captured at open() time.
    virtual const VideoInfo& info() const = 0;

    /// Human-readable description of the last failure.
    virtual const std::string& last_error() const = 0;
};

/// FFmpeg-based software video source.
///
/// Basic structure:
///   libavformat -> demux MP4/RTSP
///   libavcodec  -> H.264/H.265 software decode
///   libswscale  -> YUV420P (or other software format) to BGR24
///
/// The source is not thread-safe. One instance should be used by one decoding
/// thread. RTSP input defaults to TCP transport for more stable behaviour on
/// lossy networks; timeout/reconnect policy will be added with the RTSP
/// milestone.
class FFmpegVideoSource final : public IFrameSource {
public:
    FFmpegVideoSource();
    ~FFmpegVideoSource() override;

    FFmpegVideoSource(const FFmpegVideoSource&) = delete;
    FFmpegVideoSource& operator=(const FFmpegVideoSource&) = delete;
    FFmpegVideoSource(FFmpegVideoSource&&) = delete;
    FFmpegVideoSource& operator=(FFmpegVideoSource&&) = delete;

    bool open(const std::string& url) override;
    ReadStatus read(VideoFrame& frame) override;
    void close() override;
    const VideoInfo& info() const override;
    const std::string& last_error() const override;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace rd::io

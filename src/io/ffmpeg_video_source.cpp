#include "road_damage/io/video_source.hpp"

extern "C" {
#include <libavcodec/avcodec.h>
#include <libavformat/avformat.h>
#include <libavutil/dict.h>
#include <libavutil/error.h>
#include <libavutil/imgutils.h>
#include <libavutil/pixdesc.h>
#include <libavutil/rational.h>
#include <libswscale/swscale.h>
}

#include <cmath>
#include <cerrno>
#include <cstdio>
#include <string>

namespace rd::io {
namespace {

// Preferred BGR24 row alignment. The GPU preprocess kernel accepts any stride
// >= width * 3, so aligned rows avoid an extra repack after sws_scale.
constexpr int kDstAlignment = 32;

bool is_rtsp_url(const std::string& url) {
    return url.rfind("rtsp://", 0) == 0 || url.rfind("rtsps://", 0) == 0;
}

}  // namespace

struct FFmpegVideoSource::Impl {
    AVFormatContext* format_ctx = nullptr;
    AVCodecContext* codec_ctx = nullptr;
    SwsContext* sws_ctx = nullptr;
    AVPacket* packet = nullptr;
    AVFrame* frame = nullptr;

    // BGR24 destination for sws_scale. av_image_alloc() fills all four
    // pointers/linesizes; only plane 0 is used for packed BGR24.
    std::uint8_t* bgr_data[4] = {};
    int bgr_linesize[4] = {};
    int bgr_width = 0;
    int bgr_height = 0;

    int video_stream = -1;
    AVRational time_base{0, 1};  // Stream time base, used for PTS conversion.
    bool opened = false;

    // Demux/decode state. FFmpeg's send/receive API may return EAGAIN, so a
    // packet is kept until the decoder accepts it, and the decoder is flushed
    // with a null packet after demux EOF.
    bool packet_pending = false;
    bool demux_eof = false;
    bool flush_sent = false;

    std::uint64_t frame_id = 0;
    std::int64_t last_pts_us = 0;
    bool have_last_pts = false;

    VideoInfo info;
    std::string last_error;

    ~Impl() { close(); }

    void set_error(const std::string& what, int err) {
        last_error = what;
        if (err < 0) {
            char buffer[AV_ERROR_MAX_STRING_SIZE] = {};
            if (av_strerror(err, buffer, sizeof(buffer)) == 0) {
                last_error += ": ";
                last_error += buffer;
            }
        }
    }

    void close() {
        opened = false;

        if (bgr_data[0] != nullptr) {
            av_freep(&bgr_data[0]);
        }
        for (auto& pointer : bgr_data) {
            pointer = nullptr;
        }
        for (auto& linesize : bgr_linesize) {
            linesize = 0;
        }
        bgr_width = 0;
        bgr_height = 0;

        if (sws_ctx != nullptr) {
            sws_freeContext(sws_ctx);
            sws_ctx = nullptr;
        }

        av_frame_free(&frame);
        av_packet_free(&packet);
        avcodec_free_context(&codec_ctx);

        if (format_ctx != nullptr) {
            avformat_close_input(&format_ctx);
        }

        video_stream = -1;
        time_base = AVRational{0, 1};
        packet_pending = false;
        demux_eof = false;
        flush_sent = false;
        frame_id = 0;
        last_pts_us = 0;
        have_last_pts = false;
        info = VideoInfo{};
    }

    bool open(const std::string& url) {
        // Start from a clean state. last_error is cleared below, after close(),
        // because close() intentionally does not touch the error string.
        close();
        last_error.clear();

        if (url.empty()) {
            set_error("empty video URL", 0);
            return false;
        }

        AVDictionary* options = nullptr;
        if (is_rtsp_url(url)) {
            // TCP avoids most packet loss / reordering artifacts for the first
            // RTSP implementation. Timeouts and reconnects belong to the RTSP
            // milestone and are intentionally not hidden here.
            av_dict_set(&options, "rtsp_transport", "tcp", 0);
        }

        int ret = avformat_open_input(&format_ctx, url.c_str(), nullptr, &options);
        av_dict_free(&options);
        if (ret < 0) {
            set_error("avformat_open_input(" + url + ")", ret);
            close();
            return false;
        }

        ret = avformat_find_stream_info(format_ctx, nullptr);
        if (ret < 0) {
            set_error("avformat_find_stream_info", ret);
            close();
            return false;
        }

        const AVCodec* decoder = nullptr;
        video_stream = av_find_best_stream(format_ctx, AVMEDIA_TYPE_VIDEO, -1, -1,
                                           &decoder, 0);
        if (video_stream < 0) {
            set_error("av_find_best_stream(video)", video_stream);
            close();
            return false;
        }
        if (decoder == nullptr) {
            set_error("no decoder found for video stream", 0);
            close();
            return false;
        }

        AVStream* stream = format_ctx->streams[video_stream];
        codec_ctx = avcodec_alloc_context3(decoder);
        if (codec_ctx == nullptr) {
            set_error("avcodec_alloc_context3", AVERROR(ENOMEM));
            close();
            return false;
        }

        ret = avcodec_parameters_to_context(codec_ctx, stream->codecpar);
        if (ret < 0) {
            set_error("avcodec_parameters_to_context", ret);
            close();
            return false;
        }

        // Let libavcodec choose the thread count (0 = auto); frame + slice
        // threading keeps software decode fast without coupling to a core
        // count. This is still a software decode path.
        codec_ctx->pkt_timebase = stream->time_base;
        codec_ctx->thread_count = 0;
        codec_ctx->thread_type = FF_THREAD_FRAME | FF_THREAD_SLICE;

        ret = avcodec_open2(codec_ctx, decoder, nullptr);
        if (ret < 0) {
            const std::string decoder_name =
                (decoder->name != nullptr) ? decoder->name : "unknown";
            set_error("avcodec_open2(" + decoder_name + ")", ret);
            close();
            return false;
        }

        packet = av_packet_alloc();
        frame = av_frame_alloc();
        if (packet == nullptr || frame == nullptr) {
            set_error("av_packet_alloc / av_frame_alloc", AVERROR(ENOMEM));
            close();
            return false;
        }

        time_base = stream->time_base;
        info.width = codec_ctx->width;
        info.height = codec_ctx->height;

        const AVRational guessed_fps = av_guess_frame_rate(format_ctx, stream, nullptr);
        double fps = av_q2d(guessed_fps);
        if (!(fps > 0.0)) {
            fps = av_q2d(stream->avg_frame_rate);
        }
        info.fps = (fps > 0.0) ? fps : 0.0;

        if (stream->duration != AV_NOPTS_VALUE && stream->duration > 0) {
            info.duration_us =
                av_rescale_q(stream->duration, stream->time_base, AVRational{1, AV_TIME_BASE});
        } else if (format_ctx->duration != AV_NOPTS_VALUE && format_ctx->duration > 0) {
            info.duration_us = format_ctx->duration;
        } else {
            info.duration_us = 0;
        }

        info.codec = (decoder->name != nullptr) ? decoder->name : "";
        info.format =
            (format_ctx->iformat != nullptr && format_ctx->iformat->name != nullptr)
                ? format_ctx->iformat->name
                : "";

        opened = true;
        return true;
    }

    bool convert_and_fill(AVFrame& decoded, VideoFrame& out) {
        const int width = (decoded.width > 0) ? decoded.width : codec_ctx->width;
        const int height = (decoded.height > 0) ? decoded.height : codec_ctx->height;
        if (width <= 0 || height <= 0) {
            set_error("decoded frame has invalid dimensions", 0);
            return false;
        }

        const auto decoded_format = static_cast<AVPixelFormat>(decoded.format);
        // The YUVJ* formats are deprecated full-range aliases. Normalize them
        // to the modern format and carry the full-range flag explicitly so
        // swscale does not emit a deprecation warning and applies the correct
        // range conversion.
        AVPixelFormat source_format = decoded_format;
        int forced_full_range = -1;
        switch (decoded_format) {
            case AV_PIX_FMT_YUVJ420P:
                source_format = AV_PIX_FMT_YUV420P;
                forced_full_range = 1;
                break;
            case AV_PIX_FMT_YUVJ422P:
                source_format = AV_PIX_FMT_YUV422P;
                forced_full_range = 1;
                break;
            case AV_PIX_FMT_YUVJ444P:
                source_format = AV_PIX_FMT_YUV444P;
                forced_full_range = 1;
                break;
            case AV_PIX_FMT_YUVJ440P:
                source_format = AV_PIX_FMT_YUV440P;
                forced_full_range = 1;
                break;
            case AV_PIX_FMT_YUVJ411P:
                source_format = AV_PIX_FMT_YUV411P;
                forced_full_range = 1;
                break;
            default:
                break;
        }

        const AVPixFmtDescriptor* descriptor = av_pix_fmt_desc_get(source_format);
        if (descriptor == nullptr) {
            set_error("unknown decoded pixel format", 0);
            return false;
        }
        if ((descriptor->flags & AV_PIX_FMT_FLAG_HWACCEL) != 0) {
            set_error("hardware frames are not supported by the software source", 0);
            return false;
        }

        if (bgr_width != width || bgr_height != height) {
            if (bgr_data[0] != nullptr) {
                av_freep(&bgr_data[0]);
            }
            for (auto& pointer : bgr_data) {
                pointer = nullptr;
            }
            for (auto& linesize : bgr_linesize) {
                linesize = 0;
            }

            const int alloc_ret =
                av_image_alloc(bgr_data, bgr_linesize, width, height, AV_PIX_FMT_BGR24,
                               kDstAlignment);
            if (alloc_ret < 0) {
                set_error("av_image_alloc(BGR24)", alloc_ret);
                return false;
            }
            bgr_width = width;
            bgr_height = height;
        }

        // sws_getCachedContext rebuilds the converter only when the source
        // dimensions/format change, which also covers mid-stream resolution
        // changes without leaking the old context.
        sws_ctx = sws_getCachedContext(sws_ctx, width, height, source_format, width, height,
                                       AV_PIX_FMT_BGR24, SWS_BILINEAR, nullptr, nullptr,
                                       nullptr);
        if (sws_ctx == nullptr) {
            set_error("sws_getCachedContext(YUV->BGR24)", AVERROR(ENOMEM));
            return false;
        }

        // YUV -> BGR24 needs explicit range/matrix metadata. RGB sources are
        // already full range and do not use a YUV conversion matrix.
        if ((descriptor->flags & AV_PIX_FMT_FLAG_RGB) == 0) {
            int source_range = 0;  // most encoded video is limited range (MPEG)
            if (forced_full_range >= 0) {
                source_range = forced_full_range;
            } else if (decoded.color_range == AVCOL_RANGE_JPEG) {
                source_range = 1;
            }

            int colorspace = SWS_CS_DEFAULT;
            switch (decoded.colorspace) {
                case AVCOL_SPC_BT709:
                    colorspace = SWS_CS_ITU709;
                    break;
                case AVCOL_SPC_BT470BG:
                case AVCOL_SPC_SMPTE170M:
                    colorspace = SWS_CS_ITU601;
                    break;
                case AVCOL_SPC_BT2020_NCL:
                case AVCOL_SPC_BT2020_CL:
                    colorspace = SWS_CS_BT2020;
                    break;
                default:
                    // HD content normally uses BT.709; SD content uses BT.601.
                    colorspace = (height >= 720 && width >= 1280) ? SWS_CS_ITU709
                                                                  : SWS_CS_ITU601;
                    break;
            }

            const int* coefficients = sws_getCoefficients(colorspace);
            if (coefficients == nullptr ||
                sws_setColorspaceDetails(sws_ctx, coefficients, source_range, coefficients,
                                         1, 0, 1 << 16, 1 << 16) < 0) {
                set_error("sws_setColorspaceDetails", 0);
                return false;
            }
        }

        const int rows =
            sws_scale(sws_ctx, decoded.data, decoded.linesize, 0, height, bgr_data,
                      bgr_linesize);
        if (rows <= 0) {
            set_error("sws_scale(YUV->BGR24)", 0);
            return false;
        }

        // best_effort_timestamp is already in display order and in the stream
        // time base. If the container omits it, fall back to frame->pts and,
        // as a last resort, to a synthetic frame interval.
        std::int64_t pts = decoded.best_effort_timestamp;
        if (pts == AV_NOPTS_VALUE) {
            pts = decoded.pts;
        }

        std::int64_t pts_us = 0;
        if (pts != AV_NOPTS_VALUE) {
            pts_us = av_rescale_q(pts, time_base, AVRational{1, AV_TIME_BASE});
            last_pts_us = pts_us;
            have_last_pts = true;
        } else if (have_last_pts && info.fps > 0.0) {
            const auto interval_us =
                static_cast<std::int64_t>(std::llround(1'000'000.0 / info.fps));
            pts_us = last_pts_us + interval_us;
            last_pts_us = pts_us;
        }

        out.data = bgr_data[0];
        out.width = width;
        out.height = height;
        out.stride = bgr_linesize[0];
        out.pts_us = pts_us;
        out.frame_id = frame_id++;
        return true;
    }

    ReadStatus read(VideoFrame& out) {
        if (!opened) {
            set_error("read() called before open()", 0);
            return ReadStatus::kError;
        }

        for (;;) {
            // 1. Drain all frames the decoder can produce from the packets it
            //    already has. avcodec_receive_frame() may return EAGAIN when it
            //    needs another packet.
            av_frame_unref(frame);
            int ret = avcodec_receive_frame(codec_ctx, frame);
            if (ret == 0) {
                return convert_and_fill(*frame, out) ? ReadStatus::kOk
                                                     : ReadStatus::kError;
            }
            if (ret == AVERROR_EOF) {
                return ReadStatus::kEndOfStream;
            }
            if (ret != AVERROR(EAGAIN)) {
                set_error("avcodec_receive_frame", ret);
                return ReadStatus::kError;
            }

            // 2. Demuxer is finished: flush the decoder once, then drain its
            //    remaining frames.
            if (demux_eof) {
                if (!flush_sent) {
                    ret = avcodec_send_packet(codec_ctx, nullptr);
                    if (ret == 0) {
                        flush_sent = true;
                        continue;
                    }
                    if (ret == AVERROR(EAGAIN)) {
                        // Output is still pending; receive it first.
                        continue;
                    }
                    if (ret == AVERROR_EOF) {
                        return ReadStatus::kEndOfStream;
                    }
                    set_error("avcodec_send_packet(flush)", ret);
                    return ReadStatus::kError;
                }
                return ReadStatus::kEndOfStream;
            }

            // 3. Read the next packet. Non-video packets are discarded.
            if (!packet_pending) {
                ret = av_read_frame(format_ctx, packet);
                if (ret == AVERROR_EOF) {
                    demux_eof = true;
                    continue;
                }
                if (ret < 0) {
                    set_error("av_read_frame", ret);
                    return ReadStatus::kError;
                }
                packet_pending = true;
            }

            if (packet->stream_index != video_stream) {
                av_packet_unref(packet);
                packet_pending = false;
                continue;
            }

            // 4. Feed the packet. EAGAIN means "receive output first", so keep
            //    the packet pending and loop back.
            ret = avcodec_send_packet(codec_ctx, packet);
            if (ret == 0) {
                av_packet_unref(packet);
                packet_pending = false;
            } else if (ret == AVERROR(EAGAIN)) {
                // Keep packet_ and retry after draining decoder output.
            } else if (ret == AVERROR_EOF) {
                av_packet_unref(packet);
                packet_pending = false;
                return ReadStatus::kEndOfStream;
            } else {
                av_packet_unref(packet);
                packet_pending = false;
                set_error("avcodec_send_packet", ret);
                return ReadStatus::kError;
            }
        }
    }
};

FFmpegVideoSource::FFmpegVideoSource() : impl_(std::make_unique<Impl>()) {}

FFmpegVideoSource::~FFmpegVideoSource() = default;

bool FFmpegVideoSource::open(const std::string& url) { return impl_->open(url); }

ReadStatus FFmpegVideoSource::read(VideoFrame& frame) { return impl_->read(frame); }

void FFmpegVideoSource::close() { impl_->close(); }

const VideoInfo& FFmpegVideoSource::info() const { return impl_->info; }

const std::string& FFmpegVideoSource::last_error() const { return impl_->last_error; }

}  // namespace rd::io

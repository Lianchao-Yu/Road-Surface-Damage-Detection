#include <gtest/gtest.h>

#include <cstdlib>
#include <cstdint>
#include <string>

#include "road_damage/io/video_source.hpp"

TEST(VideoSource, OpenMissingPathFails) {
    rd::io::FFmpegVideoSource source;
    EXPECT_FALSE(source.open("/tmp/road_damage_missing_video_file.mp4"));
    EXPECT_FALSE(source.last_error().empty());
}

TEST(VideoSource, EmptyUrlFails) {
    rd::io::FFmpegVideoSource source;
    EXPECT_FALSE(source.open(""));
    EXPECT_FALSE(source.last_error().empty());
}

TEST(VideoSource, ReadBeforeOpenFails) {
    rd::io::FFmpegVideoSource source;
    rd::io::VideoFrame frame;
    EXPECT_EQ(source.read(frame), rd::io::ReadStatus::kError);
    EXPECT_FALSE(source.last_error().empty());
}

TEST(VideoSource, CloseIsIdempotent) {
    rd::io::FFmpegVideoSource source;
    source.close();
    source.close();
    SUCCEED();
}

// Optional integration fixture: set RD_TEST_VIDEO to a real MP4 to verify
// decoding, stride and timestamp propagation without committing a video file.
TEST(VideoSource, DecodesOptionalFixture) {
    const char* path = std::getenv("RD_TEST_VIDEO");
    if (path == nullptr || path[0] == '\0') {
        GTEST_SKIP() << "set RD_TEST_VIDEO to a video file to enable this test";
    }

    rd::io::FFmpegVideoSource source;
    ASSERT_TRUE(source.open(path)) << source.last_error();
    EXPECT_GT(source.info().width, 0);
    EXPECT_GT(source.info().height, 0);

    rd::io::VideoFrame frame;
    std::uint64_t expected_id = 0;
    int decoded = 0;
    while (decoded < 10) {
        const rd::io::ReadStatus status = source.read(frame);
        if (status == rd::io::ReadStatus::kEndOfStream) {
            break;
        }
        ASSERT_EQ(status, rd::io::ReadStatus::kOk) << source.last_error();
        EXPECT_EQ(frame.frame_id, expected_id);
        EXPECT_GE(frame.width, 1);
        EXPECT_GE(frame.height, 1);
        EXPECT_GE(frame.stride, frame.width * 3);
        EXPECT_NE(frame.data, nullptr);
        EXPECT_GE(frame.pts_us, 0);
        ++expected_id;
        ++decoded;
    }
    EXPECT_GT(decoded, 0);
}

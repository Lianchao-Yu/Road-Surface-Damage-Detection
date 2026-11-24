#pragma once

//namespace rd{}  ,避免命名空间冲突
struct LetterboxMeta
{
    int origWidth = 0;    // Raw picture wdith
    int origHeight = 0;   // Raw picture height
    int inputWidth = 640; // CNN input width
    int inputHeight = 640; // CNN input height
    float scale = 1.0f;    // scale ratio
    float padX = 0.0f;     // Pixels padding on X axis
    float padY = 0.0f;     // pixels padding on Y axis
};

struct Detection
{
    float x1 = 0.0f;    // left-up corner x1
    float y1 = 0.0f;    // left-up corner y1
    float x2 = 0.0f;    // right-down position x2
    float y2 = 0.0f;    // right-down position y2
    float score = 0.0f; // Probability
    int classID = -1;   // classid
};


/// BGR8 host image. `stride` is the number of bytes between rows and may be
/// larger than width * 3 (FFmpeg linesize). Pixels are 3 bytes: B, G, R.
struct ImageBgr8 {
    const std::uint8_t* data = nullptr;
    int width = 0;
    int height = 0;
    int stride = 0;
};

/// Per-channel normalization. `mean` and `std` are expressed in [0, 1] units
/// and are applied after dividing the 8-bit pixel by 255.
struct Normalization {
    float mean[3] = {0.0f, 0.0f, 0.0f};
    float std[3] = {1.0f, 1.0f, 1.0f};
};


struct PreprocessSpec {
    int dst_width = 640;
    int dst_height = 640;
    bool bgr_to_rgb = true;
    Normalization norm{};
};
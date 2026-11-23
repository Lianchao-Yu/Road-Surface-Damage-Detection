#pragma once

struct LetterboxMeta
{
    int origWidth = 0;    // Raw picture wdith
    int origHeight = 0;   // Raw picture height
    int inputWidth = 640; // CNN input width
    int inputHeight = 640;// CNN input height
    float scale = 1.0f;   // scale ratio
    float padX = 0.0f;    // Pixels padding on X axis
    float padY = 0.0f;    // pixels padding on Y axis
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
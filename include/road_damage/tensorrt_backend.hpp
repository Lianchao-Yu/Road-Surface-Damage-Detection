// TensorRTBackend.hpp
// PIMPL: This file is part of the
// Road Damage Detection project.
#pragma once
#include <string>
#include <memory>
#include <unordered_map>
#include <NvInfer.h>
#include <cuda_runtime_api.h>
#include <vector>
namespace trt
{
    enum class MemoryType
    {
        kGPU, 
        kCPU
    };

    struct TensorView //Tensor input and outoput structure
    {
        void *data = nullptr;                                  // data pointer
        nvinfer1::Dims dims{};                                 // tensor dimention
        nvinfer1::DataType dtype = nvinfer1::DataType::kFLOAT; // default data type is fp32
        MemoryType memType = MemoryType::kCPU;                 // memory type
    };

    class TensorRTBackend
    {
    public:
        TensorRTBackend();  // constructor
        ~TensorRTBackend(); // destructor

        TensorRTBackend(const TensorRTBackend &) = delete;            // delete copy constructor
        TensorRTBackend &operator=(const TensorRTBackend &) = delete; // delete copy assignment
        TensorRTBackend(TensorRTBackend &&) noexcept;                 // move constructor
        TensorRTBackend &operator=(TensorRTBackend &&) noexcept;      // move assignment
        bool TensorRTLoader(const std::string &enginePath);           // function to load TensorRT engine from a serialized file
        // bool Tensor_Infer(const std::vector<float> &hostInput, std::vector<float> &hostOutput); //function to conduct infer
        void resetResources();
        bool TensorInfer(const std::unordered_map<std::string, TensorView> &input, std::unordered_map<std::string, TensorView> &output); 

        // CUDA stream owned by the backend. External GPU preprocessing must
        // enqueue H2D / kernels on this stream so that TensorInfer() sees the
        // preprocessed tensor in stream order without an extra synchronization.
        // Returns nullptr before TensorRTLoader() succeeds.
        cudaStream_t getStream() const;

        // Device buffer preallocated by TensorRTLoader() for an input tensor.
        // It is sized for the engine's maximum profile shape. Writing the
        // preprocessed tensor directly into this buffer avoids a second
        // cudaMalloc and lets TensorInfer() consume it with MemoryType::kGPU.
        // Returns nullptr for unknown/non-input names or before loading.
        void *getInputBuffer(const std::string &name) const;
    private:
        struct Impl;  // forward declaration of implementation class
        std::unique_ptr<Impl> pImpl; // pointer to implementation
    };
}
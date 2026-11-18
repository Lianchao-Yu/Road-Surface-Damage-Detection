/*
The purpose of this file is to provide a C++ interface for the TensorRT
backend. It includes functions and classes that facilitate the integration
of TensorRT into the project, allowing for efficient inference on
NVIDIA GPUs.

work flow:
1. Load the TensorRT engine from a serialized file or build it from an
   ONNX model.
2. Create an execution context for the engine
3. Prepare input data and allocate memory for inputs and outputs.
4. Execute inference using the execution context
*/
#include "tensorrt_backend.hpp"
#include <cuda_runtime_api.h>
#include <memory>
#include <fstream>
#include <iostream>
#include <unordered_map>
#include <NvInfer.h>
#include <vector>

namespace trt
{
   class TRTLooger : public nvinfer1::ILogger // Custom logger class for TensorRT
   {
      void log(Severity severity, const char *msg) noexcept override
      {
         if (severity <= Severity::kWARNING)
         {
            std::cerr << "[TRT] " << msg << std::endl;
         }
      }
   };

   struct TensorRTBackend::Impl
   {
      int deviceId = 0;             // ID of the GPU device to use
      bool loaded = false;          // flag indicating if the engine is loaded
      cudaStream_t stream{nullptr}; // CUDA stream for asynchronous execution

      TRTLooger logger;                                              // 1. 最先创建，最后销毁
      std::unique_ptr<nvinfer1::IRuntime> runtime{nullptr};          // 2. 其次
      std::unique_ptr<nvinfer1::ICudaEngine> engine{nullptr};        // 3. 再次
      std::unique_ptr<nvinfer1::IExecutionContext> context{nullptr}; // 4. 最后创建，最先销毁

      std::vector<std::string> inputTensorsNames;            // save input tensor names
      std::vector<std::string> outputTensorsNames;           // save output tensor names
      std::unordered_map<std::string, void *> deviceBuffers; // save device buffers for input and output tensors
      std::unordered_map<std::string, size_t> bufferSizes;   // save device buffersize for input and output tensors

      ~Impl() //release cuda resource
      {
         for (auto &kv : deviceBuffers)
         {
            if (kv.second != nullptr)
            {
               cudaFree(kv.second);
            }
         }
         if (stream != nullptr)
         {
            cudaStreamDestroy(stream);
         }
      }
   };
   TensorRTBackend::TensorRTBackend() : pImpl(std::make_unique<Impl>()) {} // constructor implementation
   TensorRTBackend::~TensorRTBackend() = default;                          // destructor implementation

   bool TensorRTBackend::TensorRTLoader(const std::string &enginePath)
   {
      // 1. Load the TensorRT engine from a serialized file
      // 2. Create an execution context for the engine
      // 3. Prepare input data and allocate memory for inputs and outputs.
      //=============== 1. read the serialized engine file into a vector of chars ===============

      std::vector<char> engineData;
      // Open the engine file in binary mode and move the file pointer to the end
      std::ifstream engineFile(enginePath, std::ios::binary | std::ios::ate);
      if (!engineFile.is_open())
      {
         std::cerr << "Failed to open engine file: " << enginePath << std::endl;
         return false;
      }

      // Get the size of the engine file
      std::streamsize engineSize = engineFile.tellg();
      if (engineSize <= 0)
      {
         std::cerr << "Engine file is empty or invalid: " << enginePath << std::endl;
         return false;
      }
      // Resize the vector to hold the engine data
      engineData.resize(engineSize);

      // move the file pointer back to the beginning and read the engine data into the vector
      engineFile.seekg(0, std::ios::beg);

      if (!engineFile.read(engineData.data(), engineSize))
      {
         std::cerr << "Failed to read engine file: " << enginePath << std::endl;
         return false;
      }

      engineFile.close();

      //=============== 2. Create an execution context for the engine ===============
      // 1. 指定 GPU 设备
      cudaSetDevice(pImpl->deviceId);

      // 2. 创建 Runtime
      pImpl->runtime.reset(nvinfer1::createInferRuntime(pImpl->logger));
      if (!pImpl->runtime)
         return false;

      // 3. 反序列化生成 Engine
      pImpl->engine.reset(pImpl->runtime->deserializeCudaEngine(engineData.data(), engineData.size()));
      if (!pImpl->engine)
         return false;

      // 4. 创建 Execution Context （创建执行上下文）
      pImpl->context.reset(pImpl->engine->createExecutionContext());
      if (!pImpl->context)
         return false;

      //=============== 3. Prepare input data and allocate memory for inputs and outputs ===============

      // create a CUDA stream for asynchronous execution
      if (pImpl->stream != nullptr)
      {
         cudaStreamDestroy(pImpl->stream);
         pImpl->stream = nullptr;
      }

      // create a new CUDA stream
      if (cudaStreamCreate(&pImpl->stream) != cudaSuccess)
      {
         std::cerr << "Failed to create CUDA stream!" << std::endl;
         return false;
      }

      // 接下来继续写遍历张量、cudaMalloc 和 setTensorAddress 的代码...

      for (int i = 0; i < pImpl->engine->getNbIOTensors(); i++)
      {
         auto name = pImpl->engine->getIOTensorName(i);
         if (pImpl->engine->getTensorIOMode(name) == nvinfer1::TensorIOMode::kINPUT)
         {
            // Handle input tensor
            pImpl->inputTensorsNames.push_back(name);
         }
         if (pImpl->engine->getTensorIOMode(name) == nvinfer1::TensorIOMode::kOUTPUT)
         {
            // Handle output tensor
            pImpl->outputTensorsNames.push_back(name);
         }

         // Allocate device memory for the tensor
         void *devicePtr = nullptr;
         size_t tensorVolume = 1;
         auto dims = pImpl->engine->getTensorShape(name);
         auto dataType = pImpl->engine->getTensorDataType(name); // Get the data type of the tensor
         for (int j = 0; j < dims.nbDims; j++)
         {
            if (dims.d[j] < 0) // invalid tensor input
            {
               std::cerr << "Dynamic shape is not supported yet for tensor: " << name << std::endl;
               return false;
            }
            tensorVolume *= dims.d[j];
         }
         auto typeSize = (dataType == nvinfer1::DataType::kFLOAT) ? sizeof(float) :
                         (dataType == nvinfer1::DataType::kHALF) ? 2 :
                         (dataType == nvinfer1::DataType::kINT8) ? sizeof(int8_t) :
                         (dataType == nvinfer1::DataType::kINT32) ? sizeof(int32_t) : 0;
         if (typeSize == 0) 
         {
            std::cerr << "Unsupported data type for tensor: " << name << std::endl;
            return false;
         }
         size_t tensorSize = tensorVolume * typeSize;
         if (cudaMalloc(&devicePtr, tensorSize) != cudaSuccess)
         {
            std::cerr << "Failed to allocate device memory for tensor: " << name << std::endl;
            return false;
         }
         pImpl->context->setTensorAddress(name, devicePtr);     // Set the tensor address in the execution context
         pImpl->deviceBuffers[name] = devicePtr;            // Save the device buffer pointer
         pImpl->bufferSizes[name] = tensorSize;             // Save the size of the buffer for later use
      }
      // Mark the engine as loaded successfully
      pImpl->loaded = true;
      return true;
   }

   //推理函数 (调用loader -> 启动Tensor_Infer)
   //IO结构：单输入单输出
   bool TensorRTBackend::Tensor_Infer(const std::vector<float> &hostInput, std::vector<float> &hostOutput) 
   {
      
      //=============== 1. Copy H2D ===============
      if(!pImpl->loaded)
      {
         std::cerr << "Engine is not loaded. Cannot execute inference." << std::endl;
         return false;
      }

      const std::string &inName = pImpl->inputTensorsNames[0];
      const std::string &outName = pImpl->outputTensorsNames[0];
      if(hostInput.size() * sizeof(float) != pImpl->bufferSizes[inName])
      {
         std::cerr << "Input data size mismatch!" << std::endl;
         return false;
      }

       //可能需要pinned memory,可以作为性能测试
      if(cudaMemcpyAsync(pImpl->deviceBuffers[inName], hostInput.data(),
      pImpl->bufferSizes[inName], cudaMemcpyHostToDevice, pImpl->stream) != cudaSuccess)
      {
         std::cerr<< "Fail to Copy" <<std::endl;
         return false;
      }

      //============ 2. Execute Infer
      if(!pImpl->context->enqueueV3(pImpl->stream))
      {
        std::cerr << "TensorRT enqueueV3 failed!" << std::endl;
        return false;
      }

      //============ 3. Copy D2H (GPU -> CPU)
      hostOutput.resize(pImpl->bufferSizes[outName] / sizeof(float));
      if (cudaMemcpyAsync(hostOutput.data(), pImpl->deviceBuffers[outName],
                          pImpl->bufferSizes[outName], cudaMemcpyDeviceToHost, pImpl->stream) != cudaSuccess)
      {
         return false;
      }

      // ============ 4. Synchronize Stream ============
      if (cudaStreamSynchronize(pImpl->stream) != cudaSuccess)
      {
         std::cerr << "CUDA stream synchronize failed!" << std::endl;
         return false;
      }

      return true;
   }
};
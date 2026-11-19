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
   };
   TensorRTBackend::TensorRTBackend() : pImpl(std::make_unique<Impl>()) {} // constructor implementation
   TensorRTBackend::~TensorRTBackend() { resetResources(); };            // destructor implementation

   bool TensorRTBackend::TensorRTLoader(const std::string &enginePath)
   {
      // 1. Load the TensorRT engine from a serialized file
      // 2. Create an execution context for the engine
      // 3. Malloc memory for inputs and outputs. (根据输入维度进行动态调整分配显存)
      //PS: 动态维度即，单个张量的尺寸可变。example: [1, 1, 640, 640]  -> [1, 1, 800, 800]
      //如果onnx生成的engine支持动态维度则会以 -1占位。 example: [1, 1, -1, -1], 这里-1表示Bidth和Width可变

      //=============== 0. release resource first
      resetResources();
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
      cudaError_t devErr = cudaSetDevice(pImpl->deviceId);
      if (devErr != cudaSuccess)
      {
         std::cerr << "Failed to set CUDA device " << pImpl->deviceId
                   << ": " << cudaGetErrorString(devErr) << std::endl;
         return false;
      }

      // 2.创建runtime
      pImpl->runtime.reset(nvinfer1::createInferRuntime(pImpl->logger));
      if (!pImpl->runtime)
      {
         std::cerr << "Failed to create TensorRT Runtime!" << std::endl;
         return false;
      }

      //3. 创建engine,反序列化
      pImpl->engine.reset(pImpl->runtime->deserializeCudaEngine(engineData.data(), engineData.size()));
      if (!pImpl->engine)
      {
         std::cerr << "Failed to deserialize CUDA Engine!" << std::endl;
         resetResources();
         return false;
      }

      //4, 创建上下文
      pImpl->context.reset(pImpl->engine->createExecutionContext());
      if (!pImpl->context)
      {
         std::cerr << "Failed to create Execution Context!" << std::endl;
         resetResources();
         return false;
      }

      //=============== 3. Prepare input data and allocate memory for inputs and outputs ===============
      if (pImpl->stream != nullptr)
      {
         cudaStreamDestroy(pImpl->stream);
         pImpl->stream = nullptr;

      }

      // create a new CUDA stream
      if (cudaStreamCreate(&pImpl->stream) != cudaSuccess)
      {
         std::cerr << "Failed to create CUDA stream!" << std::endl;
         resetResources();
         return false;
      }
      //Traverse the tensor and get the input and output name
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
      }

      // ========== 4.Malloc device memory for the input tensor=============================
      //Malloc input tensor memory

      for(const auto &name : pImpl->inputTensorsNames)
      {
         auto dims = pImpl->engine->getTensorShape(name.c_str());        // Get a tensor's input dimention. [N, C, H, W]
         auto dataType = pImpl->engine->getTensorDataType(name.c_str()); // Get the data type of the tensor

         // 1. 检查是否包含动态维度 (-1) 例如[-1, 1, -1, 1]
         bool hasDynamicDim = false;
         for (int j = 0; j < dims.nbDims; j++)
         {
            if (dims.d[j] < 0)
            {
               hasDynamicDim = true;
               break;
            }
         }

         // 2. assign maximum memory(如果为动态维度，则查询支持的输入维度)
         if(hasDynamicDim)
         {
            dims = pImpl->engine->getProfileShape(name.c_str(), 0, nvinfer1::OptProfileSelector::kMAX); // Set the maximum supported size of dimention
            if (!pImpl->context->setInputShape(name.c_str(), dims))
            {
               std::cerr << "Failed to set max input shape for tensor: " << name << std::endl;
               resetResources();
               return false;
            }
         }

         // 3. Malloc device memory for input tensor
         size_t tensorVolume = 1;
         for (int j = 0; j < dims.nbDims; j++)
         {
            if(dims.d[j] <= 0)
            {
               std::cerr << "Unsupported input dimention" << name << std::endl;
               resetResources();
               return false;
            }
            tensorVolume *= static_cast<size_t>(dims.d[j]); //get the tensor volume
         }

         auto typeSize = (dataType == nvinfer1::DataType::kFLOAT) ? sizeof(float) : (dataType == nvinfer1::DataType::kHALF) ? 2
                                                                                : (dataType == nvinfer1::DataType::kINT8)   ? sizeof(int8_t)
                                                                                : (dataType == nvinfer1::DataType::kINT32)  ? sizeof(int32_t)
                                                                                                                            : 0;
         //Find unsupported type
         if (typeSize == 0)
         {
            std::cerr << "Unsupported data type for tensor: " << name << std::endl;
            resetResources();
            return false;
         }

         // Calculate the size and malloc
         size_t tensorSize = tensorVolume * typeSize;
         void *devicePtr = nullptr;
         if (cudaMalloc(&devicePtr, tensorSize) != cudaSuccess)
         {
            std::cerr << "Failed to allocate device memory for tensor: " << name << std::endl;
            resetResources();
            return false;
         }

         if (!pImpl->context->setTensorAddress(name.c_str(), devicePtr)) // Set the tensor address in the execution context
         {
            cudaFree(devicePtr);
            std::cerr << "Failed to set address for output tensor: " << name << std::endl;
            resetResources();
            return false;
         }
         pImpl->deviceBuffers[name] = devicePtr;            // Save the device buffer pointer
         pImpl->bufferSizes[name] = tensorSize;             // Save the size of the buffer for later use
      }

      // ============ 5. Malloc device memory for output tensor =============

      for (const auto &name : pImpl->outputTensorsNames)
      {
         auto dims = pImpl->context->getTensorShape(name.c_str());        // Get a tensor's output dimention. [N, C, H, W]
         auto dataType = pImpl->engine->getTensorDataType(name.c_str()); // Get the data type of the tensor

         //get the tensor volume
         size_t tensorVolume = 1; 
         for (int j = 0; j < dims.nbDims; j++)
         {
            if(dims.d[j] <= 0)
            {
               std::cerr << "Unsupported output dimention" << name << std::endl;
               resetResources();
               return false;
            }
            tensorVolume *= static_cast<size_t>(dims.d[j]); 
         }

         //get the dimention's data type
         auto typeSize = (dataType == nvinfer1::DataType::kFLOAT) ? sizeof(float) : (dataType == nvinfer1::DataType::kHALF) ? 2
                                                                                : (dataType == nvinfer1::DataType::kINT8)   ? sizeof(int8_t)
                                                                                : (dataType == nvinfer1::DataType::kINT32)  ? sizeof(int32_t)
                                                                                                                            : 0;
         // Find unsupported type
         if (typeSize == 0)
         {
            std::cerr << "Unsupported data type for tensor: " << name << std::endl;
            resetResources();
            return false;
         }

         // Calculate the size and malloc
         size_t tensorSize = tensorVolume * typeSize;
         void *devicePtr = nullptr;
         if (cudaMalloc(&devicePtr, tensorSize) != cudaSuccess)
         {
            std::cerr << "Failed to allocate device memory for tensor: " << name << std::endl;
            resetResources();
            return false;
         }

         if (!pImpl->context->setTensorAddress(name.c_str(), devicePtr)) // Set the tensor address in the execution context
         {
            cudaFree(devicePtr);
            std::cerr << "Failed to set address for output tensor: " << name << std::endl;
            resetResources();
            return false;
         }
         pImpl->deviceBuffers[name] = devicePtr;            // Save the device buffer pointer
         pImpl->bufferSizes[name] = tensorSize;             // Save the size of the buffer for later use
      }

      // =============== Mark the engine as loaded successfully ================
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

   // reset resource
   void TensorRTBackend::resetResources()
   {
      /*
      类支持移动语义（Move），当一个 TensorRTBackend 对象被 std::move 走之后，
      原对象的 pImpl 会变成 nullptr。此时原对象析构如果调用 resetResources()，
      直接访问 pImpl->context 会触发空指针解引用（Segfault）。需要加上判空语句
      */

      //判空保护
      if(pImpl == nullptr) return;
      
      // release tensor object
      pImpl->context.reset();
      pImpl->engine.reset();
      pImpl->runtime.reset();

      // release buffer
      for (auto &kv : pImpl->deviceBuffers)
      {
         if (kv.second != nullptr)
         {
            cudaFree(kv.second);
         }
      }

      // release vector and map resource
      if (pImpl->inputTensorsNames.size() != 0)
         pImpl->inputTensorsNames.clear();
      if (pImpl->outputTensorsNames.size() != 0)
         pImpl->outputTensorsNames.clear();
      if (pImpl->bufferSizes.size() != 0)
         pImpl->bufferSizes.clear();
      if (pImpl->deviceBuffers.size() != 0)
         pImpl->deviceBuffers.clear();

      // release stream
      if (pImpl->stream != nullptr)
      {
         cudaStreamDestroy(pImpl->stream);
         pImpl->stream = nullptr;
      }
   }
};
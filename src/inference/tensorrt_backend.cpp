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
#include "road_damage/tensorrt_backend.hpp"
#include <cuda_runtime_api.h>
#include <algorithm>
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
      std::unordered_map<std::string, void*> hostBuffers;    // save host buffer
   };
   TensorRTBackend::TensorRTBackend() : pImpl(std::make_unique<Impl>()) {} // constructor implementation
   TensorRTBackend::~TensorRTBackend() { resetResources(); };              // destructor implementation
   TensorRTBackend::TensorRTBackend(TensorRTBackend &&) noexcept = default;
   TensorRTBackend &TensorRTBackend::operator=(TensorRTBackend &&) noexcept = default;

   bool TensorRTBackend::TensorRTLoader(const std::string &enginePath)
   {
      // 1. Load the TensorRT engine from a serialized file
      // 2. Create an execution context for the engine
      // 3. Malloc memory for inputs and outputs. (根据输入维度进行动态调整分配显存)
      // PS: 动态维度即，单个张量的尺寸可变。example: [1, 1, 640, 640]  -> [1, 1, 800, 800]
      // 如果onnx生成的engine支持动态维度则会以 -1占位。 example: [1, 1, -1, -1], 这里-1表示Bidth和Width可变.
      //检测到如果engine支持动态维度则查询支持的最大维度，并以最大维度的size分配显存

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

      // 3. 创建engine,反序列化
      pImpl->engine.reset(pImpl->runtime->deserializeCudaEngine(engineData.data(), engineData.size()));
      if (!pImpl->engine)
      {
         std::cerr << "Failed to deserialize CUDA Engine!" << std::endl;
         resetResources();
         return false;
      }

      // 4, 创建上下文
      pImpl->context.reset(pImpl->engine->createExecutionContext());
      if (!pImpl->context)
      {
         std::cerr << "Failed to create Execution Context!" << std::endl;
         resetResources();
         return false;
      }

      //=============== 3. Classify input/outoput tensor type ===============
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
      // Traverse the tensor and get the input and output name
      for (int i = 0; i < pImpl->engine->getNbIOTensors(); i++)
      {
         auto name = pImpl->engine->getIOTensorName(i);
         if (pImpl->engine->getTensorIOMode(name) == nvinfer1::TensorIOMode::kINPUT)  //save input tensor name
         {
            // Handle input tensor
            pImpl->inputTensorsNames.push_back(name);
         }
         if (pImpl->engine->getTensorIOMode(name) == nvinfer1::TensorIOMode::kOUTPUT)  //save outpout tensor name
         {
            // Handle output tensor
            pImpl->outputTensorsNames.push_back(name);
         }
      }

      // ========== 4.Malloc device memory for the input tensor=============================
      // Malloc input tensor memory

      for (const auto &name : pImpl->inputTensorsNames)
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

         // 2. assign maximum memory(如果为动态维度，则查询支持的输入维度并设置最大维度)
         if (hasDynamicDim)
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
            if (dims.d[j] <= 0)
            {
               std::cerr << "Unsupported input dimention" << name << std::endl;
               resetResources();
               return false;
            }
            tensorVolume *= static_cast<size_t>(dims.d[j]); // get the tensor volume
         }

         //Get the typeSize, and calculate size based on different levels of precision
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
         pImpl->deviceBuffers[name] = devicePtr; // Save the device buffer pointer
         pImpl->bufferSizes[name] = tensorSize;  // Save the size of the buffer for later use
      }

      // ============ 5. Malloc device memory for output tensor =============

      for (const auto &name : pImpl->outputTensorsNames)
      {
         auto dims = pImpl->context->getTensorShape(name.c_str());       // Get a tensor's output dimention. [N, C, H, W]
         auto dataType = pImpl->engine->getTensorDataType(name.c_str()); // Get the data type of the tensor

         // get the tensor volume
         size_t tensorVolume = 1;
         for (int j = 0; j < dims.nbDims; j++)
         {
            if (dims.d[j] <= 0)
            {
               std::cerr << "Unsupported output dimention" << name << std::endl;
               resetResources();
               return false;
            }
            tensorVolume *= static_cast<size_t>(dims.d[j]);
         }

         // get the dimention's data type
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
         void *hostPtr = nullptr;

         //malloc output memory on GPU
         if (cudaMalloc(&devicePtr, tensorSize) != cudaSuccess)
         {
            std::cerr << "Failed to allocate device memory for tensor: " << name << std::endl;
            resetResources();
            return false;
         }
         // Malloc pinned memory on host
         if (cudaMallocHost(&hostPtr, tensorSize) != cudaSuccess)
         {
            std::cerr << "Failed to allocate host memory for tensor: " << name << std::endl;
            resetResources();
            return false;
         }

         // Set the tensor address in the execution context
         if (!pImpl->context->setTensorAddress(name.c_str(), devicePtr)) 
         {
            cudaFree(devicePtr);
            std::cerr << "Failed to set address for output tensor: " << name << std::endl;
            resetResources();
            return false;
         }
         //
         pImpl->deviceBuffers[name] = devicePtr; // Save the device buffer pointer
         pImpl->bufferSizes[name] = tensorSize;  // Save the size of the buffer for later use
         pImpl->hostBuffers[name] = hostPtr;
      }

      // =============== Mark the engine as loaded successfully ================
      pImpl->loaded = true;
      return true;
   }

   bool TensorRTBackend::TensorInfer(const std::unordered_map<std::string, trt::TensorView> &input,
                                     std::unordered_map<std::string, trt::TensorView> &output)
   /*
       1. 从输入 TensorView 获取本次实际输入维度 (dims)
                     |
      2. 设置输入动态维度到 context (setInputShape)
                     |
            3. 计算实际维度大小，异步拷贝显存(H2D)
                     |
               4. 异步启动推理
                     |
       5. 从 context 获取推导后的输出维度并回填，异步拷回结果 (D2H)
                     |
         6. 同步 CUDA Stream，等待所有任务完成
   */

   {
      //=============== 1. Copy H2D or D2D============================
      if (!pImpl || !pImpl->loaded)
      {
         std::cerr << "Engine is not loaded. Cannot execute inference." << std::endl;
         return false;
      }
      cudaSetDevice(pImpl->deviceId); // set GPU ID

      // Lambda funtion, compute the tensor actual byte
      auto computeTensorBytes = [](const TensorView &view) -> size_t
      {
         if (view.dims.nbDims <= 0)
         {
            return 0;
         }

         size_t volume = 1;
         for (int i = 0; i < view.dims.nbDims; ++i)
         {
            if (view.dims.d[i] <= 0)
            {
               return 0; // 动态维度尚未解析为正整数
            }
            volume *= static_cast<size_t>(view.dims.d[i]);
         }

         size_t typeSize = 0;
         switch (view.dtype)
         {
         case nvinfer1::DataType::kFLOAT:
            typeSize = sizeof(float);
            break;
         case nvinfer1::DataType::kHALF:
            typeSize = 2;
            break;
         case nvinfer1::DataType::kINT8:
            typeSize = sizeof(int8_t);
            break;
         case nvinfer1::DataType::kINT32:
            typeSize = sizeof(int32_t);
            break;
         default:
            return 0; // 不支持的数据类型
         }

         return volume * typeSize;
      };

      //============= Copy data from host/device to device ========================
      for (const auto &name : pImpl->inputTensorsNames)
      {
         auto tensorIn = input.find(name);
         if (tensorIn == input.end() || tensorIn->second.data == nullptr)
         {
            std::cerr << "Error: Missing required input tensor in map: " << name << std::endl;
            return false;
         }

         const TensorView &Inputview = tensorIn->second;
         // 显式设置本次实际维度（同时支持动态 Batch 和动态 H/W）
         if (!pImpl->context->setInputShape(name.c_str(), Inputview.dims))
         {
            std::cerr << "Invalid input shape for: " << name << std::endl;
            return false;
         }

         size_t actualBytes = computeTensorBytes(Inputview); // 计算实际size
         if (actualBytes == 0 || actualBytes > pImpl->bufferSizes[name])
         {
            std::cerr << "Input buffer overflow or invalid size for: " << name << std::endl;
            return false;
         }

         // ==== Async copy memory, H2D ==========
         // cudaMemcpyKind kind = (Inputview.memType == MemoryType::kCPU)
         //                           ? cudaMemcpyHostToDevice
         //                           : cudaMemcpyDeviceToDevice;

         // ======== case 1: GPU 2 GPU ==============
         if (Inputview.memType == MemoryType::kGPU)
         {
            if (!pImpl->context->setTensorAddress(name.c_str(), Inputview.data))
            {
               return false;
            }
         }
         // ======== case 2: GPU 2 CPU ==============
         else if (Inputview.memType == MemoryType::kCPU)
         {
            if (!pImpl->context->setTensorAddress(name.c_str(), pImpl->deviceBuffers[name]))
            {
               return false;
            }

            if (cudaMemcpyAsync(pImpl->deviceBuffers[name],
                                 Inputview.data, actualBytes, cudaMemcpyHostToDevice, pImpl->stream) != cudaSuccess)
            {
               return false;
            }
         }
         else
         {
            std::cerr << "Unsupported memory type for input: " << name << std::endl;
            return false;
         }
      }

      //============= 2. Start infer========================
      // ps: stream执行顺序： H2D copy -> Infer -> D2H, 需要绑定同一个stream

      // 1. 绑定stream
      if (!pImpl->context->enqueueV3(pImpl->stream))
      {
         std::cerr << "enqueueV3 failed!" << std::endl;
         return false;
      }

      //==============3. Finish infer, copy back===============

      // 获取推导后的输出维度，将结果拷回调用方提供的输出缓冲区
      for (const auto &name : pImpl->outputTensorsNames)
      {

         // 回填本次实际推导出的输出维度，方便上层做后处理解析
         TensorView &OutputView = output[name];
         OutputView.dims = pImpl->context->getTensorShape(name.c_str());    // Get the dimmention from context
         OutputView.dtype = pImpl->engine->getTensorDataType(name.c_str()); // Check the actual output datatype
         size_t actualBytes = computeTensorBytes(OutputView);               // Calculate outupt actual byteszie
         void *hostPtr = pImpl->hostBuffers[name];                          // Host pointer

         if (actualBytes == 0 || actualBytes > pImpl->bufferSizes[name])
         {
            std::cerr << "Output buffer overflow for: " << name << std::endl;
            return false;
         }

         // copy back to host
         if (OutputView.memType == MemoryType::kCPU) 
         {
            if (cudaMemcpyAsync(hostPtr, pImpl->deviceBuffers[name],
                                actualBytes, cudaMemcpyDeviceToHost, pImpl->stream) != cudaSuccess)
            {
               return false;
            }
            OutputView.data = hostPtr;
         }
         // copy back to GPU
         else if (OutputView.memType == MemoryType::kGPU) 
         {
            //Case B: GPU hanvn't malloc memory 
            if (OutputView.data == nullptr)
            {
               OutputView.data = pImpl->deviceBuffers[name];
            }
            //case A: GPU malloc a new memory area
            else if (OutputView.data != pImpl->deviceBuffers[name])
            {
               if (cudaMemcpyAsync(OutputView.data, pImpl->deviceBuffers[name],
                                   actualBytes, cudaMemcpyDeviceToDevice, pImpl->stream) != cudaSuccess)
               {
                  return false;
               }
            }
         }
         else
         {
            std::cerr << "Failed to copy the output" << name << std::endl;
            return false;
         }
      }

      // 4. 同步流，确保数据就绪
      return cudaStreamSynchronize(pImpl->stream) == cudaSuccess;
   }

   cudaStream_t TensorRTBackend::getStream() const
   {
      return (pImpl != nullptr) ? pImpl->stream : nullptr;
   }

   void *TensorRTBackend::getInputBuffer(const std::string &name) const
   {
      if (pImpl == nullptr)
         return nullptr;

      // Only expose actual input tensors; returning an output buffer here
      // would silently allow preprocessing to overwrite inference results.
      if (std::find(pImpl->inputTensorsNames.begin(), pImpl->inputTensorsNames.end(),
                    name) == pImpl->inputTensorsNames.end())
      {
         return nullptr;
      }

      const auto it = pImpl->deviceBuffers.find(name);
      return (it != pImpl->deviceBuffers.end()) ? it->second : nullptr;
   }

   // Reset resource
   void TensorRTBackend::resetResources()
   {
      /*
      类支持移动语义（Move），当一个 TensorRTBackend 对象被 std::move 走之后，
      原对象的 pImpl 会变成 nullptr。此时原对象析构如果调用 resetResources()，
      直接访问 pImpl->context 会触发空指针解引用（Segfault）。需要加上判空语句
      */

      // 判空保护
      if (pImpl == nullptr)
         return;

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

      // Output host buffers are allocated with cudaMallocHost() and must be
      // released with cudaFreeHost(); plain cudaFree() is not valid for pinned
      // host memory and previously leaked these allocations.
      for (auto &kv : pImpl->hostBuffers)
      {
         if (kv.second != nullptr)
         {
            cudaFreeHost(kv.second);
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
      if (pImpl->hostBuffers.size() != 0)
         pImpl->hostBuffers.clear();

      // release stream
      if (pImpl->stream != nullptr)
      {
         cudaStreamDestroy(pImpl->stream);
         pImpl->stream = nullptr;
      }
   }
};
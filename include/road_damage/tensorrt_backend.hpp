// TensorRTBackend.hpp
// PIMPL: This file is part of the
// Road Damage Detection project.
#include <string>
#include <memory>
#include <vector>
namespace trt
{
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
        bool Tensor_Infer(const std::vector<float> &hostInput, std::vector<float> &hostOutput); //function to conduct infer
    private:
        struct Impl;  // forward declaration of implementation class
        std::unique_ptr<Impl> pImpl; // pointer to implementation
    };
}
#include "road_damage/frame_slot.hpp"
#include "road_damage/async_pipeline.hpp"
#include "road_damage/tensorrt_backend.hpp"
#include "road_damage/io/video_source.hpp"
#include <thread>
#include "road_damage/frame_queue.hpp"
#include <opencv2/videoio.hpp>
#include "road_damage/shared_data.hpp"

namespace rd::pipeline
{

    struct AsyncPipeline::Impl
    {

        AsyncPipelineConfig cfg;
        std::unique_ptr<io::FFmpegVideoSource> source;
        trt::TensorRTBackend backend;
        cv::VideoWriter writer;
        std::vector<std::unique_ptr<FrameSlot>> slot_storage;
        std::unique_ptr<io::FFmpegVideoSource> source;
        std::string last_err;
        LetterboxMeta letterbox;
        
        //Source image/video dimention
        int src_width = 0;
        int src_height = 0;
        double src_fps = 30.0;

        //Model input scale
        int kModel_Width = 640;
        int kModel_Height = 640;

        //Model input & output name
        std::string inputTensorName;
        std::string outputTensorName;

        //Model input & output dimention
        nvinfer1::Dims input_dims{};
        nvinfer1::Dims output_dims{};
        


        frame_Queue<FrameSlot *> free_queue;
        frame_Queue<FrameSlot *> infer_queue;
        frame_Queue<FrameSlot *> encode_queue;

        std::thread decode_thread;
        std::thread infer_thread;
        std::thread encode_thread;

        void decodeWorker();
        void inferWorker();
        void encodeWorker();

        explicit Impl(AsyncPipelineConfig config)
            : cfg(std::move(config)),
              free_queue(static_cast<std::size_t>(std::max(1, cfg.slot_count))),
              infer_queue(static_cast<std::size_t>(std::max(1, cfg.decode_queue_capacity))),
              encode_queue(static_cast<std::size_t>(std::max(1, cfg.encode_queue_capacity)))
        {
        }

        // ============= function =============
        void stop()
        {
            free_queue.stop();
            infer_queue.stop();
            encode_queue.stop();

            if (decode_thread.joinable()) // wait until finished
                decode_thread.join();
            if (infer_thread.joinable()) // wait
                infer_thread.join();
            if (encode_thread.joinable()) // wait
                encode_thread.join();
        }

        ~Impl()
        {
            stop();
        }

        // function

        // Init() implementation
        bool init()
        {
            // 1. open video source
            source = std::make_unique<io::FFmpegVideoSource>();
            if (!source->open(cfg.video_path))
            {
                last_err = "Failed to open video source: " + source->last_error();
                return false;
            }

            const io::VideoInfo &vinfo = source->info();
            src_width = vinfo.width;
            src_height = vinfo.height;
            src_fps = (vinfo.fps > 1.0) ? vinfo.fps : 30.0;
            letterbox = computeLetterboxMeta(src_width, src_height, kModel_Width, kModel_Height);

            // 2. Load TensorRT engine
            if (!backend.TensorRTLoader(cfg.engine_path))
            {
                last_err = "Failed to load TensorRT engine: " + cfg.engine_path;
                return false;
            }

            // get io tensor name
            const auto &in_names = backend.getInputNames();
            const auto &out_names = backend.getOutputNames();
            if (in_names.empty() || out_names.empty())
            {
                last_err = "TensorRT engine has no input or output tensors";
                return false;
            }

            inputTensorName = in_names[0];
            outputTensorName = out_names[0];

            input_dims = backend.getTensorShape(inputTensorName);
            output_dims = backend.getTensorShape(outputTensorName);
            if (input_dims.nbDims > 0 && input_dims.d[0] < 0)
                input_dims.d[0] = 1;
            if (output_dims.nbDims > 0 && output_dims.d[0] < 0)
                output_dims.d[0] = 1;

                
            if (!backend.getStream() || !backend.getInputBuffer(inputTensorName))
            {
                last_err = "TensorRT stream or input buffer '" + std::string(inputTensorName) + "' is null";
                return false;
            }

            // 3. Turn on the video encode 
            if (cfg.write_video)
            {
                if (cfg.output_path.empty())
                {
                    last_err = "write_video is true but output_path is empty";
                    return false;
                }
                int fourcc = cv::VideoWriter::fourcc('m', 'p', '4', 'v');
                if (!writer.open(cfg.output_path, fourcc, src_fps, cv::Size(src_width, src_height), true))
                {
                    last_err = "Failed to open VideoWriter: " + cfg.output_path;
                    return false;
                }
            }

            //
        }

        // Run() implementation
        bool run()
        {
        }
        // stop() implementation
        void stop() {

        };
    };

    // ============ Out layer forwarding ==============

    // stop
    void AsyncPipeline::stop()
    {
        if (impl_)
            impl_->stop();
    }

    // constructor
    AsyncPipeline::AsyncPipeline(AsyncPipelineConfig config)
        : impl_(std::make_unique<Impl>(std::move(config)))
    {
    }

    // Destructor
    AsyncPipeline::~AsyncPipeline() = default;

    // init
    bool AsyncPipeline::init()
    {
        return impl_ ? impl_->init() : false;
    }
    // run
    bool AsyncPipeline::run()
    {
        return impl_ ? impl_->run() : false;
    }
    // stop
    void AsyncPipeline::stop()
    {
        if (impl_)
            impl_->stop();
    }
}

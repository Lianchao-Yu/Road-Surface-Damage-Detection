#pragma once
#include "road_damage/postprocess_cpu.hpp"
#include <algorithm>
namespace postprocessing
{
    namespace
    {
        float Compute_IoU(const Detection &a, const Detection &b)
        {
            float X1 = std::max(a.x1, b.x1); // left-up X
            float X2 = std::min(a.x2, b.x2); // right-down X
            float Y1 = std::max(a.y1, b.y1); // left-up Y
            float Y2 = std::min(a.y2, b.y2); // right-down Y
            float interW = std::max(0.0f, X2 - X1);
            float interH = std::max(0.0f, Y2 - Y1);

            float overlap =interW * interH; //Calculate overlapping area
            if(overlap <= 0 ) return 0.0f;

            float areaA = (a.x2 - a.x1) * (a.y2 - a.y1); 
            float areaB = (b.x2 - b.x1) * (b.y2 - b.y1);
            float UnionArea = areaA + areaB - overlap;
            return (UnionArea>0)? overlap/UnionArea : 0.0f;
        }
        std::vector<Detection> NMS(std::vector<Detection> & proposal, float iouThreshold)
        {
            if(proposal.empty()) return{}; //return type is vector, thus cannot purely "return"

            //Sort
            std::sort(proposal.begin(), proposal.end(), [](const Detection &a, const Detection &b){return a.score > b.score;});
            std::vector<Detection> result;         //save the result
            result.reserve(proposal.size());
            std::vector<bool> suppression(proposal.size(), false);  //save suppression status
            
            for(int i = 0; i< proposal.size(); i++)
            {
                if(suppression[i]) continue;
                result.push_back(proposal[i]); //Add the un-suppressed box
                for(int j=i+1; j<proposal.size(); j++)
                {
                    if(suppression[j]) continue;
                    if(proposal[i].classID != proposal[j].classID) continue;
                    if(Compute_IoU(proposal[i], proposal[j]) >= iouThreshold)
                    {
                        suppression[j] = true;
                    }
                }                  
            }
            //Optimization            
            // int active = proposal.size();
            // for(int i = 0; i < active; i++){
            //     int write = i+1;
            //     result.push_back(proposal[i]);
            //     for(int j=i+1; j<active; j++){
            //         if(proposal[i].classID != proposal[j].classID || Compute_IoU(proposal[i], proposal[j]) < iouThreshold)
                    
            //         {
            //             if(write !=j) proposal[write]=proposal[j];
            //             write++;
            //         }
                    
            //     }
            //     active=write;
            // }
            
            return result;
        }

        
        std::vector<Detection> PostProcessing_yolo(std::unordered_map<std::string, trt::TensorView> OutputView, const LetterboxMeta &letterbox,
                                                   float confThreshold = 0.25f, float iouThreshold = 0.45f)
        {
            // ============== step 1: Filtering out the redundant bounding box =================

            // Verify input
            if (OutputView.empty())
                return {}; // if empty, return
            const auto &view = OutputView.begin()->second;

            if (view.data == nullptr ||
                view.memType != trt::MemoryType::kCPU ||
                view.dtype != nvinfer1::DataType::kFLOAT ||
                view.dims.nbDims != 3)
            {
                return {};
            }

            // Get yolo's layout: [1, 4 + numClasses, numAnchors]
            const size_t numClasses = static_cast<int>(view.dims.d[1]) - 4;
            const size_t numAnchors = static_cast<int>(view.dims.d[2]);
            if (numClasses <= 0 || numAnchors <= 0 || letterbox.scale <= 0.0f)
            {
                return {};
            }
            const float *ptr = static_cast<const float *>(view.data);    // read data pointer
            const float invScale = 1.0f / letterbox.scale;               // inverse scale
            const float maxW = static_cast<float>(letterbox.origWidth);  // Original Width
            const float maxH = static_cast<float>(letterbox.origHeight); // Original Height
            std::vector<Detection> proposals;
            proposals.reserve(64);

            //Traverse all anchors(bbox)
            for(int i = 0; i < numAnchors; i++)
            {
                float maxScore = 0.0f;
                int bestClass = -1;
                for(int j = 0; j < numClasses; j++)
                {
                    const float score = ptr[(4 + j) * numAnchors + i];
                    if (score > maxScore)
                    {
                        maxScore = score;
                        bestClass = j;
                    }
                }
                // Filter out low confidence box
                if (maxScore < confThreshold)
                    continue;

                // Get proposed bbox
                float cx = ptr[0 * numAnchors + i]; // Central x
                float cy = ptr[1 * numAnchors + i]; // Central y
                float w = ptr[2 * numAnchors + i];  // Box width
                float h = ptr[3 * numAnchors + i];  // Box height

                // 转为角点坐标 -> 减去 Letterbox 灰边偏移 -> 除以缩放比例还原回原图尺寸
                float x1 = (cx - 0.5f * w - letterbox.padX) * invScale;
                float y1 = (cy - 0.5f * h - letterbox.padY) * invScale;
                float x2 = (cx + 0.5f * w - letterbox.padX) * invScale;
                float y2 = (cy + 0.5f * h - letterbox.padY) * invScale;

                // Clamp the edge, prevent going beyond the boundaries
                x1 = std::clamp(x1, 0.0f, maxW);
                y1 = std::clamp(y1, 0.0f, maxH);
                x2 = std::clamp(x2, 0.0f, maxW);
                y2 = std::clamp(y2, 0.0f, maxH);

                //Remove zero width or height box
                if(x2 <= x1 || y2 <= y1) continue;
                proposals.push_back(Detection{x1, y1, x2, y2, maxScore, bestClass});
            }
            // ============== step 2: Calculate IoU and remove overlapping box in a same class(NMS) =================
            return NMS(proposals, iouThreshold);
        }
    }
};

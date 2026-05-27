#pragma once

#include <opencv2/opencv.hpp>

namespace ISPStages
{
struct DemosaicResult
{
    cv::Mat bgr32;
    cv::Mat saturationBgr;
};

DemosaicResult demosaicBayerAverage(const cv::Mat &wbRaw32,
                                    const cv::Mat &saturationMask,
                                    int bayerPattern);
} // namespace ISPStages


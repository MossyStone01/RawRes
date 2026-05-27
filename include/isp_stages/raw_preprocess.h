#pragma once

#include <opencv2/opencv.hpp>

namespace ISPStages
{
struct RawPreprocessResult
{
    cv::Mat normalized;
    cv::Mat saturationMask;
};

RawPreprocessResult normalizeAndWhiteBalanceBayer(
    const cv::Mat &bayer16, int blackLevel, int whiteLevel, int bayerPattern,
    float redGain, float greenGain, float blueGain,
    const cv::Vec3f &highlightLinearityLimitBgr);
} // namespace ISPStages


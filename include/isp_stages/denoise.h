#pragma once

#include <opencv2/opencv.hpp>

namespace ISPStages
{
cv::Mat applyPreviewDenoiser(const cv::Mat &linearBgr, int denoiser);
} // namespace ISPStages


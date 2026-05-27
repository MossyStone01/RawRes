#pragma once

#include <opencv2/opencv.hpp>

namespace ISPStages
{
cv::Mat applyGammaCorrection(const cv::Mat &linearBgr, double gamma);

cv::Mat convertLinearBgrTo8Bit(const cv::Mat &linearBgr);
} // namespace ISPStages


#pragma once

#include <opencv2/opencv.hpp>

namespace ISPStages
{
cv::Mat convertCameraBgrToLinearBgr(const cv::Mat &cameraBgr,
                                    const cv::Matx33f &rgbCam);
} // namespace ISPStages


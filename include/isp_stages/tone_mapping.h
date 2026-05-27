#pragma once

#include <opencv2/opencv.hpp>

namespace ISPStages
{
void applyHighlightDesaturation(cv::Mat &linearBgr,
                                const cv::Mat &saturationBgr);

cv::Mat applyExposure(const cv::Mat &linearBgr, float exposureEv);

cv::Mat applyExtendedReinhardToneMap(const cv::Mat &linearBgr);
} // namespace ISPStages


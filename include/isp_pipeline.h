#pragma once

#include <opencv2/opencv.hpp>

class ISPPipeline
{
  public:
    static cv::Mat makePreview(const cv::Mat &bayer16,
                               const cv::Matx33f &rgbCam, int blackLevel,
                               int whiteLevel, double gamma = 2.2,
                               int bayerPattern = 0, float redGain = 1.0f,
                               float greenGain = 1.0f, float blueGain = 1.0f,
                               int denoiser = 0,
                               cv::Vec3f highlightLinearityLimitBgr =
                                   cv::Vec3f(0.0f, 0.0f, 0.0f));
};

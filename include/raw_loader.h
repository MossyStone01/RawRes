#pragma once

#include <opencv2/opencv.hpp>
#include <string>

struct RawImageData
{
    cv::Mat bayer16; // CV_16UC1
    int width = 0;
    int height = 0;
    int blackLevel = 0;
    int whiteLevel = 0;
    int bayerPattern = 0;
    float wbRed = 1.0f;
    float wbGreen = 1.0f;
    float wbBlue = 1.0f;
    cv::Matx33f rgbCam;
    cv::Vec3f highlightLinearityLimitBgr = cv::Vec3f(0.0f, 0.0f, 0.0f);
    cv::Mat saturationMask;
};

class RawLoader
{
  public:
    static RawImageData load(const std::string &path);
};

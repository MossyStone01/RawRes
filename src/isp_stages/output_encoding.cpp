#include "isp_stages/output_encoding.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace ISPStages
{
cv::Mat applyGammaCorrection(const cv::Mat &linearBgr, double gamma)
{
    if (linearBgr.empty() || linearBgr.type() != CV_32FC3)
    {
        throw std::runtime_error("Invalid linear BGR image for gamma correction.");
    }
    if (gamma <= 0.0)
    {
        throw std::runtime_error("Gamma must be greater than zero.");
    }

    cv::Mat gammaCorrected = linearBgr.clone();
    const float invGamma = 1.0f / static_cast<float>(gamma);

    for (int y = 0; y < gammaCorrected.rows; ++y)
    {
        cv::Vec3f *row = gammaCorrected.ptr<cv::Vec3f>(y);
        for (int x = 0; x < gammaCorrected.cols; ++x)
        {
            row[x][0] = std::pow(std::clamp(row[x][0], 0.0f, 1.0f), invGamma);
            row[x][1] = std::pow(std::clamp(row[x][1], 0.0f, 1.0f), invGamma);
            row[x][2] = std::pow(std::clamp(row[x][2], 0.0f, 1.0f), invGamma);
        }
    }

    return gammaCorrected;
}

cv::Mat convertLinearBgrTo8Bit(const cv::Mat &linearBgr)
{
    if (linearBgr.empty() || linearBgr.type() != CV_32FC3)
    {
        throw std::runtime_error("Invalid linear BGR image for 8-bit output.");
    }

    cv::Mat preview8;
    linearBgr.convertTo(preview8, CV_8UC3, 255.0);
    return preview8;
}
} // namespace ISPStages


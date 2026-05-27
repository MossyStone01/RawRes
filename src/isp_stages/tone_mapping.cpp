#include "isp_stages/tone_mapping.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace ISPStages
{
namespace
{
float luminanceFromBgr(const cv::Vec3f &bgr)
{
    const float r = std::max(0.0f, bgr[2]);
    const float g = std::max(0.0f, bgr[1]);
    const float b = std::max(0.0f, bgr[0]);

    return 0.2126f * r + 0.7152f * g + 0.0722f * b;
}

float findMaxLuminance(const cv::Mat &linearBgr)
{
    float maxLuminance = 0.0f;

    for (int y = 0; y < linearBgr.rows; ++y)
    {
        const cv::Vec3f *row = linearBgr.ptr<cv::Vec3f>(y);
        for (int x = 0; x < linearBgr.cols; ++x)
        {
            maxLuminance = std::max(maxLuminance, luminanceFromBgr(row[x]));
        }
    }

    return std::max(maxLuminance, 1.0f);
}
} // namespace

void applyHighlightDesaturation(cv::Mat &linearBgr,
                                const cv::Mat &saturationBgr)
{
    if (linearBgr.empty() || linearBgr.type() != CV_32FC3)
    {
        throw std::runtime_error(
            "Invalid linear BGR image for highlight desaturation.");
    }
    if (saturationBgr.empty() || saturationBgr.type() != CV_8UC3 ||
        saturationBgr.size() != linearBgr.size())
    {
        throw std::runtime_error(
            "Invalid saturation mask for highlight desaturation.");
    }

    for (int y = 0; y < linearBgr.rows; ++y)
    {
        cv::Vec3f *row = linearBgr.ptr<cv::Vec3f>(y);
        const cv::Vec3b *satRow = saturationBgr.ptr<cv::Vec3b>(y);

        for (int x = 0; x < linearBgr.cols; ++x)
        {
            const cv::Vec3b &sat = satRow[x];
            const int maxSaturationLevel =
                std::max({static_cast<int>(sat[0]), static_cast<int>(sat[1]),
                          static_cast<int>(sat[2])});
            if (maxSaturationLevel == 0)
            {
                continue;
            }

            cv::Vec3f &bgr = row[x];
            const float luminance = luminanceFromBgr(bgr);
            const cv::Vec3f neutral(luminance, luminance, luminance);
            const float strength = std::clamp(
                static_cast<float>(maxSaturationLevel) / 4.0f, 0.0f, 1.0f);

            bgr = bgr * (1.0f - strength) + neutral * strength;
        }
    }
}

cv::Mat applyExposure(const cv::Mat &linearBgr, float exposureEv)
{
    if (linearBgr.empty() || linearBgr.type() != CV_32FC3)
    {
        throw std::runtime_error("Invalid linear BGR image for exposure.");
    }

    cv::Mat exposed;
    linearBgr.convertTo(exposed, CV_32FC3, std::pow(2.0f, exposureEv));
    return exposed;
}

cv::Mat applyExtendedReinhardToneMap(const cv::Mat &linearBgr)
{
    if (linearBgr.empty() || linearBgr.type() != CV_32FC3)
    {
        throw std::runtime_error("Invalid linear BGR image for tone mapping.");
    }

    const float whitePoint = findMaxLuminance(linearBgr);
    const float whitePointSquared = whitePoint * whitePoint;

    cv::Mat toneMapped = linearBgr.clone();

    for (int y = 0; y < toneMapped.rows; ++y)
    {
        cv::Vec3f *row = toneMapped.ptr<cv::Vec3f>(y);

        for (int x = 0; x < toneMapped.cols; ++x)
        {
            cv::Vec3f &bgr = row[x];
            const float luminance = luminanceFromBgr(bgr);

            if (luminance <= 0.0f)
            {
                bgr = cv::Vec3f(0.0f, 0.0f, 0.0f);
                continue;
            }

            const float mappedLuminance =
                (luminance * (1.0f + luminance / whitePointSquared)) /
                (1.0f + luminance);
            const float scale = mappedLuminance / luminance;

            bgr[0] = std::clamp(bgr[0] * scale, 0.0f, 1.0f);
            bgr[1] = std::clamp(bgr[1] * scale, 0.0f, 1.0f);
            bgr[2] = std::clamp(bgr[2] * scale, 0.0f, 1.0f);
        }
    }

    return toneMapped;
}
} // namespace ISPStages


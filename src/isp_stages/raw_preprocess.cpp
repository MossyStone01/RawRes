#include "isp_stages/raw_preprocess.h"

#include "isp_stages/bayer_pattern.h"

#include <algorithm>
#include <stdexcept>

namespace ISPStages
{
RawPreprocessResult normalizeAndWhiteBalanceBayer(
    const cv::Mat &bayer16, int blackLevel, int whiteLevel, int bayerPattern,
    float redGain, float greenGain, float blueGain,
    const cv::Vec3f &highlightLinearityLimitBgr)
{
    if (bayer16.empty() || bayer16.type() != CV_16UC1)
    {
        throw std::runtime_error("Invalid Bayer image.");
    }

    RawPreprocessResult result;
    result.normalized = cv::Mat::zeros(bayer16.size(), CV_32FC1);
    result.saturationMask = cv::Mat::zeros(bayer16.size(), CV_8UC1);

    const float denom = std::max(1, whiteLevel - blackLevel);

    for (int y = 0; y < bayer16.rows; ++y)
    {
        const ushort *src = bayer16.ptr<ushort>(y);
        uchar *sat = result.saturationMask.ptr<uchar>(y);
        float *dst = result.normalized.ptr<float>(y);

        for (int x = 0; x < bayer16.cols; ++x)
        {
            const int bgrChannel = bayerChannelAt(y, x, bayerPattern);
            const float saturationLevel = saturationLevelForChannel(
                bgrChannel, highlightLinearityLimitBgr, blackLevel, whiteLevel);
            sat[x] = static_cast<float>(src[x]) >= saturationLevel ? 1 : 0;

            float v =
                static_cast<float>(src[x]) - static_cast<float>(blackLevel);
            v = std::max(0.0f, v);
            v /= denom;
            v *= whiteBalanceGainForPixel(y, x, bayerPattern, redGain,
                                          greenGain, blueGain);

            dst[x] = v;
        }
    }

    return result;
}
} // namespace ISPStages


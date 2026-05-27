#include "isp_stages/demosaic.h"

#include "isp_stages/bayer_pattern.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace ISPStages
{
namespace
{
struct ChannelSample
{
    float value = 0.0f;
    uchar saturationLevel = 0;
};

ChannelSample averageReliableBayerChannel(const cv::Mat &wbRaw32,
                                          const cv::Mat &saturationMask, int y,
                                          int x, int targetChannel,
                                          int bayerPattern)
{
    float reliableSum = 0.0f;
    int reliableCount = 0;
    float fallbackSum = 0.0f;
    int fallbackCount = 0;
    int saturatedCount = 0;

    for (int dy = -1; dy <= 1; ++dy)
    {
        const int yy = y + dy;
        if (yy < 0 || yy >= wbRaw32.rows)
        {
            continue;
        }

        const float *row = wbRaw32.ptr<float>(yy);
        const uchar *maskRow = saturationMask.ptr<uchar>(yy);

        for (int dx = -1; dx <= 1; ++dx)
        {
            const int xx = x + dx;
            if (xx < 0 || xx >= wbRaw32.cols)
            {
                continue;
            }

            if (bayerChannelAt(yy, xx, bayerPattern) != targetChannel)
            {
                continue;
            }

            fallbackSum += row[xx];
            ++fallbackCount;

            if (maskRow[xx] != 0)
            {
                ++saturatedCount;
            }
            else
            {
                reliableSum += row[xx];
                ++reliableCount;
            }
        }
    }

    const uchar saturationLevel =
        fallbackCount > 0 ? static_cast<uchar>(std::clamp(
                                static_cast<int>(std::lround(
                                    4.0f * static_cast<float>(saturatedCount) /
                                    static_cast<float>(fallbackCount))),
                                0, 4))
                          : static_cast<uchar>(0);

    if (reliableCount > 0)
    {
        return {reliableSum / static_cast<float>(reliableCount),
                saturationLevel};
    }

    return {fallbackCount > 0 ? fallbackSum / static_cast<float>(fallbackCount)
                              : 0.0f,
            saturationLevel};
}
} // namespace

DemosaicResult demosaicBayerAverage(const cv::Mat &wbRaw32,
                                    const cv::Mat &saturationMask,
                                    int bayerPattern)
{
    if (wbRaw32.empty() || wbRaw32.type() != CV_32FC1)
    {
        throw std::runtime_error("Invalid 32F Bayer image for demosaicing.");
    }
    if (saturationMask.empty() || saturationMask.type() != CV_8UC1 ||
        saturationMask.size() != wbRaw32.size())
    {
        throw std::runtime_error("Invalid saturation mask for demosaicing.");
    }

    cv::Mat bgr32 = cv::Mat::zeros(wbRaw32.size(), CV_32FC3);
    cv::Mat saturationBgr = cv::Mat::zeros(wbRaw32.size(), CV_8UC3);

    for (int y = 0; y < wbRaw32.rows; ++y)
    {
        cv::Vec3f *dst = bgr32.ptr<cv::Vec3f>(y);
        cv::Vec3b *satDst = saturationBgr.ptr<cv::Vec3b>(y);

        for (int x = 0; x < wbRaw32.cols; ++x)
        {
            for (int channel = 0; channel < 3; ++channel)
            {
                const ChannelSample sample = averageReliableBayerChannel(
                    wbRaw32, saturationMask, y, x, channel, bayerPattern);
                dst[x][channel] = sample.value;
                satDst[x][channel] = sample.saturationLevel;
            }
        }
    }

    return {bgr32, saturationBgr};
}
} // namespace ISPStages


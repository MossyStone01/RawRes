#include "isp_pipeline.h"
#include "noise_removal/guided_filter.h"
#include <algorithm>
#include <cmath>
#include <stdexcept>

// TODO : HDR 지원하기

namespace
{
struct ChannelSample
{
    float value = 0.0f;
    uchar saturationLevel = 0;
};

struct DemosaicResult
{
    cv::Mat bgr32;
    cv::Mat saturationBgr;
};

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
            const float strength =
                std::clamp(static_cast<float>(maxSaturationLevel) / 4.0f,
                           0.0f, 1.0f);

            bgr = bgr * (1.0f - strength) + neutral * strength;
        }
    }
}

float fallbackSaturationLevel(int blackLevel, int whiteLevel)
{
    return static_cast<float>(blackLevel) +
           static_cast<float>(whiteLevel - blackLevel) * 0.95f;
}

float saturationLevelForChannel(int bgrChannel,
                                const cv::Vec3f &highlightLinearityLimitBgr,
                                int blackLevel, int whiteLevel)
{
    const float metadataLimit = highlightLinearityLimitBgr[bgrChannel];
    return metadataLimit > 0.0f
               ? metadataLimit
               : fallbackSaturationLevel(blackLevel, whiteLevel);
}

int bayerChannelAt(int y, int x, int bayerPattern)
{
    const bool evenY = (y % 2 == 0);
    const bool evenX = (x % 2 == 0);

    switch (bayerPattern)
    {
    case 0: // RGGB -> R G / G B
        if (evenY && evenX)
        {
            return 2;
        }
        if (!evenY && !evenX)
        {
            return 0;
        }
        return 1;

    case 1: // BGGR -> B G / G R
        if (evenY && evenX)
        {
            return 0;
        }
        if (!evenY && !evenX)
        {
            return 2;
        }
        return 1;

    case 2: // GRBG -> G R / B G
        if (evenY && !evenX)
        {
            return 2;
        }
        if (!evenY && evenX)
        {
            return 0;
        }
        return 1;

    case 3: // GBRG -> G B / R G
        if (evenY && !evenX)
        {
            return 0;
        }
        if (!evenY && evenX)
        {
            return 2;
        }
        return 1;

    default:
        throw std::runtime_error("Invalid Bayer pattern for demosaicing.");
    }
}

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
        fallbackCount > 0
            ? static_cast<uchar>(
                  std::clamp(static_cast<int>(std::lround(
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

DemosaicResult
Demosaicing(const cv::Mat &wbRaw32, const cv::Mat &saturationMask,
            int bayerPattern) // TODO : implement other demosacing patterns
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
} // namespace

cv::Mat ISPPipeline::makePreview(const cv::Mat &bayer16,
                                 const cv::Matx33f &rgbCam, int blackLevel,
                                 int whiteLevel, double gamma, int bayerPattern,
                                 float redGain, float greenGain, float blueGain,
                                 int denoiser,
                                 cv::Vec3f highlightLinearityLimitBgr)
{
    if (bayer16.empty() || bayer16.type() != CV_16UC1)
    {
        throw std::runtime_error("Invalid Bayer image.");
    }

    // TODO : Implement active area crop

    cv::Mat normalized = cv::Mat::zeros(bayer16.size(), CV_32FC1);
    cv::Mat saturationMask = cv::Mat::zeros(bayer16.size(), CV_8UC1);

    const float denom = std::max(1, whiteLevel - blackLevel);
    // Normalise and white balance in Bayer space.
    for (int y = 0; y < bayer16.rows; ++y)
    {
        const ushort *src = bayer16.ptr<ushort>(y);
        uchar *sat = saturationMask.ptr<uchar>(y);
        float *dst = normalized.ptr<float>(y);

        for (int x = 0; x < bayer16.cols; ++x)
        {
            const int bgrChannel = bayerChannelAt(y, x, bayerPattern);
            const float saturationLevel = saturationLevelForChannel(
                bgrChannel, highlightLinearityLimitBgr, blackLevel,
                whiteLevel);
            sat[x] =
                static_cast<float>(src[x]) >= saturationLevel ? 1 : 0;

            const bool evenY = (y % 2 == 0);
            const bool evenX = (x % 2 == 0);

            float v =
                static_cast<float>(src[x]) - static_cast<float>(blackLevel);
            v = std::max(0.0f, v);
            v /= denom;

            float channelGain = greenGain;

            // WB
            switch (bayerPattern)
            {
            case 0: // RGGB -> R G / G B
                if (evenY && evenX)
                {
                    channelGain = redGain;
                }
                else if (!evenY && !evenX)
                {
                    channelGain = blueGain;
                }
                break;

            case 1: // BGGR -> B G / G R
                if (evenY && evenX)
                {
                    channelGain = blueGain;
                }
                else if (!evenY && !evenX)
                {
                    channelGain = redGain;
                }
                break;

            case 2: // GRBG -> G R / B G
                if (evenY && !evenX)
                {
                    channelGain = redGain;
                }
                else if (!evenY && evenX)
                {
                    channelGain = blueGain;
                }
                break;

            case 3: // GBRG -> G B / R G
                if (evenY && !evenX)
                {
                    channelGain = blueGain;
                }
                else if (!evenY && evenX)
                {
                    channelGain = redGain;
                }
                break;

            default:
                break;
            }

            v *= channelGain;

            dst[x] = v;
        }
    }

    // Demosaicing to camera BGR (device BGR).
    DemosaicResult demosaicResult =
        Demosaicing(normalized, saturationMask, bayerPattern);
    cv::Mat bgr32 = demosaicResult.bgr32;
    cv::Mat saturationBgr = demosaicResult.saturationBgr;

    cv::Mat correctedBgr(bgr32.size(), CV_32FC3);

    for (int y = 0; y < bgr32.rows; ++y)
    {
        cv::Vec3f *bgrRow = bgr32.ptr<cv::Vec3f>(y);
        cv::Vec3f *correctedBgrRow = correctedBgr.ptr<cv::Vec3f>(y);

        // To Linear RGB From Device BGR
        for (int x = 0; x < bgr32.cols; ++x)
        {
            cv::Vec3f bgr = bgrRow[x];
            cv::Vec3f rgb(bgr[2], bgr[1], bgr[0]);

            cv::Vec3f corrected = rgbCam * rgb;

            corrected[0] = std::max(corrected[0], 0.0f);
            corrected[1] = std::max(corrected[1], 0.0f);
            corrected[2] = std::max(corrected[2], 0.0f);

            correctedBgrRow[x][0] = corrected[2];
            correctedBgrRow[x][1] = corrected[1];
            correctedBgrRow[x][2] = corrected[0];
        }
    }

    cv::Mat denoisedImg;

    if (denoiser < 1) // No denoise
    {
        denoisedImg = correctedBgr;
    }
    else if (denoiser == 1) // guided filter
    {
        // Test code

        cv::Mat temp;
        cv::cvtColor(correctedBgr, temp, cv::COLOR_BGR2GRAY);
        denoisedImg = NoiseRemoval::GuidedFilter(3, correctedBgr, temp, 0.001f);
    }
    else if (denoiser == 2) // BM3D
    {
        denoisedImg = correctedBgr;
    }
    else
    {
        denoisedImg = correctedBgr;
    }

    applyHighlightDesaturation(denoisedImg, saturationBgr);

    // Tone mapping before gamma correction.
    denoisedImg *= pow(2.0f, 1);
    cv::Mat toneMapped = applyExtendedReinhardToneMap(denoisedImg);

    cv::Mat gammaCorrected = toneMapped.clone();

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

    cv::Mat preview7;
    gammaCorrected.convertTo(preview7, CV_8UC3, 255.0);
    // TODO : Comparing with original data and restorated data

    return preview7;
}

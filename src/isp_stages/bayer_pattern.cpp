#include "isp_stages/bayer_pattern.h"

#include <algorithm>
#include <stdexcept>

namespace ISPStages
{
namespace
{
float fallbackSaturationLevel(int blackLevel, int whiteLevel)
{
    return static_cast<float>(blackLevel) +
           static_cast<float>(whiteLevel - blackLevel) * 0.95f;
}

float whiteBalanceGainForChannel(int bgrChannel, float redGain, float greenGain,
                                 float blueGain)
{
    switch (bgrChannel)
    {
    case 0:
        return blueGain;
    case 1:
        return greenGain;
    case 2:
        return redGain;
    default:
        throw std::runtime_error("Invalid BGR channel for white balance.");
    }
}
} // namespace

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
        throw std::runtime_error("Invalid Bayer pattern.");
    }
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

float whiteBalanceGainForPixel(int y, int x, int bayerPattern, float redGain,
                               float greenGain, float blueGain)
{
    return whiteBalanceGainForChannel(
        bayerChannelAt(y, x, bayerPattern), redGain, greenGain, blueGain);
}
} // namespace ISPStages


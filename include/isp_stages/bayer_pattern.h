#pragma once

#include <opencv2/opencv.hpp>

namespace ISPStages
{
int bayerChannelAt(int y, int x, int bayerPattern);

float saturationLevelForChannel(int bgrChannel,
                                const cv::Vec3f &highlightLinearityLimitBgr,
                                int blackLevel, int whiteLevel);

float whiteBalanceGainForPixel(int y, int x, int bayerPattern, float redGain,
                               float greenGain, float blueGain);
} // namespace ISPStages


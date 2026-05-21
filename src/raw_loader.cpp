#include "raw_loader.h"

#include <algorithm>
#include <iostream>
#include <libraw/libraw.h>
#include <stdexcept>

RawImageData RawLoader::load(const std::string &path)
{
    LibRaw rawProcessor;

    int ret = rawProcessor.open_file(path.c_str());
    if (ret != LIBRAW_SUCCESS)
    {
        throw std::runtime_error("Failed to open RAW file: " + path);
    }

    ret = rawProcessor.unpack();
    if (ret != LIBRAW_SUCCESS)
    {
        throw std::runtime_error("Failed to unpack RAW file: " + path);
    }

    rawProcessor.imgdata.params.use_camera_wb = 1;

    int rawWidth = rawProcessor.imgdata.sizes.raw_width;
    int rawHeight = rawProcessor.imgdata.sizes.raw_height;

    int width = rawProcessor.imgdata.sizes.width;
    int height = rawProcessor.imgdata.sizes.height;

    int left = rawProcessor.imgdata.sizes.left_margin;
    int top = rawProcessor.imgdata.sizes.top_margin;

    std::cout << "Raw Width : " << rawWidth << std::endl;
    std::cout << "Raw Height : " << rawHeight << std::endl;
    std::cout << "Width : " << width << std::endl;
    std::cout << "Height : " << height << std::endl;

    const char *cdesc = rawProcessor.imgdata.idata.cdesc;
    int c00 = rawProcessor.COLOR(top, left);
    int c01 = rawProcessor.COLOR(top, left + 1);
    int bayerPattern;

    int blackLevel = rawProcessor.imgdata.color.black;
    int whiteLevel = rawProcessor.imgdata.color.maximum;
    const float *camMul = rawProcessor.imgdata.color.cam_mul;
    const unsigned *linearMax = rawProcessor.imgdata.color.linear_max;
    const float fallbackSaturationLevel =
        static_cast<float>(blackLevel) +
        static_cast<float>(whiteLevel - blackLevel) * 0.95f;
    const float green1Limit =
        linearMax[1] > 0 ? static_cast<float>(linearMax[1]) : 0.0f;
    const float green2Limit =
        linearMax[3] > 0 ? static_cast<float>(linearMax[3]) : 0.0f;
    const float greenLimit =
        green1Limit > 0.0f && green2Limit > 0.0f
            ? std::min(green1Limit, green2Limit)
            : std::max(green1Limit, green2Limit);
    const cv::Vec3f highlightLinearityLimitBgr(
        linearMax[2] > 0 ? static_cast<float>(linearMax[2]) : 0.0f,
        greenLimit,
        linearMax[0] > 0 ? static_cast<float>(linearMax[0]) : 0.0f);

    if (cdesc[c00] == 'R')
    {
        bayerPattern = 0; // RGGB
    }
    else if (cdesc[c00] == 'B')
    {
        bayerPattern = 1; // BGGR
    }
    else
    {
        if (cdesc[c01] == 'R')
        {
            bayerPattern = 2; // GRBG
        }
        else
        {
            bayerPattern = 3; // GBRG
        }
    }

    cv::Mat bayer(height, width, CV_16UC1);
    cv::Mat saturationMask = cv::Mat::zeros(height, width, CV_8UC1);

    ushort *rawData = rawProcessor.imgdata.rawdata.raw_image;
    if (!rawData)
    {
        throw std::runtime_error("RAW image buffer is null.");
    }

    for (int y = 0; y < height; ++y)

    {

        ushort *dst = bayer.ptr<ushort>(y);
        uchar *satMask = saturationMask.ptr<uchar>(y);

        for (int x = 0; x < width; ++x)

        {
            int rawX = x + left;

            int rawY = y + top;

            dst[x] = rawData[rawY * rawWidth + rawX];
            const int rawChannel = rawProcessor.COLOR(rawY, rawX);
            const float saturationLevel =
                linearMax[rawChannel] > 0
                    ? static_cast<float>(linearMax[rawChannel])
                    : fallbackSaturationLevel;
            satMask[x] =
                static_cast<float>(dst[x]) >= saturationLevel ? 1 : 0;
        }
    }

    float wbRed = 1.0f;
    float wbGreen = 1.0f;
    float wbBlue = 1.0f;

    if (camMul[0] > 0.0f && camMul[1] > 0.0f && camMul[2] > 0.0f &&
        camMul[3] > 0.0f)
    {
        const float greenRef = (camMul[1] + camMul[3]) / 2;
        wbRed = camMul[0] / greenRef;
        wbBlue = camMul[2] / greenRef;
    }

    wbRed = std::clamp(wbRed, 0.1f, 8.0f);
    wbGreen = std::clamp(wbGreen, 0.1f, 8.0f);
    wbBlue = std::clamp(wbBlue, 0.1f, 8.0f);

    RawImageData result;
    result.bayer16 = bayer;
    result.width = width;
    result.height = height;
    result.blackLevel = blackLevel;
    result.whiteLevel = whiteLevel;
    result.bayerPattern = bayerPattern;
    result.wbRed = wbRed;
    result.wbGreen = wbGreen;
    result.wbBlue = wbBlue;
    result.highlightLinearityLimitBgr = highlightLinearityLimitBgr;
    result.saturationMask = saturationMask;

    for (int i = 0; i < 3; ++i)
    {
        for (int j = 0; j < 3; ++j)
        {
            result.rgbCam(i, j) = rawProcessor.imgdata.color.rgb_cam[i][j];
        }
    }

    rawProcessor.recycle();
    return result;
}

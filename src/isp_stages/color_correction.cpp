#include "isp_stages/color_correction.h"

#include <algorithm>
#include <stdexcept>

namespace ISPStages
{
cv::Mat convertCameraBgrToLinearBgr(const cv::Mat &cameraBgr,
                                    const cv::Matx33f &rgbCam)
{
    if (cameraBgr.empty() || cameraBgr.type() != CV_32FC3)
    {
        throw std::runtime_error("Invalid camera BGR image for color correction.");
    }

    cv::Mat correctedBgr(cameraBgr.size(), CV_32FC3);

    for (int y = 0; y < cameraBgr.rows; ++y)
    {
        const cv::Vec3f *bgrRow = cameraBgr.ptr<cv::Vec3f>(y);
        cv::Vec3f *correctedBgrRow = correctedBgr.ptr<cv::Vec3f>(y);

        for (int x = 0; x < cameraBgr.cols; ++x)
        {
            const cv::Vec3f bgr = bgrRow[x];
            const cv::Vec3f rgb(bgr[2], bgr[1], bgr[0]);

            cv::Vec3f corrected = rgbCam * rgb;

            corrected[0] = std::max(corrected[0], 0.0f);
            corrected[1] = std::max(corrected[1], 0.0f);
            corrected[2] = std::max(corrected[2], 0.0f);

            correctedBgrRow[x][0] = corrected[2];
            correctedBgrRow[x][1] = corrected[1];
            correctedBgrRow[x][2] = corrected[0];
        }
    }

    return correctedBgr;
}
} // namespace ISPStages


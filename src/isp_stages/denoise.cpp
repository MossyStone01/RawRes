#include "isp_stages/denoise.h"

#include "noise_removal/guided_filter.h"

#include <stdexcept>

namespace ISPStages
{
cv::Mat applyPreviewDenoiser(const cv::Mat &linearBgr, int denoiser)
{
    if (linearBgr.empty() || linearBgr.type() != CV_32FC3)
    {
        throw std::runtime_error("Invalid linear BGR image for denoising.");
    }

    if (denoiser < 1)
    {
        return linearBgr.clone();
    }

    if (denoiser == 1)
    {
        cv::Mat source = linearBgr.clone();
        cv::Mat guide;
        cv::cvtColor(source, guide, cv::COLOR_BGR2GRAY);
        return NoiseRemoval::GuidedFilter(3, source, guide, 0.001f);
    }

    if (denoiser == 2)
    {
        return linearBgr.clone();
    }

    return linearBgr.clone();
}
} // namespace ISPStages


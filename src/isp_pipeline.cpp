#include "isp_pipeline.h"

#include "isp_stages/color_correction.h"
#include "isp_stages/demosaic.h"
#include "isp_stages/denoise.h"
#include "isp_stages/output_encoding.h"
#include "isp_stages/raw_preprocess.h"
#include "isp_stages/tone_mapping.h"

// TODO : HDR 지원하기

namespace
{
constexpr float kPreviewExposureEv = 1.0f;
} // namespace

cv::Mat ISPPipeline::makePreview(const cv::Mat &bayer16,
                                 const cv::Matx33f &rgbCam, int blackLevel,
                                 int whiteLevel, double gamma, int bayerPattern,
                                 float redGain, float greenGain, float blueGain,
                                 int denoiser,
                                 cv::Vec3f highlightLinearityLimitBgr)
{
    const ISPStages::RawPreprocessResult raw = ISPStages::normalizeAndWhiteBalanceBayer(
        bayer16, blackLevel, whiteLevel, bayerPattern, redGain, greenGain,
        blueGain, highlightLinearityLimitBgr);

    const ISPStages::DemosaicResult demosaic =
        ISPStages::demosaicBayerAverage(raw.normalized, raw.saturationMask,
                                        bayerPattern);

    const cv::Mat correctedBgr =
        ISPStages::convertCameraBgrToLinearBgr(demosaic.bgr32, rgbCam);

    cv::Mat denoisedBgr =
        ISPStages::applyPreviewDenoiser(correctedBgr, denoiser);

    ISPStages::applyHighlightDesaturation(denoisedBgr,
                                          demosaic.saturationBgr);

    const cv::Mat exposedBgr =
        ISPStages::applyExposure(denoisedBgr, kPreviewExposureEv);
    const cv::Mat toneMappedBgr =
        ISPStages::applyExtendedReinhardToneMap(exposedBgr);
    const cv::Mat gammaCorrectedBgr =
        ISPStages::applyGammaCorrection(toneMappedBgr, gamma);

    return ISPStages::convertLinearBgrTo8Bit(gammaCorrectedBgr);
}


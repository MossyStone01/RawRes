#include <chrono>
#include <ctime>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>

#include "isp_pipeline.h"
#include "raw_loader.h"
#include <opencv2/opencv.hpp>

namespace
{
constexpr const char *kPreviewWindowName = "RAW Preview";

std::tm makeLocalTime(std::time_t timeValue)
{
    std::tm localTime{};
#if defined(_WIN32)
    localtime_s(&localTime, &timeValue);
#else
    localtime_r(&timeValue, &localTime);
#endif
    return localTime;
}

std::filesystem::path saveImg(const std::string &filePath, const cv::Mat &img)
{
    namespace fs = std::filesystem;
    const fs::path resultsDir = fs::current_path() / "results";
    fs::create_directories(resultsDir);

    const fs::path inputPath(filePath);
    const auto now = std::chrono::system_clock::now();
    const std::time_t nowTime = std::chrono::system_clock::to_time_t(now);
    const std::tm localTime = makeLocalTime(nowTime);

    std::ostringstream timestampStream;
    timestampStream << std::put_time(&localTime, "%Y%m%d_%H%M%S");

    const fs::path outputPath =
        resultsDir / (inputPath.stem().string() + "_" + timestampStream.str() +
                      "_result.png");

    if (!cv::imwrite(outputPath.string(), img))
    {
        throw std::runtime_error("Failed to save result image.");
    }

    std::cout << "Saved result image: " << outputPath << std::endl;
    return outputPath;
}
} // namespace

int main(int argc, char **argv)
{
    std::string rawPath;
    if (argc > 1)
    {
        rawPath = argv[1];
    }
    else
    {
        std::cout << "Enter a path of a photo" << std::endl;
        std::getline(std::cin >> std::ws, rawPath);
    }

    try
    {
        RawImageData raw = RawLoader::load(rawPath);

        std::cout << "Loaded RAW file: " << rawPath << std::endl;
        std::cout << "Size: " << raw.width << " x " << raw.height << std::endl;
        std::cout << "Black level: " << raw.blackLevel << std::endl;
        std::cout << "White level: " << raw.whiteLevel << std::endl;
        std::cout << "Highlight linearity limit (B/G/R): "
                  << raw.highlightLinearityLimitBgr[0] << " / "
                  << raw.highlightLinearityLimitBgr[1] << " / "
                  << raw.highlightLinearityLimitBgr[2] << std::endl;
        std::cout << "As-shot WB (R/G/B): " << raw.wbRed << " / " << raw.wbGreen
                  << " / " << raw.wbBlue << std::endl;

        switch (raw.bayerPattern)
        {
        case 0:
            std::cout << "Bayer pattern : RGGB" << std::endl;
            break;
        case 1:
            std::cout << "Bayer pattern : BGGR" << std::endl;
            break;
        case 2:
            std::cout << "Bayer pattern : GRBG" << std::endl;
            break;
        case 3:
            std::cout << "Bayer pattern : GBRG" << std::endl;
            break;
        default:
            std::cout << "Bayer pattern : Invalid" << std::endl;
            break;
        }

        cv::namedWindow(kPreviewWindowName,
                        cv::WINDOW_NORMAL | cv::WINDOW_KEEPRATIO);

        cv::Mat preview = ISPPipeline::makePreview(
            raw.bayer16, raw.rgbCam, raw.blackLevel, raw.whiteLevel, 2.2,
            raw.bayerPattern, raw.wbRed, raw.wbGreen, raw.wbBlue, 1,
            raw.highlightLinearityLimitBgr);
        saveImg(rawPath, preview);

        cv::resizeWindow(kPreviewWindowName, preview.cols, preview.rows);
        cv::imshow(kPreviewWindowName, preview);

        while (cv::waitKey(1) != 27) // esc
        {
        };
    }
    catch (const std::exception &e)
    {
        std::cerr << "[ERROR] " << e.what() << std::endl;
        return 1;
    }

    return 0;
}

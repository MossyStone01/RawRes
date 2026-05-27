cmake_minimum_required(VERSION 3.16)

project(RawRes VERSION 0.1 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

find_package(OpenCV REQUIRED)
find_package(PkgConfig QUIET)

if(PkgConfig_FOUND)
    pkg_check_modules(LIBRAW QUIET libraw)
endif()

find_path(LIBRAW_INCLUDE_DIR
    NAMES libraw/libraw.h
    HINTS ${LIBRAW_INCLUDEDIR} ${LIBRAW_INCLUDE_DIRS}
)

find_library(LIBRAW_LIBRARY
    NAMES raw_r raw libraw_r libraw
    HINTS ${LIBRAW_LIBDIR} ${LIBRAW_LIBRARY_DIRS}
)

if(NOT LIBRAW_INCLUDE_DIR OR NOT LIBRAW_LIBRARY)
    message(FATAL_ERROR
        "LibRaw not found. Install LibRaw and make sure CMake can locate its headers and library."
    )
endif()

add_executable(RawRes
    src/main.cpp
    src/raw_loader.cpp
    src/isp_pipeline.cpp
    src/isp_stages/bayer_pattern.cpp
    src/isp_stages/color_correction.cpp
    src/isp_stages/demosaic.cpp
    src/isp_stages/denoise.cpp
    src/isp_stages/output_encoding.cpp
    src/isp_stages/raw_preprocess.cpp
    src/isp_stages/tone_mapping.cpp
    src/noise_removal/bm3d.cpp
    src/noise_removal/guided_filter.cpp
)

target_include_directories(RawRes PRIVATE
    include
    ${OpenCV_INCLUDE_DIRS}
    ${LIBRAW_INCLUDE_DIR}
)

target_link_libraries(RawRes PRIVATE
    ${OpenCV_LIBS}
    ${LIBRAW_LIBRARY}
)

if(WIN32)
    target_compile_definitions(RawRes PRIVATE _CRT_SECURE_NO_WARNINGS)
endif()

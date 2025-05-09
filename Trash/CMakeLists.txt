# Minimum CMake Version
cmake_minimum_required(VERSION 3.26)

# Project Name and Metadata
project(FaceLandmarks VERSION 1.0 LANGUAGES CXX)

# Set C++ Standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Add Custom Paths for Conan or Dependencies
list(APPEND CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/build/conan")
list(APPEND CMAKE_PREFIX_PATH "${CMAKE_SOURCE_DIR}/build/conan")

# Find Required Packages
find_package(OpenCV 4.6.0 REQUIRED COMPONENTS core highgui imgproc)
find_library(TENSORFLOW_LITE_LIB tensorflowlite REQUIRED HINTS ${CMAKE_SOURCE_DIR}/third_party/tflite/lib)
find_library(OPENBLAS_LIB openblas REQUIRED HINTS /usr/lib /usr/lib64 /lib /lib64)
find_library(DLIB_LIBRARIES dlib REQUIRED HINTS /usr/local/lib)

# Include directories for external libraries
include_directories(
        /usr/local/include         # Dlib headers
        ${CMAKE_SOURCE_DIR}/third_party/tflite/include # TensorFlow Lite headers
        ${OpenCV_INCLUDE_DIRS}     # OpenCV headers
)
# Define the Face Library
add_library(face_lib
        FaceLib/face_detector.cpp
        FaceLib/landmark_extractor.cpp
)

# Link libraries for Face Library
target_link_libraries(face_lib PUBLIC
        ${TENSORFLOW_LITE_LIB}
        ${DLIB_LIBRARIES}
        ${OpenCV_LIBS}
        ${OPENBLAS_LIB}
)

# Define the Demo Executable
add_executable(demo demo.cpp)

# Link libraries for Demo Executable
target_link_libraries(demo PUBLIC
        face_lib
        ${TENSORFLOW_LITE_LIB}
        ${DLIB_LIBRARIES}
        ${OpenCV_LIBS}
        ${OPENBLAS_LIB}
)

# Define the Take Photos Executable
add_executable(take_photos take_photos.cpp)

# Link libraries for Take Photos Executable
target_link_libraries(take_photos PUBLIC
        face_lib
        ${TENSORFLOW_LITE_LIB}
        ${DLIB_LIBRARIES}
        ${OpenCV_LIBS}
        ${OPENBLAS_LIB}
)

# Output Build Information
message(STATUS "OpenCV Version: ${OpenCV_VERSION}")
message(STATUS "TensorFlow Lite Library Found: ${TENSORFLOW_LITE_LIB}")
message(STATUS "Dlib Library Found: ${DLIB_LIBRARIES}")
message(STATUS "OpenBLAS Library Found: ${OPENBLAS_LIB}")

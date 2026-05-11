cmake_minimum_required(VERSION 3.12)
project(g1_29dof_controller)

set(CMAKE_CXX_STANDARD 17)

set(UNITREE_SDK2_PREFIX "$ENV{UNITREE_SDK2_PREFIX}" CACHE PATH "Path to a local unitree_sdk2 install prefix")
if(UNITREE_SDK2_PREFIX)
  list(PREPEND CMAKE_PREFIX_PATH "${UNITREE_SDK2_PREFIX}" "${UNITREE_SDK2_PREFIX}/lib/cmake")
endif()
if(DEFINED ENV{CONDA_PREFIX})
  list(APPEND CMAKE_IGNORE_PREFIX_PATH "$ENV{CONDA_PREFIX}")
endif()
list(APPEND CMAKE_IGNORE_PREFIX_PATH "$ENV{HOME}/miniconda3")

find_package(unitree_sdk2 REQUIRED CONFIG)
find_package(Eigen3 REQUIRED)
find_package(Boost REQUIRED COMPONENTS program_options)
find_package(yaml-cpp REQUIRED CONFIG PATHS /usr/lib/x86_64-linux-gnu/cmake/yaml-cpp NO_DEFAULT_PATH)
find_package(fmt REQUIRED CONFIG PATHS /usr/lib/x86_64-linux-gnu/cmake/fmt NO_DEFAULT_PATH)

set(ONNXRUNTIME_ROOT ${PROJECT_SOURCE_DIR}/../../thirdparty/onnxruntime-linux-x64-1.22.0)

include_directories(include)

file(GLOB_RECURSE ADD_SRC_LIST
  ${PROJECT_SOURCE_DIR}/src/*.cpp
)

add_library(${PROJECT_NAME}_lib ${ADD_SRC_LIST})
target_include_directories(${PROJECT_NAME}_lib
  PUBLIC
    ${PROJECT_SOURCE_DIR}/../../include
    ${ONNXRUNTIME_ROOT}/include
)
target_compile_definitions(${PROJECT_NAME}_lib
  PUBLIC
    SPDLOG_HEADER_ONLY
)
target_link_libraries(${PROJECT_NAME}_lib
  PUBLIC
    Eigen3::Eigen
    unitree_sdk2
    ddsc
    ddscxx
    Boost::program_options
    yaml-cpp
    fmt::fmt
    rt
    pthread
    ${ONNXRUNTIME_ROOT}/lib/libonnxruntime.so.1.22.0
)

add_executable(g1_ctrl main.cpp)
target_link_libraries(g1_ctrl PRIVATE ${PROJECT_NAME}_lib)

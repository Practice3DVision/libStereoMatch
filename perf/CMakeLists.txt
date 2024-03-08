cmake_minimum_required(VERSION 3.20)

include(FetchContent)
FetchContent_Declare(
    benchmark
    URL ${PROJECT_SOURCE_DIR}/3rdParty/benchmark-1.8.3.zip
)

project(PerfermanceTest)

set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)

FetchContent_MakeAvailable(benchmark)

add_executable(
    PerfermanceCostCompute
    ${CMAKE_CURRENT_SOURCE_DIR}/perfCostCompute.cpp
)

target_link_libraries(
    PerfermanceCostCompute
    PRIVATE
    benchmark::benchmark
    StereoMatch
)

add_executable(
    PerfermanceDispCompute
    ${CMAKE_CURRENT_SOURCE_DIR}/perfDispCompute.cpp
)

target_link_libraries(
    PerfermanceDispCompute
    PRIVATE
    benchmark::benchmark
    StereoMatch
)

add_executable(
    PerfermanceCostAggregation
    ${CMAKE_CURRENT_SOURCE_DIR}/perfCostAggregation.cpp
)

target_link_libraries(
    PerfermanceCostAggregation
    PRIVATE
    benchmark::benchmark
    StereoMatch
)

add_executable(
    PerfermanceDispOptimiztion
    ${CMAKE_CURRENT_SOURCE_DIR}/perfDispOptimiztion.cpp
)

target_link_libraries(
    PerfermanceDispOptimiztion
    PRIVATE
    benchmark::benchmark
    StereoMatch
)
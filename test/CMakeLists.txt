cmake_minimum_required(VERSION 3.20)

include(FetchContent)
FetchContent_Declare(
    googletest
    URL ${PROJECT_SOURCE_DIR}/3rdParty/googletest-1.12.0.zip
)

project(PrecisionTest)

set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)

FetchContent_MakeAvailable(googletest)
enable_testing()

add_executable(
    TestCostCompute
    ${CMAKE_CURRENT_SOURCE_DIR}/testCostCompute.cpp
)

target_link_libraries(
    TestCostCompute
    PRIVATE
    gtest_main
    StereoMatch
)

include(GoogleTest)
gtest_discover_tests(TestCostCompute)
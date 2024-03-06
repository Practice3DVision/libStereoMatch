cmake_minimum_required(VERSION 3.20)

project(libStereoMatch VERSION 0.1.0)

set(CMAKE_EXPORT_COMPILE_COMMANDS TRUE)

option(BUILD_SHARED "compile libStereoMatch as a dynamic library." OFF)
option(BUILD_TEST "perform accuracy testing." ON)
option(BUILD_PERF "perform performance testing." ON)

configure_file(
    ${CMAKE_CURRENT_SOURCE_DIR}/include/libStereoMatchConfig.h.in
    ${CMAKE_CURRENT_SOURCE_DIR}/include/libStereoMatchConfig.h
)

message("Version of libStereoMatch: ${PROJECT_VERSION}")
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/src)

if(BUILD_TEST)
    message("Perform accuracy testing.")
    add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/test)
endif()

if(BUILD_PERF)
    message("Perform performance testing.")
    add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/perf)
endif()
cmake_minimum_required(VERSION 3.20)

set(PROJECT_ROOT_HEADER_DIR ${PROJECT_SOURCE_DIR}/include)

project(StereoMatch LANGUAGES CXX)

find_package(OpenCV REQUIRED)

file(GLOB_RECURSE HEADERS ${CMAKE_CURRENT_SOURCE_DIR}/*.h ${PROJECT_ROOT_HEADER_DIR}/*.h)
file(GLOB_RECURSE SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp ${CMAKE_CURRENT_SOURCE_DIR}/*.hpp)
source_group("Headers" FILES ${HEADERS})
source_group("Sources" FILES ${SOURCES})

if(BUILD_SHARED)
    add_library(${PROJECT_NAME} SHARED)
else()
    add_library(${PROJECT_NAME} STATIC)
endif()

target_sources(${PROJECT_NAME} PUBLIC ${HEADERS} PRIVATE ${SOURCES})
target_compile_options(${PROJECT_NAME} PRIVATE /openmp)
target_link_libraries(${PROJECT_NAME} PUBLIC ${OpenCV_LIBRARIES})
target_include_directories(${PROJECT_NAME} PUBLIC ${PROJECT_ROOT_HEADER_DIR} ${CMAKE_CURRENT_SOURCE_DIR})
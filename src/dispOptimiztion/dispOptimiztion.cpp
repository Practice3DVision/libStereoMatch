#include "dispOptimiztion.h"

#include <opencv2/opencv.hpp>

#include <unordered_map>

using namespace cv;
using namespace std;

namespace libSM {

/**
 * @brief remove small connected domains
 *
 * @param dispMap               //disparity map
 * @param out                   //out disparity map
 * @param dispDomainThreshold   //parallax map connectivity threshold
 * @param smallAreaThreshold    //threshold for the number of pixels in a small
 * connected domain
 */
void removeSmallArea(const Mat &dispMap, Mat &out,
                     const float dispDomainThreshold,
                     const int smallAreaThreshold) {
    CV_Assert(!dispMap.empty());

    if (out.empty())
        out = dispMap.clone();

    vector<bool> visited(dispMap.rows * dispMap.cols, false);

    for (int i = 0; i < dispMap.rows; ++i) {
        auto ptrDispMap = dispMap.ptr<float>(i);

        for (int j = 0; j < dispMap.cols; ++j) {
            if (visited[dispMap.cols * i + j] || abs(ptrDispMap[j]) < 0.001f) {
                continue;
            }

            vector<pair<int, int>> area;
            int areaSize = area.size();
            area.emplace_back(make_pair(j, i));
            visited[dispMap.cols * i + j] = true;

            do {
                int preSize = areaSize;
                areaSize = area.size();

                for (int k = preSize; k < areaSize; ++k) {
                    auto curPixelX = area[k].first;
                    auto curPixelY = area[k].second;
                    auto curPixelDisp =
                        dispMap.ptr<float>(curPixelY)[curPixelX];

                    for (int y = -1; y <= 1; ++y) {
                        for (int x = -1; x <= 1; ++x) {
                            auto visitPixelX = curPixelX + x;
                            auto visitPixelY = curPixelY + y;

                            if (visitPixelX < 0 ||
                                visitPixelX > dispMap.cols - 1 ||
                                visitPixelY < 0 ||
                                visitPixelY > dispMap.rows - 1) {
                                continue;
                            }

                            if (visited[dispMap.cols * visitPixelY +
                                        visitPixelX] ||
                                abs(dispMap.ptr<float>(
                                    visitPixelY)[visitPixelX]) < 0.001f) {
                                continue;
                            }

                            if (abs(dispMap.ptr<float>(
                                        visitPixelY)[visitPixelX] -
                                    curPixelDisp) < dispDomainThreshold) {
                                area.push_back(
                                    make_pair(visitPixelX, visitPixelY));
                                visited[dispMap.cols * visitPixelY +
                                        visitPixelX] = true;
                            }
                        }
                    }
                }
            } while (areaSize < area.size());

            if (area.size() < smallAreaThreshold) {
                for (auto loc : area) {
                    out.ptr<float>(loc.second)[loc.first] = NONE_PIXEL;
                }
            }
        }
    }
}

/**
 * @brief median filter for float-type image
 *
 * @param dispMap disparity map
 * @param out filtered disparity map
 * @param k kernel size
 */
void medianFilter(const Mat &dispMap, Mat &out, const int k) {
    const int halfSize = k / 2;
    const int medianIndex = (k * k) / 2;

#pragma omp parallel for schedule(dynamic) default(shared)
    for (int i = halfSize; i < dispMap.rows - halfSize; ++i) {
        auto ptrOut = out.ptr<float>(i);

        for (int j = halfSize; j < dispMap.cols - halfSize; ++j) {
            vector<float> disp;

            for (int y = -halfSize; y <= halfSize; ++y) {
                for (int x = -halfSize; x <= halfSize; ++x) {
                    disp.emplace_back(dispMap.ptr<float>(i + y)[j + x]);
                }
            }

            std::sort(disp.begin(), disp.end());
            ptrOut[j] = disp[medianIndex];
        }
    }
}

/**
 * @brief fill disparity map
 *
 * @param dispMap disparity map
 * @param out filled disparity map
 */
void dispFill(const Mat &dispMap, Mat &out) {
    const vector<pair<int, int>> direction = {
        make_pair(0, -1), make_pair(1, -1), make_pair(1, 0),
        make_pair(1, 1),  make_pair(0, 1),  make_pair(-1, 1),
        make_pair(-1, 0), make_pair(-1, -1)};

#pragma omp parallel for schedule(dynamic) default(shared)
    for (int i = 0; i < dispMap.rows; ++i) {
        auto ptrDispMap = dispMap.ptr<float>(i);
        auto ptrOut = out.ptr<float>(i);

        for (int j = 0; j < dispMap.cols; ++j) {
            auto state = make_pair(IS_OCCLUDED_PIXEL(ptrDispMap[j]),
                                   IS_MISMATCHED_PIXEL(ptrDispMap[j]) || IS_NONE_PIXEL(ptrDispMap[j]));
            if (state.first || state.second) {
                vector<float> disp;

                for (int d = 0; d < direction.size(); ++d) {
                    auto curLocX = j;
                    auto curLocY = i;
                    bool isFind = false;

                    do {
                        curLocX += direction[d].first;
                        curLocY += direction[d].second;

                        if (curLocX < 0 || curLocX > dispMap.cols - 1 ||
                            curLocY < 0 || curLocY > dispMap.rows - 1) {
                            break;
                        }

                        auto curVal = dispMap.ptr<float>(curLocY)[curLocX];

                        if (!IS_OCCLUDED_PIXEL(curVal) &&
                            !IS_MISMATCHED_PIXEL(curVal) &&
                            !IS_NONE_PIXEL(curVal)) {
                            disp.emplace_back(curVal);
                            isFind = true;
                            break;
                        }
                    } while (true);

                    if (!isFind) {
                        disp.emplace_back(0);
                    }
                }

                std::sort(disp.begin(), disp.end());

                ptrOut[j] = state.first ? disp[1] : disp[disp.size() / 2];
            }
        }
    }
}

void dispOptimiz(const Mat &dispMap, Mat &out, const DispOptParams params) {
    CV_Assert(!dispMap.empty());

    if (out.empty())
        out = dispMap.clone();

    if (params.enableRemoveSmallArea) {
        removeSmallArea(dispMap, out, params.dispDomainThreshold,
                        params.smallAreaThreshold);
    }

    if (params.enableDispFill) {
        Mat temp = out.clone();
        dispFill(temp, out);
    }

    if (params.enableMedianFilter) {
        Mat temp = out.clone();
        medianFilter(temp, out, params.k);
    } else if (params.enableBilateralFilter) {
        Mat temp = out.clone();
        bilateralFilter(temp, out, params.d, params.sigmaColor,
                        params.sigmaSpace);
    }
}

} // namespace libSM
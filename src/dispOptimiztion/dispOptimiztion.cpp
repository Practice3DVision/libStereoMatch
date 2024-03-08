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
                    out.ptr<float>(loc.second)[loc.first] = 0.f;
                }
            }
        }
    }
}

void dispOptimiz(const Mat &dispMap, Mat &out, const DispOptParams params) {
    CV_Assert(!dispMap.empty());

    if (out.empty())
        out = dispMap.clone();

    if (params.bilateralFilter) {
        bilateralFilter(dispMap, out, params.d, params.sigmaColor,
                        params.sigmaSpace);
    }

    if (params.removeSmallArea) {
        removeSmallArea((params.bilateralFilter) ? out : dispMap, out,
                        params.dispDomainThreshold, params.smallAreaThreshold);
    }
}

} // namespace libSM
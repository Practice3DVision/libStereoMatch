#include "dispCompute.h"

#include <opencv2/opencv.hpp>

using namespace cv;
using namespace std;

namespace libSM {
/**
 * @brief check if the difference in value is too small
 *
 * @param majorMinCost      minimum cost
 * @param minorMinCost      sub minimum cost
 * @param uniquenessRatio   uniqueness ratio
 * @return true             pass
 * @return false            not passed
 */
bool uniqueCheck(const float majorMinCost, const float minorMinCost,
                 const float uniquenessRatio = 0.8f) {
    return (minorMinCost - majorMinCost) >
           majorMinCost * (1.f - uniquenessRatio);
}

/**
 * @brief left and right consistency test
 *
 * @param ptrCostMap        the cost of the current line
 * @param leftBestDisp      left image best parallax
 * @param lx                left image x-coordinate
 * @param rx                matched right image x-coordinate
 * @param dispRange         parallax interval
 * @param cols              number of image columns
 * @param lrCheckThreshod   left and right consistency threshold
 * @return true             pass
 * @return false            not passed
 */
bool lrCheck(const float *ptrCostMap, const int lx, const int rx,
             const int minDisp, const int maxDisp, const int cols,
             const int lrCheckThreshod = 1) {
    int rightBestDisp = lx - rx;
    float minCost = FLT_MAX;
    const int dispRange = maxDisp - minDisp;

    for (int d = -dispRange + 1; d <= 0; ++d) {
        auto curLx = rx - d + minDisp;
        if (curLx < 0 || curLx > cols - 1) {
            continue;
        }

        auto curCost = ptrCostMap[dispRange * curLx - d];
        if (curCost < minCost) {
            minCost = curCost;
            rightBestDisp = d;
        }
    }

    return abs(rx - rightBestDisp + minDisp - lx) <= lrCheckThreshod;
}

/**
 * @brief sub-pixel fitting
 *
 * @param ptrCostMap the cost of the current line
 * @param lx         left image x-coordinate
 * @param disp       parallax of current left image pixels
 * @param minCost    the minimum cost of pixels in the current left image
 * @param dispRange  parallax interval
 * @return float     sub-pixel parallax
 */
float subpixelFitting(const float *ptrCostMap, const int lx, const int disp,
                      const float minCost, const int dispRange) {
    auto preCost = ptrCostMap[dispRange * lx + disp - 1];
    auto aftCost = ptrCostMap[dispRange * lx + disp + 1];
    auto denom = max(0.001f, preCost + aftCost - 2 * minCost);
    return disp + (preCost - aftCost) / (denom * 2.f);
}

void winnerTakesAll(const Mat &costMap, Mat &dispMap,
                    const DispComputeParams params) {
    CV_Assert_N(!costMap.empty());

    if (dispMap.empty())
        dispMap = Mat(costMap.size(), CV_32FC1, cv::Scalar(0.f));

#pragma omp parallel for schedule(dynamic) default(shared)
    for (int i = 0; i < costMap.rows; ++i) {

        auto ptrCostMap = costMap.ptr<float>(i);
        auto ptrDispMap = dispMap.ptr<float>(i);

        for (int j = 0; j < costMap.cols; ++j) {

            float majorMinCost = FLT_MAX, minorMinCost = FLT_MAX;
            int majorDisp = 0;

            for (int d = 0; d < costMap.channels(); ++d) {
                auto curCost = ptrCostMap[costMap.channels() * j + d];
                if (curCost < majorMinCost) {
                    minorMinCost = majorMinCost;
                    majorMinCost = curCost;
                    majorDisp = d;
                }
            }

            if (params.enableUniqueCheck &&
                !uniqueCheck(majorMinCost, minorMinCost,
                             params.uniquenessRatio)) {
                ptrDispMap[j] = 0.f;
                continue;
            }

            if (params.enableLRCheck &&
                !lrCheck(ptrCostMap, j, j - (majorDisp + params.minDisp),
                         params.minDisp, params.maxDisp, costMap.cols,
                         params.lrCheckThreshod)) {
                ptrDispMap[j] = 0.f;
                continue;
            }

            if (params.enableSubpixelFitting &&
                (majorDisp != 0 && majorDisp != costMap.channels() - 1)) {
                ptrDispMap[j] =
                    subpixelFitting(ptrCostMap, j, majorDisp, majorMinCost,
                                    costMap.channels()) +
                    params.minDisp;
            } else {
                ptrDispMap[j] = majorDisp + params.minDisp;
            }
        }
    }
}
} // namespace libSM
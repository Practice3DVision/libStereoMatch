/**
 * @file dispCompute.h
 * @author Liu Yunhuang (1369215984@qq.com)
 * @brief
 * @version 0.1
 * @date 2024-03-06
 *
 * @copyright Copyright (c) 2024
 *
 */

#ifndef __DISP_COMPUTE_H_
#define __DISP_COMPUTE_H_

#include <typeDef.h>

namespace cv {
class Mat;
}

namespace libSM {
/**
 * @brief disparity computation control parameters
 *
 */
struct DispComputeParams {
    DispComputeParams()
        : lrCheck(true), uniqueCheck(true), subpixelFitting(true), minDisp(0),
          maxDisp(64), uniquenessRatio(0.95f), lrCheckThreshod(1) {}
    bool lrCheck;          // left-right consistency check
    bool uniqueCheck;      // consistency check
    bool subpixelFitting;  // subpixel fitting
    float uniquenessRatio; // uniqueness ratio
    int lrCheckThreshod;   // left and right consistency threshold
    int minDisp;           // minimum disparity value
    int maxDisp;           // maximum disparity value
};

/**
 * @brief winner-takes-all algorithm
 *
 * @param costMap //cost space
 * @param dispMap //disparity map
 * @param params  //disparity computation control parameters
 */
void LIBSM_API winnerTakesAll(IN const cv::Mat &costMap, OUT cv::Mat &dispMap,
                              IN const DispComputeParams params);
} // namespace libSM

#endif //!__DISP_COMPUTE_H_
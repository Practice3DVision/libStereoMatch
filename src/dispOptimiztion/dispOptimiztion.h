/**
 * @file dispOptimiztion.h
 * @author Liu Yunhuang (1369215984@qq.com)
 * @brief
 * @version 0.1
 * @date 2024-03-07
 *
 * @copyright Copyright (c) 2024
 *
 */

#ifndef __DISP_OPTIMIZTION_H_
#define __DISP_OPTIMIZTION_H_

#include <typeDef.h>

namespace cv {
    class Mat;
}

namespace libSM {
    /**
     * @brief parallax optimization parameters
     * 
     */
    struct DispOptParams {
        bool bilateralFilter;    //enable bilateral filter
        bool removeSmallArea;    //enable remove small area
        int smallAreaThreshold;  //small area threshild
        int dispDomainThreshold; //parallax connected domain threshold
        int d;                   //bilateral filtering filter field diameter
        float sigmaColor;        //standard deviation of Gaussian function in color space
        float sigmaSpace;        //standard deviation of Gaussian function in coordinate space
    };

    /**
     * @brief optimize disparity map
     * 
     * @param dispMap disparity map
     * @param out out disparity map
     * @param params parallax optimization parameters
     */
    void LIBSM_API dispOptimiz(IN const cv::Mat& dispMap, OUT cv::Mat& out, IN const DispOptParams params);
}

#endif //!__DISP_OPTIMIZTION_H_
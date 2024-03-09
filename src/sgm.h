/**
 * @file sgm.h
 * @author Liu Yunhuang (1369215984@qq.com)
 * @brief
 * @version 0.1
 * @date 2024-03-08
 *
 * @copyright Copyright (c) 2024
 *
 */

#ifndef __SGM_H_
#define __SGM_H_

#include "algorithm.h"

namespace cv {
class Mat;
}

namespace libSM {
/**
 * @brief Semi Global Matching Algorithm
 *
 * @paper H. Hirschmuller, "Stereo Processing by Semiglobal Matching and Mutual
 * Information," in IEEE Transactions on Pattern Analysis and Machine
 * Intelligence, vol. 30, no. 2, pp. 328-341, Feb. 2008,
 * doi: 10.1109/TPAMI.2007.1166.
 *
 */
class LIBSM_API SGM : public Algorithm {
  public:
    /**
     * @brief control parameters
     *
     */
    struct Params {
        Params()
            : enableHonrizon(true), enableVertiacl(true), enablePostive45(true),
              enableNegtive45(true), enableBilateralFilter(false),
              enableRemoveSmallArea(true), enableLRCheck(true),
              enableUniqueCheck(true), enableSubpixelFitting(true),
              enableMedianFilter(true), enableDispFill(true), windowWidth(9),
              windowHeight(7), minDisp(0), maxDisp(64), P1(10), P2(150),
              lrCheckThreshod(1), uniquenessRatio(0.95), smallAreaThreshold(20),
              dispDomainThreshold(1), k(3), d(10), sigmaColor(10),
              sigmaSpace(10) {}
        bool enableHonrizon;        // enable aggregation on horizontal line
        bool enableVertiacl;        // enable aggregation on vertical line
        bool enablePostive45;       // enable aggregation on postive 45 line
        bool enableNegtive45;       // enable aggregation on negtive 45 line
        bool enableBilateralFilter; // enable bilateral filter
        bool enableRemoveSmallArea; // enable remove small area
        bool enableLRCheck;         // left-right consistency check
        bool enableUniqueCheck;     // consistency check
        bool enableSubpixelFitting; // subpixel fitting
        bool enableMedianFilter;    // enable median filter
        bool enableDispFill;        // enable fill the background or prospect
        int windowWidth;            // the width of the cost calculation window
        int windowHeight;           // the height of the cost calculation window
        int minDisp;                // minimum disparity value
        int maxDisp;                // maximum disparity value.
        float P1;            // penalty coefficient for disparity continuity
        float P2;            // penalty coefficient for disparity no continuity
        int lrCheckThreshod; // left and right consistency threshold
        float uniquenessRatio;   // uniqueness ratio
        int smallAreaThreshold;  // small area threshild
        int dispDomainThreshold; // parallax connected domain threshold
        int k;                   // median filtering filter kernel size
        int d;                   // bilateral filtering filter field diameter
        float sigmaColor; // standard deviation of Gaussian function in color
                          // space
        float sigmaSpace; // standard deviation of Gaussian function in
                          // coordinate space
    };
    virtual ~SGM() {}
    /**
     * @brief create the SGM algorithm
     *
     * @param params control params
     * @return Ptr<Algorithm> SGM algorithm
     */
    static Ptr<Algorithm> create(const Params params);
    /**
     * @brief perform stereo matching
     *
     * @param left left image
     * @param right right image
     * @param dispMap disparity Map
     */
    virtual void match(IN const cv::Mat &left, IN const cv::Mat &right,
                       OUT cv::Mat &dispMap) override = 0;
};
} // namespace libSM

#endif //!__SGM_H_
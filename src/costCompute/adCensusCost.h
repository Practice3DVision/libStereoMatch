/**
 * @file adCensusCost.h
 * @author Liu Yunhuang (1369215984@qq.com)
 * @brief
 * @version 0.1
 * @date 2024-03-06
 *
 * @copyright Copyright (c) 2024
 *
 */

#ifndef __AD_CENSUS_COST_H_
#define __AD_CENSUS_COST_H_

#include <libStereoMatchConfig.h>

#include "costCompute.h"

namespace libSM {
/**
 * @brief ADCensus Cost Calculator
 *
 */
class LIBSM_API ADCensusCost : public CostComputer {
  public:
    /**
     * @brief parameters in the ADCensus cost calculator.
     *
     */
    struct Params {
        Params()
            : windowWidth(9), windowHeight(7), minDisp(0), maxDisp(128),
              adWeight(10), censusWeight(30) {}
        int windowWidth;    // the width of the cost calculation window
        int windowHeight;   // the height of the cost calculation window
        int minDisp;        // minimum disparity value
        int maxDisp;        // maximum disparity value.
        float adWeight;     // weight of ad
        float censusWeight; // weight of census
    };
    /**
     * @brief create a cost calculator
     *
     * @param params parameters
     * @return Ptr<CostComputer> cost calculator
     */
    static Ptr<CostComputer>
    create(IN const Params params); // create a cost calculator
    /**
     * @brief cost calculation
     *
     * @param left rectified left image
     * @param right rectified right image
     * @param out cost three-dimensional space
     */
    virtual void compute(IN const cv::Mat &left, IN const cv::Mat &right,
                         OUT cv::Mat &out) override = 0;
};
} // namespace libSM

#endif //! __AD_CENSUS_COST_H_
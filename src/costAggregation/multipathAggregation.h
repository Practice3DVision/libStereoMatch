/**
 * @file multipathAggregation.h
 * @author Liu Yunhuang (1369215984@qq.com)
 * @brief
 * @version 0.1
 * @date 2024-03-07
 *
 * @copyright Copyright (c) 2024
 *
 */

#ifndef __MULTI_PATH_AGGREGATION_H_
#define __MULTI_PATH_AGGREGATION_H_

#include "costAggregation.h"

namespace libSM {
/**
 * @brief multi-path cost aggregator(which is used in SGM)
 *
 */
class LIBSM_API MultipathAggregation : public CostAggregation {
  public:
    /**
     * @brief cost aggregation parameters
     *
     */
    struct Params {
        Params() : enableHonrizon(true), enableVertiacl(true), enableNegtive45(true), enablePostive45(true), P1(10.f), P2(150.f) {}
        bool enableHonrizon;  // enable aggregation on horizontal line
        bool enableVertiacl;  // enable aggregation on vertical line
        bool enablePostive45; // enable aggregation on postive 45 line
        bool enableNegtive45; // enable aggregation on negtive 45 line
        float P1;             // penalty coefficient for disparity continuity
        float P2;             // penalty coefficient for disparity no continuity
    };
    virtual ~MultipathAggregation() {}
    /**
     * @brief create MultiPathAggregation
     *
     * @param params cost aggregation parameters
     * @return Ptr<CostAggregation> CostAggregation's Ptr
     */
    static Ptr<CostAggregation> create(IN const Params params);
    /**
     * @brief aggregation cost
     *
     * @param left  left image
     * @param cost cost space
     * @param aggregationCost aggregated cost
     */
    virtual void aggregation(IN const cv::Mat &left,
                             IN const cv::Mat &cost,
                             OUT cv::Mat &aggregationCost) override = 0;
};
} // namespace libSM

#endif //!__MULTI_PATH_AGGREGATION_H_
/**
 * @file costAggregation.h
 * @author Liu Yunhuang (1369215984@qq.com)
 * @brief
 * @version 0.1
 * @date 2024-03-07
 *
 * @copyright Copyright (c) 2024
 *
 */

#ifndef __COST_AGGREGATION_H_
#define __COST_AGGREGATION_H_

#include <typeDef.h>

namespace cv {
class Mat;
}

namespace libSM {
/**
 * @brief cost aggregator
 *
 */
class LIBSM_API CostAggregation {
  public:
    /**
     * @brief aggregation cost
     *
     * @param left  left image
     * @param cost cost space
     * @param aggregationCost aggregated cost
     */
    virtual void aggregation(IN const cv::Mat &left,
                             IN const cv::Mat &cost,
                             OUT cv::Mat &aggregationCost) = 0;
};
} // namespace libSM

#endif //!__COST_AGGREGATION_H_
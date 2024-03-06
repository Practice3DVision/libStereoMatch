/**
 * @file costCompute.h
 * @author Liu Yunhuang (1369215984@qq.com)
 * @brief 
 * @version 0.1
 * @date 2024-03-06
 * 
 * @copyright Copyright (c) 2024
 * 
 */

#ifndef __COST_COMPUTE_H_
#define __COST_COMPUTE_H_

#include <typeDef.h>

namespace cv {
class Mat;
}

namespace libSM {
/**
 * @brief abstract base class for cost calculation
 * 
 */
class LIBSM_API CostComputer {
  public:
    virtual ~CostComputer() {}
    /**
     * @brief cost calculation
     *
     * @param left rectified left image
     * @param right rectified right image
     * @param out cost three-dimensional space
     */
    virtual void compute(IN const cv::Mat &left, IN const cv::Mat &right,
                         OUT cv::Mat &out) = 0;
};
} // namespace libSM

#endif //!__COST_COMPUTE_H_
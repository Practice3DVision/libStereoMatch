#include "adCensusCost.h"
#include "adCost.h"
#include "censusCost.h"

#include <omp.h>

#include <opencv2/opencv.hpp>

using namespace cv;

namespace libSM {
/**
 * @brief implementation class for the ADCensus Cost interface.
 *
 */
class ADCensusCostImpl : public ADCensusCost {
  public:
    ADCensusCostImpl(const Params params) : params_(params) {}
    void compute(const Mat &left, const Mat &right, Mat &out) override;

  private:
    Params params_;
};

void ADCensusCostImpl::compute(const Mat &left, const Mat &right, Mat &out) {
    CV_Assert_N(!left.empty(), !right.empty(), left.size == right.size);

    Mat adCost, censusCost;

    {
        ADCost::Params adParams;
        adParams.minDisp = params_.minDisp;
        adParams.maxDisp = params_.maxDisp;
        adParams.minDisp = params_.minDisp;
        adParams.maxDisp = params_.maxDisp;
        auto adCostComputor = ADCost::create(adParams);
        adCostComputor->compute(left, right, adCost);
    }

    {
        CensusCost::Params censusParams;
        censusParams.minDisp = params_.minDisp;
        censusParams.maxDisp = params_.maxDisp;
        censusParams.minDisp = params_.minDisp;
        censusParams.maxDisp = params_.maxDisp;
        auto censusCostComputor = CensusCost::create(censusParams);
        censusCostComputor->compute(left, right, censusCost);
    }

    const int halfWidth = params_.windowWidth / 2;
    const int halfHeight = params_.windowHeight / 2;
    const int dispRange = params_.maxDisp - params_.minDisp;

    if (out.empty())
        out = Mat(left.size(), CV_32FC(dispRange));

#pragma omp parallel for default(shared) schedule(static)
    for (int i = 0; i < out.rows; ++i) {
        for (int j = 0; j < out.cols; ++j) {

            if (j < halfWidth || j > out.cols - halfWidth - 1 ||
                i < halfHeight || i > out.rows - halfHeight - 1) {
                for (int d = 0; d < dispRange; ++d) {
                    out.ptr<float>(i)[dispRange * j + d] = FLT_MAX;
                }
                continue;
            }

            for (int d = 0; d < dispRange; ++d) {
                out.ptr<float>(i)[dispRange * j + d] =
                    1 -
                    exp(-adCost.ptr<float>(i)[dispRange * j + d] /
                        params_.adWeight) +
                    1 -
                    exp(-censusCost.ptr<float>(i)[dispRange * j + d] /
                        params_.censusWeight);
            }
        }
    }
}

Ptr<CostComputer> ADCensusCost::create(const Params params) {
    return Ptr<ADCensusCostImpl>(new ADCensusCostImpl(params));
}
} // namespace libSM
#include "sgm.h"
#include "costCompute/censusCost.h"
#include "costAggregation/multipathAggregation.h"
#include "dispCompute/dispCompute.h"
#include "dispOptimiztion/dispOptimiztion.h"

#include <opencv2/opencv.hpp>

using namespace cv;
using namespace std;

namespace libSM {
/**
 * @brief SGM algorithm's implement
 * 
 */
class SGMImpl : public SGM {
  public:
    SGMImpl(const Params params) : params_(params) {}
    void match(const cv::Mat &left, const cv::Mat &right,
               cv::Mat &dispMap) override;
  private:
    Params params_;
};

void SGMImpl::match(const cv::Mat &left, const cv::Mat &right, cv::Mat &dispMap) {
    CV_Assert_N(!left.empty(), !right.empty(), left.type() == CV_8UC1 || left.type() == CV_8UC3, right.type() == CV_8UC1 || right.type() == CV_8UC3);

    Mat leftProcess, rightProcess;

    if(left.type() == CV_8UC3)
        cvtColor(left, leftProcess, COLOR_BGR2GRAY);
    else
        leftProcess = left;

    if(right.type() == CV_8UC3)
        cvtColor(right, rightProcess, COLOR_BGR2GRAY);
    else
        rightProcess = right;
    
    //cost compute
    Mat cost;
    {
        auto params = CensusCost::Params();
        params.windowWidth = params_.windowWidth;
        params.windowHeight = params_.windowHeight;
        params.minDisp = params_.minDisp;
        params.maxDisp = params_.maxDisp;

        auto adCensusComputer = CensusCost::create(params);
        adCensusComputer->compute(leftProcess, rightProcess, cost);
    }

    //cost aggregation
    Mat aggregatedCost;
    {
        auto params = MultipathAggregation::Params();
        params.P1 = params_.P1;
        params.P2 = params_.P2;
        params.enableHonrizon = params_.enableHonrizon;
        params.enableVertiacl = params.enableVertiacl;
        params.enableNegtive45 = params_.enableNegtive45;
        params.enablePostive45 = params_.enablePostive45;

        auto multipathAggregator = MultipathAggregation::create(params);
        multipathAggregator->aggregation(leftProcess, cost, aggregatedCost);
    }

    cost.release();

    //disparity compute
    Mat disp;
    {
        auto params = DispComputeParams();
        params.enableLRCheck = params_.enableLRCheck;
        params.enableUniqueCheck = params_.enableUniqueCheck;
        params.enableSubpixelFitting = params_.enableSubpixelFitting;
        params.lrCheckThreshod = params_.lrCheckThreshod;
        params.uniquenessRatio = params_.uniquenessRatio;
        params.minDisp = params_.minDisp;
        params.maxDisp = params_.maxDisp;

        winnerTakesAll(aggregatedCost, disp, params);
    }

    aggregatedCost.release();

    //disparity optimiztion
    {
        auto params = DispOptParams();
        params.enableRemoveSmallArea = params_.enableRemoveSmallArea;
        params.dispDomainThreshold = params_.dispDomainThreshold;
        params.smallAreaThreshold = params_.smallAreaThreshold;
        params.enableBilateralFilter = params_.enableBilateralFilter;
        params.d = params_.d;
        params.sigmaColor = params_.sigmaColor;
        params.sigmaSpace = params_.sigmaSpace;
        params.enableMedianFilter = params_.enableMedianFilter;
        params.k = params_.k;
        params.enableDispFill = params_.enableDispFill;

        dispOptimiz(disp, dispMap, params);
    }
}

Ptr<libSM::Algorithm> SGM::create(const Params params) {
    return Ptr<libSM::Algorithm>(new SGMImpl(params));
}

} // namespace libSM
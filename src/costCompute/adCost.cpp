#include "adCost.h"

#include <omp.h>

#include <opencv2/opencv.hpp>

using namespace cv;

namespace libSM {
/**
 * @brief implementation class for the ADCost interface.
 *
 */
class ADCostImpl : public ADCost {
  public:
    ADCostImpl(const Params params) : params_(params) {}
    void compute(const Mat &left, const Mat &right, Mat &out) override;

  private:
    /**
     * @brief clculate the AD cost within the window
     * 
     * @param leftROI Left image ROI
     * @param right right image
     * @param rx right image x-coordinate
     * @param ry right image y-coordinate
     * @return float ad cost of the window
     */
    float getWindowPixelsCost(const Mat &leftROI, const Mat &right, const int rx,
                              const int ry);
    Params params_;
};

float ADCostImpl::getWindowPixelsCost(const Mat &leftROI, const Mat &right,
                                      const int rx, const int ry) {
    const int halfWidth = params_.windowWidth / 2;
    const int halfHeight = params_.windowHeight / 2;

    float cost = 0.f;

    for (int i = -halfHeight; i <= halfHeight; ++i) {
        for (int j = -halfWidth; j <= halfWidth; ++j) {
            if (leftROI.type() == CV_8UC3) {
                auto leftPixel = leftROI.ptr<Vec3b>(halfHeight + i)[halfWidth + j];
                auto rightPixel = right.ptr<Vec3b>(ry + i)[rx + j];
                cost += (abs(static_cast<float>(leftPixel[0]) - rightPixel[0]) +
                         abs(static_cast<float>(leftPixel[1]) - rightPixel[1]) +
                         abs(static_cast<float>(leftPixel[2]) - rightPixel[2])) / 3.f;
            } else {
                cost += abs(static_cast<float>(leftROI.ptr<uchar>(
                                halfHeight + i)[halfWidth + j]) -
                            right.ptr<uchar>(ry + i)[rx + j]);
            }
        }
    }

    return cost;
}

void ADCostImpl::compute(const Mat &left, const Mat &right, Mat &out) {
    CV_Assert_N(!left.empty(), !right.empty(), left.size == right.size);

    const int dispRange = params_.maxDisp - params_.minDisp;

    if (out.empty())
        out = Mat(left.size(), CV_32FC(dispRange));

    const int halfWidth = params_.windowWidth / 2;
    const int halfHeight = params_.windowHeight / 2;

#pragma omp parallel for default(shared) schedule(static)
    for (int i = 0; i < out.rows; ++i) {
        for (int j = 0; j < out.cols; ++j) {

            if(j < halfWidth || j > out.cols - halfWidth - 1 || i < halfHeight || i > out.rows - halfHeight - 1) {
                for (int d = 0; d < dispRange; ++d) { out.ptr<float>(i)[dispRange * j + d] = FLT_MAX; }
                continue;
            }

            auto leftROI = left(Rect(j - halfWidth, i - halfHeight,
                                    params_.windowWidth, params_.windowHeight));

            for (int d = 0; d < dispRange; ++d) {
                if(j - d < halfWidth || j - d > out.cols - halfWidth) {
                    out.ptr<float>(i)[dispRange * j + d] = FLT_MAX;
                    continue;
                }

                out.ptr<float>(i)[dispRange * j + d] =
                    getWindowPixelsCost(leftROI, right, j - d, i);
            }
        }
    }
}

Ptr<CostComputer> ADCost::create(const Params params) {
    return Ptr<ADCostImpl>(new ADCostImpl(params));
}
} // namespace libSM
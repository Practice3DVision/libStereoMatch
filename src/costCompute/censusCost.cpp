#include "censusCost.h"

#include <omp.h>

#include <opencv2/opencv.hpp>

using namespace cv;

namespace libSM {
/**
 * @brief implementation class for the CensusCost interface.
 *
 */
class CensusCostImpl : public CensusCost {
  public:
    CensusCostImpl(const Params params) : params_(params) {}
    void compute(const Mat &left, const Mat &right, Mat &out) override;

  private:
    /**
     * @brief clculate the AD cost within the window
     *
     * @param img image
     * @param x image x-coordinate
     * @param y image y-coordinate
     * @return uint64_t census of the window
     */
    uint64_t getWindowPixelsCensus(const Mat &img, const int x, const int y);
    /**
     * @brief calculate hamming distance
     *
     * @param lhs left census
     * @param rhs right census
     * @return uint64_t
     */
    uint8_t hammingDistance(uint64_t lhs, uint64_t rhs);
    Params params_;
};

uint64_t CensusCostImpl::getWindowPixelsCensus(const Mat &img, const int x,
                                               const int y) {
    const int halfWidth = params_.windowWidth / 2;
    const int halfHeight = params_.windowHeight / 2;

    const uchar centerGray = img.ptr<uchar>(y)[x];
    uint64_t census = 0;

    for (int i = -halfHeight; i <= halfHeight; ++i) {
        for (int j = -halfWidth; j <= halfWidth; ++j) {
            census += (img.ptr<uchar>(y + i)[x + j] > centerGray);
            if(i != halfHeight || j != halfWidth) {
                census <<= 1;
            }
        }
    }

    return census;
}

uint8_t CensusCostImpl::hammingDistance(uint64_t lhs, uint64_t rhs) {
    uint8_t distance = 0;
    const int itemCount = params_.windowHeight * params_.windowWidth;
    int iterationCount = 0;

    while (iterationCount++ < itemCount) {
        distance += (lhs & 0x01) ^ (rhs & 0x01);
        lhs >>= 1;
        rhs >>= 1;
    }

    return distance;
}

void CensusCostImpl::compute(const Mat &left, const Mat &right, Mat &out) {
    CV_Assert_N(!left.empty(), !right.empty(), left.size == right.size,
                left.type() == CV_8UC1, right.type() == CV_8UC1);

    const int dispRange = params_.maxDisp - params_.minDisp;

    if (out.empty())
        out = Mat(left.size(), CV_32FC(dispRange));

    Mat leftCensus = Mat(left.size(), CV_8UC(8), cv::Scalar(0));
    Mat rightCensus = Mat(right.size(), CV_8UC(8), cv::Scalar(0));

    const int halfWidth = params_.windowWidth / 2;
    const int halfHeight = params_.windowHeight / 2;

#pragma omp parallel for default(shared) schedule(static)
    for (int i = halfHeight; i < out.rows - halfHeight; ++i) {
        auto ptrLeftCensus = leftCensus.ptr<uint64_t>(i);
        auto ptrRightCensus = rightCensus.ptr<uint64_t>(i);
        for (int j = halfWidth; j < out.cols - halfWidth; ++j) {
            ptrLeftCensus[j] = getWindowPixelsCensus(left, j, i);
            ptrRightCensus[j] = getWindowPixelsCensus(right, j, i);
        }
    }

#pragma omp parallel for default(shared) schedule(static)
    for (int i = halfHeight; i < out.rows - halfHeight; ++i) {
        auto ptrLeftCensus = leftCensus.ptr<uint64_t>(i);
        auto ptrRightCensus = rightCensus.ptr<uint64_t>(i);
        for (int j = halfWidth; j < out.cols - halfWidth; ++j) {
            auto leftCensusVal = ptrLeftCensus[j];
            for (int d = 0; d < dispRange; ++d) {
                if (j - d < halfWidth || j - d > out.cols - halfWidth) {
                    out.ptr<float>(i)[dispRange * j + d] = FLT_MAX;
                    continue;
                }

                out.ptr<float>(i)[dispRange * j + d] = hammingDistance(leftCensusVal, ptrRightCensus[j - d]);
            }
        }
    }
}

Ptr<CostComputer> CensusCost::create(const Params params) {
    return Ptr<CensusCostImpl>(new CensusCostImpl(params));
}
} // namespace libSM
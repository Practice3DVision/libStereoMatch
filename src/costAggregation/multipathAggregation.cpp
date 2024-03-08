#include "multipathAggregation.h"

#include <opencv2/opencv.hpp>

using namespace cv;
using namespace std;

namespace libSM {
class MultipathAggregationImpl : public MultipathAggregation {
  public:
    MultipathAggregationImpl(const Params params) : params_(params){};
    void aggregation(const cv::Mat &left, const cv::Mat &cost,
                     cv::Mat &aggregationCost) override;

  private:
    /**
     * @brief aggregation cost horizontally
     *
     * @param left  left image
     * @param cost cost space
     * @param aggregationCost aggregated cost
     * @param leftToRight from left to right
     */
    void aggregationHorizontal(const cv::Mat &left, const Mat &cost,
                               Mat &aggregationCost, bool leftToRight = true);
    /**
     * @brief aggregation cost vertically
     *
     * @param left  left image
     * @param cost cost space
     * @param aggregationCost aggregated cost
     * @param leftToRight from up to bottom
     */
    void aggregationVertical(const cv::Mat &left, const Mat &cost,
                             Mat &aggregationCost, bool upToBottom = true);
    /**
     * @brief aggregation cost on the negative 45-degree line
     *
     * @param left  left image
     * @param cost cost space
     * @param aggregationCost aggregated cost
     * @param topLeftToBottomRight from top-left to bottom-right
     */
    void aggregationNegative45(const cv::Mat &left, const Mat &cost,
                               Mat &aggregationCost,
                               bool topLeftToBottomRight = true);
    /**
     * @brief aggregation cost on the 45-degree line
     *
     * @param left  left image
     * @param cost cost space
     * @param aggregationCost aggregated cost
     * @param topRightToBottomLeft from top-right to bottom-left
     */
    void aggregationPostive45(const cv::Mat &left, const Mat &cost,
                              Mat &aggregationCost,
                              bool topRightToBottomLeft = true);
    Params params_;
};

void MultipathAggregationImpl::aggregationHorizontal(const cv::Mat &left,
                                                     const Mat &cost,
                                                     Mat &aggregationCost,
                                                     bool leftToRight) {
    const int beginLoc = leftToRight ? 0 : cost.cols - 1;
    const int endLoc = leftToRight ? cost.cols : 0;
    const int direction = leftToRight ? 1 : -1;

#pragma omp parallel for schedule(dynamic) default(shared)
    for (int i = 0; i < cost.rows; ++i) {
        auto ptrCost = cost.ptr<float>(i);
        auto ptrAggregationCost = aggregationCost.ptr<float>(i);
        auto ptrLeft = left.ptr<uchar>(i);

        vector<float> lastCost(cost.channels() + 2, FLT_MAX);
        float lastMin = FLT_MAX;

        for (int d = 0; d < cost.channels(); ++d) {
            auto curCost = ptrCost[cost.channels() * beginLoc + d];
            lastMin = min(lastMin, curCost);
            lastCost[d + 1] = curCost;
        }

        for (int j = beginLoc + direction; j != endLoc; j += direction) {
            float curLocMinCost = FLT_MAX;

            for (int d = 0; d < cost.channels(); ++d) {
                auto lastCurDispCost = lastCost[d + 1];
                auto lastPreDispCost = lastCost[d] + params_.P1;
                auto lastAftDispCost = lastCost[d + 2] + params_.P1;
                auto lastElseDispCost =
                    lastMin +
                    max(params_.P2 /
                            (max(abs(ptrLeft[j] - ptrLeft[j - direction]), 1)),
                        params_.P1);

                auto curCost = ptrCost[cost.channels() * j + d] +
                               min(min(lastCurDispCost, lastPreDispCost),
                                   min(lastAftDispCost, lastElseDispCost)) -
                               lastMin;

                curLocMinCost = min(curLocMinCost, curCost);
                ptrAggregationCost[cost.channels() * j + d] = curCost;
            }

            for (int d = 0; d < cost.channels(); ++d) {
                lastCost[d + 1] = ptrAggregationCost[cost.channels() * j + d];
            }

            lastMin = curLocMinCost;
        }
    }
}

void MultipathAggregationImpl::aggregationVertical(const cv::Mat &left,
                                                   const Mat &cost,
                                                   Mat &aggregationCost,
                                                   bool upToBottom) {
    const int beginLoc = upToBottom ? 0 : cost.rows - 1;
    const int endLoc = upToBottom ? cost.rows : 0;
    const int direction = upToBottom ? 1 : -1;

#pragma omp parallel for schedule(dynamic) default(shared)
    for (int j = 0; j < cost.cols; ++j) {
        vector<float> lastCost(cost.channels() + 2, FLT_MAX);
        float lastMin = FLT_MAX;
        uchar lastPixel = left.ptr<uchar>(beginLoc)[j];

        for (int d = 0; d < cost.channels(); ++d) {
            auto curCost = cost.ptr<float>(beginLoc)[cost.channels() * j + d];
            lastCost[d + 1] = curCost;

            if (curCost < lastMin) {
                lastMin = curCost;
            }
        }

        for (int i = beginLoc + direction; i != endLoc; i = i + direction) {
            auto ptrCurCost = cost.ptr<float>(i);
            auto ptrCurLeft = left.ptr<uchar>(i);
            auto ptrAggregationCost = aggregationCost.ptr<float>(i);
            float curLocMinCost = FLT_MAX;

            for (int d = 0; d < cost.channels(); ++d) {
                auto lastCurDispCost = lastCost[d + 1];
                auto lastPreDispCost = lastCost[d] + params_.P1;
                auto lastAftDispCost = lastCost[d + 2] + params_.P1;
                auto lastElseDispCost =
                    lastMin +
                    max(params_.P2 / (max(abs(ptrCurLeft[j] - lastPixel), 1)),
                        params_.P1);

                auto curCost = ptrCurCost[cost.channels() * j + d] +
                               min(min(lastCurDispCost, lastPreDispCost),
                                   min(lastAftDispCost, lastElseDispCost)) -
                               lastMin;

                curLocMinCost = min(curLocMinCost, curCost);
                ptrAggregationCost[cost.channels() * j + d] = curCost;
            }

            for (int d = 0; d < cost.channels(); ++d) {
                lastCost[d + 1] = ptrAggregationCost[cost.channels() * j + d];
            }

            lastMin = curLocMinCost;
            lastPixel = ptrCurLeft[j];
        }
    }
}

void MultipathAggregationImpl::aggregationPostive45(const cv::Mat &left,
                                                    const Mat &cost,
                                                    Mat &aggregationCost,
                                                    bool topRightToBottomLeft) {
    const int beginLocY = topRightToBottomLeft ? 0 : cost.rows - 1;
    const int endLocY = topRightToBottomLeft ? cost.rows : 0;
    const int directionY = topRightToBottomLeft ? 1 : -1;
    const int beginLocX = topRightToBottomLeft ? cost.cols - 1 : 0;
    const int endLocX = topRightToBottomLeft ? 0 : cost.cols;
    const int directionX = topRightToBottomLeft ? -1 : 1;

    // Not directly using '!=' to avoid OpenMP's inability to statically
    // determine the iteration count.
#pragma omp parallel for schedule(dynamic) default(shared)
    for (int indexX = 0; indexX < cost.cols; ++indexX) {
        int j = beginLocX + indexX * directionX;
        vector<float> lastCost(cost.channels() + 2, FLT_MAX);
        float lastMin = FLT_MAX;
        uchar lastPixel = left.ptr<uchar>(beginLocY)[j];

        for (int d = 0; d < cost.channels(); ++d) {
            auto curCost = cost.ptr<float>(beginLocY)[cost.channels() * j + d];
            lastCost[d + 1] = curCost;

            if (curCost < lastMin) {
                lastMin = curCost;
            }
        }

        int curLinej = j + directionX;

        if (curLinej < 0) {
            curLinej = cost.cols - 1;
        } else if (curLinej > cost.cols - 1) {
            curLinej = 0;
        }

        for (int i = beginLocY + directionY; i != endLocY; i += directionY) {
            auto ptrCurCost = cost.ptr<float>(i);
            auto ptrCurLeft = left.ptr<uchar>(i);
            auto ptrAggregationCost = aggregationCost.ptr<float>(i);
            float curLocMinCost = FLT_MAX;

            for (int d = 0; d < cost.channels(); ++d) {
                auto lastCurDispCost = lastCost[d + 1];
                auto lastPreDispCost = lastCost[d] + params_.P1;
                auto lastAftDispCost = lastCost[d + 2] + params_.P1;
                auto lastElseDispCost =
                    lastMin +
                    max(params_.P2 /
                            (max(abs(ptrCurLeft[curLinej] - lastPixel), 1)),
                        params_.P1);

                auto curCost = ptrCurCost[cost.channels() * curLinej + d] +
                               min(min(lastCurDispCost, lastPreDispCost),
                                   min(lastAftDispCost, lastElseDispCost)) -
                               lastMin;

                curLocMinCost = min(curLocMinCost, curCost);
                ptrAggregationCost[cost.channels() * curLinej + d] = curCost;
            }

            for (int d = 0; d < cost.channels(); ++d) {
                lastCost[d + 1] =
                    ptrAggregationCost[cost.channels() * curLinej + d];
            }

            lastMin = curLocMinCost;
            lastPixel = ptrCurLeft[curLinej];
            curLinej += directionX;

            if (curLinej < 0) {
                curLinej = cost.cols - 1;
            } else if (curLinej > cost.cols - 1) {
                curLinej = 0;
            }
        }
    }
}

void MultipathAggregationImpl::aggregationNegative45(
    const cv::Mat &left, const Mat &cost, Mat &aggregationCost,
    bool topLeftToBottomRight) {
    const int beginLocY = topLeftToBottomRight ? 0 : cost.rows - 1;
    const int endLocY = topLeftToBottomRight ? cost.rows : 0;
    const int directionY = topLeftToBottomRight ? 1 : -1;
    const int beginLocX = topLeftToBottomRight ? 0 : cost.cols - 1;
    const int endLocX = topLeftToBottomRight ? cost.cols : 0;
    const int directionX = topLeftToBottomRight ? 1 : -1;
    
    // Not directly using '!=' to avoid OpenMP's inability to statically
    // determine the iteration count.
#pragma omp parallel for schedule(dynamic) default(shared)
    for (int indexX = 0; indexX < cost.cols; ++indexX) {
        int j = beginLocX + indexX * directionX;
        vector<float> lastCost(cost.channels() + 2, FLT_MAX);
        float lastMin = FLT_MAX;
        uchar lastPixel = left.ptr<uchar>(beginLocY)[j];

        for (int d = 0; d < cost.channels(); ++d) {
            auto curCost = cost.ptr<float>(beginLocY)[cost.channels() * j + d];
            lastCost[d + 1] = curCost;

            if (curCost < lastMin) {
                lastMin = curCost;
            }
        }

        int curLinej = j + directionX;

        if (curLinej < 0) {
            curLinej = cost.cols - 1;
        } else if (curLinej > cost.cols - 1) {
            curLinej = 0;
        }

        for (int i = beginLocY + directionY; i < endLocY; i += directionY) {
            auto ptrCurCost = cost.ptr<float>(i);
            auto ptrCurLeft = left.ptr<uchar>(i);
            auto ptrAggregationCost = aggregationCost.ptr<float>(i);
            float curLocMinCost = FLT_MAX;

            for (int d = 0; d < cost.channels(); ++d) {
                auto lastCurDispCost = lastCost[d + 1];
                auto lastPreDispCost = lastCost[d] + params_.P1;
                auto lastAftDispCost = lastCost[d + 2] + params_.P1;
                auto lastElseDispCost =
                    lastMin +
                    max(params_.P2 /
                            (max(abs(ptrCurLeft[curLinej] - lastPixel), 1)),
                        params_.P1);

                auto curCost = ptrCurCost[cost.channels() * curLinej + d] +
                               min(min(lastCurDispCost, lastPreDispCost),
                                   min(lastAftDispCost, lastElseDispCost)) -
                               lastMin;

                curLocMinCost = min(curLocMinCost, curCost);
                ptrAggregationCost[cost.channels() * curLinej + d] = curCost;
            }

            for (int d = 0; d < cost.channels(); ++d) {
                lastCost[d + 1] =
                    ptrAggregationCost[cost.channels() * curLinej + d];
            }

            lastPixel = ptrCurLeft[curLinej];
            lastMin = curLocMinCost;
            curLinej += directionX;

            if (curLinej < 0) {
                curLinej = cost.cols - 1;
            } else if (curLinej > cost.cols - 1) {
                curLinej = 0;
            }
        }
    }
}

void MultipathAggregationImpl::aggregation(const cv::Mat &left,
                                           const cv::Mat &cost,
                                           cv::Mat &aggregationCost) {
    CV_Assert(!cost.empty());

    if (aggregationCost.empty())
        aggregationCost = Mat(cost.size(), cost.type(), Scalar(0.f));

    if (params_.enableHonrizon) {
        Mat temp = Mat(cost.size(), cost.type(), Scalar(0.f));
        aggregationHorizontal(left, cost, temp, true);
        aggregationCost += temp;
        aggregationHorizontal(left, cost, temp, false);
        aggregationCost += temp;
    }

    if (params_.enableVertiacl) {
        Mat temp = Mat(cost.size(), cost.type(), Scalar(0.f));
        aggregationVertical(left, cost, temp, true);
        aggregationCost += temp;
        aggregationVertical(left, cost, temp, false);
        aggregationCost += temp;
    }

    if (params_.enablePostive45) {
        Mat temp = Mat(cost.size(), cost.type(), Scalar(0.f));
        aggregationPostive45(left, cost, temp, true);
        aggregationCost += temp;
        aggregationPostive45(left, cost, temp, false);
        aggregationCost += temp;
    }

    if (params_.enableNegtive45) {
        Mat temp = Mat(cost.size(), cost.type(), Scalar(0.f));
        aggregationNegative45(left, cost, temp, true);
        aggregationCost += temp;
        aggregationNegative45(left, cost, temp, false);
        aggregationCost += temp;
    }
}

Ptr<CostAggregation> MultipathAggregation::create(const Params params) {
    return Ptr<CostAggregation>(new MultipathAggregationImpl(params));
}

} // namespace libSM
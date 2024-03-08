#include <gtest/gtest.h>

#include <opencv2/opencv.hpp>

#include <libStereoMatch.h>

using namespace cv;
using namespace std;
using namespace libSM;

const string CONES_DATA_SET_PATH = "../../data/cones/";
const string TEDDY_DATA_SET_PATH = "../../data/teddy/";

class Cones : public testing::Test {
    protected:
        void SetUp() override {
            left = imread(CONES_DATA_SET_PATH + "im2.png", IMREAD_UNCHANGED);
            right = imread(CONES_DATA_SET_PATH + "im6.png", IMREAD_UNCHANGED);
        }

    public:
        Mat left;
        Mat right;
        void transformToGray() {
            if(left.type() == CV_8UC3) 
                cvtColor(left, left, COLOR_BGR2GRAY);
            
            if(right.type() == CV_8UC3) 
                cvtColor(right, right, COLOR_BGR2GRAY);
        }
};

class Teddy : public testing::Test {
    protected:
        void SetUp() override {
            left = imread(TEDDY_DATA_SET_PATH + "im2.png", IMREAD_UNCHANGED);
            right = imread(TEDDY_DATA_SET_PATH + "im6.png", IMREAD_UNCHANGED);
        }
    public:
        Mat left;
        Mat right;
        void transformToGray() {
            if(left.type() == CV_8UC3) 
                cvtColor(left, left, COLOR_BGR2GRAY);
            
            if(right.type() == CV_8UC3) 
                cvtColor(right, right, COLOR_BGR2GRAY);
        }
};

TEST_F(Cones, testMultipathAggregation) {
    transformToGray();

    Mat cost;
    {
        auto params = CensusCost::Params();
        params.windowWidth = 9;
        params.windowHeight = 7;
        params.minDisp = 0;
        params.maxDisp = 64;

        auto adCensusComputer = CensusCost::create(params);
        adCensusComputer->compute(left, right, cost);
    }

    Mat aggregatedCost;
    {
        auto params = MultipathAggregation::Params();
        params.P1 = 10.f;
        params.P2 = 150.f;
        params.enableHonrizon = true;
        params.enableNegtive45 = true;
        params.enablePostive45 = true;
        params.enableVertiacl = true;

        auto multipathAggregator = MultipathAggregation::create(params);
        multipathAggregator->aggregation(left, cost, aggregatedCost);
    }

    Mat disp;
    {
        auto params = DispComputeParams();
        params.lrCheck = true;
        params.uniqueCheck = true;
        params.subpixelFitting = true;
        params.lrCheckThreshod = 1;
        params.uniquenessRatio = 0.95f;
        params.minDisp = 0;
        params.maxDisp = 64;

        winnerTakesAll(aggregatedCost, disp, params);
    }


    Mat optimizedDisp;
    {
        auto params = DispOptParams();
        params.removeSmallArea = true;
        params.dispDomainThreshold = 2;
        params.smallAreaThreshold = 10;
        params.bilateralFilter = true;
        params.d = 10;
        params.sigmaColor = 10;
        params.sigmaSpace = 20;

        dispOptimiz(disp, optimizedDisp, params);
    }

    ASSERT_LE(abs(optimizedDisp.ptr<float>(301)[308] - 40), 1.f);
}
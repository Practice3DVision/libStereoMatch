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

TEST_F(Cones, testWinnerTakesAll) {
    Mat cost;
    {
        auto params = ADCensusCost::Params();
        params.windowWidth = 9;
        params.windowHeight = 7;
        params.minDisp = 0;
        params.maxDisp = 64;
        params.adWeight = 10.f;
        params.censusWeight = 30.f;

        transformToGray();

        auto adCensusComputer = ADCensusCost::create(params);
        adCensusComputer->compute(left, right, cost);
    }

    Mat disp;
    {
        auto params = DispComputeParams();
        params.lrCheck = true;
        params.uniqueCheck = true;
        params.subpixelFitting = true;
        params.lrCheckThreshod = 1;
        params.uniquenessRatio = 0.98f;
        params.minDisp = 0;
        params.maxDisp = 64;

        winnerTakesAll(cost, disp, params);

    }

    ASSERT_LE(abs(disp.ptr<float>(301)[308] - 40), 1.f);
}
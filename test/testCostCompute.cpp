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

TEST_F(Cones, testADCostColor) {
    auto params = ADCost::Params();
    params.windowWidth = 9;
    params.windowHeight = 7;
    params.minDisp = 0;
    params.maxDisp = 4;
    auto adCostComputer = ADCost::create(params);
    Mat out;
    adCostComputer->compute(left, right, out);

    ASSERT_LE(abs(out.ptr<Vec4f>(9)[9][0] - 1671.3), 1);
}

TEST_F(Cones, testADCostGray) {
    auto params = ADCost::Params();
    params.windowWidth = 9;
    params.windowHeight = 7;
    params.minDisp = 0;
    params.maxDisp = 4;
    auto adCostComputer = ADCost::create(params);
    Mat out;
    transformToGray();
    adCostComputer->compute(left, right, out);

    ASSERT_LE(abs(out.ptr<Vec4f>(17)[17][0] - 860), 1);
}

TEST_F(Cones, testCensusCost) {
    auto params = CensusCost::Params();
    params.windowWidth = 9;
    params.windowHeight = 7;
    params.minDisp = 0;
    params.maxDisp = 4;
    auto censusCostComputer = CensusCost::create(params);
    Mat out;
    transformToGray();
    censusCostComputer->compute(left, right, out);

    ASSERT_EQ(out.ptr<Vec4f>(25)[25][0], 24);
    ASSERT_EQ(out.ptr<Vec4f>(25)[25][1], 24);
}

TEST_F(Cones, testADCensusCost) {
    auto params = ADCensusCost::Params();
    params.windowWidth = 9;
    params.windowHeight = 7;
    params.minDisp = 0;
    params.maxDisp = 4;
    params.adWeight = 10.f;
    params.censusWeight = 30.f;
    auto adCensusComputer = ADCensusCost::create(params);
    Mat out;
    transformToGray();
    adCensusComputer->compute(left, right, out);
}
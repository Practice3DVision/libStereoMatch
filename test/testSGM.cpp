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

TEST_F(Cones, testSGMGray) {
    transformToGray();

    auto params = SGM::Params();
    auto sgm = SGM::create(params);

    Mat disparityMap;
    sgm->match(left, right, disparityMap);

    ASSERT_LE(abs(disparityMap.ptr<float>(301)[308] - 40), 1.f);
}

TEST_F(Cones, testSGMColor) {
    auto params = SGM::Params();
    auto sgm = SGM::create(params);

    Mat disparityMap;
    sgm->match(left, right, disparityMap);

    ASSERT_LE(abs(disparityMap.ptr<float>(301)[308] - 40), 1.f);
}
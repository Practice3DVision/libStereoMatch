#include <benchmark/benchmark.h>

#include <libStereoMatch.h>

#include <opencv2/opencv.hpp>

using namespace libSM;
using namespace cv;
using namespace std;

const string CONES_DATA_SET_PATH = "../../data/cones/";
const string TEDDY_DATA_SET_PATH = "../../data/teddy/";

class Cones : public benchmark::Fixture {
  protected:
    void SetUp(const benchmark::State &) override {
        left = imread(CONES_DATA_SET_PATH + "im2.png", IMREAD_UNCHANGED);
        right = imread(CONES_DATA_SET_PATH + "im6.png", IMREAD_UNCHANGED);
    }

  public:
    Mat left;
    Mat right;
    void transformToGray() {
        if (left.type() == CV_8UC3)
            cvtColor(left, left, COLOR_BGR2GRAY);

        if (right.type() == CV_8UC3)
            cvtColor(right, right, COLOR_BGR2GRAY);
    }
};

BENCHMARK_DEFINE_F(Cones, perfADCostColor)(benchmark::State &state) {
    auto params = ADCost::Params();
    params.windowWidth = 9;
    params.windowHeight = 7;
    params.minDisp = 0;
    params.maxDisp = state.range(0);
    auto adCostComputer = ADCost::create(params);
    Mat out;
    for (auto _ : state) {
        adCostComputer->compute(left, right, out);
    }
};

BENCHMARK_DEFINE_F(Cones, perfADCostGray)(benchmark::State &state) {
    auto params = ADCost::Params();
    params.windowWidth = 9;
    params.windowHeight = 7;
    params.minDisp = 0;
    params.maxDisp = state.range(0);
    auto adCostComputer = ADCost::create(params);
    Mat out;
    transformToGray();
    for (auto _ : state) {
        adCostComputer->compute(left, right, out);
    }
};

BENCHMARK_DEFINE_F(Cones, perfCensusCost)(benchmark::State& state) {
    auto params = CensusCost::Params();
    params.windowWidth = 9;
    params.windowHeight = 7;
    params.minDisp = 0;
    params.maxDisp = state.range(0);
    auto censusCostComputer = CensusCost::create(params);
    Mat out;
    transformToGray();
    for (auto _ : state) {
        censusCostComputer->compute(left, right, out);
    }
}

BENCHMARK_DEFINE_F(Cones, perfADCensusCost)(benchmark::State& state) {
    auto params = ADCensusCost::Params();
    params.windowWidth = 9;
    params.windowHeight = 7;
    params.minDisp = 0;
    params.maxDisp = state.range(0);
    params.adWeight = 10.f;
    params.censusWeight = 30.f;
    auto adcensusCostComputer = ADCensusCost::create(params);
    Mat out;
    transformToGray();
    for (auto _ : state) {
        adcensusCostComputer->compute(left, right, out);
    }
}

BENCHMARK_REGISTER_F(Cones, perfADCostColor)->MeasureProcessCPUTime()->UseRealTime()->Unit(benchmark::TimeUnit::kSecond)->DenseRange(32, 256, 32);
BENCHMARK_REGISTER_F(Cones, perfADCostGray)->MeasureProcessCPUTime()->UseRealTime()->Unit(benchmark::TimeUnit::kSecond)->DenseRange(32, 256, 32);
BENCHMARK_REGISTER_F(Cones, perfCensusCost)->MeasureProcessCPUTime()->UseRealTime()->Unit(benchmark::TimeUnit::kSecond)->DenseRange(32, 256, 32);
BENCHMARK_REGISTER_F(Cones, perfADCensusCost)->MeasureProcessCPUTime()->UseRealTime()->Unit(benchmark::TimeUnit::kSecond)->DenseRange(32, 256, 32);

BENCHMARK_MAIN();
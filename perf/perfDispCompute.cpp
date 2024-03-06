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

BENCHMARK_DEFINE_F(Cones, perfWinnerTakesAll)(benchmark::State& state) {
    Mat cost;
    {
        auto params = ADCensusCost::Params();
        params.windowWidth = 9;
        params.windowHeight = 7;
        params.minDisp = 0;
        params.maxDisp = 128;
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
        params.maxDisp = state.range(0);

        for (auto _ : state) {
            winnerTakesAll(cost, disp, params);
        }
    }
}

BENCHMARK_REGISTER_F(Cones, perfWinnerTakesAll)->MeasureProcessCPUTime()->UseRealTime()->Unit(benchmark::TimeUnit::kSecond)->DenseRange(32, 256, 32);

BENCHMARK_MAIN();
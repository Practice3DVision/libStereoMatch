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

BENCHMARK_DEFINE_F(Cones, perfDispOptimiztion)(benchmark::State& state) {
    transformToGray();

    auto params = SGM::Params();
    params.maxDisp = state.range(0);
    auto sgm = SGM::create(params);

    Mat disparityMap;
    
    for (auto _ : state) {
        sgm->match(left, right, disparityMap);
    }
}

BENCHMARK_REGISTER_F(Cones, perfDispOptimiztion)->MeasureProcessCPUTime()->UseRealTime()->Unit(benchmark::TimeUnit::kSecond)->DenseRange(32, 256, 32);

BENCHMARK_MAIN();
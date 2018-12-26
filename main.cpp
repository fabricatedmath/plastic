#include <iostream>
#include <Eigen/Dense>
#include "eigen_boost_serialization.hpp"
#include "state.h"
#include "test.h"
#include "init.h"
#include "dataset.h"
#include <random>

using namespace std;
using namespace Eigen;

int main(int argc, char* argv[]) {

    std::default_random_engine generator;
    std::poisson_distribution<int> distribution(4.1);
    auto poisson = [&] (int) {return distribution(generator);};

    RowVectorXi v = RowVectorXi::NullaryExpr(10, poisson );
    std::cout << v << "\n";
    exit(0);

    MutableState mutableState;
    {
        mutableState.w = Init<float,int>::initW();
        mutableState.wff = Init<float,int>::initWff();
        mutableState.incomingSpikes = Init<float,int>::initIncomingSpikes();
        mutableState.firings = Init<float,int>::initFirings();

        mutableState.v = Init<float,int>::initV();
        mutableState.vprev = Init<float,int>::initVPrev();
        mutableState.vthresh = Init<float,int>::initVThresh();
        mutableState.vlongtrace = Init<float,int>::initVLongtrace();
        mutableState.vpos = Init<float,int>::initVPos();
        mutableState.vneg = Init<float,int>::initVNeg();

        mutableState.xplastLat = Init<float,int>::initXPlastLat();
        mutableState.xplastFF = Init<float,int>::initXPlastFF();

        mutableState.wadap = Init<float,int>::initWadap();
        mutableState.z = Init<float,int>::initZ();

        mutableState.isSpiking = Init<float,int>::initIsSpiking();
    }

    StaticState staticState;
    {
        staticState.input = Dataset<float>::retrieveTransformedDataset();
        staticState.delays = Init<float,int>::initDelays();
        staticState.altds = Init<float,int>::initALTDS();
    }

    Buffers buffers;
    {
        buffers.lgnfirings = Init<float,int>::initLgnFiringsBuffer();
        buffers.neuronInputs = Init<float,int>::initNeuronInputsBuffer();

        buffers.eachNeurLTD = Init<float,int>::initEachNeurLTD();
        buffers.eachNeurLTP = Init<float,int>::initEachNeurLTP();

    }

    cout << staticState.input.row(0).head(10) << endl;

    //    wrapper(mutableState, staticState, buffers);
}

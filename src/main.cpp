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

template<typename F, typename I>
void run() {
    MutableState<F,I> mutableState;
    {
        mutableState.w = Init<F,I>::initW();
        mutableState.wff = Init<F,I>::initWff();
        mutableState.incomingSpikes = Init<F,I>::initIncomingSpikes();
        mutableState.firings = Init<F,I>::initFirings();

        mutableState.v = Init<F,I>::initV();
        mutableState.vprev = Init<F,I>::initVPrev();
        mutableState.vthresh = Init<F,I>::initVThresh();
        mutableState.vlongtrace = Init<F,I>::initVLongtrace();
        mutableState.vpos = Init<F,I>::initVPos();
        mutableState.vneg = Init<F,I>::initVNeg();

        mutableState.xplastLat = Init<F,I>::initXPlastLat();
        mutableState.xplastFF = Init<F,I>::initXPlastFF();

        mutableState.wadap = Init<F,I>::initWadap();
        mutableState.z = Init<F,I>::initZ();

        mutableState.isSpiking = Init<F,I>::initIsSpiking();
    }

    StaticState<F,I> staticState;
    {
        staticState.input = Dataset<F>::retrieveTransformedDataset();
        staticState.delays = Init<F,I>::initDelays();
        staticState.altds = Init<F,I>::initALTDS();
    }

    Buffers<F,I> buffers;
    {
        buffers.lgnfirings = Init<F,I>::initLgnFiringsBuffer();
        buffers.poissonNoise = Init<F,I>::initPoissonNoiseBuffer();
        buffers.neuronInputs = Init<F,I>::initNeuronInputsBuffer();

        buffers.eachNeurLTD = Init<F,I>::initEachNeurLTD();
        buffers.eachNeurLTP = Init<F,I>::initEachNeurLTP();
    }

    wrapper(mutableState, staticState, buffers);
}

int main(int argc, char* argv[]) {
    run<float,int>();
}

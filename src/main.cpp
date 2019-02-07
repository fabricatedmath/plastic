#include <iostream>
#include <Eigen/Dense>
#include "eigen_boost_serialization.hpp"
#include "state.h"
#include "test.h"
#include "init.h"
#include "loader.h"
#include "dataset.h"
#include <random>
#include <boost/program_options.hpp>

namespace po = boost::program_options;

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

    run(mutableState, staticState, buffers);
}

template<typename F, typename I>
void runTesting() {
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

    RandomHistorical<F> randomHistorical;
    {
        randomHistorical.uniformMatrix = MatrixRX<F>::Random(NBSTEPSSTIM, FFRFSIZE);
        randomHistorical.posPoissonMatrix = MatrixRX<unsigned int>::Random(NBSTEPSPERPRES, NBNEUR);
        randomHistorical.negPoissonMatrix = MatrixRX<unsigned int>::Random(NBSTEPSPERPRES, NBNEUR);
    }

    run(mutableState, staticState, buffers, randomHistorical);
}

int main(int argc, char* argv[]) {

    MatrixRX<float> matrix = MatrixRX<float>::Random(2,2);
    cout << matrix << endl;
    storeMatrix("matrix",matrix);
    MatrixRX<float> matrix2 = loadMatrix<float>("matrix");
    cout << matrix2 << endl;


    exit(1);
    po::options_description desc("Usage");
    desc.add_options()
        ("help", "produce help message")
        ("double,D", "Use Doubles (default Floats)")
        ("testing,T", "Run in testing mode")
        ;

    po::variables_map opts;
    po::store(po::parse_command_line(argc, argv, desc), opts);
    if (opts.count("help")) {
        cout << desc << "\n";
        return 1;
    }

    if(opts.count("testing")) {
        if (opts.count("double")) {
            cout << "Running Doubles, Testing" << endl;
            runTesting<double,int>();
        } else {
            cout << "Running Floats, Testing" << endl;
            runTesting<float,int>();
        }
    } else {
        if (opts.count("double")) {
            cout << "Running Doubles" << endl;
            run<double,int>();
        } else {
            cout << "Running Floats" << endl;
            run<float,int>();
        }
    }
}

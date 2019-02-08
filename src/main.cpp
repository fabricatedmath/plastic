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

        mutableState.v = Init<F,I>::initV();
        mutableState.vthresh = Init<F,I>::initVThresh();
        mutableState.vlongtrace = Init<F,I>::initVLongtrace();
        mutableState.vpos = Init<F,I>::initVPos();
        mutableState.vneg = Init<F,I>::initVNeg();

        mutableState.xplastLat = Init<F,I>::initXPlastLat();
        mutableState.xplastFF = Init<F,I>::initXPlastFF();

        mutableState.wadap = Init<F,I>::initWadap();
        mutableState.z = Init<F,I>::initZ();
    }

    StaticState<F,I> staticState;
    {
        staticState.input = Dataset<F>::retrieveTransformedDataset();
        staticState.delays = Init<F,I>::initDelays();
        staticState.altds = Init<F,I>::initALTDS();
    }

    Buffers<F,I> buffers;
    {
        buffers.incomingSpikes = Init<F,I>::initIncomingSpikes();
        buffers.firings = Init<F,I>::initFirings();

        buffers.lgnfirings = Init<F,I>::initLgnFiringsBuffer();
        buffers.poissonNoise = Init<F,I>::initPoissonNoiseBuffer();
        buffers.neuronInputs = Init<F,I>::initNeuronInputsBuffer();

        buffers.eachNeurLTD = Init<F,I>::initEachNeurLTD();
        buffers.eachNeurLTP = Init<F,I>::initEachNeurLTP();
    }

    run(mutableState, staticState, buffers);
}

void runTesting(int i) {
    MutableState<double,int> mutableState;
    {
        string is = "-" + to_string(i);
        mutableState.w = loadMatrix<double>("data/w" + is);
        mutableState.wff = loadMatrix<double>("data/wff" + is);

        mutableState.v = loadVector<double>("data/v" + is);
        mutableState.vthresh = loadVector<double>("data/vthresh" + is);
        mutableState.vlongtrace = loadVector<double>("data/vlongtrace" + is);
        mutableState.vpos = loadVector<double>("data/vpos" + is);
        mutableState.vneg = loadVector<double>("data/vneg" + is);

        mutableState.xplastLat = loadVector<double>("data/xplastLat" + is);
        mutableState.xplastFF = loadVector<double>("data/xplastFF" + is);

        mutableState.wadap = loadVector<double>("data/wadap" + is);
        mutableState.z = loadVector<double>("data/z" + is);
    }

    StaticState<double,int> staticState;
    {
        staticState.input = Dataset<double>::retrieveTransformedDataset();
        staticState.delays = loadMatrix<int>("data/delays");
        staticState.altds = loadMatrix<double>("data/altds");
    }

    Buffers<double,int> buffers;
    {
        buffers.incomingSpikes = Init<double,int>::initIncomingSpikes();
        buffers.firings = Init<double,int>::initFirings();

        buffers.lgnfirings = Init<double,int>::initLgnFiringsBuffer();
        buffers.poissonNoise = Init<double,int>::initPoissonNoiseBuffer();
        buffers.neuronInputs = Init<double,int>::initNeuronInputsBuffer();

        buffers.eachNeurLTD = Init<double,int>::initEachNeurLTD();
        buffers.eachNeurLTP = Init<double,int>::initEachNeurLTP();
    }

    RandomHistorical<double> randomHistorical;
    {
        randomHistorical.uniformMatrix = loadMatrix<double>("data/randlgnrates");
        randomHistorical.posPoissonMatrix = loadMatrix<unsigned int>("data/posnoise");
        randomHistorical.negPoissonMatrix = loadMatrix<unsigned int>("data/negnoise");
    }

    run(mutableState, staticState, buffers, randomHistorical);
}

int main(int argc, char* argv[]) {
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
            runTesting(0);
        } else {
            cout << "Running Floats, Testing" << endl;
            cout << "no option to do this, dying" << endl;
            exit(1);
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

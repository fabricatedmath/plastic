#include <iostream>
#include <Eigen/Dense>
#include "eigen_boost_serialization.hpp"
#include "state.h"
#include "test.h"
#include "init.h"
#include "dataset.h"

using namespace std;
using namespace Eigen;

int main(int argc, char* argv[]) {
    MutableState mutableState;
    {
        MatrixW<float,int> w = Init<float,int>::initW();
        MatrixWff<float,int> wff = Init<float,int>::initWff();
        MatrixIncomingSpikes<float,int> incomingSpikes =
            Init<float,int>::initIncomingSpikes();
        VectorFirings<float,int> firings = Init<float,int>::initFirings();

        mutableState.w = w;
        mutableState.wff = wff;
        mutableState.incomingSpikes = incomingSpikes;
        mutableState.firings = firings;
    }

    StaticState staticState;
    {
        MatrixTransformedDataset<float> transformedDataset =
            Dataset<float>::retrieveTransformedDataset();
        MatrixDelays<float,int> delays = Init<float,int>::initDelays();

        staticState.input = transformedDataset;
        staticState.delays = delays;

        cout << staticState.input.row(0).head(10) << endl;
    }

    Buffers buffers;
    {
        MatrixLgnFiringsBuffer<float,int> lgnfiringsBuffer =
            Init<float,int>::initLgnFiringsBuffer();
        VectorNeuronInputsBuffer<float,int> neuronInputsBuffer =
            Init<float,int>::initNeuronInputsBuffer();

        buffers.lgnfirings = lgnfiringsBuffer;
        buffers.neuronInputs = neuronInputsBuffer;
    }

    wrapper(mutableState, staticState, buffers);
}

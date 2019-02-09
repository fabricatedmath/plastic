#pragma once

#include "state.h"

MutableState<double,int> loadMutableState(int i) {
    const string is = "-" + to_string(i);

    MutableState<double,int> mutableState;

    mutableState.w = loadMatrix<double>("data/w" + is);
    mutableState.wff = loadMatrix<double>("data/wff" + is);

    mutableState.vthresh = loadVector<double>("data/vthresh" + is);
    mutableState.vlongtrace = loadVector<double>("data/vlongtrace" + is);
    mutableState.vpos = loadVector<double>("data/vpos" + is);
    mutableState.vneg = loadVector<double>("data/vneg" + is);

    mutableState.xplastLat = loadVector<double>("data/xplastLat" + is);
    mutableState.xplastFF = loadVector<double>("data/xplastFF" + is);

    mutableState.wadap = loadVector<double>("data/wadap" + is);
    mutableState.z = loadVector<double>("data/z" + is);

    return mutableState;
}

StaticState<double,int> loadStaticState() {
    StaticState<double,int> staticState;

    staticState.input = Dataset<double>::retrieveTransformedDataset();
    staticState.delays = loadMatrix<int>("data/delays");
    staticState.altds = loadMatrix<double>("data/altds");

    return staticState;
}

RandomHistorical<double> loadRandomHistorical() {
    RandomHistorical<double> randomHistorical;

    randomHistorical.uniformMatrix = loadMatrix<double>("data/randlgnrates");
    randomHistorical.posPoissonMatrix = loadMatrix<unsigned int>("data/posnoise");
    randomHistorical.negPoissonMatrix = loadMatrix<unsigned int>("data/negnoise");

    return randomHistorical;
}

VectorX<double> loadV(int i) {
    const string is = "-" + to_string(i);

    return loadVector<double>("data/v" + is);
}

VectorX<double> loadInput(int i) {
    const string is = "-" + to_string(i);

    return loadVector<double>("data/i" + is);
}

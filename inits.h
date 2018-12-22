#pragma once
#include <fstream>
#include <Eigen/Dense>
#include "eigen_boost_serialization.hpp"

#include <boost/archive/binary_oarchive.hpp>
#include <boost/archive/binary_iarchive.hpp>

#include "type.h"
#include "constants.h"

using namespace std;

typedef Matrix<double,NBNEUR,NBNEUR,RowMajor> MatrixW;

MatrixW initW() {
    MatrixW w =  MatrixXd::Zero(NBNEUR, NBNEUR);
    w.bottomRows(NBI).leftCols(NBE).setRandom(); // Inhbitory neurons receive excitatory inputs from excitatory neurons
    w.rightCols(NBI).setRandom(); // Everybody receives inhibition (including inhibitory neurons)
    w.bottomRows(NBI).rightCols(NBI) =  -w.bottomRows(NBI).rightCols(NBI).cwiseAbs() * WII_MAX;
    w.topRows(NBE).rightCols(NBI) = -w.topRows(NBE).rightCols(NBI).cwiseAbs() * WIE_MAX;
    w.bottomRows(NBI).leftCols(NBE) = w.bottomRows(NBI).leftCols(NBE).cwiseAbs() * WEI_MAX;
    w = w - w.cwiseProduct(MatrixXd::Identity(NBNEUR, NBNEUR)); // Diagonal lateral weights are 0 (no autapses !)
    return w;
}

typedef Matrix<double,NBNEUR,FFRFSIZE,RowMajor> MatrixWff;

MatrixWff initWff() {
    MatrixWff wff = MatrixXd::Zero(NBNEUR, FFRFSIZE);
    wff =  (WFFINITMIN + (WFFINITMAX-WFFINITMIN) * MatrixXd::Random(NBNEUR, FFRFSIZE).cwiseAbs().array()).cwiseMin(MAXW);
    wff.bottomRows(NBI).setZero();
    return wff;
}

typedef Matrix<int8_t, Dynamic, FFRFSIZE, RowMajor> MatrixDataset;

MatrixDataset loadDataset() {
    int ffrfSize = FFRFSIZE;

    ifstream DataFile ("./patchesCenteredScaledBySumTo126ImageNetONOFFRotatedNewInt8.bin.dat", ios::in | ios::binary | ios::ate);
    if (!DataFile.is_open()) {
        throw ios_base::failure("Failed to open the binary data file!");
        //return -1;
    }
    ifstream::pos_type  fsize = DataFile.tellg();
    char *membuf = new char[fsize];
    DataFile.seekg (0, ios::beg);
    DataFile.read(membuf, fsize);
    DataFile.close();
    int8_t* imagedata = (int8_t*) membuf;

    int totaldatasize = fsize / sizeof(int8_t); // To change depending on whether the data is float/single (4) or double (8)
    int numRows = totaldatasize/ffrfSize;
    int numCols = ffrfSize;

    cout << "Data read!" << " total read: " << totaldatasize << endl;
    Map<VectorXu> mf (imagedata,totaldatasize);
    VectorXu v(mf);
    Map<MatrixDataset> mout(v.data(),numRows,numCols);
    return MatrixDataset(mout);
}

typedef Matrix<double,Dynamic,2*FFRFSIZE,RowMajor> MatrixTransformedDataset;

MatrixTransformedDataset transformDataset(MatrixDataset d) {
    Matrix<double,Dynamic,FFRFSIZE,RowMajor> dd = d.cast<double>();

    MatrixTransformedDataset out(dd.rows(), 2*dd.cols());
    out << dd, dd;

    out.leftCols(dd.cols()) = out.leftCols(dd.cols()).cwiseMax(0);
    out.rightCols(dd.cols()) = out.rightCols(dd.cols()).cwiseMin(0).cwiseAbs();

    out = (1.0 + out.array()).log();

    for(int i = 0; i < out.rows(); i++) {
        out.row(i) = out.row(i) / out.row(i).maxCoeff();
    }

    out = (INPUTMULT * out.array());

    return out;
}

void storeTransformedDataset(MatrixTransformedDataset transformedDataset) {
    std::ofstream ofs("transformedDataset");
    boost::archive::binary_oarchive oa(ofs);
    oa << transformedDataset;
}

bool checkForCachedTransformedDataset() {
    ifstream DataFile ("transformedDataset", ios::in | ios::binary | ios::ate);
    return DataFile.is_open();
}

MatrixTransformedDataset loadTransformedDataset() {
    MatrixTransformedDataset transformedDataset;
    std::ifstream ifs("transformedDataset");
    boost::archive::binary_iarchive ia(ifs);
    ia & transformedDataset;
    return transformedDataset;
}

MatrixTransformedDataset retrieveTransformedDataset() {
    MatrixTransformedDataset transformedDataset;
    if (checkForCachedTransformedDataset()) {
        cout << "Transformed Dataset found in cache, retrieving" << endl;
        transformedDataset = loadTransformedDataset();
    } else {
        cout << "Transformed Dataset not found in cache, building.." << endl;
        MatrixDataset d = loadDataset();
        transformedDataset = transformDataset(d);
        cout << "Caching Transformed Dataset" << endl;
        storeTransformedDataset(transformedDataset);
    }
    return transformedDataset;
}

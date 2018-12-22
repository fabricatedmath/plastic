#pragma once
#include <fstream>
#include <Eigen/Dense>

#include "type.h"
#include "constants.cuh"

using namespace std;

MatrixXd initW() {
    MatrixXd w =  MatrixXd::Zero(NBNEUR, NBNEUR); //MatrixXd::Random(NBNEUR, NBNEUR).cwiseAbs();
    //w.fill(1.0);
    w.bottomRows(NBI).leftCols(NBE).setRandom(); // Inhbitory neurons receive excitatory inputs from excitatory neurons
    w.rightCols(NBI).setRandom(); // Everybody receives inhibition (including inhibitory neurons)
    w.bottomRows(NBI).rightCols(NBI) =  -w.bottomRows(NBI).rightCols(NBI).cwiseAbs() * WII_MAX;
    w.topRows(NBE).rightCols(NBI) = -w.topRows(NBE).rightCols(NBI).cwiseAbs() * WIE_MAX;
    w.bottomRows(NBI).leftCols(NBE) = w.bottomRows(NBI).leftCols(NBE).cwiseAbs() * WEI_MAX;
    w = w - w.cwiseProduct(MatrixXd::Identity(NBNEUR, NBNEUR)); // Diagonal lateral weights are 0 (no autapses !)
    return w;
}

MatrixXd initWff() {
    MatrixXd wff = MatrixXd::Zero(NBNEUR, FFRFSIZE);
    wff =  (WFFINITMIN + (WFFINITMAX-WFFINITMIN) * MatrixXd::Random(NBNEUR, FFRFSIZE).cwiseAbs().array()).cwiseMin(MAXW);
    wff.bottomRows(NBI).setZero();
    return wff;
}

MatrixRXu loadDataset() {
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
    Map<MatrixRXu> mout(v.data(),numRows,numCols);
    return MatrixRXu(mout);
}

MatrixRXd loadTransformedDataset() {
    MatrixRXu d = loadDataset();
    MatrixRXd dd = d.cast<double>();

    MatrixRXd out(dd.rows(), 2*dd.cols());
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

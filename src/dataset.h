#pragma once

template<typename T>
class Dataset {
public:
    typedef Matrix<T,Dynamic,2*FFRFSIZE,RowMajor> MatrixTransformedDataset;

    static MatrixTransformedDataset retrieveTransformedDataset() {
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

private:
    BOOST_STATIC_ASSERT(is_same<float,T>::value || is_same<double,T>::value);

    typedef Matrix<int8_t, Dynamic, FFRFSIZE, RowMajor> MatrixDataset;
    typedef Matrix<int8_t, 1, Dynamic> VectorXu;

    static MatrixDataset loadDataset() {
        const int ffrfSize = FFRFSIZE;
        const string fileName = string("./dataset/patchesCenteredScaledBySumTo126ImageNetONOFFRotatedNewInt8.bin.dat");
        ifstream DataFile (fileName, ios::in | ios::binary | ios::ate);
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


    static MatrixTransformedDataset transformDataset(MatrixDataset d) {
        Matrix<double,Dynamic,FFRFSIZE,RowMajor> dd = d.cast<double>();

        Matrix<double,Dynamic,2*FFRFSIZE,RowMajor> out(dd.rows(), 2*dd.cols());
        out << dd, dd;

        out.leftCols(dd.cols()) = out.leftCols(dd.cols()).cwiseMax(0);
        out.rightCols(dd.cols()) = out.rightCols(dd.cols()).cwiseMin(0).cwiseAbs();

        out = (1.0 + out.array()).log();

        for(int i = 0; i < out.rows(); i++) {
            out.row(i) = out.row(i) / out.row(i).maxCoeff();
        }

        out = (INPUTMULT * out.array());

        return out.cast<T>();
    }

    static void storeTransformedDataset(MatrixTransformedDataset transformedDataset) {
        string fileName = string("dataset/transformedDataset.").append(typeid(T).name());
        std::ofstream ofs(fileName);
        boost::archive::binary_oarchive oa(ofs);
        oa << transformedDataset;
    }

    static bool checkForCachedTransformedDataset() {
        string fileName = string("dataset/transformedDataset.").append(typeid(T).name());
        ifstream DataFile (fileName, ios::in | ios::binary | ios::ate);
        return DataFile.is_open();
    }

    static MatrixTransformedDataset loadTransformedDataset() {
        MatrixTransformedDataset transformedDataset;
        string fileName = string("dataset/transformedDataset.").append(typeid(T).name());
        std::ifstream ifs(fileName);
        boost::archive::binary_iarchive ia(ifs);
        ia & transformedDataset;
        return transformedDataset;
    }
};

template <typename T>
using MatrixTransformedDataset = typename Dataset<T>::MatrixTransformedDataset;

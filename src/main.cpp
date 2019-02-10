#include <iostream>
#include <iomanip>
#include <Eigen/Dense>
#include "eigen_boost_serialization.hpp"
#include "state.h"
#include "test.h"
#include "init.h"
#include "loader.h"
#include "dataset.h"
#include "testing.h"
#include <random>
#include <boost/program_options.hpp>

namespace po = boost::program_options;

using namespace std;
using namespace Eigen;

template<typename F, typename I>
void run() {
    MutableState<F,I> ms = Init<F,I>::initMutableState();
    StaticState<F,I> ss = Init<F,I>::initStaticState();
    Buffers<F,I> b = Init<F,I>::initBuffers();

    run(ms, ss, b);
}

void runTesting(int iter) {
    int numRuns = NBSTEPSPERPRES;
    MutableState<double,int> ms = loadMutableState(iter);
    cout << "here" << endl;
    StaticState<double,int> ss = loadStaticState();
    cout << "here" << endl;
    Buffers<double,int> b = Init<double,int>::initBuffers();
    cout << "here" << endl;
    RandomHistorical<double> r = loadRandomHistorical();

    cout << "here" << endl;

    auto tup = run(ms, ss, b, r);

    ms = std::get<0>(tup);
    b = std::get<1>(tup);

    for (int i = 1; i < numRuns; i++) {
        //printf("%d\n", i);
        //print(i);
    }

    cout << "here" << endl;

    MutableState<double,int> msnext = loadMutableState(iter+numRuns);


    cout << std::setprecision(15);

    int i;

    cout << ms.wadap.transpose() << endl;
    cout << msnext.wadap.transpose() << endl;

    cout << "wadap err+: " << (ms.wadap - msnext.wadap).maxCoeff(&i) << " @ " << i << endl;
    cout << "wadap err+: " << (ms.wadap - msnext.wadap).minCoeff(&i) << " @ " << i << endl;
    cout << endl;

    cout << "xplastff err+: " << (ms.xplastFF - msnext.xplastFF).maxCoeff(&i) << " @ " << i << endl;
    cout << "xplastff err-: " << (ms.xplastFF - msnext.xplastFF).minCoeff(&i) << " @ " << i << endl;
    cout << endl;

    cout << "xplastlat err+: " << (ms.xplastLat - msnext.xplastLat).maxCoeff(&i) << " @ " << i << endl;
    cout << "xplastlat err-: " << (ms.xplastLat - msnext.xplastLat).minCoeff(&i) << " @ " << i << endl;
    cout << endl;

    VectorX<double> neurLTD = loadNeurLTD(iter+numRuns);

    cout << "neurLTD err+: " << (b.eachNeurLTD - neurLTD).maxCoeff(&i) << " @ " << i << endl;
    cout << "neurLTD err-: " << (b.eachNeurLTD - neurLTD).minCoeff(&i) << " @ " << i << endl;
    cout << endl;


    VectorX<double> neurLTP = loadNeurLTP(iter+numRuns);

    cout << "neurLTP err+: " << (b.eachNeurLTP - neurLTP).maxCoeff(&i) << " @ " << i << endl;
    cout << "neurLTP err-: " << (b.eachNeurLTP - neurLTP).minCoeff(&i) << " @ " << i << endl;
    cout << endl;

    //cout << b.lgnfirings << endl;
    for (int i = 1; i < 251; i++) {
        VectorX<double> lgnfirings = loadLgnfirings(i);
        VectorX<double> lgnfiringsRow = b.lgnfirings.row(i-1).cast<double>();
        VectorX<double> diff = (lgnfirings - lgnfiringsRow).cwiseAbs();
        if (diff.maxCoeff() > 0.001) {
            cout << "lgnfirings don't match" << endl;
            exit(1);
        }
    }

    for (int i = 251; i < 350; i++) {
        VectorX<double> lgnfirings = loadLgnfirings(i);
        if (lgnfirings.maxCoeff() > 0.001) {
            cout << "lgnfirings aren't zero after 250" << i << endl;
            exit(1);
        }
    }
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

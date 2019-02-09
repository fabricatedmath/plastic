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

void runTesting(int i) {
    MutableState<double,int> ms = loadMutableState(i);
    cout << ms.wadap.head(4).transpose() << endl;

    StaticState<double,int> ss = loadStaticState();
    Buffers<double,int> b = Init<double,int>::initBuffers();
    RandomHistorical<double> r = loadRandomHistorical();

    ms = run(ms, ss, b, r);
    cout << ms.wadap.head(4).transpose() << endl;

    MutableState<double,int> msnext = loadMutableState(i+2);
    VectorX<double> input = loadInput(i+2);

    cout << msnext.wadap.head(4).transpose() << endl;
    cout << input.head(4).transpose() << endl;
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

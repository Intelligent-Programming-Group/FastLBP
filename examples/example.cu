/// @file example.cu
/// @brief A simple example of using the library.
/// 
/// @date 2024-07-13

#include <iostream>
#include <ext/causal_bp.h>
#include <lbp/bp.h>
#include <lbp/factorgraph.h>

using namespace lbp;
using namespace std;

int main(int argc, char *argv[]) {
    if (argc != 2) {
        cout << "Usage: " << argv[0] << " <filename.fg>" << endl << endl;
        cout << "Reads factor graph <filename.fg>." << endl << endl;
        return 1;
    }

    // Read FactorGraph from the file specified by the first command line argument
    FactorGraph fg;
    fg.ReadFromFile(argv[1]);
    // size_t maxstates = 1000000;

    // Set some constants
    size_t maxiter = 100000;
    Real tol = 1e-6;
    size_t verb = 1;
    Real maxtime = 300;

    // Store the constants in a PropertySet object
    PropertySet opts;
    opts.set("maxiter", maxiter);  // Maximum number of iterations
    opts.set("tol", tol);          // Tolerance for convergence
    opts.set("verbose", verb);     // Verbosity (amount of output generated)
    opts.set("maxtime", maxtime);

    // Construct a BP (belief propagation) object from the FactorGraph fg
    // using the parameters specified by opts and two additional properties,
    // specifying the type of updates the BP algorithm should perform and
    // whether they should be done in the Real or in the logdomain
    BP bp(fg, opts);
    // // Initialize belief propagation algorithm
    bp.init();
    // // Run belief propagation algorithm
    bp.run();

    // Report variable marginals for fg, calculated by the belief propagation algorithm
    cout << "Approximate (loopy belief propagation) variable marginals:" << endl;
    for ( size_t i = 0; i < fg.nrVars(); i++ ) { // iterate over all variables in fg
        cout << bp.belief(fg.var(i)) << endl; // display the belief of bp for that variable
    }

    // // Report factor marginals for fg, calculated by the belief propagation algorithm
    // cout << "Approximate (loopy belief propagation) factor marginals:" << endl;
    // for( size_t I = 0; I < fg.nrFactors(); I++ ) // iterate over all factors in fg
    //     cout << bp.belief(fg.factor(I).vars()) << endl; // display the belief of bp for the variables in that factor

    return 0;
}

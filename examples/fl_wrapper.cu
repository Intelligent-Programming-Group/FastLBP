#include <ext/causal_bp.h>
#include <kernel/causal_bp_seq.h>
#include <lbp/properties.h>
#include <utils/utils.h>
using namespace lbp;

#include <nvml.h>
#include <cassert>
#include <ctime>
#include <iostream>
#include <map>
#include <string>
#include <vector>
#include <queue>
#include <fstream>
#include <iomanip>
using namespace std;

static bool is_causal;
static FactorGraph fg;
static CausalFactorGraph causal_fg;
static PropertySet opts;
static map<int, bool> clamps;
static string factorGraphFileName;

void runBP() {
    cudaFree(0);
    unique_ptr<CausalBP> causal_bp;
    for (const auto& clamp : clamps) {
        int varIndex = clamp.first;
        bool varValue = clamp.second;
        if (is_causal) {
            for (size_t i = 0; i < causal_fg.nrVars(); i++) {
                if (causal_fg.var(i).label() == varIndex) {
                    causal_fg.clamp(i, varValue ? 1 : 0);
                }
            }
        }
        else {
            LOG_FATAL("We don't support clamp for BP");
            // bp->clamp(varIndex, varValue ? 1 : 0);
        }
    }
    
    CausalFactorGraph new_causal_fg(causal_fg.factors());
    // clog << __LOGSTR__ << "debug" << endl;
    Real tic1 = toc();
    if (is_causal)
        causal_bp.reset(new CausalBP(new_causal_fg, opts));
    if (is_causal)
        causal_bp->init();

    Real tolerance = 1e-6;
    size_t maxIters = 500, histLength = 1;
    // clog << __LOGSTR__ << "BP " << tolerance << " " << minIters << " " << maxIters << " " << histLength << endl;

    Real yetToConvergeFraction = 
            is_causal ? causal_bp->run(tolerance, maxIters, histLength)
                    : 1;
    // clog << __LOGSTR__ << "yetToConvergeFraction: " << yetToConvergeFraction << endl;
    // clog << __LOGSTR__ << "number of variables: " << causal_fg.nrVars() << endl;
    Real tic2 = toc();
    clog << tic2 - tic1 << endl;
    // Report variable marginals for fg, calculated by the belief propagation algorithm
    size_t maxVarLabel = 0;
    for ( size_t i = 0; i < new_causal_fg.nrVars(); i++ ) {
        if (new_causal_fg.var(i).label() > maxVarLabel) {
            maxVarLabel = new_causal_fg.var(i).label();
        }
    }
    // clog << __LOGSTR__ << "maxVarLabel: " << maxVarLabel << endl;
    vector<Real> marginals(maxVarLabel + 1, -1);
    // clog << __LOGSTR__ << "arrive here" << endl;
    // causal_bp->transferMessagesToHost();
    for ( size_t i = 0; i < new_causal_fg.nrVars(); i++ ) { // iterate over all variables in fg
        marginals[new_causal_fg.var(i).label()] = causal_bp->newBelief(i); // display the belief of bp for that variable
    }

    // clog << __LOGSTR__ << "final number of factors for variable " << new_causal_fg.nrFactors() << endl;
    for (size_t i = 0; i < marginals.size(); i++) {
        if (marginals[i] >= 0 && marginals[i] <= 1) {
            cout << i << " " << setprecision(17) << marginals[i] << endl;
        } else if (clamps.find(i) != clamps.end()) {
            cout << i << " " << (clamps[i] ? 1.0 : 0.0) << endl;
        } else {
            cout << i << " " << 0.5 << endl;
        }
    }
}

void clamp(int varIndex, string varValueStr) {
    assert(varValueStr == "true" || varValueStr == "false");
    // clog << __LOGSTR__ << "O " << varIndex << " " << varValueStr << endl;

    bool varValue = (varValueStr == "true");
    clamps[varIndex] = varValue;
}

int main(int argc, char *argv[]) {
    if (argc != 3) {
        cerr << __LOGSTR__ << "Incorrect number of arguments." << endl;
        // show_help(argv[0]);
        return 1;
    }
    factorGraphFileName = argv[1];
    is_causal = factorGraphFileName.substr(
            factorGraphFileName.find_last_of(".") + 1
            ) == "causal_fg";
    clog << __LOGSTR__ << "Hello!" << endl
                       << "wrapper.cpp compiled on " << __DATE__ << " at " << __TIME__ << "." << endl;
    if (is_causal)
        causal_fg.ReadFromFile(factorGraphFileName.c_str());
    else
        fg.ReadFromFile(factorGraphFileName.c_str());
    clog << __LOGSTR__ << "Finished reading factor graph." << endl;

    // Read observation from file
    ifstream observeFile(argv[2]);
    if (!observeFile.is_open()) {
        cerr << "Cannot open observation file!\n";
        return 1;
    }
    int id;
    string boolStr;
    while (observeFile >> id >> boolStr) {
        clamp(id, boolStr);
    }
    
    opts.set("maxiter", static_cast<size_t>(10000000));
    opts.set("maxtime", Real(57600));
    opts.set("tol", Real(1e-6));
    opts.set("verb", static_cast<size_t>(1));
    opts.set("updates", string("PARALL")); // "PARALL", "SEQFIX", or "TOPO"
    opts.set("logdomain", false);
    
    runBP();

    // clog << __LOGSTR__ << "Bye!" << endl;
    return 0;
}

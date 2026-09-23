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
using namespace std;

static bool is_causal;
static FactorGraph fg;
static CausalFactorGraph causal_fg;
static PropertySet opts;
static unique_ptr<CausalBP> causal_bp;
static map<int, bool> clamps;
static string factorGraphFileName;

void initBP() {
    CausalFactorGraph new_causal_fg(causal_fg);
    for (const auto& clamp : clamps) {
        int varIndex = clamp.first;
        bool varValue = clamp.second;
        if (is_causal)
            new_causal_fg.clamp(varIndex, varValue ? 1 : 0);
        else {
            LOG_FATAL("We don't support clamp for BP");
            // bp->clamp(varIndex, varValue ? 1 : 0);
        }
    }
    if (is_causal)
        causal_bp.reset(new CausalBP(new_causal_fg, opts));
    if (is_causal)
        causal_bp->init();
}

void queryVariable() {
    int varIndex;
    cin >> varIndex;
    clog << __LOGSTR__ << "Q " << varIndex << endl;

    // auto ans = bp.belief(fg.var(varIndex)).get(1);
    Real ans = is_causal ? causal_bp->newBelief(varIndex)
                    : 0;
    clog << __LOGSTR__ << "Returning " << ans << "." << endl;
    cout << ans << endl;
}

// void queryFactor() {
//     int factorIndex, valueIndex;
//     cin >> factorIndex >> valueIndex;
//     clog << __LOGSTR__ << "FQ " << factorIndex << " " << valueIndex << endl;

//     auto ans = is_causal ? causal_bp->beliefF(factorIndex).get(valueIndex)
//                     : bp->beliefF(factorIndex).get(valueIndex); 
//     clog << __LOGSTR__ << "Returning " << ans << "." << endl;
//     cout << ans << endl;
// }

void runBP() {
    Real tolerance;
    size_t minIters, maxIters, histLength;
    cin >> tolerance >> minIters >> maxIters >> histLength;
    clog << __LOGSTR__ << "BP " << tolerance << " " << minIters << " " << maxIters << " " << histLength << endl;

    initBP();
    Real yetToConvergeFraction = 
            is_causal ? causal_bp->run(tolerance, minIters, maxIters, histLength)
                    : 1;
    cout << yetToConvergeFraction << endl;
}

void clamp() {
    int varIndex;
    string varValueStr;
    cin >> varIndex >> varValueStr;
    assert(varValueStr == "true" || varValueStr == "false");
    clog << __LOGSTR__ << "O " << varIndex << " " << varValueStr << endl;

    bool varValue = (varValueStr == "true");
    clamps[varIndex] = varValue;
    cout << "O " << varIndex << " " << varValueStr << endl;
}

void unclamp() {
    int varIndex;
    cin >> varIndex;
    assert(clamps.find(varIndex) != clamps.end());
    clamps.erase(varIndex);
    clog << __LOGSTR__ << "UC " << varIndex << endl;
    cout << "UC " << varIndex << endl;
}

void show_help(const char *program) {
    cerr << "Usage: " << program << " <filename> [<seed> <--auto|--manual id>]\n"
        << "Options:\n"
        << "  --auto        Automatically select which GPU device to run on.\n"
        << "  --manual id   Manually specify the GPU device ID to run on.\n";
}

int main(int argc, char *argv[]) {
    if (argc < 2) {
        cerr << __LOGSTR__ << "Insufficient number of arguments." << endl;
        show_help(argv[0]);
        return 1;
    } else if (argc > 2) {
        size_t seed = std::stoi(argv[2]);
        clog << __LOGSTR__ << "Setting random seed: " << seed << endl; 
        rnd_gen.seed(static_cast<unsigned int>(seed));
        if (argc > 3) {
            int device_id = 0;
            std::string arg = argv[3];
            if (arg == "--auto") {
                nvmlInit();
                int device_count;
                nvmlDevice_t device;
                nvmlUtilization_t utilization;
                cudaGetDeviceCount(&device_count);
                unsigned int min_util = 101;
                for (int i = 0; i < device_count; i++) {
                    nvmlDeviceGetHandleByIndex(i, &device);
                    nvmlDeviceGetUtilizationRates(device, &utilization);
                    unsigned int cur_util = utilization.gpu;
                    if (cur_util < min_util) {
                        min_util = cur_util;
                        device_id = i;
                    }
                }
                nvmlShutdown();
            } else if (arg == "--manual") {
                if (argc <= 4) {
                    cerr << __LOGSTR__ << "--manual requires one integer argument for the GPU device ID." << endl;
                    show_help(argv[0]);
                    return 1;
                }
                device_id = std::stoi(argv[4]);
            } else {
                cerr << __LOGSTR__ << "Wrong arguments." << endl;
                show_help(argv[0]);
                return 1;
            }
            clog << __LOGSTR__ << "Setting device ID: " << device_id << endl;
            cudaError_t err = cudaSetDevice(device_id);
            if (err != cudaSuccess) {
                clog << __LOGSTR__ << "Failed to set device " << device_id << ": " << cudaGetErrorString(err) << endl;
            }
        }
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
    
    opts.set("maxiter", static_cast<size_t>(10000000));
    opts.set("maxtime", Real(57600));
    opts.set("tol", Real(1e-6));
    opts.set("verb", static_cast<size_t>(1));
    opts.set("updates", string("SEQFIX")); // "SEQRND", or "PARALL", or "SEQFIX"
    opts.set("logdomain", false);

    // topoSort();

    string cmdType;
    while (cin >> cmdType) {
        clog << __LOGSTR__ << "Read command " << cmdType << endl;
        if (cmdType == "Q") {
            queryVariable();
        }/* else if (cmdType == "FQ") {
            queryFactor();
        } */else if (cmdType == "BP") {
            runBP();
        } else if (cmdType == "O") {
            clamp();
        } else if (cmdType == "UC") {
            unclamp();
        } else if (cmdType == "BYE") {
            break;
        } else {
            assert(cmdType == "NL");
            cout << endl;
        }
    }

    clog << __LOGSTR__ << "Bye!" << endl;
    return 0;
}

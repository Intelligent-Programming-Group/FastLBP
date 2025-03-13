/// @file causal_fg.cpp
/// @brief
///
/// @date 2024-10-19

#include <fstream>
#include <iomanip>
#include <map>
#include <set>
#include <unordered_map>
#include <ext/causal_fg.h>
#include <utils/log.h>
#include <utils/utils.h>

namespace lbp {

void CausalFactorGraph::clamp(size_t i, size_t x) {
    std::map<size_t, CausalFactor> newFacs;
    for ( const Neighbor &I: nbV(i) ) {
        CausalFactor newFac = factor(I).gen_clamped(var(i), x);
        newFacs.emplace(I, newFac);
    }
    setFactors(newFacs);

    return;
}

void CausalFactorGraph::constructGraph( size_t nrEdges ) {
    // create a mapping for indices
    std::unordered_map<size_t, size_t> hashmap;

    for( size_t i = 0; i < vars().size(); i++ ) {
        hashmap[var(i).label()] = i;
    }

    // create edge list
    std::vector<Edge> edges;
    edges.reserve( nrEdges );
    for( size_t i2 = 0; i2 < nrFactors(); i2++ ) {
        const VarSet& ns = factor(i2).vars();
        for( VarSet::const_iterator q = ns.begin(); q != ns.end(); q++ )
            edges.push_back( Edge(hashmap[q->label()], i2) );
    }

    // create bipartite graph
    // When dumping causal graph, sub-ids has been uniqued, so no need for check
    _G.construct( nrVars(), nrFactors(), edges.begin(), edges.end(), false );
}

CausalFactorGraph::CausalFactorGraph(const std::vector<CausalFactor> &P): _G() {
    // add factors, obtain variables
    std::set<Var> varset;
    _factors.reserve( P.size() );
    size_t nrEdges = 0;
    for( std::vector<CausalFactor>::const_iterator p2 = P.begin(); p2 != P.end(); p2++ ) {
        _factors.push_back( *p2 );
        copy( p2->vars().begin(), p2->vars().end(), inserter( varset, varset.begin() ) );
        nrEdges += p2->vars().size();
    }

    // add vars
    _vars.reserve( varset.size() );
    for( std::set<Var>::const_iterator p1 = varset.begin(); p1 != varset.end(); p1++ )
        _vars.push_back( *p1 );

    // create graph structure
    constructGraph( nrEdges );
}

const Var& CausalFactorGraph::var( size_t i ) const {
    return _vars[i];
}

const CausalFactor &CausalFactorGraph::factor(size_t I) const {
    return _factors[I];
}

const std::vector<Var> &CausalFactorGraph::vars() const {
    return _vars;
}

const Neighbors &CausalFactorGraph::nbV(size_t i) const {
    return _G.nb1(i);
}

const Neighbor &CausalFactorGraph::nbV( size_t i, size_t _I ) const {
    return _G.nb1(i)[_I];
}

const Neighbors& CausalFactorGraph::nbF( size_t I ) const { 
    return _G.nb2(I);
}

size_t CausalFactorGraph::nrVars() const {
    return _vars.size();
}

size_t CausalFactorGraph::nrFactors() const {
    return _factors.size();
}

size_t CausalFactorGraph::nrEdges() const {
    return _G.nrEdges();
}

size_t CausalFactorGraph::findVar(const Var &n) const {
    size_t i = find( _vars.begin(), _vars.end(), n ) - _vars.begin();
    if( i == nrVars() ) {
        std::cerr << "var not found \n";
    }
    return i;
}

void CausalFactorGraph::ReadFromFile(const char *filename) {
    std::ifstream infile;
    infile.open(filename);
    if (infile.is_open()) {
        infile >> *this;
        infile.close();
    } else {
        LOG_FATAL("Cannot read from file %s.", filename);
    }
}

std::istream& operator>> ( std::istream& is, CausalFactorGraph& fg ) {
    long verbose = 0;

    std::vector<CausalFactor> facs;
    size_t nr_Factors;
    std::string line;
    
    getline(is, line);
    read_annotation(is, line);
    nr_Factors = std::stoi(line);
    if(is.fail()) {
        LOG_FATAL("Cannot read number of causal factors");
    }
    if(verbose >= 1) {
        LOG_INFO("Reading %zu causal factors...", nr_Factors);
    }
    
    std::set<size_t> heads, all_vars;
    for( size_t I = 0; I < nr_Factors; I++ ) {
        getline(is, line);
        if(is.fail() || line.size() > 0) {
            LOG_ERROR("Expecting empty line");
        }
        
        if( verbose >= 2 ) {
            LOG_INFO("Reading causal factor %zu...", I);
        }

        size_t head_id;
        getline(is, line);
        read_annotation(is, line);
        head_id = std::stoi(line);
        if(verbose >= 2) {
            LOG_INFO("head: %zu", head_id);
        }
        Var head{head_id, 2};
        heads.insert(head_id);
        all_vars.insert(head_id);
        
        char type_ch;
        CausalFactor::CausalType type;
        getline(is, line);
        read_annotation(is, line);
        type_ch = line.front();
        switch (type_ch) {
            case CausalFactor::Singleton: {
                type = CausalFactor::Singleton;
                if( verbose >= 2 ) {
                    LOG_INFO("type: singleton");
                }
                break;
            }
            case CausalFactor::DefiniteAnd: {
                type = CausalFactor::DefiniteAnd;
                if( verbose >= 2 ) {
                    LOG_INFO("type: definitive-and");
                }
                break;
            }
            case CausalFactor::DefiniteOr: {
                type = CausalFactor::DefiniteOr;
                if( verbose >= 2 ) {
                    LOG_INFO("type: definitive-or");
                }
                break;
            }
            default: {
                LOG_FATAL("Unreachable");
            }
        }

        if (type == CausalFactor::Singleton) {
            Real prob = 0;
            getline(is, line);
            read_annotation(is, line);
            prob = static_cast<Real>(std::stold(line));
            if( verbose >= 2 ) {
                LOG_INFO("");
                std::cerr << "probability: " << std::setw(std::cerr.precision()+4) << prob << std::endl;
            }
            facs.emplace_back(head, prob);
        } else {
            Real prob = 1;
            Real prob_default = 0;
            if (line.length() > 1) {
                size_t split_pos = line.find(';');
                if (split_pos > 1) {
                    prob = static_cast<Real>(std::stold(line.substr(1, split_pos - 1)));
                }
                if (split_pos != std::string::npos && split_pos + 1 != line.length()) {
                    prob_default = static_cast<Real>(std::stold(line.substr(split_pos+1)));
                }
                if( verbose >= 2 ) {
                    LOG_INFO("");
                    std::cerr << " probability: " << std::setw(std::cerr.precision()+4) << prob
                        << " default: " << std::setw(std::cerr.precision()+4) << prob_default
                        << std::endl;
                }
            } else {
                if( verbose >= 2 ) {
                    LOG_INFO(" definite!");
                }
            }
            size_t body_len;    
            getline(is, line);
            read_annotation(is, line);
            body_len = std::stoi(line);
            if( verbose >= 2 ) {
                LOG_INFO("body len: %zu", body_len);
            }

            getline(is, line);
            read_annotation(is, line);
            std::istringstream body_ss(line);
            std::vector<size_t> body_ids;
            for( size_t bi = 0; bi < body_len; bi++ ) {
                long body_id;
                body_ss >> body_id;
                body_ids.push_back(body_id);
            }
            if( verbose >= 2 ) {
                LOG_INFO("");
                std::cerr << "  body: " << body_ids << std::endl;
            }
            
            VarSet body;
            for( size_t body_id : body_ids ) {
                Var body_i{body_id, 2};
                body.insert(body_i);
                all_vars.insert(body_id);
            }
            facs.emplace_back(head, body, type == CausalFactor::DefiniteAnd, prob, prob_default);
        }
    }
    
    for (size_t singleton_id : all_vars) {
        if (heads.find(singleton_id) == heads.end()) {
            Var single_var{singleton_id, 2};
            facs.emplace_back(single_var, 0.5);
        }
    }

    if( verbose >= 3 ) {
        LOG_INFO("");
        std::cerr << "factors:" << facs << std::endl;
    }

    fg = CausalFactorGraph(facs);

    return is;
}

} // namespace lbp 

/// @file util.h
/// @brief Defines general utility functions.
///
/// @date 2024-07-21

#pragma once

// Assume POSIX compliant system. We need the following for querying the 
// system time
#include <sys/time.h>

#include <map>
#include <iostream>
#include <set>
#include <vector>

namespace lbp {

typedef double Real;

struct Index2Real {
    size_t ind, v;
    Real prob;
};

struct Index3 {
    size_t ind, v, src_ind;
};

struct Index5Real2 {
    size_t ind, v, src_ind, start_ind, end_ind;
    Real prob_default, prob;
};

struct Index5Real4 {
    size_t ind, v, src_ind, start_ind, end_ind;
    Real prob_default, prob, mask0, mask1;
};

struct Index6Real2 {
    size_t ind, v, src_ind, start_ind, end_ind, head;
    Real prob_default, prob;
};

struct Index6Real4 {
    size_t ind, v, src_ind, start_ind, end_ind, head;
    Real prob_default, prob, mask0, mask1;
};

/// @brief Writes a \c std::map<> to a \c std::ostream
/// @tparam T 
/// @param os 
/// @param x 
/// @return 
template<class T>
std::ostream& operator << (std::ostream& os, const std::vector<T> & x) {
    os << "(";
    for ( typename std::vector<T>::const_iterator it = x.begin(); it != x.end(); it++ )
        os << (it != x.begin() ? ", " : "") << *it;
    os << ")";
    return os;
}

/// @brief Writes a \c std::set<> to a \c std::ostream
/// @tparam T 
/// @param os 
/// @param x 
/// @return 
template<class T>
std::ostream& operator << (std::ostream& os, const std::set<T> & x) {
    os << "{";
    for( typename std::set<T>::const_iterator it = x.begin(); it != x.end(); it++ )
        os << (it != x.begin() ? ", " : "") << *it;
    os << "}";
    return os;
}

/// @brief Writes a \c std::map<> to a \c std::ostream
/// @tparam T1 
/// @tparam T2 
/// @param os 
/// @param x 
/// @return 
template<class T1, class T2>
std::ostream& operator << (std::ostream& os, const std::map<T1,T2> & x) {
    os << "{";
    for ( typename std::map<T1,T2>::const_iterator it = x.begin(); it != x.end(); it++ )
        os << (it != x.begin() ? ", " : "") << it->first << "->" << it->second;
    os << "}";
    return os;
}

/// @brief Read the following annotation from `is`.
/// @param is input stream to read
/// @param line string to store the lines
inline void read_annotation(std::istream& is, std::string line) {
    while(is.peek() == '#') {
        getline(is, line);
    }
}

/// @brief Returns wall clock time in seconds
/// @return user+system time in seconds
inline Real toc() {
    struct timeval tv;
    struct timezone tz;
    gettimeofday(&tv, &tz);
    return (Real)(tv.tv_sec + (Real)tv.tv_usec / 1000000.0);
}

}

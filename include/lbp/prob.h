/// @file prob.h
/// @brief Defines TProb<> and Prob classes which represent (probability) 
/// vectors (e.g., probability distributions of discrete random variables)
///
/// @date 2024-07-23

#pragma once

#include <stddef.h>
#include <algorithm>
#include <functional>
#include <iostream>
#include <vector>
#include <utils/utils.h>

namespace lbp {

/// @brief Function object that returns the value itself
/// @tparam T 
template<typename T> 
struct fo_id {
    /// Returns \a x
    T operator()( const T &x ) const {
        return x;
    }
};

/// @brief Function object similar to std::divides(), but different in that 
/// dividing by zero results in zero
/// @tparam T 
template<typename T> 
struct fo_divides0 {
    /// Returns (\a y == 0 ? 0 : (\a x / \a y))
    T operator()( const T &x, const T &y ) const {
        if (y == (T)0) {
            return (T)0;
        } else {
            return x / y;
        }
    }
};

/// @brief Represents a vector with entries of type `T`.
/// @tparam T 
template<typename T>
class TProb {
private:
    /// @brief Type of data structure used for storing the values
    typedef std::vector<T> container_type;
    /// @brief Shorthand
    typedef TProb<T> this_type;
    /// @brief The data structure that stores the values
    container_type _p;
public:
    TProb(/* args */) = default;
    ~TProb() = default;
    /// @brief 
    /// @param n 
    explicit TProb(size_t n);
    /// @brief Construct vector of length `n` with each entry set to `p`
    /// @param n 
    /// @param p 
    explicit TProb(size_t n, T p);
    /// @brief Construct vector from another vector
    /// @tparam S type of elements in \a v (should be castable to type \a T)
    /// @param v vector used for initialization.
    template <typename S>
    TProb(const std::vector<S> &v): _p() {
        _p.reserve( v.size() );
        _p.insert( _p.begin(), v.begin(), v.end() );
    }

    /// Constant iterator over the elements
    typedef typename container_type::const_iterator const_iterator;
    /// Returns constant iterator that points to the first element
    const_iterator begin() const { return _p.begin(); }
    /// Returns constant iterator that points beyond the last element
    const_iterator end() const { return _p.end(); }

    /// @brief Sets `i`'th entry to `val`
    /// @param i 
    /// @param val 
    void set(size_t i, T val);

    /// @brief 
    /// @return 
    const std::vector<T> &toVec() const;

    /// Returns a copy of the \a i 'th entry
    T operator[]( size_t i ) const { return _p[i]; }
        
    /// @brief 
    /// @return length of the vector (i.e., the number of entries)
    size_t size() const;

    template<typename unOp> 
    T accumulateSum( T init, unOp op ) const {
        T t = op(init);
        for( const_iterator it = begin(); it != end(); it++ )
            t += op(*it);
        return t;
    }

    /// Returns sum of all entries
    T sum() const { return accumulateSum( (T)0, fo_id<T>() ); }

    T normalize() {
        T Z = 0;
        Z = sum();
        if (Z == (T)0) {
            std::cerr << "Not normalizable!" << std::endl;
        } else {
            *this /= Z;
        }
        return Z;
    }

    /// Applies unary operation \a op pointwise
    template<typename unaryOp> 
    this_type& pwUnaryOp( unaryOp op ) {
        std::transform( _p.begin(), _p.end(), _p.begin(), op );
        return *this;
    }

    /// Divides each entry by scalar \a x, where division by 0 yields 0
    this_type& operator/= (T x) {
        if( x != 1 )
            return pwUnaryOp( std::bind( fo_divides0<T>(), std::placeholders::_1, x ) );
        else
            return *this;
    }

    template <typename binaryOp> 
    this_type &pwBinaryOp( const this_type &q, binaryOp op ) {
        if (size() != q.size()) {
            std::cerr << "The sizes of two vectors performing pointwise operation are not the same!" << std::endl;
        }
        std::transform( _p.begin(), _p.end(), q._p.begin(), _p.begin(), op );
        return *this;
    }

    this_type &operator*=(const this_type &q) { return pwBinaryOp( q, std::multiplies<T>() ); }
};

/// @brief Represents a vector with entries of type Real.
typedef TProb<Real> Prob;

template class TProb<Real>;
    
} // namespace lbp


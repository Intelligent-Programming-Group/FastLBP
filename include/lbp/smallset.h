/// @file smallset.h
/// @brief Defines the SmallSet<> class, which represents a set (optimized 
/// for a small number of elements).
///
/// @date 2024-07-23

#pragma once

#include <stddef.h>
#include <algorithm>
#include <vector>
#include <lbp/var.h>

namespace lbp {

template <typename T>
class SmallSet {
private:
    /// @brief The elements in this set
    std::vector<T> _elements;

protected:

public:
    /// @brief Default constructor (constructs an empty set)
    SmallSet(/* args */) = default;
    /// @brief Construct a set consisting of one element
    /// @param t 
    SmallSet(const T &t);
    ~SmallSet() = default;

    /// @brief Construct a `SmallSet` from a range of elements.
    /// @tparam TIterator TIterator Iterates over instances of type T.
    /// @param begin Points to first element to be added.
    /// @param end Points just beyond last element to be added.
    /// @param sizeHint For efficiency, the number of elements can be 
    /// speficied by `sizeHint`.
    template<typename TIterator>
    SmallSet(TIterator begin, TIterator end, size_t sizeHint) {
        _elements.reserve(sizeHint);
        _elements.insert(_elements.begin(), begin, end);
        std::sort(_elements.begin(), _elements.end());
        typename std::vector<T>::iterator new_end = std::unique(_elements.begin(), _elements.end());
        _elements.erase(new_end, _elements.end());
    }

    /// @brief Inserts `t` into `*this`
    /// @param t 
    /// @return 
    SmallSet &insert(const T &t);

    /// Erases \a t from \c *this
    SmallSet& erase( const T& t ) {
        return (*this /= t);
    }

    /// @brief Constant iterator over the elements
    typedef typename std::vector<T>::const_iterator const_iterator;
    /// @brief Iterator over the elements
    typedef typename std::vector<T>::iterator iterator;

    /// Set-intersection operator: returns all elements in \c *this that are also contained in \a x
    SmallSet operator& ( const SmallSet& x ) const {
        SmallSet res;
        std::set_intersection( _elements.begin(), _elements.end(), x._elements.begin(), x._elements.end(), inserter( res._elements, res._elements.begin() ) );
        return res;
    }

    /// @brief 
    /// @param x 
    /// @return \c true if \a x is a subset of \c *this
    bool operator>>(const SmallSet &x) const;

    /// Erases one element
    SmallSet& operator/= ( const T &t ) {
        typename std::vector<T>::iterator pos = lower_bound( _elements.begin(), _elements.end(), t );
        if( pos != _elements.end() )
            if( *pos == t ) // found element, delete it
                _elements.erase( pos );
        return *this;
    }

    /// Returns \c true if \c *this contains the element \a t
    bool contains( const T &t ) const {
        return std::binary_search( _elements.begin(), _elements.end(), t );
    }

    /// @brief 
    /// @return constant iterator that points to the first element
    const_iterator begin() const;

    /// @brief 
    /// @return constant iterator that points beyond the last element
    const_iterator end() const;

    /// @brief 
    /// @return number of elements
    typename std::vector<T>::size_type size() const;
};

template class SmallSet<Var>;

}

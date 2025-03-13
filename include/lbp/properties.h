/// @file properties.h
///
/// @brief Defines the Property and PropertySet classes, which are mainly used 
/// for managing parameters of inference algorithms
/// @date 2024-09-09

#pragma once

#include <any>
#include <map>
#include <string>

namespace lbp {

/// @brief Type of the key of a Property
typedef std::string PropertyKey;

/// @brief Type of the value of a Property
typedef std::any PropertyValue;

class PropertySet: private std::map<PropertyKey, PropertyValue> {
private:
    /* data */
public:
    PropertySet(/* args */) = default;
    ~PropertySet() = default;

    /// @brief Sets a property (a key \a key with a corresponding value \a val)
    /// @param key 
    /// @param val 
    /// @return 
    PropertySet &set(const PropertyKey &key, const PropertyValue &val);

    /// @brief Gets the value corresponding to \a key
    /// @param key 
    /// @return 
    const PropertyValue &get(const PropertyKey &key) const;

    /// @brief Gets the value corresponding to `key`, cast to `ValueType`, 
    /// converting from a string if necessary.
    /// @tparam ValueType Type to which the value should be cast/converted
    /// @param key 
    /// @return 
    template<typename ValueType>
    ValueType getStringAs(const PropertyKey &key) const {
        PropertyValue val = get(key);
        if (val.type() == typeid(ValueType)) {
            return std::any_cast<ValueType>(val);
        } else {
            std::cerr << "get properties error" << std::endl;
        }
        return ValueType();
    }

    /// @brief Check if a property with the given \a key is defined
    /// @param key 
    /// @return 
    bool hasKey(const PropertyKey &key) const;
};

} // namespace lbp 

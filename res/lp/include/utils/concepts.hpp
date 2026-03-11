#pragma once

#include<concepts>
#include<array>
#include<iostream>
#include<type_traits>


namespace lp{
namespace types{

template<typename T>
concept field= requires (T a, T b){
    // Addition + Identity + Inverse
    {a + b} -> std::same_as<T>;     // addition returns T
    {T(0)}  -> std::same_as<T>;     // default constructible to 0
    {-a}    -> std::same_as<T>;     // additive inverse
    {a - b} -> std::same_as<T>;     // [optional] but convenient

    // Product + Identity + Inverse
    {a * b} -> std::same_as<T>;     // product returns T
    {T(1)}  -> std::same_as<T>;     // default constructible to 1
    {a / b} -> std::same_as<T>;     // Inverse must exist

    // required for generic code
    requires std::is_copy_constructible_v<T>;
    // 0 != 1 in fields ever!
    requires requires {requires T(0) != T(1);};

    // equal and not equal should be there for code purposes
    {a == b} -> std::convertible_to<bool>;
    {a != b} -> std::convertible_to<bool>;
};
template<typename T>
concept ord_field= field<T> && requires (T a, T b){
    { a < b } -> std::convertible_to<bool>;
    { a > b } -> std::convertible_to<bool>;
    { a <=b } -> std::convertible_to<bool>;
    { a >=b } -> std::convertible_to<bool>;
};

template<typename T>
concept number = std::integral<T> || std::floating_point<T>;


}

}
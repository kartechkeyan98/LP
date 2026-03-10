#pragma once

#include<concepts>
#include<array>
#include<iostream>
#include<type_traits>

#include<matrix/matrix.hpp>

namespace lp{
namespace core{


template<typename T>
concept number = std::integral<T> || std::floating_point<T>;

// Core algebraic operations concept (meant to detect scalars only)
template<typename T>
concept algebraic = requires(T a, const T& b) {
    // Basic arithmetic operations
    { a + b } -> std::convertible_to<T>;
    { a - b } -> std::convertible_to<T>;
    { a * b } -> std::convertible_to<T>;
    { a / b } -> std::convertible_to<T>;
    
    // Assignment versions
    { a += b } -> std::convertible_to<T>;
    { a -= b } -> std::convertible_to<T>;
    { a *= b } -> std::convertible_to<T>;
    { a /= b } -> std::convertible_to<T>;
    
    // Unary operations
    { +a } -> std::convertible_to<T>;
    { -a } -> std::convertible_to<T>;
    
    // Comparison (often needed for processing)
    { a == b } -> std::convertible_to<bool>;
    { a != b } -> std::convertible_to<bool>;
    
    // Construction and assignment
    requires std::is_default_constructible_v<T>;
    requires std::is_copy_constructible_v<T>;
    requires std::is_copy_assignable_v<T>;
    requires std::is_move_constructible_v<T>;
    requires std::is_move_assignable_v<T>;
};



template<typename T>
concept scalar = !is_matrix_v<T> && algebraic<T>;

}

}
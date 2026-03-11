#pragma once

#include<utils/utils.hpp>
#include<utils/concepts.hpp>
#include<utils/errors.hpp>

#include<matrix/matrix.hpp>
#include<matrix/kernels.hpp>

// not putting them in namespace so that even without
// doing things like using namespace lp::..., these 
// operator overloads will work!

/**
 * Matrix Ops [Unary]
 */
template<lp::types::field T>
lp::core::matrix<T> operator-(const lp::core::matrix<T>& A){
    lp::core::matrix<T> res(A.shape(), A.get_alignment());
    lp::core::matrix_uniop(A, res, [](const T& x){return -x;});
    return res;
}
template<std::integral T>
lp::core::matrix<T> operator~(const lp::core::matrix<T>& A){
    lp::core::matrix<T> res(A.shape(), A.get_alignment());
    lp::core::matrix_uniop(A, res, [](const T& x){return ~x;});
    return res;
}

/**
 *  Matrix-Matrix Ops [Binary]
 **/ 
// Arithmetic
template<lp::types::field T, lp::types::field U>
auto operator+(const lp::core::matrix<T>& A, const lp::core::matrix<U>& B){
    #ifdef LP_DEBUG
    if(A.shape()!= B.shape()) 
        throw std::runtime_error(error::dimension_mismatch(A.shape(), B.shape()))
    #endif
    using V= std::common_type_t<T,U>;
    lp::core::matrix<V> C(A.shape(), std::max(A.get_alignment(), B.get_alignment()));
    lp::core::matrix_binop(A, B, C, std::plus<>());
    return C;
}
template<lp::types::field T, lp::types::field U>
auto operator-(const lp::core::matrix<T>& A, const lp::core::matrix<U>& B){
    #ifdef LP_DEBUG
    if(A.shape()!= B.shape()) 
        throw std::runtime_error(error::dimension_mismatch(A.shape(), B.shape()))
    #endif
    using V= std::common_type_t<T,U>;
    lp::core::matrix<V> C(A.shape(), std::max(A.get_alignment(), B.get_alignment()));
    lp::core::matrix_binop(A, B, C, std::minus<>());
    return C;
}
template<lp::types::field T, lp::types::field U>
auto operator*(const lp::core::matrix<T>& A, const lp::core::matrix<U>& B){
    #ifdef LP_DEBUG
    if(A.shape()!= B.shape()) 
        throw std::runtime_error(error::dimension_mismatch(A.shape(), B.shape()))
    #endif
    using V= std::common_type_t<T,U>;
    lp::core::matrix<V> C(A.shape(), std::max(A.get_alignment(), B.get_alignment()));
    lp::core::matrix_binop(A, B, C, std::multiplies<>());
    return C;
}
template<lp::types::field T, lp::types::field U>
auto operator/(const lp::core::matrix<T>& A, const lp::core::matrix<U>& B){
    #ifdef LP_DEBUG
    if(A.shape()!= B.shape()) 
        throw std::runtime_error(error::dimension_mismatch(A.shape(), B.shape()))
    #endif
    using V= std::common_type_t<T,U>;
    lp::core::matrix<V> C(A.shape(), std::max(A.get_alignment(), B.get_alignment()));
    lp::core::matrix_binop(A, B, C, std::divides<>());
    return C;
}

/** Special Ones */
// Modulo
template<std::integral T, std::integral U>
auto operator%(const lp::core::matrix<T>& A, const lp::core::matrix<U>& B){
    #ifdef LP_DEBUG
    if(A.shape()!= B.shape()) 
        throw std::runtime_error(error::dimension_mismatch(A.shape(), B.shape()))
    #endif
    using V= std::common_type_t<T,U>;
    lp::core::matrix<V> C(A.shape(), std::max(A.get_alignment(), B.get_alignment()));
    lp::core::matrix_binop(A, B, C, std::modulus<>());
    return C;
}
// Bitwise
template<std::integral T, std::integral U>
auto operator<<(const lp::core::matrix<T>& A, const lp::core::matrix<U>& B){
    #ifdef LP_DEBUG
    if(A.shape()!= B.shape()) 
        throw std::runtime_error(error::dimension_mismatch(A.shape(), B.shape()))
    #endif
    using V= std::common_type_t<T,U>;
    lp::core::matrix<V> C(A.shape(), std::max(A.get_alignment(), B.get_alignment()));
    lp::core::matrix_binop(A, B, C, [](const T& a, const U& b){return a << b;});
    return C;
}
template<std::integral T, std::integral U>
auto operator>>(const lp::core::matrix<T>& A, const lp::core::matrix<U>& B){
    #ifdef LP_DEBUG
    if(A.shape()!= B.shape()) 
        throw std::runtime_error(error::dimension_mismatch(A.shape(), B.shape()))
    #endif
    using V= std::common_type_t<T,U>;
    lp::core::matrix<V> C(A.shape(), std::max(A.get_alignment(), B.get_alignment()));
    lp::core::matrix_binop(A, B, C, [](const T& a, const U& b){return a >> b;});
    return C;
}
template<std::integral T, std::integral U>
auto operator&(const lp::core::matrix<T>& A, const lp::core::matrix<U>& B){
    #ifdef LP_DEBUG
    if(A.shape()!= B.shape()) 
        throw std::runtime_error(error::dimension_mismatch(A.shape(), B.shape()))
    #endif
    using V= std::common_type_t<T,U>;
    lp::core::matrix<V> C(A.shape(), std::max(A.get_alignment(), B.get_alignment()));
    lp::core::matrix_binop(A, B, C, [](const T& a, const U& b){return a & b;});
    return C;
}
template<std::integral T, std::integral U>
auto operator|(const lp::core::matrix<T>& A, const lp::core::matrix<U>& B){
    #ifdef LP_DEBUG
    if(A.shape()!= B.shape()) 
        throw std::runtime_error(error::dimension_mismatch(A.shape(), B.shape()))
    #endif
    using V= std::common_type_t<T,U>;
    lp::core::matrix<V> C(A.shape(), std::max(A.get_alignment(), B.get_alignment()));
    lp::core::matrix_binop(A, B, C, [](const T& a, const U& b){return a | b;});
    return C;
}
template<std::integral T, std::integral U>
auto operator^(const lp::core::matrix<T>& A, const lp::core::matrix<U>& B){
    #ifdef LP_DEBUG
    if(A.shape()!= B.shape()) 
        throw std::runtime_error(error::dimension_mismatch(A.shape(), B.shape()))
    #endif
    using V= std::common_type_t<T,U>;
    lp::core::matrix<V> C(A.shape(), std::max(A.get_alignment(), B.get_alignment()));
    lp::core::matrix_binop(A, B, C, [](const T& a, const U& b){return a ^ b;});
    return C;
}


/**
 * Matrix-Scalar Ops [Binary] (Scalar Right)
 */
// Arithmetic
template<lp::types::field T, lp::types::field U>
auto operator+(const lp::core::matrix<T>& A, const U& s){
    using V= std::common_type_t<T,U>;
    lp::core::matrix<V> C(A.shape(), A.get_alignment());
    lp::core::scalar_binop(A, s, C, std::plus<>());
    return C;
}
template<lp::types::field T, lp::types::field U>
auto operator-(const lp::core::matrix<T>& A, const U& s){
    using V= std::common_type_t<T,U>;
    lp::core::matrix<V> C(A.shape(), A.get_alignment());
    lp::core::scalar_binop(A, s, C, std::minus<>());
    return C;
}
template<lp::types::field T, lp::types::field U>
auto operator*(const lp::core::matrix<T>& A, const U& s){
    using V= std::common_type_t<T,U>;
    lp::core::matrix<V> C(A.shape(), A.get_alignment());
    lp::core::scalar_binop(A, s, C, std::multiplies<>());
    return C;
}
template<lp::types::field T, lp::types::field U>
auto operator/(const lp::core::matrix<T>& A, const U& s){
    using V= std::common_type_t<T,U>;
    lp::core::matrix<V> C(A.shape(), A.get_alignment());
    lp::core::scalar_binop(A, s, C, std::divides<>());
    return C;
}
// Modulo
template<std::integral T, std::integral U>
auto operator%(const lp::core::matrix<T>& A, const U& s){
    using V= std::common_type_t<T,U>;
    lp::core::matrix<V> C(A.shape(), A.get_alignment());
    lp::core::scalar_binop(A, s, C, std::modulus<>());
    return C;
}
//Bitwise
template<std::integral T, std::integral U>
auto operator<<(const lp::core::matrix<T>& A, const U& s){
    using V= std::common_type_t<T,U>;
    lp::core::matrix<V> C(A.shape(), A.get_alignment());
    lp::core::scalar_binop(A, s, C, [](const T& a, const U& b){return a<<b;});
    return C;
}
template<std::integral T, std::integral U>
auto operator>>(const lp::core::matrix<T>& A, const U& s){
    using V= std::common_type_t<T,U>;
    lp::core::matrix<V> C(A.shape(), A.get_alignment());
    lp::core::scalar_binop(A, s, C, [](const T& a, const U& b){return a>>b;});
    return C;
}
template<std::integral T, std::integral U>
auto operator&(const lp::core::matrix<T>& A, const U& s){
    using V= std::common_type_t<T,U>;
    lp::core::matrix<V> C(A.shape(), A.get_alignment());
    lp::core::scalar_binop(A, s, C, [](const T& a, const U& b){return a&b;});
    return C;
}
template<std::integral T, std::integral U>
auto operator|(const lp::core::matrix<T>& A, const U& s){
    using V= std::common_type_t<T,U>;
    lp::core::matrix<V> C(A.shape(), A.get_alignment());
    lp::core::scalar_binop(A, s, C, [](const T& a, const U& b){return a|b;});
    return C;
}
template<std::integral T, std::integral U>
auto operator^(const lp::core::matrix<T>& A, const U& s){
    using V= std::common_type_t<T,U>;
    lp::core::matrix<V> C(A.shape(), A.get_alignment());
    lp::core::scalar_binop(A, s, C, [](const T& a, const U& b){return a^b;});
    return C;
}

/**
 * Scalar Matrix [Binary] (Scalar Left)
 */
// Arithmetic
template<lp::types::field T, lp::types::field U>
auto operator+(const T& s, const lp::core::matrix<U>& A){
    using V= std::common_type_t<T,U>;
    lp::core::matrix<V> C(A.shape(), A.get_alignment());
    lp::core::scalar_left_binop(s, A, C, std::plus<>());
    return C;
}
template<lp::types::field T, lp::types::field U>
auto operator-(const T& s, const lp::core::matrix<U>& A){
    using V= std::common_type_t<T,U>;
    lp::core::matrix<V> C(A.shape(), A.get_alignment());
    lp::core::scalar_left_binop(s, A, C, std::minus<>());
    return C;
}
template<lp::types::field T, lp::types::field U>
auto operator*(const T& s, const lp::core::matrix<U>& A){
    using V= std::common_type_t<T,U>;
    lp::core::matrix<V> C(A.shape(), A.get_alignment());
    lp::core::scalar_left_binop(s, A, C, std::multiplies<>());
    return C;
}
template<lp::types::field T, lp::types::field U>
auto operator/(const T& s, const lp::core::matrix<U>& A){
    using V= std::common_type_t<T,U>;
    lp::core::matrix<V> C(A.shape(), A.get_alignment());
    lp::core::scalar_left_binop(s, A, C, std::divides<>());
    return C;
}

//Modulo
template<std::integral T, std::integral U>
auto operator%(const T& s, const lp::core::matrix<U>& A){
    using V= std::common_type_t<T,U>;
    lp::core::matrix<V> C(A.shape(), A.get_alignment());
    lp::core::scalar_left_binop(s, A, C, std::modulus<>());
    return C;
}

//Bitwise
template<std::integral T, std::integral U>
auto operator<<(const T& s, const lp::core::matrix<U>& A){
    using V= std::common_type_t<T,U>;
    lp::core::matrix<V> C(A.shape(), A.get_alignment());
    lp::core::scalar_left_binop(s, A, C, [](const T& a, const U& b){return a<<b;});
    return C;
}
template<std::integral T, std::integral U>
auto operator>>(const T& s, const lp::core::matrix<U>& A){
    using V= std::common_type_t<T,U>;
    lp::core::matrix<V> C(A.shape(), A.get_alignment());
    lp::core::scalar_left_binop(s, A, C, [](const T& a, const U& b){return a>>b;});
    return C;
}
template<std::integral T, std::integral U>
auto operator&(const T& s, const lp::core::matrix<U>& A){
    using V= std::common_type_t<T,U>;
    lp::core::matrix<V> C(A.shape(), A.get_alignment());
    lp::core::scalar_left_binop(s, A, C, [](const T& a, const U& b){return a&b;});
    return C;
}
template<std::integral T, std::integral U>
auto operator|(const T& s, const lp::core::matrix<U>& A){
    using V= std::common_type_t<T,U>;
    lp::core::matrix<V> C(A.shape(), A.get_alignment());
    lp::core::scalar_left_binop(s, A, C, [](const T& a, const U& b){return a|b;});
    return C;
}
template<std::integral T, std::integral U>
auto operator^(const T& s, const lp::core::matrix<U>& A){
    using V= std::common_type_t<T,U>;
    lp::core::matrix<V> C(A.shape(), A.get_alignment());
    lp::core::scalar_left_binop(s, A, C, [](const T& a, const U& b){return a^b;});
    return C;
} 





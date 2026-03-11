#pragma once

#include<cmath>
#include<functional>

#include<utils/utils.hpp>
#include<utils/concepts.hpp>
#include<utils/errors.hpp>

#include<matrix/matrix.hpp>
#include<matrix/kernels.hpp>

namespace lp::core{

template<types::field T, types::field U>
matrix<U> abs(const matrix<T>& A){
    matrix<U> res(A.shape(), alignof(U));
    matrix_uniop(A, res, [](T x){return std::abs(x);});
    return res;
}
template<types::field T, types::field U>
matrix<U> sqrt(const matrix<T>& A){
    matrix<U> res(A.shape(), alignof(U));
    matrix_uniop(A, res, [](T x){return std::sqrt(x);});
    return res;
}
template<types::field T, types::field U>
matrix<U> pow(const matrix<T>& A, double p){
    matrix<U> res(A.shape(), alignof(U));
    matrix_uniop(A, res, [p](T x){return std::pow(x, p);});
    return res;
}



}
#pragma once

#include<cmath>
#include<functional>

#include<utils/utils.hpp>
#include<utils/concepts.hpp>
#include<utils/errors.hpp>

#include<matrix/matrix.hpp>
#include<matrix/kernels.hpp>

namespace lp::core{

template<types::field T>
matrix<T> abs(const matrix<T>& A){
    matrix<T> res(A.shape(), alignof(T));
    matrix_uniop(A, res, [](T x){return std::abs(x);});
    return res;
}
template<types::field T>
matrix<T> sqrt(const matrix<T>& A){
    matrix<T> res(A.shape(), alignof(T));
    matrix_uniop(A, res, [](T x){return std::sqrt(x);});
    return res;
}
template<types::field T>
matrix<T> pow(const matrix<T>& A, double p){
    matrix<T> res(A.shape(), alignof(T));
    matrix_uniop(A, res, [p](T x){return std::pow(x, p);});
    return res;
}



}
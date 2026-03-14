#pragma once

#include<cmath>
#include<functional>
#include<concepts>
#include<type_traits>
#include<utility>

#include<utils/utils.hpp>
#include<utils/concepts.hpp>
#include<utils/errors.hpp>

#include<matrix/matrix.hpp>
#include<matrix/kernels.hpp>

namespace lp::core{

template<typename T>
auto abs(const T& x){
    if constexpr(requires {std::abs(x);})return std::abs(x);
    else if constexpr(types::ord_field<T>)return (x < T(0))*(-x) + x*(x>=T(0));
    else static_assert(false, "magnitude not defined!"); 
}


template<types::field T>
matrix<T> abs(const matrix<T>& A){
    matrix<T> res(A.shape(), alignof(T));
    matrix_uniop(A, res, [](T x){return abs(x);});
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

template<types::field T>
void swap(matrix<T>& A, matrix<T> &B){
    // assuming A, B have same shape
    #ifdef LP_DEBUG
    if (A.shape() != B.shape())
        throw std::runtime_error(error::dimension_mismatch(A.shape(), B.shape()));
    #endif
    Shape sh= A.shape();

    if(A.is_contiguous() && B.is_contiguous()){

        size_t total= sh[0]*sh[1];
        T* a= A.mem();
        T* b= B.mem();

        if(A.is_aligned() && B.is_aligned()){
            #pragma omp parallel for simd
            for(size_t i=0;i<total;i++){
                T tmp = a[i];
                a[i] = b[i];
                b[i] = tmp;
            }
        }else{
            #pragma omp parallel for schedule(static, arch::chunk<T>)
            for(size_t i=0;i<total;i++){
                std::swap(a[i], b[i]);
            }
        }
    }else{
        #pragma omp parallel for collapse(2)
        for(size_t i=0;i<sh[0];i++){
            for(size_t j=0;j<sh[1];j++){
                std::swap(A(i,j), B(i,j));
            }
        }
    }
}



}
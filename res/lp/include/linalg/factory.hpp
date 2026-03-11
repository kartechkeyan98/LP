#pragma once

#include<omp.h>
#include<concepts>
#include<random>

#include<utils/utils.hpp>
#include<utils/concepts.hpp>
#include<utils/errors.hpp>

#include<matrix/matrix.hpp>

namespace lp{
namespace linalg{

constexpr size_t small_size_total= 10000;

// simple stuff
template<types::field T>
core::matrix<T> zeros(size_t m, size_t n=1, size_t align= 64){
    core::matrix<T> res(m, n, align);
    size_t total= m*n;
    T* data= res.mem();

    if(total < small_size_total){
        for(size_t i=0;i<total;i++){
            data[i]= static_cast<T>(0);
        }
        return res;
    }

    // res is contiguous, so this cache friendly way is good
    #pragma omp parallel for simd
    for(size_t i=0;i<total;i++){
        data[i]= static_cast<T>(0);
    } 

    return res;
}
template<types::field T>
core::matrix<T> eye(size_t n, size_t align= 64){
    core::matrix<T> res= zeros<T>(n,n, align);

    if(n*n < small_size_total){
        for(size_t i=0;i<n;i++){
            res(i,i)= static_cast<T>(1);
        } 
        return res;
    }

    #pragma omp parallel for
    for(size_t i=0;i<n;i++){
        res(i,i)= static_cast<T>(1);
    } 

    return res;
}
template<types::field T>
core::matrix<T> full(const T& val, size_t m, size_t n=1, size_t align= 64){
    core::matrix<T> res(m,n, align);
    size_t total= m*n;
    T* data= res.mem();

    if(total < small_size_total){
        for(size_t i=0;i<total;i++){
            data[i]= val;
        }
        return res;
    }

    #pragma omp parallel for simd
    for(size_t i=0;i<total;i++){
        data[i]= val;
    } 

    return res;
}



}
}




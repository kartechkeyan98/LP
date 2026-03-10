#pragma once

// file for computing hnf of a matrix
#include<concepts>
#include<vector>

// matrix stuffs
#include<matrix/linalg.hpp>
// algorithms
#include<algos/gcd.hpp>

namespace lp{

template<std::integral T>
linalg::matrix<int64_t> hermite_normal_form(const linalg::matrix<T>& A){
    matrix<int64_t> a= A.copy<int64_t>();
    Shape ash= a.shape();

    for(size_t i=0;i<ash[0];++i){
        // 1. clear elements to the right of a(i,i) > 0
        // firstly find the first non-zero column after i and swap it with the
        // i-th column, if a(i,i)= 0
        for(size_t j=i;j<ash[1];++j){
            if(a(i,i)== 0)continue;
            matrix<T> temp= a(j);
            a(j)= a(i);
            a(i)= temp;
            break;
        }
        for(size_t j=i+1;j<ash[1];++j){
            /**
             * The goal here is: for col i, j (>i),
             * You need an invertible transformation such that
             * [a_i, a_j] -> [d, 0]
             * So, this is done by find coefficients: 
             * d = x11*a_i + x12*a_j
             * 0 = x21*a_i + x22*a_j
             * 
             * And for transformation to be invertible, 
             * det(X) = x11*x22 - x21*x12 != 0 (is 1 for convenience)
             * This is called a unimodular transformation.
             * So, the operations must be: 
             * A[:,i] = x11*A[:,i] + x12*A[:,j]
             * A[:,j] = x21*A[:,i] + x22*A[:,j] 
             */
            
            

            

        }
    }
}
}
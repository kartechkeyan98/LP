#pragma once

#include<iostream>
#include<array>

#include<matrix/linalg.hpp>
using namespace lp::linalg;

// return a matrix of shape (3,1)
// contains {g, x, y} -> g = ax + by is gcd(a,b) 
template<std::integral T>
matrix<T> gcd(const T& a, const T&b){
    // convert to absolute values
    T ar= std::abs(a);
    T br= std::abs(b);
    // take care of sign
    T as= (a>0)?1:-1;
    T bs= (b>0)?1:-1;

    /**
     * a = 1 * a + 0 * b
     * b = 0 * a + 1 * b
     * 
     * Update Rules:
     * next = prev - q*curr
     * prev= curr, curr= next
     */
    T x_prev= 1, y_prev= 0;
    T x_curr= 0, y_curr= 1;

    // gcd loop
    while(br!=0){
        T q= ar/br;

        // update coeffs
        T x_next= x_prev - q*x_curr;
        T y_next= y_prev - q*y_curr;
        x_prev= x_curr, y_prev= y_curr;
        x_curr= x_next, y_prev= y_next;

        // update nums
        T t= br;
        br= ar%br;
        ar= t; 
    }

    matrix<T> res(3,1);
    res(0,0)= ar, res(1,0)= as*x_prev, res(2,0)= bs*y_prev;
    return res;
}
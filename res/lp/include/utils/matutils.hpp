#pragma once

#include<iostream>
#include<iomanip>
#include<cmath>
#include<string>

#include<utils/concepts.hpp>

#include<matrix/matrix.hpp>
#include<matrix/reductions.hpp>


template<lp::types::field T>
std::ostream& operator<<(std::ostream& out, const lp::core::matrix<T>& A){
    lp::core::Shape sh= A.shape();

    // Some basic thing for printing and all for floats!
    lp::core::matrix<T> mx;
    T tol;
    if constexpr (std::is_floating_point_v<T>){
        mx= lp::core::max(A);
        tol= static_cast<T>(sh[1])*std::numeric_limits<T>::epsilon()*std::max(T(1), mx(0,0));
    }

    for(size_t i=0;i<sh[0];i++){
    for(size_t j=0;j<sh[1];j++){
        T val= A(i,j);
        // floating point value moments!
        if constexpr (std::is_floating_point_v<T>){
            if(std::abs(val)<tol && std::signbit(val)) val= T(0.);
        }
        out<<val<<" ";
    }
    out<<std::endl;
    }
    return out;
}


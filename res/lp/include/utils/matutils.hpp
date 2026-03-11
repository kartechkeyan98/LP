#pragma once

#include<iostream>
#include<string>

#include<matrix/matrix.hpp>

template<typename T>
std::ostream& operator<<(std::ostream& out, const lp::core::matrix<T>& A){
    lp::core::Shape sh= A.shape();

    for(size_t i=0;i<sh[0];i++){
    for(size_t j=0;j<sh[1];j++){
        out<<A(i,j)<<" ";
    }
    out<<std::endl;
    }
    return out;
}


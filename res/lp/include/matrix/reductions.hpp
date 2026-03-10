#pragma once

#include<functional>
#include<iostream>

#include<matrix/matrix.hpp>
#include<matrix/kernels.hpp>
#include<matrix/concepts.hpp>



namespace lp{
namespace core{

template<algebraic T>
matrix<T> sum(const matrix<T>& A, int axis= -1){
    return reduce(A, axis, std::plus<T>(), T(0));
}
template<algebraic T>
matrix<T> prod(const matrix<T>& A, int axis= -1){
    return reduce(A, axis, std::multiplies<T>(), T(1));
}
template<algebraic T>
matrix<T> mean(const matrix<T>& A, int axis= -1){
    size_t count;
    Shape sh= A.shape();
    if(axis== 0) count= sh[1];
    else if(axis== 1)count= sh[0];
    else count= sh[0]*sh[1];

    matrix<T> res= reduce(A, axis, std::plus<T>(), T(0));
    res/=count;
    return res;
}


template<ord_algebraic T>
matrix<T> max(const matrix<T> &A, int axis= -1){
    return reduce(
        A, axis, 
        [](T& a, const T& b){return std::max(a,b);}, 
        std::numeric_limits<T>::lowest()
    );
}
template<ord_algebraic T>
matrix<T> min(const matrix<T> &A, int axis= -1){
    return reduce(
        A, axis, 
        [](T& a, const T& b){return std::min(a,b);}, 
        std::numeric_limits<T>::max()
    );
}

template<algebraic T>
matrix<T> norm(const matrix<T>& A, double p= 2., int axis= -1){
    matrix<T> r= reduce(
        A, axis,
        [p](T& a, const T& b){return a + std::pow(b, p);},
        T(0)
    );
    matrix<T> res(r.shape(), r.get_alignment());
    matrix_uniop(
        r, res, 
        [p](const T& a){return std::pow(a, 1./p);}
    );
    return res;
}

}
}
#pragma once

#include<matrix/matrix.hpp>
#include<matrix/arithmetic.hpp>
#include<matrix/elementwise.hpp>
#include<matrix/concepts.hpp>

namespace lp{
namespace linalg{

// we are using ordered algebraic for numerical stability
template<core::ord_algebraic T>
core::matrix<T> rref(const core::matrix<T>& A, T tol= T(-1)){
    core::Shape sh= A.shape();

    // set tolerance if not given properly
    if(tol < T(0)){
        // find a suitable value for tolerance
        core::matrix<T> mx= core::max(core::abs(A));
        tol= static_cast<T>(sh[1])*std::numeric_limits<T>::epsilon()*std::max(T(1), mx(0,0));
    }

    

}

}
}
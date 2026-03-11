#pragma once

#include<utils/concepts.hpp>
#include<utils/utils.hpp>

#include<matrix/matrix.hpp>
#include<matrix/arithmetic.hpp>
#include<matrix/elementwise.hpp>


namespace lp{
namespace linalg{

// we are using ordered algebraic for numerical stability
template<types::ord_field T>
core::matrix<T> rref(const core::matrix<T>& A, T tol= T(-1)){
    core::Shape sh= A.shape();

    // set tolerance if not given properly
    if(tol < T(0)){
        // find a suitable value for tolerance
        core::matrix<T> mx= core::max(core::abs(A));
        tol= static_cast<T>(sh[1])*std::numeric_limits<T>::epsilon()*std::max(T(1), mx(0,0));
    }

    for(size_t i=0, n= min(sh[0], sh[1]);i<n;i++){
        // find the pivot for current row!
        
    }


}

}
}
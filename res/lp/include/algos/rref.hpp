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
    #ifdef LP_DEBUG
    if(A.is_empty()){
        throw std::runtime_error("RREF cannot process empty matrix!");
    }
    #endif
    core::Shape sh= A.shape();
    size_t m= sh[0];
    size_t n= sh[1];
    size_t rank=0;
    core::matrix<T> res= A.copy();
    // set tolerance if not given properly
    if(tol < T(0)){
        // find a suitable value for tolerance
        core::matrix<T> mx= core::max(core::abs(res));
        tol= static_cast<T>(sh[1])*std::numeric_limits<T>::epsilon()*std::max(T(1), mx(0,0));
    }

    for(size_t i=0;i<n&&rank<m;i++){
        // we want row rank to be <= m
        auto col_seg= res.submatrix(rank,i, m-rank, 1);       // m-rank x 1
        auto max_idx= core::argmax(core::abs(col_seg),0);   // 1 x 1

        size_t pivot_rel= max_idx(0,0);
        size_t pivot_row= rank + pivot_rel;
        T pivot_val= res(pivot_row, i);
        // if pivot value less than tolerance, no pivot in this column!
        if(std::abs(pivot_val)<=tol)continue;

        if(pivot_row!=rank){
            auto r1= res.row_view(rank);
            auto r2= res.row_view(pivot_row);
            // only place where copying occurs!
            auto t= r1.copy();
            r1= r2;
            r2= t;
        }

        // scale row rank!
        auto rp= res.row_view(rank);
        rp*= T(1)/pivot_val;

        // eliminate all the other rows for this column!
        for(size_t r=0;r<m;r++){
            if(r==rank)continue;
            T factor= res(r,i);
            // already eliminated!
            if(std::abs(factor)<=tol)continue;
            auto rr= res.row_view(r);
            auto rrank= res.row_view(rank);
            rr-= factor*rrank;
        }
        rank++;
    }
    return res;
}

}
}
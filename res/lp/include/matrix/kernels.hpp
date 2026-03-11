#pragma once


#include<omp.h>
#include<stdexcept>
#include<vector>
#include<functional>
#include<type_traits>

#include<cblas.h>

#include<utils/utils.hpp>
#include<utils/concepts.hpp>
#include<utils/errors.hpp>

#include<matrix/matrix.hpp>

namespace lp{
namespace core{

/**
 * Element-Wise Operation Kernels
 */
template<types::field T, types::field U, types::field V, typename binop>
void matrix_binop(const matrix<T>& A, const matrix<U>& B, matrix<V>& C, binop op){
    // assuming A, B and C have same shape
    Shape sh= A.shape();

    if(A.is_contiguous() && B.is_contiguous() && C.is_contiguous()){

        size_t total= sh[0]*sh[1];
        const T* a= A.mem();
        const U* b= B.mem();
        V* c= C.mem();

        if(A.is_aligned() && B.is_aligned() && C.is_aligned()){
            #pragma omp parallel for simd
            for(size_t i=0;i<total;i++){
                c[i]= static_cast<V>(op(a[i], b[i]));
            }
        }else{
            #pragma omp parallel for schedule(static, arch::chunk<V>)
            for(size_t i=0;i<total;i++){
                c[i]= static_cast<V>(op(a[i], b[i]));
            }
        }
    }else{
        #pragma omp parallel for collapse(2)
        for(size_t i=0;i<sh[0];i++){
            for(size_t j=0;j<sh[1];j++){
                C(i,j)= static_cast<V>(op(A(i,j), B(i,j)));
            }
        }
    }
}
template<types::field T, types::field U, types::field V, typename binop>
void scalar_binop(const matrix<T>& A, const U& s, matrix<V>& C, binop op){
    // assuming A, B and C have same shape
    Shape sh= A.shape();

    if(A.is_contiguous() && C.is_contiguous()){
        size_t total= sh[0]*sh[1];
        const T* a= A.mem();
        V* c= C.mem();

        if(A.is_aligned() && C.is_aligned()){
            #pragma omp parallel for simd
            for(size_t i=0;i<total;i++){
                c[i]= static_cast<V>(op(a[i], s));
            }
        }else{
            #pragma omp parallel for schedule(static, arch::chunk<V>) 
            for(size_t i=0;i<total;i++){
                c[i]= static_cast<V>(op(a[i], s));
            }
        }
    }else{
        #pragma omp parallel for collapse(2)
        for(size_t i=0;i<sh[0];i++){
            for(size_t j=0;j<sh[1];j++){
                C(i,j)= static_cast<V>(op(A(i,j), s));
            }
        }
    }
}
template<types::field T, types::field U, typename uniop>
void matrix_uniop(const matrix<T>& A, matrix<U>& C, uniop op){
    // assuming that the shapes are same!
    Shape sh= A.shape();
    if(A.is_contiguous() && C.is_contiguous()){
        size_t total= sh[0]*sh[1];
        const T* a= A.mem();
        U* c= C.mem();

        if(A.is_aligned() && C.is_aligned()){
            #pragma omp parallel for simd
            for(size_t i=0;i<total;i++){
                c[i]= static_cast<U>(op(a[i]));
            }
        }else{
            #pragma omp parallel for schedule(static, arch::chunk<U>)
            for(size_t i=0;i<total;i++){
                c[i]= static_cast<U>(op(a[i]));
            }
        }
    }else{
        #pragma omp parallel for collapse(2)
        for (size_t i = 0; i < sh[0]; ++i) {
            for (size_t j = 0; j < sh[1]; ++j) {
                C(i,j)= static_cast<U>(op(A(i,j)));
            }
        }
    }
}
template<types::field T, types::field U, types::field V, typename binop>
void scalar_left_binop(const T& s, const matrix<U>& A, matrix<V>& C, binop op){
    // assuming A, B and C have same shape
    Shape sh= A.shape();

    if(A.is_contiguous() && C.is_contiguous()){
        size_t total= sh[0]*sh[1];
        const T* a= A.mem();
        V* c= C.mem();

        if(A.is_aligned() && C.is_aligned()){
            #pragma omp parallel for simd
            for(size_t i=0;i<total;i++){
                c[i]= static_cast<V>(op(s, a[i]));
            }
        }else{
            #pragma omp parallel for schedule(static, arch::chunk<V>) 
            for(size_t i=0;i<total;i++){
                c[i]= static_cast<V>(op(s, a[i]));
            }
        }
    }else{
        #pragma omp parallel for collapse(2)
        for(size_t i=0;i<sh[0];i++){
            for(size_t j=0;j<sh[1];j++){
                C(i,j)= static_cast<V>(op(s, A(i,j)));
            }
        }
    }
}

/**
 * Reduction Kernels [10 Mar 2025]
 */
template<types::field T, typename binop>
matrix<T> reduce(const matrix<T>& A, int axis, binop op, T identity){
    #ifdef LP_DEBUG
    if(!(axis==0 || axis== 1 || axis== -1)){
        throw std::runtime_error(error::invalid_axis());
    }
    // -------------------------------
    // -------- Empty Matrices -------
    // -------------------------------
    if(A.is_empty()){  // empty matrices handling!
        throw std::runtime_error("Cannot Reduce Empty Matrix!");
    }
    #endif

    Shape sh= A.shape();
    // -------------------------------
    // ---- Non Empty Matrices -------
    // -------------------------------
    if(axis==0){ 
        // axis= 0 -> reduce cols -> 1 x n
        matrix<T> res(1, sh[1], A.get_alignment());
        #pragma omp parallel for schedule(static, arch::chunk<T>)
        for(size_t j=0;j<sh[1];j++){
            T acc= identity;
            // check if columns are contiguous
            if(A.stride()[0]== 1){  // ie rstride= 1
                // find pointer to j-th column!
                const T* col_ptr= A.mem() + A.stride()[2] + j*A.stride()[1];
                for(size_t i=0;i<sh[0];i++){
                    acc= op(acc, col_ptr[i]);
                }
            }else{
                // fall back to non-contiguous thing!
                for(size_t i=0;i<sh[0];i++){
                    acc= op(acc, A(i,j));
                }
            }
            res(0,j)= acc;
        }
        return res;
    }else if(axis== 1){
        // axis= 1 -> reduce rows -> m x 1
        matrix<T> res(sh[0], 1, A.get_alignment());
        #pragma omp parallel for schedule(static, arch::chunk<T>)
        for(size_t i=0;i<sh[0];i++){
            T acc= identity;
            if(A.stride()[1]== 1){  // ie the rows are contiguous
                const T* row_ptr= A.mem() + A.stride()[2] + i*A.stride()[0];
                for(size_t j=0;j<sh[1];j++){
                    acc= op(acc, row_ptr[j]);
                }
            }else{                  // fallback to normal for loop
                for(size_t j=0;j<sh[1];j++){
                    acc= op(acc, A(i,j));
                }
            }
            res(i,0)= acc;
        }
        return res;
    }else{
        // full reduction!
        size_t total= sh[0]*sh[1];
        T result= identity;
        matrix<const T> At= A.transpose();
        
        // trying to get contiguous view of all elements
        const T* data= nullptr;
        if(A.is_contiguous()){
            data= A.mem() + A.stride()[2];
        }else if(At.is_contiguous()){
            data= At.mem() + At.stride()[2];
        }

        if(data){
            // contiguous data: use parallel reduction
            // with private accummulators
            int nthreads= 1;
            #pragma omp parallel
            {
                #pragma omp single
                nthreads= omp_get_num_threads();
            }
            std::vector<T> partials(nthreads, identity);

            #pragma omp parallel
            {
                int tid= omp_get_thread_num();
                T local= identity;
                #pragma omp for schedule(static, arch::chunk<T>) nowait
                for(size_t idx= 0;idx<total;idx++){
                    local= op(local, data[idx]);
                }
                partials[tid]= local;
            }
            // combine the results
            for(size_t i=0;i<nthreads;i++){
                result= op(result, partials[i]);
            }
        }else{
            // contiguous data: use parallel reduction
            // with private accummulators
            int nthreads= 1;
            #pragma omp parallel
            {
                #pragma omp single
                nthreads= omp_get_num_threads();
            }
            std::vector<T> partials(nthreads, identity);

            #pragma omp parallel
            {
                int tid= omp_get_thread_num();
                T local= identity;
                #pragma omp for collapse(2) schedule(static, arch::chunk<T>) nowait
                for(size_t i= 0;i<sh[0];i++){
                    for(size_t j=0;j<sh[1];j++){
                        local= op(local, A(i,j));
                    }
                }
                partials[tid]= local;
            }
            // combine the results
            for(size_t i=0;i<nthreads;i++){
                result= op(result, partials[i]);
            }
        }

        matrix<T> res(1,1,A.get_alignment());
        res(0,0)=result;
        return res;
    }
    
   
    
}


/**
 * Matmul Kernels
 */

}
}
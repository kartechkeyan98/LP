#pragma once

#include<omp.h>
#ifdef _WIN32
#include<malloc.h>
#else
#include<cstdlib>
#endif

#include<utility>
#include<cstddef>

#include<memory>
#include<array>
#include<functional>
#include<concepts>


#include<matrix/utils.hpp>
#include<matrix/errors.hpp>

namespace lp{
namespace core{

template<typename T>
class matrix{
private:
    std::shared_ptr<T[]> data;
    size_t rows, cols;
    size_t rstride, cstride;
    size_t offset;
    // simd purposes
    size_t alignment;   // in bytes [B]

    // make all instantiations of matrix friends (for const views)
    // basically private members of matrix<T> can be modified 
    // by matrix<U>
    template<typename U> friend class matrix;

    // memory allocation (aligned)
    static std::shared_ptr<T[]> aligned_alloc(size_t count, size_t align){
        if(align < alignof(T)) {
            align= alignof(T);
        }
        size_t size= count*sizeof(T);
        void* ptr= nullptr;

        #if defined(__linux__)
        if(align < sizeof(void*))align= sizeof(void*);
        #endif
        
        // handle zero size matrices
        #ifdef _WIN32
        if(size==0){
            auto deleter= [](T* p){ _aligned_free(p);};
            return std::shared_ptr<T[]>(nullptr, deleter);
        }
        #else
        if(size==0){
            auto deleter= [](T* p){free(p);};
            return std::shared_ptr<T[]>(nullptr, deleter);
        }
        #endif

        #ifdef _WIN32
        ptr= _aligned_malloc(size, align);
        if(!ptr) throw std::bad_alloc();
        auto deleter= [](T* p){ _aligned_free(p);};
        #else
        if(posix_memalign(&ptr, align, size) !=0)
            throw std::bad_alloc();
        auto deleter= [](T* p){free(p);};
        #endif 
        return std::shared_ptr<T[]>(reinterpret_cast<T*>(ptr), deleter);
    }
    //deep copies
    template<typename U>
    static void deep_copy(matrix<T>& dst, const matrix<U>& src){
        // check for dimension mismatch
        if(dst.shape() != src.shape()){
            throw std::runtime_error(error::dimension_mismatch(src.shape(), dst.shape()));
        }
        if(dst.rows== 0||dst.cols== 0)return;

        size_t total= dst.rows*dst.cols;
        T* d= &dst.data[dst.offset];
        const U* s= src.mem() + src.stride()[2];

        bool dst_contig= dst.is_contiguous();
        bool src_contig= src.is_contiguous();

        if(dst_contig && src_contig){
            if(dst.is_aligned()&&src.is_aligned()){
                #pragma omp parallel for simd
                for(size_t i=0;i<total;i++){
                    d[i]= static_cast<T>(s[i]);
                }
            }else{
                #pragma omp parallel for schedule(static, arch::chunk<T>)
                for (size_t i = 0; i < total; ++i) {
                    d[i] = static_cast<T>(s[i]);
                }
            }
        }else{
            // At least one is non‑contiguous – use nested loops
            #pragma omp parallel for collapse(2)
            for (size_t i = 0; i < dst.rows; ++i) {
                for (size_t j = 0; j < dst.cols; ++j) {
                    dst(i, j) = static_cast<T>(src(i, j));
                }
            }
        }
        return;
    }

    // arithmetic in-place operations
    template<typename U, typename BinaryOp>
    matrix& matrix_inplace(const matrix<U>& rhs, BinaryOp op){
        Shape sh= rhs.shape();
        if(rows!= sh[0] || cols!= sh[1]){
            throw std::runtime_error("Matrix Dimensions do not match!");
        }
        Stride sr= rhs.stride();
        T* dst= &data[this->offset];
        const U* src= rhs.mem() + sr[2];

        

        if(this->is_contiguous() && rhs.is_contiguous()){
            // then cache friendly things can be done
            size_t total= rows*cols;
            if(this->is_aligned() && rhs.is_aligned()){
                #pragma omp parallel for simd
                for(size_t i=0;i<total;++i){
                    dst[i]= op(dst[i], src[i]);
                }
            }else{
                #pragma omp parallel for schedule(static, arch::chunk<T>)
                for(size_t i=0;i<total;++i){
                    dst[i]= op(dst[i], src[i]);
                }
            }
            
        }else{
            #pragma omp parallel for collapse(2)
            for(size_t i=0;i<rows;i++){
                for(size_t j=0;j<cols;j++){
                    dst[i*rstride + j*cstride]
                    = op(
                        dst[i*rstride + j*cstride], 
                        src[i*sr[0] + j*sr[1]]
                    );  // This shit can be inlined
                }
            }
        }
        return *this;
    }
    template<typename U, typename BinaryOp>
    matrix& scalar_inplace(const U& s, BinaryOp op){
        T* dst= &data[this->offset];

        if(this->is_contiguous()){
            // then cache friendly things can be done
            size_t total= rows*cols;
            if(this->is_aligned()){
                #pragma omp parallel for simd
                for(size_t i=0;i<total;++i){
                    dst[i]= op(dst[i], s);
                }
            }else{
                #pragma omp parallel for schedule(static, arch::chunk<T>)
                for(size_t i=0;i<total;++i){
                    dst[i]= op(dst[i], s);
                }
            }
            
        }else{
            #pragma omp parallel for collapse(2)
            for(size_t i=0;i<rows;i++){
                for(size_t j=0;j<cols;j++){
                    dst[i*rstride + j*cstride]
                    = op(dst[i*rstride + j*cstride], s);  
                    // This shit can be inlined
                }
            }
        }
        return *this;
    }
public:
/** Construction Semantics */
    // default
    matrix()noexcept
    : rows(0), cols(0),
    rstride(0), cstride(0), offset(0),
    alignment(0), data(aligned_alloc(0,0)){}
    // r,c,a constructor
    matrix(size_t r, size_t c, size_t align= 64)
    : rows(r), cols(c),
    rstride(c), cstride(1), offset(0), 
    alignment(align), data(aligned_alloc(r*c, align)){}
    // Shape, alignment constructor
    matrix(const Shape& sh, size_t align= 64)
    : matrix(sh[0],sh[1],align){}

/** Copy and Move Semantics */
    matrix(const matrix&)= default; // let default constructor be shallow
    matrix(matrix&& other)noexcept  // move constructor, leaves source empty
    :rows(std::exchange(other.rows, 0)), cols(std::exchange(other.cols, 0)),
    rstride(std::exchange(other.rstride, 0)), cstride(std::exchange(other.cstride, 0)),
    offset(std::exchange(other.offset, 0)), data(std::move(other.data)),
    alignment(std::exchange(other.alignment, 0)){}

    // copying for const views...
    template<typename U>
    matrix(const matrix<U>& other)requires std::is_same_v<T, const U>
    : rows(other.rows), cols(other.cols), 
    rstride(other.rstride), cstride(other.cstride), offset(other.offset),
    data(other.data), alignment(other.alignment){}

    // same type copy is shallow copy [numpy semantics]
    matrix& operator=(const matrix& other){
        if(data== nullptr){             // uninitialized matrix case
            rows = other.rows;
            cols = other.cols;
            rstride = other.rstride;
            cstride = other.cstride;
            offset = other.offset;
            data = other.data;           // share the same storage
            alignment = other.alignment; // alignment is part of the allocation
            return *this;
        }else{                           // initialized
            if (rows != other.rows || cols != other.cols) {
                throw std::runtime_error(error::dimension_mismatch(other.shape(), this->shape()));
            }
            return this->operator=<T>(other);
        }
    }
    matrix& operator=(matrix&& other) noexcept { // move assignment (leaves source empty)
        if (this != &other) {
            rows = std::exchange(other.rows, 0);
            cols = std::exchange(other.cols, 0);
            rstride = std::exchange(other.rstride, 0);
            cstride = std::exchange(other.cstride, 0);
            offset = std::exchange(other.offset, 0);
            data = std::move(other.data);
            alignment = std::exchange(other.alignment, 0);
        }
        return *this;
    }
    // different type copy is deep copy [numpy semantics]
    template<typename U>
    matrix& operator=(const matrix<U>& a){
        if (reinterpret_cast<const void*>(this) == reinterpret_cast<const void*>(&a))
            return *this;
        deep_copy(*this, a);
        return *this;
    }

/** Accessors and Views */
    // a(r,c) = data[offset + r * rstride + c * cstride]
    T& operator()(size_t r, size_t c){
        #ifdef LP_DEBUG
        if(r >= rows || c >= cols)
            throw std::runtime_error(error::out_of_bounds(r,c,{rows,cols}));
        #endif
        return data[offset + r * rstride + c * cstride];
    }
    const T& operator()(size_t r, size_t c)const{
        #ifdef LP_DEBUG
        if(r >= rows || c >= cols)
            throw std::runtime_error(error::out_of_bounds(r,c,{rows,cols}));
        #endif
        return data[offset + r * rstride + c * cstride];
    }

    // shape and stride vector
    Shape shape()const{return {rows,cols};}
    Stride stride()const{return {rstride, cstride, offset};}
    const T* mem()const{return data.get();}
    T* mem(){return data.get();}

    // utils
    bool is_contiguous()const{
        if (rows == 0 || cols == 0) return true;   // empty matrices are trivially contiguous
        return rstride==cols && cstride==1;
    }
    size_t get_alignment()const{return alignment;}
    bool is_aligned()const{
        // without contiguity, no SIMD!
        if (alignment == 0) return false;   // no alignment guarantee
        uintptr_t addr= reinterpret_cast<uintptr_t>(data.get() + offset);
        return (addr % alignment) == 0 && this->is_contiguous();
    }

    // views and submatrices (non-const)
    matrix col_view(size_t c){
        #ifdef LP_DEBUG
        if(c >= cols)
            throw std::runtime_error(error::out_of_bounds(0,c,{rows,cols}));
        #endif
        matrix<T> view= *this;
        view.cols= 1, view.rows= this->rows;
        view.offset= this->offset + (c * this->cstride);
        view.rstride= this->rstride;
        view.cstride= 0;
        // rows, rstride stay the same
        return view;
    }
    matrix row_view(size_t r){
        #ifdef LP_DEBUG
        if(r >= rows)
            throw std::runtime_error(error::out_of_bounds(r,0,{rows,cols}));
        #endif
        matrix<T> view= *this;
        view.rows= 1, view.cols= this->cols;
        view.offset= offset + (r* this->rstride);
        view.rstride= 0;
        view.cstride= this->cstride;
        return view;
    }
    matrix submatrix(size_t r, size_t c, size_t nrows, size_t ncols){
        // strided thing works for contiguous/uniform slicing only
        // non-uniform slicing cannot be done with stride
        #ifdef LP_DEBUG
        if(r >= rows || c >= cols)
            throw std::runtime_error(error::out_of_bounds(r,c,{rows,cols}));
        else if((nrows>0 && r>rows-nrows) || (ncols>0 && c>cols-ncols))
            throw std::runtime_error(error::out_of_bounds(r+nrows-1,c+ncols-1,{rows,cols}));
        #endif
        matrix<T> view= *this;
        view.rows= nrows;
        view.cols= ncols;
        view.offset= this->offset + (r*rstride) + (c*cstride);
        return view;
    }
    // access functions (for row and column!)
    matrix col(size_t i){return this->col_view(i);}
    matrix row(size_t i){return this->row_view(i);}

    // const views and submatrices
    matrix<const T> col_view(size_t c)const{
        #ifdef LP_DEBUG
        if(c>=cols)throw std::runtime_error(error::out_of_bounds(0,c,{rows, cols}));
        #endif
        matrix<const T> view(*this);    // convert to const (shares data though)
        // this is why it was declared as friend class
        // so that we can modify matrix<const T> private attribs from
        // matrix<T> methods!
        view.cols= 1, view.rows= this->rows;
        view.offset= this->offset + c*this->cstride;    // every column is this much off
        view.rstride= this->rstride, view.cstride= 0;
        // the alignment and all don't change
        return view;
    }
    matrix<const T> row_view(size_t r)const{
        #ifdef LP_DEBUG
        if(r>=rows) throw std::runtime_error(error::out_of_bounds(r, 0, {rows,cols}));
        #endif
        matrix<const T> view(*this);
        view.cols= this->cols, view.rows= 1;
        view.offset= this->offset + r * this->rstride;
        view.rstride= 0, view.cstride= this->cstride;
        return view;
    }
    matrix<const T> submatrix(size_t r, size_t c, size_t nrows, size_t ncols)const{
        #ifdef LP_DEBUG
        if(r >= rows || c >= cols)
            throw std::runtime_error(error::out_of_bounds(r,c,{rows,cols}));
        else if((nrows>0 && r>rows-nrows) || (ncols>0 && c>cols-ncols))
            throw std::runtime_error(error::out_of_bounds(r+nrows-1,c+ncols-1,{rows,cols}));
        #endif
        matrix<const T> view= *this;
        view.rows= nrows;
        view.cols= ncols;
        view.offset= this->offset + (r*rstride) + (c*cstride);
        return view;
    }
    // access functions for row and col (const version!)
    matrix<const T> col(size_t i)const{return this->col_view(i);}
    matrix<const T> row(size_t i)const{return this->row_view(i);}
    
    
    // copying (deep copy to same type)
    matrix copy()const{
        matrix res(rows, cols);
        deep_copy(res, *this);
        return res;
    }
    // deep copy to different type
    template<typename U>
    matrix<U> to()const{
        matrix<U> res(rows, cols);
        deep_copy(res, *this);
        return res;
    }

/** Arithmetic (in-place) */
    // arithmetic inplace operators (matrix-matrix)
    template<typename U>
    matrix& operator+=(const matrix<U>& rhs){matrix_inplace(rhs, std::plus<>()); return *this;}
    template<typename U>
    matrix& operator-=(const matrix<U>& rhs){matrix_inplace(rhs, std::minus<>()); return *this;}
    template<typename U>
    matrix& operator*=(const matrix<U>& rhs){matrix_inplace(rhs, std::multiplies<>()); return *this;}
    template<typename U>
    matrix& operator/=(const matrix<U>& rhs){matrix_inplace(rhs, std::divides<>()); return *this;}
    // arithmetic inplace operators (matrix-scalar)
    template<typename U>
    matrix& operator+=(const U& s){scalar_inplace(s, std::plus<>()); return *this;}
    template<typename U>
    matrix& operator-=(const U& s){scalar_inplace(s, std::minus<>()); return *this;}
    template<typename U>
    matrix& operator*=(const U& s){scalar_inplace(s, std::multiplies<>()); return *this;}
    template<typename U>
    matrix& operator/=(const U& s){scalar_inplace(s, std::divides<>()); return *this;}

    // special arithmetic operators
    // modulo
    template<std::integral U> requires std::integral<T>
    matrix& operator%=(const matrix<U>& rhs){matrix_inplace(rhs, std::modulus<>()); return *this;}
    template<std::integral U> requires std::integral<T>
    matrix& operator%=(const U& rhs){scalar_inplace(rhs, std::modulus<>()); return *this;} 
    
    
    // One of the most important operations!
    matrix transpose(){
        matrix res(*this);
        res.rows= cols, res.cols= rows;
        res.rstride= 1, res.cstride= cols;
        return res;
    }
    matrix<const T> transpose()const{
        matrix<const T> res(*this);
        res.rows= cols, res.cols= rows;
        res.rstride= 1, res.cstride= cols;
        return res;
    }
};

// define when something is matrix!
template<typename> struct is_matrix : std::false_type {};
template<typename T> struct is_matrix<matrix<T>> : std::true_type {};
template<typename T> inline constexpr bool is_matrix_v = is_matrix<T>::value;

}
}


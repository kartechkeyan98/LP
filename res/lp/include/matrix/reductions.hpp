#pragma once

#include<functional>
#include<iostream>
#include<cmath>

#include<utils/utils.hpp>
#include<utils/concepts.hpp>
#include<utils/errors.hpp>

#include<matrix/matrix.hpp>
#include<matrix/kernels.hpp>
#include<matrix/elementwise.hpp>




namespace lp{
namespace core{

// Reductions for fields type scalars
template<types::field T>
matrix<T> sum(const matrix<T>& A, int axis= -1){
    return reduce(A, axis, std::plus<T>(), T(0));
}
template<types::field T>
matrix<T> prod(const matrix<T>& A, int axis= -1){
    return reduce(A, axis, std::multiplies<T>(), T(1));
}
template<types::field T>
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
template<types::field T>
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

// Reduction for ordered field type scalars
template<types::ord_field T>
matrix<T> max(const matrix<T> &A, int axis= -1){
    return reduce(
        A, axis, 
        [](T& a, const T& b){return std::max(a,b);}, 
        std::numeric_limits<T>::lowest()
    );
}
template<types::ord_field T>
matrix<T> amax(const matrix<T> &A, int axis= -1){
    return reduce(
        A, axis, 
        [](T& a, const T& b){return std::max(core::abs(a), core::abs(b));}, 
        T(0)
    );
}
template<types::ord_field T>
matrix<T> min(const matrix<T> &A, int axis= -1){
    return reduce(
        A, axis, 
        [](T& a, const T& b){return std::min(a,b);}, 
        std::numeric_limits<T>::max()
    );
}
template<types::ord_field T>
matrix<T> amin(const matrix<T> &A, int axis= -1){
    return reduce(
        A, axis, 
        [](T& a, const T& b){return std::min(core::abs(a),core::abs(b));}, 
        std::numeric_limits<T>::max()
    );
}

// Argmax and the like
inline size_t linear_index(size_t i, size_t j, size_t cols) { return i * cols + j; }
template<types::ord_field T>
matrix<size_t> argmax(const matrix<T>& A, int axis=-1) {
    #ifdef LP_DEBUG
    if (!(axis == -1 || axis == 0 || axis == 1))
        throw std::runtime_error(error::invalid_axis());

    if (A.is_empty())
        throw std::runtime_error("argmax: empty matrix");
    #endif

    size_t m = A.shape()[0];
    size_t n = A.shape()[1];

    if (axis == 0) {
        // Reduce rows → result 1×n (row index of max in each column)
        matrix<size_t> res(1, n);
        #pragma omp parallel for schedule(static, arch::chunk<size_t>)
        for (size_t j = 0; j < n; ++j) {
            T best_val = std::numeric_limits<T>::lowest();
            size_t best_i = 0;
            // If column is contiguous (rstride == 1), use pointer
            if (A.stride()[0] == 1) {
                const T* col_ptr = A.mem() + A.stride()[2] + j * A.stride()[1];
                for (size_t i = 0; i < m; ++i) {
                    if (col_ptr[i] > best_val) {
                        best_val = col_ptr[i];
                        best_i = i;
                    }
                }
            } else {
                for (size_t i = 0; i < m; ++i) {
                    T val = A(i, j);
                    if (val > best_val) {
                        best_val = val;
                        best_i = i;
                    }
                }
            }
            res(0, j) = best_i;
        }
        return res;
    }
    else if (axis == 1) {
        // Reduce columns → result m×1 (col index of max in each row)
        matrix<size_t> res(m, 1);
        #pragma omp parallel for schedule(static, arch::chunk<size_t>)
        for (size_t i = 0; i < m; ++i) {
            T best_val = std::numeric_limits<T>::lowest();
            size_t best_j = 0;
            if (A.stride()[1] == 1) {   // row contiguous
                const T* row_ptr = A.mem() + A.stride()[2] + i * A.stride()[0];
                for (size_t j = 0; j < n; ++j) {
                    if (row_ptr[j] > best_val) {
                        best_val = row_ptr[j];
                        best_j = j;
                    }
                }
            } else {
                for (size_t j = 0; j < n; ++j) {
                    T val = A(i, j);
                    if (val > best_val) {
                        best_val = val;
                        best_j = j;
                    }
                }
            }
            res(i, 0) = best_j;
        }
        return res;
    }
    else { // axis == -1, global argmax → 2×1 matrix [row; col]
        size_t total = m * n;
        T best_val = std::numeric_limits<T>::lowest();
        size_t best_idx = 0;   // linear index

        // Try to get a contiguous view of all elements
        const T* data = nullptr;
        if (A.is_contiguous()) {
            data = A.mem() + A.stride()[2];
        } else {
            auto At = A.transpose();   // cheap view
            if (At.is_contiguous()) {
                data = At.mem() + At.stride()[2];
            }
        }

        if (data) {
            // Contiguous data: parallel reduction with private accumulators
            int nthreads = 1;
            #pragma omp parallel
            { 
                #pragma omp single 
                nthreads = omp_get_num_threads(); 
            }

            std::vector<T> partial_vals(nthreads, std::numeric_limits<T>::lowest());
            std::vector<size_t> partial_idxs(nthreads, 0);

            #pragma omp parallel
            {
                int tid = omp_get_thread_num();
                T local_val = std::numeric_limits<T>::lowest();
                size_t local_idx = 0;
                #pragma omp for schedule(static, arch::chunk<T>) nowait
                for (size_t idx = 0; idx < total; ++idx) {
                    if (data[idx] > local_val) {
                        local_val = data[idx];
                        local_idx = idx;
                    }
                }
                partial_vals[tid] = local_val;
                partial_idxs[tid] = local_idx;
            }

            // Combine partial results
            best_val = partial_vals[0];
            best_idx = partial_idxs[0];
            for (int i = 1; i < nthreads; ++i) {
                if (partial_vals[i] > best_val) {
                    best_val = partial_vals[i];
                    best_idx = partial_idxs[i];
                }
            }
        } else {
            // Non‑contiguous: use nested loops
            int nthreads = 1;
            #pragma omp parallel
            { 
                #pragma omp single 
                nthreads = omp_get_num_threads(); 
            }

            std::vector<T> partial_vals(nthreads, std::numeric_limits<T>::lowest());
            std::vector<size_t> partial_idxs(nthreads, 0);

            #pragma omp parallel
            {
                int tid = omp_get_thread_num();
                T local_val = std::numeric_limits<T>::lowest();
                size_t local_idx = 0;
                #pragma omp for collapse(2) schedule(static, arch::chunk<T>) nowait
                for (size_t i = 0; i < m; ++i) {
                    for (size_t j = 0; j < n; ++j) {
                        T val = A(i, j);
                        if (val > local_val) {
                            local_val = val;
                            local_idx = linear_index(i, j, n);
                        }
                    }
                }
                partial_vals[tid] = local_val;
                partial_idxs[tid] = local_idx;
            }

            // Combine partial results
            best_val = partial_vals[0];
            best_idx = partial_idxs[0];
            for (int i = 1; i < nthreads; ++i) {
                if (partial_vals[i] > best_val) {
                    best_val = partial_vals[i];
                    best_idx = partial_idxs[i];
                }
            }
        }

        // Convert linear index to (row, col)
        size_t best_row = best_idx / n;
        size_t best_col = best_idx % n;

        matrix<size_t> res(2, 1);
        res(0, 0) = best_row;
        res(1, 0) = best_col;
        return res;
    }
}
template<types::ord_field T>
matrix<size_t> argmin(const matrix<T>& A, int axis=-1) {
    #ifdef LP_DEBUG
    if (!(axis == -1 || axis == 0 || axis == 1))
        throw std::runtime_error(error::invalid_axis());

    if (A.is_empty())
        throw std::runtime_error("argmin: empty matrix");
    #endif
    size_t m = A.shape()[0];
    size_t n = A.shape()[1];
    if (axis == 0) {
        // Reduce rows → result 1×n (row index of max in each column)
        matrix<size_t> res(1, n);
        #pragma omp parallel for schedule(static, arch::chunk<size_t>)
        for (size_t j = 0; j < n; ++j) {
            T best_val = std::numeric_limits<T>::max();
            size_t best_i = 0;
            // If column is contiguous (rstride == 1), use pointer
            if (A.stride()[0] == 1) {
                const T* col_ptr = A.mem() + A.stride()[2] + j * A.stride()[1];
                for (size_t i = 0; i < m; ++i) {
                    if (col_ptr[i] < best_val) {
                        best_val = col_ptr[i];
                        best_i = i;
                    }
                }
            } else {
                for (size_t i = 0; i < m; ++i) {
                    T val = A(i, j);
                    if (val < best_val) {
                        best_val = val;
                        best_i = i;
                    }
                }
            }
            res(0, j) = best_i;
        }
        return res;
    }
    else if (axis == 1) {
        // Reduce columns → result m×1 (col index of max in each row)
        matrix<size_t> res(m, 1);
        #pragma omp parallel for schedule(static, arch::chunk<size_t>)
        for (size_t i = 0; i < m; ++i) {
            T best_val = std::numeric_limits<T>::max();
            size_t best_j = 0;
            if (A.stride()[1] == 1) {   // row contiguous
                const T* row_ptr = A.mem() + A.stride()[2] + i * A.stride()[0];
                for (size_t j = 0; j < n; ++j) {
                    if (row_ptr[j] < best_val) {
                        best_val = row_ptr[j];
                        best_j = j;
                    }
                }
            } else {
                for (size_t j = 0; j < n; ++j) {
                    T val = A(i, j);
                    if (val < best_val) {
                        best_val = val;
                        best_j = j;
                    }
                }
            }
            res(i, 0) = best_j;
        }
        return res;
    }
    else { // axis == -1, global argmax → 2×1 matrix [row; col]
        size_t total = m * n;
        T best_val = std::numeric_limits<T>::max();
        size_t best_idx = 0;   // linear index

        // Try to get a contiguous view of all elements
        const T* data = nullptr;
        if (A.is_contiguous()) {
            data = A.mem() + A.stride()[2];
        } else {
            auto At = A.transpose();   // cheap view
            if (At.is_contiguous()) {
                data = At.mem() + At.stride()[2];
            }
        }

        if (data) {
            // Contiguous data: parallel reduction with private accumulators
            int nthreads = 1;
            #pragma omp parallel
            { 
                #pragma omp single 
                nthreads = omp_get_num_threads(); 
            }

            std::vector<T> partial_vals(nthreads, std::numeric_limits<T>::max());
            std::vector<size_t> partial_idxs(nthreads, 0);

            #pragma omp parallel
            {
                int tid = omp_get_thread_num();
                T local_val = std::numeric_limits<T>::max();
                size_t local_idx = 0;
                #pragma omp for schedule(static, arch::chunk<T>) nowait
                for (size_t idx = 0; idx < total; ++idx) {
                    if (data[idx] < local_val) {
                        local_val = data[idx];
                        local_idx = idx;
                    }
                }
                partial_vals[tid] = local_val;
                partial_idxs[tid] = local_idx;
            }

            // Combine partial results
            best_val = partial_vals[0];
            best_idx = partial_idxs[0];
            for (int i = 1; i < nthreads; ++i) {
                if (partial_vals[i] < best_val) {
                    best_val = partial_vals[i];
                    best_idx = partial_idxs[i];
                }
            }
        } else {
            // Non‑contiguous: use nested loops
            int nthreads = 1;
            #pragma omp parallel
            { 
                #pragma omp single 
                nthreads = omp_get_num_threads(); 
            }

            std::vector<T> partial_vals(nthreads, std::numeric_limits<T>::max());
            std::vector<size_t> partial_idxs(nthreads, 0);

            #pragma omp parallel
            {
                int tid = omp_get_thread_num();
                T local_val = std::numeric_limits<T>::max();
                size_t local_idx = 0;
                #pragma omp for collapse(2) schedule(static, arch::chunk<T>) nowait
                for (size_t i = 0; i < m; ++i) {
                    for (size_t j = 0; j < n; ++j) {
                        T val = A(i, j);
                        if (val < local_val) {
                            local_val = val;
                            local_idx = linear_index(i, j, n);
                        }
                    }
                }
                partial_vals[tid] = local_val;
                partial_idxs[tid] = local_idx;
            }

            // Combine partial results
            best_val = partial_vals[0];
            best_idx = partial_idxs[0];
            for (int i = 1; i < nthreads; ++i) {
                if (partial_vals[i] < best_val) {
                    best_val = partial_vals[i];
                    best_idx = partial_idxs[i];
                }
            }
        }

        // Convert linear index to (row, col)
        size_t best_row = best_idx / n;
        size_t best_col = best_idx % n;

        matrix<size_t> res(2, 1);
        res(0, 0) = best_row;
        res(1, 0) = best_col;
        return res;
    }
}
template<types::ord_field T>
matrix<size_t> iamin(const matrix<T>& A, int axis=-1) {
    #ifdef LP_DEBUG
    if (!(axis == -1 || axis == 0 || axis == 1))
        throw std::runtime_error(error::invalid_axis());

    if (A.is_empty())
        throw std::runtime_error("argmin: empty matrix");
    #endif
    size_t m = A.shape()[0];
    size_t n = A.shape()[1];
    if (axis == 0) {
        // Reduce rows → result 1×n (row index of max in each column)
        matrix<size_t> res(1, n);
        #pragma omp parallel for schedule(static, arch::chunk<size_t>)
        for (size_t j = 0; j < n; ++j) {
            T best_val = std::numeric_limits<T>::max();
            size_t best_i = 0;
            // If column is contiguous (rstride == 1), use pointer
            if (A.stride()[0] == 1) {
                const T* col_ptr = A.mem() + A.stride()[2] + j * A.stride()[1];
                for (size_t i = 0; i < m; ++i) {
                    if (core::abs(col_ptr[i]) < best_val) {
                        best_val = core::abs(col_ptr[i]);
                        best_i = i;
                    }
                }
            } else {
                for (size_t i = 0; i < m; ++i) {
                    T val = core::abs(A(i, j));
                    if (val < best_val) {
                        best_val = val;
                        best_i = i;
                    }
                }
            }
            res(0, j) = best_i;
        }
        return res;
    }
    else if (axis == 1) {
        // Reduce columns → result m×1 (col index of max in each row)
        matrix<size_t> res(m, 1);
        #pragma omp parallel for schedule(static, arch::chunk<size_t>)
        for (size_t i = 0; i < m; ++i) {
            T best_val = std::numeric_limits<T>::max();
            size_t best_j = 0;
            if (A.stride()[1] == 1) {   // row contiguous
                const T* row_ptr = A.mem() + A.stride()[2] + i * A.stride()[0];
                for (size_t j = 0; j < n; ++j) {
                    if (core::abs(row_ptr[j]) < best_val) {
                        best_val = core::abs(row_ptr[j]);
                        best_j = j;
                    }
                }
            } else {
                for (size_t j = 0; j < n; ++j) {
                    T val = core::abs(A(i, j));
                    if (val < best_val) {
                        best_val = val;
                        best_j = j;
                    }
                }
            }
            res(i, 0) = best_j;
        }
        return res;
    }
    else { // axis == -1, global argmax → 2×1 matrix [row; col]
        size_t total = m * n;
        T best_val = std::numeric_limits<T>::max();
        size_t best_idx = 0;   // linear index

        // Try to get a contiguous view of all elements
        const T* data = nullptr;
        if (A.is_contiguous()) {
            data = A.mem() + A.stride()[2];
        } else {
            auto At = A.transpose();   // cheap view
            if (At.is_contiguous()) {
                data = At.mem() + At.stride()[2];
            }
        }

        if (data) {
            // Contiguous data: parallel reduction with private accumulators
            int nthreads = 1;
            #pragma omp parallel
            { 
                #pragma omp single 
                nthreads = omp_get_num_threads(); 
            }

            std::vector<T> partial_vals(nthreads, std::numeric_limits<T>::max());
            std::vector<size_t> partial_idxs(nthreads, 0);

            #pragma omp parallel
            {
                int tid = omp_get_thread_num();
                T local_val = std::numeric_limits<T>::max();
                size_t local_idx = 0;
                #pragma omp for schedule(static, arch::chunk<T>) nowait
                for (size_t idx = 0; idx < total; ++idx) {
                    if (core::abs(data[idx]) < local_val) {
                        local_val = core::abs(data[idx]);
                        local_idx = idx;
                    }
                }
                partial_vals[tid] = local_val;
                partial_idxs[tid] = local_idx;
            }

            // Combine partial results
            best_val = partial_vals[0];
            best_idx = partial_idxs[0];
            for (int i = 1; i < nthreads; ++i) {
                if (partial_vals[i] < best_val) {
                    best_val = partial_vals[i];
                    best_idx = partial_idxs[i];
                }
            }
        } else {
            // Non‑contiguous: use nested loops
            int nthreads = 1;
            #pragma omp parallel
            { 
                #pragma omp single 
                nthreads = omp_get_num_threads(); 
            }

            std::vector<T> partial_vals(nthreads, std::numeric_limits<T>::max());
            std::vector<size_t> partial_idxs(nthreads, 0);

            #pragma omp parallel
            {
                int tid = omp_get_thread_num();
                T local_val = std::numeric_limits<T>::max();
                size_t local_idx = 0;
                #pragma omp for collapse(2) schedule(static, arch::chunk<T>) nowait
                for (size_t i = 0; i < m; ++i) {
                    for (size_t j = 0; j < n; ++j) {
                        T val = core::abs(A(i, j));
                        if (val < local_val) {
                            local_val = val;
                            local_idx = linear_index(i, j, n);
                        }
                    }
                }
                partial_vals[tid] = local_val;
                partial_idxs[tid] = local_idx;
            }

            // Combine partial results
            best_val = partial_vals[0];
            best_idx = partial_idxs[0];
            for (int i = 1; i < nthreads; ++i) {
                if (partial_vals[i] < best_val) {
                    best_val = partial_vals[i];
                    best_idx = partial_idxs[i];
                }
            }
        }

        // Convert linear index to (row, col)
        size_t best_row = best_idx / n;
        size_t best_col = best_idx % n;

        matrix<size_t> res(2, 1);
        res(0, 0) = best_row;
        res(1, 0) = best_col;
        return res;
    }
}
template<types::ord_field T>
matrix<size_t> iamax(const matrix<T>& A, int axis=-1) {
    #ifdef LP_DEBUG
    if (!(axis == -1 || axis == 0 || axis == 1))
        throw std::runtime_error(error::invalid_axis());

    if (A.is_empty())
        throw std::runtime_error("argmin: empty matrix");
    #endif
    Shape sh= A.shape();
    size_t m = sh[0];
    size_t n = sh[1];
    if (axis == 0) {
        // Reduce rows → result 1×n (row index of max in each column)
        matrix<size_t> res(1, n);
        #pragma omp parallel for schedule(static, arch::chunk<size_t>)
        for (size_t j = 0; j < n; ++j) {
            T best_val = T(0);
            size_t best_i = 0;
            // If column is contiguous (rstride == 1), use pointer
            if (A.stride()[0] == 1) {
                const T* col_ptr = A.mem() + A.stride()[2] + j * A.stride()[1];
                for (size_t i = 0; i < m; ++i) {
                    if (core::abs(col_ptr[i]) > best_val) {
                        best_val = core::abs(col_ptr[i]);
                        best_i = i;
                    }
                }
            } else {
                for (size_t i = 0; i < m; ++i) {
                    T val = core::abs(A(i, j));
                    if (val > best_val) {
                        best_val = val;
                        best_i = i;
                    }
                }
            }
            res(0, j) = best_i;
        }
        return res;
    }
    else if (axis == 1) {
        // Reduce columns → result m×1 (col index of max in each row)
        matrix<size_t> res(m, 1);
        #pragma omp parallel for schedule(static, arch::chunk<size_t>)
        for (size_t i = 0; i < m; ++i) {
            T best_val = T(0);
            size_t best_j = 0;
            if (A.stride()[1] == 1) {   // row contiguous
                const T* row_ptr = A.mem() + A.stride()[2] + i * A.stride()[0];
                for (size_t j = 0; j < n; ++j) {
                    if (core::abs(row_ptr[j]) > best_val) {
                        best_val = core::abs(row_ptr[j]);
                        best_j = j;
                    }
                }
            } else {
                for (size_t j = 0; j < n; ++j) {
                    T val = core::abs(A(i, j));
                    if (val > best_val) {
                        best_val = val;
                        best_j = j;
                    }
                }
            }
            res(i, 0) = best_j;
        }
        return res;
    }
    else { // axis == -1, global argmax → 2×1 matrix [row; col]
        size_t total = m * n;
        T best_val = T(0);
        size_t best_idx = 0;   // linear index

        // Try to get a contiguous view of all elements
        const T* data = nullptr;
        if (A.is_contiguous()) {
            data = A.mem() + A.stride()[2];
        } else {
            auto At = A.transpose();   // cheap view
            if (At.is_contiguous()) {
                data = At.mem() + At.stride()[2];
            }
        }

        if (data) {
            // Contiguous data: parallel reduction with private accumulators
            int nthreads = 1;
            #pragma omp parallel
            { 
                #pragma omp single 
                nthreads = omp_get_num_threads(); 
            }

            std::vector<T> partial_vals(nthreads, T(0));
            std::vector<size_t> partial_idxs(nthreads, 0);

            #pragma omp parallel
            {
                int tid = omp_get_thread_num();
                T local_val = T(0);
                size_t local_idx = 0;
                #pragma omp for schedule(static, arch::chunk<T>) nowait
                for (size_t idx = 0; idx < total; ++idx) {
                    if (core::abs(data[idx]) > local_val) {
                        local_val = core::abs(data[idx]);
                        local_idx = idx;
                    }
                }
                partial_vals[tid] = local_val;
                partial_idxs[tid] = local_idx;
            }

            // Combine partial results
            best_val = partial_vals[0];
            best_idx = partial_idxs[0];
            for (int i = 1; i < nthreads; ++i) {
                if (partial_vals[i] > best_val) {
                    best_val = partial_vals[i];
                    best_idx = partial_idxs[i];
                }
            }
        } else {
            // Non‑contiguous: use nested loops
            int nthreads = 1;
            #pragma omp parallel
            { 
                #pragma omp single 
                nthreads = omp_get_num_threads(); 
            }

            std::vector<T> partial_vals(nthreads, T(0));
            std::vector<size_t> partial_idxs(nthreads, 0);

            #pragma omp parallel
            {
                int tid = omp_get_thread_num();
                T local_val = T(0);
                size_t local_idx = 0;
                #pragma omp for collapse(2) schedule(static, arch::chunk<T>) nowait
                for (size_t i = 0; i < m; ++i) {
                    for (size_t j = 0; j < n; ++j) {
                        T val = core::abs(A(i, j));
                        if (val > local_val) {
                            local_val = val;
                            local_idx = linear_index(i, j, n);
                        }
                    }
                }
                partial_vals[tid] = local_val;
                partial_idxs[tid] = local_idx;
            }

            // Combine partial results
            best_val = partial_vals[0];
            best_idx = partial_idxs[0];
            for (int i = 1; i < nthreads; ++i) {
                if (partial_vals[i] > best_val) {
                    best_val = partial_vals[i];
                    best_idx = partial_idxs[i];
                }
            }
        }

        // Convert linear index to (row, col)
        size_t best_row = best_idx / n;
        size_t best_col = best_idx % n;

        matrix<size_t> res(2, 1);
        res(0, 0) = best_row;
        res(1, 0) = best_col;
        return res;
    }
}

}
}
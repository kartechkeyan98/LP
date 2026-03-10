#pragma once

#include<concepts>
#include<random>

#include<matrix/concepts.hpp>
#include<matrix/matrix.hpp>


namespace lp{
namespace random{

template<core::algebraic T, typename Dist>
T random(Dist dist, unsigned seed= std::random_device{}()){
    std::mt19937 rng(seed);
    return static_cast<T>(dist(rng));
}

template<core::algebraic T>
T uniform(
    T min= static_cast<T>(0), T max= static_cast<T>(1), 
    unsigned seed= std::random_device{}()
){
    using Dist= typename std::conditional<
        std::is_integral<T>::value,
        std::uniform_int_distribution<T>,
        std::uniform_real_distribution<T>
    >::type;
    Dist dist(min, max);
    T res= random<T>(dist, seed);
    return res;
}
template<core::algebraic T> requires std::floating_point<T>
T randn(
    T mu= static_cast<T>(0), T stddev= static_cast<T>(1), 
    unsigned seed= std::random_device{}()
){
    using Dist= std::normal_distribution<T>;
    Dist dist(mu, stddev);
    T res= random<T>(dist, seed);
    return res;
}


// random matrices
template<core::algebraic T, typename Dist>
void fill_random(core::matrix<T>& mat, Dist dist, unsigned seed= std::random_device{}()){
    // determine the number of threads
    int num_threads= 1;
    #pragma omp parallel
    {
        #pragma omp single
        num_threads= omp_get_num_threads();
    }

    // one rnng per thread (pro way of doing things)
    std::vector<std::mt19937> rngs;
    rngs.reserve(num_threads);
    for(int i=0;i<num_threads;i++)rngs.emplace_back(seed+i);

    // matrix dims
    size_t rows = mat.shape()[0];
    size_t cols = mat.shape()[1];
    size_t total = rows * cols;

    // nothing to do
    if(rows==0||cols==0)return;

    if(mat.is_contiguous()){
        T* data= mat.mem();

        #pragma omp parallel
        {
            int tid= omp_get_thread_num();
            auto& local_rng= rngs[tid];
            Dist local_dist= dist;  // each thread gets own copy of distro
            #pragma omp for schedule(static, arch::chunk<T>)
            for(size_t i=0;i<total;i++){
                data[i]= local_dist(local_rng);
            }
        }
    }else{
        // non-contiguous data! Cannot do the cache friendly way
        #pragma omp parallel
        {
            int tid= omp_get_thread_num();
            auto& local_rng= rngs[tid];
            Dist local_dist= dist;  // threads get copy of distro (in case stateful distro)
            #pragma omp for collapse(2)
            for(size_t i=0;i<rows;i++){
                for(size_t j=0;j<cols;j++)mat(i,j)=local_dist(local_rng);
            }
        }
    }
    return;
}
template<core::algebraic T>
core::matrix<T> uniform_mat(
    size_t m, size_t n=1, 
    T min=static_cast<T>(0), T max= static_cast<T>(1),
    size_t align= 64, unsigned seed= std::random_device{}()
){
    using Dist= typename std::conditional<
        std::is_integral<T>::value,
        std::uniform_int_distribution<T>,
        std::uniform_real_distribution<T>
    >::type;
    core::matrix<T> mat(m,n,align);
    Dist dist(min, max);
    fill_random(mat, dist, seed);
    return mat;
}
template<core::algebraic T> requires std::floating_point<T>
core::matrix<T> randn_mat(
    size_t m, size_t n=1,
    T mu= static_cast<T>(0), T stddev= static_cast<T>(1),
    size_t align= 64, unsigned seed= std::random_device{}()
){
    using Dist= std::normal_distribution<T>;
    core::matrix<T> mat(m,n,align);
    Dist dist(mu, stddev);
    fill_random(mat, dist, seed);
    return mat;
}


}
}
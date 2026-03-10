#pragma once

#include<functional>
#include<iostream>

#include<matrix/matrix.hpp>
#include<matrix/kernels.hpp>
#include<matrix/concepts.hpp>



namespace lp{
namespace core{

template<typename T>
matrix<T> sum(const matrix<T>& A, int axis= -1){
    return reduce(A, axis, std::plus<T>(), T(0));
}

}
}
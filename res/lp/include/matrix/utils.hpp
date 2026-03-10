#pragma once

#include<array>
#include<iostream>

namespace lp{
namespace core{

using Shape= std::array<size_t, 2>;
using Stride= std::array<size_t, 3>;

bool operator==(const Shape& sh1, const Shape& sh2){return sh1[0]== sh2[0] && sh1[1]== sh2[1];}
bool operator!=(const Shape& sh1, const Shape& sh2){return !(sh1==sh2);}



template<typename T>
using vec3= std::array<T, 3>;
template<typename T>
using vec2= std::array<T, 2>;

}

namespace arch{

constexpr size_t cache_line= 64;
template<typename V>
constexpr size_t elem_size = sizeof(V);
template<typename V>
constexpr size_t chunk= (cache_line/elem_size<V>) > 0 ? (cache_line/elem_size<V>): 1;

}
}

std::ostream& operator<<(std::ostream& out, const lp::core::Shape& sh){
    out<<"("<<sh[0]<<", "<<sh[1]<<")";
    return out;
}


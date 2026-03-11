#pragma once

#include<string>
#include<sstream>

#include<utils/utils.hpp>

namespace lp{
namespace error{

std::string dimension_mismatch(const core::Shape& s1, const core::Shape& s2){
    std::stringstream ss;
    ss<<"\033[1;31m[Error]\033[0m Dimension Mismatch! Incompatible: \033[1m("
    <<s1[0]<<", "<<s1[1]<<") & ("
    <<s2[0]<<", "<<s2[1]<<")\033[0m\n";
    return ss.str();
}

std::string out_of_bounds(size_t i, size_t j, const core::Shape& sh){
    std::stringstream ss;
    ss<<"\033[1;31m[Error]\033[0m Invalid Index(s): \n";
    if(i < 0 || i >= sh[0]){ 
        ss << "row index: \033[1m"<< i 
        << "\033[0m not in range \033[1m(0, "<<sh[0]<<")\033[0m\n";
    }
    if(j < 0 || j >= sh[1]){
        ss << "col index: \033[1m"<< j 
        << "\033[0m not in range \033[1m(0, "<<sh[1]<<")\033[0m\n";
    }
    return ss.str();
}

std::string empty_submatrix(){
    std::stringstream ss;
    ss<<"\033[1;31m[Error]\033[0m Cannot return empty submatrix!\n";
    return ss.str();
}

std::string invalid_axis(){
    std::stringstream ss;
    ss<<"\033[1;31m[Error]\033[0m Invalid Axis value, can only be \033[1m[0, 1, -1]\033[0m!\n";
    return ss.str();
}

}
}
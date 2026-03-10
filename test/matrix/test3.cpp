#include<iostream>

#include<lp.hpp>
using namespace lp;

int main(void){
    core::matrix<float> m= lp::linalg::full<float>(7, 1000, 5000); 
    core::matrix<float> s= lp::core::sum(m, -1);
    std::cout<<s.shape()<<std::endl;
    std::cout<<s(0,0)<<std::endl;
}

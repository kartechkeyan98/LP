#include<iostream>

#include<lp.hpp>
using namespace lp;

int main(void){
    core::matrix<float> m= lp::random::randn_mat<float>(10, 5); 
    std::cout<<m<<std::endl;
    core::matrix<float> s= lp::core::max(m, 1);
    std::cout<<s.shape()<<std::endl;
    std::cout<<s<<std::endl;
}

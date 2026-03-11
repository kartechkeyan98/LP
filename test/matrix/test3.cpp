#include<iostream>

#include<lp.hpp>
using namespace lp;

int main(void){
    core::matrix<int> m= lp::random::uniform_mat<int>(6, 6, -100, 100); 
    std::cout<<m<<std::endl;
    core::matrix<size_t> s= lp::core::argmax(lp::core::abs(m));
    std::cout<<s.shape()<<std::endl;
    std::cout<<s<<std::endl;
}

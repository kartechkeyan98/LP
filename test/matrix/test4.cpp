#include<iostream>

#include<lp.hpp>
using namespace lp;

int main(void){
    core::matrix<float> m= lp::random::uniform_mat<float>(6, 6, -100, 100); 
    std::cout<<m<<std::endl;
    core::matrix<float> f= lp::linalg::rref(m);
    std::cout<<f<<std::endl;
}

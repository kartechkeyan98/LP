#include<iostream>
#include<iomanip>

#include<lp.hpp>
using namespace lp;

int main(void){
    core::matrix<float> m= lp::random::uniform_mat<float>(6, 5, -100, 100); 
    std::cout<<m<<std::endl;
    core::matrix<float> f= lp::linalg::rref(m);
    std::cout<<std::fixed<<std::setprecision(3)<<f<<std::endl;
}

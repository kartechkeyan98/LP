#include<iostream>
#include<iomanip>
#include<chrono>

#include<lp.hpp>
using namespace lp;

int main(void){
    core::matrix<float> m= lp::random::uniform_mat<float>(10000, 10000, -100, 100); 
    // std::cout<<std::fixed<<std::setprecision(4)<<m<<std::endl;

    auto start= std::chrono::high_resolution_clock::now();
    core::matrix<float> f= lp::linalg::rref(m);
    auto end= std::chrono::high_resolution_clock::now();

    std::chrono::duration<double, std::milli> elapsed= end-start;
    std::cout<<"RREF time: "<<elapsed.count()<<" ms\n";

    // std::cout<<std::fixed<<std::setprecision(4)<<f<<std::endl;
}

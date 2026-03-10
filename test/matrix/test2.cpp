#include<iostream>
#include<cstdint>

// including the library
#include<lp.hpp>
using namespace lp::core;   // for core structures like matrix
using namespace lp::linalg; // for factory functions like uniform and all

// gtest
#include<gtest/gtest.h>

/**
 * This test is for arithmetic operations and all
 * So, everything is related to elementwise ops!
 */
class MatrixRandomTest: public ::testing::Test{
protected:
    unsigned base_seed; // base seed for rng
    size_t iterations;  // no.of test cases basically
    void SetUp()override{
        // set these to be constant for reproducible tests
        omp_set_num_threads(8);
        base_seed= 12345;
        iterations= 200;
    }
    using R= double;
    using U= float;
};
size_t random_dim(std::mt19937& rng){
    std::uniform_int_distribution<size_t>dim_dist(1, 10000);
    return dim_dist(rng);
}

/**
 * In-place matrix-matrix operations
 */
TEST_F(MatrixRandomTest, InPlaceMatrixMatrix){
    std::mt19937 rng(base_seed);
    const double atol= 1e-10;
    for(size_t iter= 0;iter<iterations;iter++){
        std::cout<<"["<<iter+1<<"/"<<iterations<<"]";
        // create test dimensions
        size_t rows= random_dim(rng);
        size_t cols= random_dim(rng);
        matrix<R> A=lp::random::uniform_mat(rows, cols, R(-1000), R(1000), 64, base_seed + iter);
        matrix<R> B=lp::random::uniform_mat(rows, cols, R(-1000), R(1000), 64, base_seed + iter + 1000);

        matrix<R> Acopy= A.copy();
        A+=B;

        const R *a= A.mem(), *b=B.mem(), *ac= Acopy.mem();
        int mismatch=0;

        #pragma omp parallel for reduction(||:mismatch)
        for(size_t i=0;i<rows*cols;i++){
            if(fabs(a[i]-b[i]-ac[i])>=atol)mismatch=1;
        }
        EXPECT_EQ(mismatch, 0);
        std::cout<<"\b\b\b\b\b\b\b\b\b\b\b";
    }
}
TEST_F(MatrixRandomTest, InPlaceMatrixScalar){
    std::mt19937 rng(base_seed);
    const double atol= 1e-10;
    for(size_t iter= 0;iter<iterations;iter++){
        std::cout<<"["<<iter+1<<"/"<<iterations<<"]";
        // create test dimensions
        size_t rows= random_dim(rng);
        size_t cols= random_dim(rng);
        matrix<R> A= lp::random::uniform_mat(rows, cols, R(-1000), R(1000), 64, base_seed + iter);
        R s= lp::random::uniform<R>(-1000, 1000, base_seed + iter + 1000);
        
        matrix<R> Acopy= A.copy();
        A+=s;

        const R *a= A.mem(), *ac= Acopy.mem();
        int mismatch=0;

        #pragma omp parallel for reduction(||:mismatch)
        for(size_t i=0;i<rows*cols;i++){
            if(fabs(a[i]-s-ac[i])>=atol)mismatch=1;
        }
        EXPECT_EQ(mismatch, 0);
        std::cout<<"\b\b\b\b\b\b\b\b\b\b\b";
    }
}
TEST_F(MatrixRandomTest, BinaryOpsMatrixMatrix){
    std::mt19937 rng(base_seed);
    const double atol= 1e-10;
    for(size_t iter= 0;iter<iterations;iter++){
        std::cout<<"["<<iter+1<<"/"<<iterations<<"]";
        // create test dimensions
        size_t rows= random_dim(rng);
        size_t cols= random_dim(rng);
        matrix<R> A= lp::random::uniform_mat<R>(rows, cols, -1000, 1000, 64, base_seed + iter);
        matrix<R> B= lp::random::randn_mat<R>(rows, cols, 500., 30., A.get_alignment(), base_seed + iter + 100);
        
        matrix<R> C= A * B;

        const R *a= A.mem(), *b= B.mem(), *c= C.mem();
        int mismatch=0;

        #pragma omp parallel for reduction(||:mismatch)
        for(size_t i=0;i<rows*cols;i++){
            if(fabs(c[i]-(a[i]*b[i]))>=atol)mismatch=1;
        }
        EXPECT_EQ(mismatch, 0);
        std::cout<<"\b\b\b\b\b\b\b\b\b\b\b";
    }
}

TEST_F(MatrixRandomTest, MatrixTranspose){
    std::mt19937 rng(base_seed);
    iterations= 1000;
    const double atol= 1e-10;
    for(size_t iter= 0;iter<iterations;iter++){
        std::cout<<"["<<iter+1<<"/"<<iterations<<"]";
        // create test dimensions
        size_t rows= random_dim(rng);
        size_t cols= random_dim(rng);
        matrix<R> A= lp::random::uniform_mat<R>(rows, cols, -1000, 1000, 64, base_seed + iter);
        
        matrix<R> C= A.transpose();
        int mismatch=0;

        #pragma omp parallel for collapse(2) reduction(||:mismatch)
        for(size_t i=0;i<rows;i++){
            for(size_t j=0;j<cols;j++){
                if(fabs(A(i,j)-C(j,i))>=atol)mismatch= 1;
            }
        }
        EXPECT_EQ(mismatch, 0);
        std::cout<<"\b\b\b\b\b\b\b\b\b\b\b";
    }
}



int main(int argc, char** argv){
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}

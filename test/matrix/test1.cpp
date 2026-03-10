#include<iostream>
#include<cstdint>

// our stuff
#include<lp.hpp>
using namespace lp::core;
using namespace lp::linalg;

// gtest
#include<gtest/gtest.h>

class MatrixRandomTest: public ::testing::Test{
protected:
    unsigned base_seed; // base seed for rng
    size_t iterations;  // no.of test cases basically
    void SetUp()override{
        // set these to be constant for reproducible tests
        omp_set_num_threads(4);
        base_seed= 12345;
        iterations= 1000;
    }
    using R= double;
    using U= float;
};
size_t random_dim(std::mt19937& rng){
    std::uniform_int_distribution<size_t>dim_dist(1, 10000);
    return dim_dist(rng);
}

/**
 * Constructor Tests [27 Feb 2026]
 */
TEST_F(MatrixRandomTest, DefaultConstructor){
    matrix<R> m;
    Shape  def_shape= {0,0};
    Stride def_stride= {0,0,0};
    EXPECT_EQ(m.shape(),  def_shape);
    EXPECT_EQ(m.stride(), def_stride);
    EXPECT_TRUE(m.is_contiguous());
    EXPECT_EQ(m.get_alignment(), 0);
    EXPECT_TRUE(!m.mem());
}
TEST_F(MatrixRandomTest, SizeConstructor){
    std::mt19937 dim_rng(base_seed);
    for(size_t i=0;i<iterations;i++){
        std::cout<<"["<<i+1<<"/"<<iterations<<"]";
        size_t rows= random_dim(dim_rng);
        size_t cols= random_dim(dim_rng);
        matrix<R> m(rows, cols, alignof(R));

        Shape mshape= m.shape(), ashape={rows,cols};
        Stride mstride= m.stride(), astride={cols,1,0};

        // shape and stride must be equal!
        EXPECT_EQ(mshape, ashape);
        EXPECT_EQ(mstride, astride);

        // alignment and contiguity
        EXPECT_EQ(m.get_alignment(), alignof(R));
        EXPECT_TRUE(m.is_aligned());
        std::cout<<"\b\b\b\b\b\b\b\b\b\b\b";
    }
    std::cout<<std::endl;
}
TEST_F(MatrixRandomTest, ShapeConstructor){
    std::mt19937 dim_rng(base_seed);
    for(size_t i=0;i<iterations;i++){
        std::cout<<"["<<i+1<<"/"<<iterations<<"]";
        size_t rows= random_dim(dim_rng);
        size_t cols= random_dim(dim_rng);
        matrix<R> m({rows, cols}, alignof(R));

        Shape mshape= m.shape(), ashape={rows,cols};
        Stride mstride= m.stride(), astride={cols,1,0};

        // shape and stride must be equal!
        EXPECT_EQ(mshape, ashape);
        EXPECT_EQ(mstride, astride);

        // alignment and contiguity
        EXPECT_EQ(m.get_alignment(), alignof(R));
        EXPECT_TRUE(m.is_aligned());
        std::cout<<"\b\b\b\b\b\b\b\b\b\b\b";
    }
    std::cout<<std::endl;
}
TEST_F(MatrixRandomTest, CopyConstructor){
    // copy constructor only works for same type
    // not for different types!
    std::mt19937 dim_rng(base_seed);
    for(size_t i=0;i<iterations;i++){
        std::cout<<"["<<i+1<<"/"<<iterations<<"]";
        size_t rows= random_dim(dim_rng);
        size_t cols= random_dim(dim_rng);
        matrix<R> m= lp::random::uniform_mat(rows, cols, R(-6), R(6), 64, base_seed + i);
        matrix<R> d(m);

        // verify shape and stride info for source
        EXPECT_EQ(m.shape(), Shape({rows, cols}));
        EXPECT_EQ(m.stride(), Stride({cols, 1, 0}));
        EXPECT_EQ(m.get_alignment(), 64);
        EXPECT_TRUE(m.is_aligned());

        // verify d and m have the same meta data
        EXPECT_EQ(m.shape(),d.shape());
        EXPECT_EQ(m.stride(),d.stride());
        EXPECT_EQ(m.get_alignment(), d.get_alignment());
        // expecting d and m to have same data as well...
        EXPECT_EQ(m.mem(),d.mem());
        std::cout<<"\b\b\b\b\b\b\b\b\b\b\b";
    }
    std::cout<<std::endl;
}
TEST_F(MatrixRandomTest, ZeroDimMatrix){
    lp::core::matrix<int> m(0, 10);
    EXPECT_EQ(m.shape()[0], 0);
    EXPECT_EQ(m.shape()[1], 10);
    EXPECT_TRUE(m.is_contiguous());     // empty rows → contiguous
    EXPECT_TRUE(m.is_aligned()); 
    EXPECT_TRUE(!m.mem());              // the data should be nullptr
}
TEST_F(MatrixRandomTest, MoveConstructor){
    std::mt19937 dim_rng(base_seed);
    for(size_t i=0;i<iterations;i++){
        std::cout<<"["<<i+1<<"/"<<iterations<<"]";
        size_t rows= random_dim(dim_rng);
        size_t cols= random_dim(dim_rng);
        matrix<R> m= lp::random::uniform_mat(rows, cols, R(-6), R(6), 64, base_seed+i);

        // verify shape and stride info for source
        EXPECT_EQ(m.shape(), Shape({rows, cols}));
        EXPECT_EQ(m.stride(), Stride({cols, 1, 0}));
        EXPECT_EQ(m.get_alignment(), 64);
        EXPECT_TRUE(m.is_aligned());
        
        void* ptr= reinterpret_cast<void*>(m.mem());
        matrix<R> d(std::move(m));

        // things from d
        EXPECT_EQ(d.shape(), Shape({rows, cols}));
        EXPECT_EQ(d.stride(), Stride({cols, 1, 0}));
        EXPECT_EQ(d.get_alignment(), 64);
        EXPECT_TRUE(d.is_aligned());
        EXPECT_EQ(ptr, reinterpret_cast<void*>(d.mem()));

        // things from m
        EXPECT_EQ(m.shape(), Shape({0, 0}));
        EXPECT_EQ(m.stride(), Stride({0, 0, 0}));
        EXPECT_EQ(m.get_alignment(), 0);
        EXPECT_TRUE(!m.mem());
        std::cout<<"\b\b\b\b\b\b\b\b\b\b\b";
    }
    std::cout<<std::endl;
}


/**
 * Assignment Operator Tests [5 Feb 2026]
 */
TEST_F(MatrixRandomTest, CopyAssignmentSameType){
    std::mt19937 dim_rng(base_seed);
    double atol= 1e-5;
    for(size_t i=0;i<iterations;i++){
        std::cout<<"["<<i+1<<"/"<<iterations<<"]";
        size_t rows= random_dim(dim_rng);
        size_t cols= random_dim(dim_rng);
        matrix<R> m= lp::random::uniform_mat(rows, cols, R(-6), R(6), 64, base_seed+i);

        // verify shape and stride info for source
        EXPECT_EQ(m.shape(), Shape({rows, cols}));
        EXPECT_EQ(m.stride(), Stride({cols, 1, 0}));
        EXPECT_EQ(m.get_alignment(), 64);
        EXPECT_TRUE(m.is_aligned());
        
        // copying assignment from d!
        matrix<R> d(m.shape(), 64);
        d= m;

        // things from d
        EXPECT_EQ(d.shape(), m.shape());
        EXPECT_EQ(d.stride(), m.stride());
        EXPECT_EQ(d.get_alignment(), m.get_alignment());
        EXPECT_TRUE(d.is_aligned());
        EXPECT_NE(d.mem(), m.mem());    // since d is already init, deep copy

        // verifying equality!
        int mismatch= 0;
        #pragma omp parallel for reduction(||:mismatch)
        for(size_t t=0;t<rows*cols;t++){
            size_t j= t%cols;
            size_t i= t/cols;
            if(fabs(R(d(i,j))-m(i,j))>=atol){
                mismatch= 1;
            } 
        }
        EXPECT_EQ(mismatch, 0);
        
        std::cout<<"\b\b\b\b\b\b\b\b\b\b\b";
    }
    std::cout<<std::endl;
}
TEST_F(MatrixRandomTest, CopyAssignmentGeneral){
    std::mt19937 dim_rng(base_seed);
    double atol= 1e-5;
    for(size_t i=0;i<iterations;i++){
        std::cout<<"["<<i+1<<"/"<<iterations<<"]";
        size_t rows= random_dim(dim_rng);
        size_t cols= random_dim(dim_rng);
        lp::core::matrix<R> m= lp::random::uniform_mat(rows, cols, R(-6), R(6), 64, base_seed+i);

        // verify shape and stride info for source
        EXPECT_EQ(m.shape(), Shape({rows, cols}));
        EXPECT_EQ(m.stride(), Stride({cols, 1, 0}));
        EXPECT_EQ(m.get_alignment(), 64);
        EXPECT_TRUE(m.is_aligned());
        
        // copying assignment from d!
        matrix<U> d(m.shape(), m.get_alignment());
        d= m;

        // things from d
        EXPECT_EQ(d.shape(), m.shape());
        EXPECT_EQ(d.stride(), m.stride());
        EXPECT_EQ(d.get_alignment(), m.get_alignment());
        EXPECT_TRUE(d.is_aligned());
        
        // verifying equality!
        int mismatch= 0;
        #pragma omp parallel for reduction(||:mismatch)
        for(size_t t=0;t<rows*cols;t++){
            size_t j= t%cols;
            size_t i= t/cols;
            if(fabs(R(d(i,j))-m(i,j))>=atol){
                mismatch= 1;
            } 
        }
        EXPECT_EQ(mismatch, 0);
        std::cout<<"\b\b\b\b\b\b\b\b\b\b\b";
    }
    std::cout<<std::endl;
}
TEST_F(MatrixRandomTest, AssignmentDimensionMismatch){
    lp::core::matrix<double> a(2, 3);
    lp::core::matrix<double> b(3, 2);
    EXPECT_THROW(b = a, std::runtime_error);
}




int main(int argc, char **argv){
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
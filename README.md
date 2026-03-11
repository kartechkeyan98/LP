# LP

This is a linear algebra library written in C++ [C++20 required!] which uses **OpenMP** and **SIMD** (wherever possible). 
The library follows a minimum copy philosophy since it is meant to work with large matrices on CPU, by large, **10k~100k**
type range.

The suite of operations deal with elementwise, reduction and matmul operations for which a variaety of kernels are provided.

To run a source file (`.cpp` or `.c`), basically run the `Makefile` after updating the source file in that.

## Running Tests/Source Files

1. **Linux**
```Makefile
make
```

2. **Windows (MinGW)**
```Makefile
mingw32-make.exe
```


> NOTE: Matmul is work in progress

# ==================================================
# Cross-Platform Makefile for matrix library tests
# Works on Windows (MinGW) and Linux
# ==================================================

# Compilers and tools
CXX         	= g++
CXXFLAGS    	= -std=c++20 -O2
LDFLAGS     	= -fopenmp

# directories to different things
LP_HOME     	= ./res/lp
OPENBLAS_HOME  ?= ./res/openblas
GTEST_HOME     ?= ./res/gtest

# platform detection
UNAME_S    		:= $(shell uname -s 2>/dev/null || echo Windows)

ifeq ($(UNAME_S),Linux)
	# linux settings
	GTEST_LIB   = $(GTEST_HOME)/lib/linux
	OPENBLAS_LIB= $(OPENBLAS_HOME)/lib
	OPENBLAS_LD = -lopenblas
	PLATFORM_LD = -pthread -lm
	GTEST_LD    = -lgtest -lgtest_main
	TARGET_EXT  = .out
	RM_CMD      = rm
else
	# windows settings
	GTEST_LIB   = $(GTEST_HOME)/lib/windows
	OPENBLAS_LIB= $(OPENBLAS_HOME)/lib
	OPENBLAS_LD = -lopenblas
	GTEST_LD    = -lgtest -lgtest_main
	PLATFORM_LD = 
	TARGET_EXT  = .exe
	RM_CMD      = del
endif

INCFLAGS= -I$(LP_HOME)/include -I$(OPENBLAS_HOME)/include -I$(GTEST_HOME)/include
LIBFLAGS= -L$(GTEST_LIB) -L$(OPENBLAS_LIB)

TARGET  = a$(TARGET_EXT)
SOURCES = ./test/matrix/test3.cpp

all: $(TARGET)

$(TARGET): $(SOURCES)
	$(CXX) $(CXXFLAGS) $(INCFLAGS) -o $@ $^ $(LIBFLAGS) $(LDFLAGS) $(GTEST_LD) $(OPENBLAS_LD)

clean: 
	$(RM_CMD) $(TARGET)

run: $(TARGET)
	./$(TARGET)

.PHONY: all clean run
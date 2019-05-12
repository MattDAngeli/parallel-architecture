# A simple CUDA makefile
#
# Author: Naga Kandasamy
# Date: 02/14/2017
#
# CUDA depends on two things:
#  1) The CUDA nvcc compiler, which needs to be on your path,
#	or called directly, which we do here
#  2) The CUDA shared library being available at runtime,
#	which we make available by setting the LD_LIBRARY_PATH
#	variable for the duration of the makefile.
#   
#   Note that you can set your PATH and LD_LIBRARY_PATH variables as part of your
# .bash_profile so that you can compile and run without using this makefile.

NVCCFLAGS	:= -O3 -gencode arch=compute_60,code=sm_60
NVCC		:= /usr/local/cuda/bin/nvcc
LD_LIBRARY_PATH	:= /usr/local/cuda/lib64

all: matrix_multiply

matrix_multiply: matrix_multiply.cu matrix_multiply_gold.cpp 
	$(NVCC) -o matrix_multiply matrix_multiply.cu matrix_multiply_gold.cpp $(NVCCFLAGS)

clean:
	rm matrix_multiply 

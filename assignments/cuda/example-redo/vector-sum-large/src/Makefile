NVCC		:= /usr/local/cuda/bin/nvcc
NVCCFLAGS	:= -O3 -gencode arch=compute_60,code=sm_60
LD_LIB_PATH	:= /usr/local/cuda/lib64

all: vector_addition

vector_addition: main.cu vector_addition_gold.cpp
	$(NVCC) -o vector_addition main.cu vector_addition_gold.cpp $(NVCCFLAGS)

clean:
	rm vector_addition

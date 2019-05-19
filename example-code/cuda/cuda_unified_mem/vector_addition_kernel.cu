#ifndef _VECTOR_ADDITION_KERNEL_H_
#define _VECTOR_ADDITION_KERNEL_H_

#include <cuda.h>
#include <curand_kernel.h>

__global__ void 
vector_addition_kernel (float *A, float *B, float *C, int num_elements)
{
    /* Obtain the index of the thread within the thread block, 
       corresponding to the tile index. */
    int tid = blockIdx.x * blockDim.x + threadIdx.x; 	
    int stride = blockDim.x * gridDim.x; /* Compute the stride length. */
		  
    while (tid < num_elements) {
        C[tid] = A[tid] + B[tid];
        tid += stride;
    }
		  
    return; 
}

/* Kernel that initializes the values for arrays A and B. */
__global__ void 
init_kernel (float *A, float *B, int num_elements, curandState *state)
{

    int tid = blockIdx.x * blockDim.x + threadIdx.x; 	
    int stride = blockDim.x * gridDim.x; /* Compute the stride length. */
    curandState local_state = state[tid];
		  
    while (tid < num_elements) {
        A[tid] = floorf (5 * curand (&local_state)); 
        B[tid] = floorf (5 * curand (&local_state));
        tid += stride;
    }

    return;
}

/* Kernel to set up the random number generator. */
__global__ void 
setup_kernel (curandState *state)
{
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    /* Each thread gets same seed, a different sequence 
       number, no offset */
    curand_init(1234, tid, 0, &state[tid]);
    return;
}

#endif // #ifndef _VECTOR_ADDITION_KERNEL_H

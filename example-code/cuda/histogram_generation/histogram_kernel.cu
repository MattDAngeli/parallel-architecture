#ifndef _HISTOGRAM_KERNEL_H_
#define _HISTOGRAM_KERNEL_H_

/* Each thread block generates a histogram within shared memory. These histrograms are then 
   accumulated into the global histogram data structure stored in global memory .
   */
__global__ 
void histogram_kernel_fast(int *input_data, int *histogram, int num_elements, int histogram_size)
{
    __shared__ unsigned int s[HISTOGRAM_SIZE];
	
    /* Initialize the shared memory area. */ 
    if(threadIdx.x < histogram_size)
        s[threadIdx.x] = 0;
		
    __syncthreads();

    unsigned int offset = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int stride = blockDim.x * gridDim.x;
	
    while (offset < num_elements) {
        atomicAdd (&s[input_data[offset]], 1);
        offset += stride;
    }	  
	
    __syncthreads();

    /* Accumulate the histogram in shared memory into global memory. */
    if (threadIdx.x < histogram_size) 
        atomicAdd (&histogram[threadIdx.x], s[threadIdx.x]);

    return;
}

/* The shared histrogram data structure is stored in global memory and each thread directly 
   accumulates into the structure using an atomic operation. There is a lot of contention between threads
   in this implementation. 
   */
__global__ 
void histogram_kernel_slow (int *input_data, int *histogram, int num_elements, int histogram_size)
{
    unsigned int offset = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int stride = blockDim.x * gridDim.x; 
	
    while (offset < num_elements) {
        atomicAdd (&histogram[input_data[offset]], 1);
        offset += stride;
    }	

  return;  
}

#endif // #ifndef _HISTOGRAM_KERNEL_H

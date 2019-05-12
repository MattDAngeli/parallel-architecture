#ifndef _VECTOR_ADDITION_KERNEL_H_
#define _VECTOR_ADDITION_KERNEL_H_

__global__ void vector_addition_kernel ( float *A, float *B, float *C, int n_elements )
{
   // Obtain index of thread within thread block corresponding to tile index
   int thread_id = blockIdx.x * blockDim.x + threadIdx.x;

   // Compute stride length
   int stride = blockDim.x * gridDim.x;

   while (thread_id < n_elements) {
      C[thread_id] = A[thread_id] + B[thread_id];
      thread_id += stride;
   }

   return;
}

#endif // #ifndef _VECTOR_ADDITION_KERNEL_H_

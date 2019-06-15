/* Write the GPU code to perform the step(s) involved in counting sort. */

#include "counting_sort.h"

__global__ 
void counting_sort_kernel (
      int *out, int *scan, int *scan_pong, int *hist, const int *in, 
      const int num_elements )
{
   __shared__ unsigned int s[RANGE];

   int tid = blockIdx.x * blockDim.x + threadIdx.x;
   
   // Initialize shared memory area
   if( threadIdx.x < RANGE)
      s[threadIdx.x] = 0;

   __syncthreads();

   unsigned int offset = blockIdx.x * blockDim.x + threadIdx.x;
   unsigned int stride = blockDim.x * gridDim.x;

   while (offset < num_elements) {
      atomicAdd( &s[in[offset]], 1);
      offset += stride;
   }

   __syncthreads();

   // Accumulate histogram from shared memory into global memory
   if (threadIdx.x < RANGE)
      atomicAdd( &hist[threadIdx.x], s[threadIdx.x] );

   __syncthreads();

   // Create inclusive scan of histogram
   if (tid < RANGE){
      int scan_stride = 1;
      
      scan_pong[tid] = hist[tid];

      if (tid == 0)
         scan[tid] = hist[tid];
      
      __syncthreads();

      while (scan_stride < RANGE) {
         if (tid >= scan_stride) {
            scan[tid] = scan_pong[tid] + scan_pong[tid - scan_stride];
            scan_pong[tid] = scan[tid];
         }

         scan_stride *= 2;
         __syncthreads();
      }
      __syncthreads();
   }
   
   if (threadIdx.x < RANGE) {
      int t = tid;
      while (t < num_elements) {
         int histIdx = in[t];
         int histVal = hist[histIdx];
         while (histVal) {
            int outIdx = scan[in[t]] - histVal;
            out[outIdx] = in[t];
            histVal--;
         }
         t += RANGE;
         __syncthreads();
      }
   }
   
   __syncthreads();

   return;
}

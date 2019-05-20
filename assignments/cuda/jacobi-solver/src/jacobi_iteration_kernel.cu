#include "jacobi_iteration.h"

/* FIXME: Write the device kernel to solve the Jacobi iterations. */


__global__ void jacobi_iteration_kernel_hybrid ( 
      float *x, float *new_x, const float *A, const float *B )
{
   int row = blockDim.x * blockIdx.x + threadIdx.x;
   int col;
   x[row] = B[row];
   unsigned int done = 0;
   __shared__ double ssd[MATRIX_SIZE];
   double mse;

   while (!done) {
      double sum = -A[row * MATRIX_SIZE + row] * x[row];

      for (col = 0; col < MATRIX_SIZE; col++)
         sum += A[row * MATRIX_SIZE + col] * x[col];

      new_x[row] = (B[row] - sum) / A[row * MATRIX_SIZE + row];

      ssd[row] = 0.0;
      ssd[row] += (new_x[row] - x[row]) * (new_x[row] - x[row]);
      x[row] = new_x[row];
      
      __syncthreads();

      for (int stride = MATRIX_SIZE / 2; stride >= 1; stride /= 2) {
        if (row < stride)
           ssd[row] += ssd[row + stride];

         __syncthreads();
      }

      mse = sqrt( ssd[0] );
      if (mse < THRESHOLD)
         done = 1;

      __syncthreads();
   }

   return;
   
}


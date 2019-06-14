#include "jacobi_iteration.h"

/* FIXME: Write the device kernels to solve the Jacobi iterations. */


__global__ void jacobi_iteration_kernel_naive ( 
      float *x, double *mse, const float *A, const float *B )
{
   int row = blockDim.x * blockIdx.x + threadIdx.x;
   int col;
   double new_x;
   __shared__ double ssd[MATRIX_SIZE];

   for (row = row; row < MATRIX_SIZE; row += THREAD_BLOCK_SIZE) {

      double sum = -A[row * MATRIX_SIZE + row] * x[row];

      for (col = 0; col < MATRIX_SIZE; col++)
         sum += A[row * MATRIX_SIZE + col] * x[col];

      new_x = (B[row] - sum) / A[row * MATRIX_SIZE + row];

      ssd[row] = 0.0;
      ssd[row] += (new_x - x[row]) * (new_x - x[row]);
      x[row] = new_x;

      __syncthreads();

      for (int stride = MATRIX_SIZE / 2; stride >=1; stride /= 2) {
         if (row < stride)
            ssd[row] += ssd[row + stride];

         __syncthreads();
      }
   }

   __syncthreads();

   *mse = sqrt( ssd[0] );

   return;
}

__global__ void jacobi_iteration_kernel_optimized (
      float *x, double *mse, const float *A, const float *B )
{
   __shared__ float A_tile[TILE_SIZE][TILE_SIZE];
   __shared__ float x_tile[TILE_SIZE];
   __shared__ double ssd[MATRIX_SIZE];

   int threadX = threadIdx.x;
   int threadY = threadIdx.y;
   int blockY = blockIdx.y;
   int row = blockDim.y * blockY + threadY;
   double new_x;

   if (row < MATRIX_SIZE) {
      double sum = -A[row * MATRIX_SIZE + row] * x[row];

      for (int i = 0; i < MATRIX_SIZE; i += TILE_SIZE) {

         A_tile[threadY][threadX] = A[row * MATRIX_SIZE + i + threadX];
         
         if (threadY == 0) {
            x_tile[threadX] = x[i + threadX];
         }

         __syncthreads();

         // Compute partial sum for current tile
         if (threadX == 0) {
            for (int j = 0; j < TILE_SIZE; j++)
               sum += A_tile[threadY][j] * x_tile[j];
         }
         __syncthreads();
      }

      new_x = (B[row] - sum) / A[row * MATRIX_SIZE + row];

      ssd[row] = 0.0;
      ssd[row] += (new_x - x[row]) * (new_x - x[row]);
      x[row] = new_x;

      __syncthreads();

      for (int stride = MATRIX_SIZE / 2; stride >= 1; stride /=2) {
         if (row < stride)
            ssd[row] += ssd[row + stride];

         __syncthreads();
      }

      *mse = sqrt( ssd[0] );
   }

   return;
}


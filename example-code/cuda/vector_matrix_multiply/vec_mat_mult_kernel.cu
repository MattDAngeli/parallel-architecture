/* Vector-Matrix multiplication: Y = A * X.
 * Device code.
 * Author: Naga Kandasamy
 * Date created: January 29, 2011
 * Date modified: May 4, 2019
*/

#ifndef _MATRIXMUL_KERNEL_H_
#define _MATRIXMUL_KERNEL_H_

#include "vec_mat_mult.h"

/* Kernel shows the use of shared memory. Threads maintain good reference patterns 
   to global memory via coalesced accesses. 
 */
__global__ 
void multiply_kernel_optimized (const float* A, const float* X, float* Y, int num_rows, int num_columns)
{
    /* Declare shared memory for the thread block. */
	__shared__ float aTile[TILE_SIZE][TILE_SIZE];
	__shared__ float xTile[TILE_SIZE];

	// Calculate thread index, block index and position in matrix
	int threadX = threadIdx.x;
	int threadY = threadIdx.y;
	int blockY = blockIdx.y;
	int row = blockDim.y * blockY + threadY;

    if (row < num_rows) {
        float sum = 0.0;

        for (int i = 0; i < num_columns; i += TILE_SIZE) {
            /* Bring TILE_SIZE elements per row of A into shared memory. */
            aTile[threadY][threadX] = A[num_columns * row + i + threadX]; 		
            
            /* Bring TILE_SIZE elements of the vector X into shared memory. */
            if(threadY == 0) 
                xTile[threadX] = X[i + threadX]; 
		
            /* Barrier sync to ensure that shared memory has been populated. */
            __syncthreads();

            /* Compute partial sum for the current tile. */
            if (threadX == 0) {
                for (int k = 0; k < TILE_SIZE; k += 1) 
                    sum += aTile[threadY][k] * xTile[k]; 		      
            }
            __syncthreads();
        }

        /* Store sum. */
        if (threadX == 0) 
            Y[row] = sum;
    }

    return;
}

/* This kernel uses global memory to compute Y = AX. The reference patterns 
 to global memory by the threads are not coalesced. */
__global__ 
void multiply_kernel_naive (const float *A, const float *X, float *Y, int num_rows, int num_columns)
{		  
    int threadY = threadIdx.y; 	  
    int blockY = blockIdx.y;
		  
    int row = blockDim.y * blockY + threadY; /* Obtain row number. */

    if (row < num_rows) {
        float sum = 0.0;
        for (int i = 0; i < num_columns; i++) {				 
            sum += A[num_columns * row + i] * X[i];
        }
		  
        Y[row] = sum;
    }

    return;
}

#endif // #ifndef _MATRIXMUL_KERNEL_H_

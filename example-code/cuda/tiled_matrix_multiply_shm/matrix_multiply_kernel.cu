/* Matrix multiplication: P = M * N.
 * Device code.

    Author: Naga Kandasamy
    Date: 2/16/2017
 */

#ifndef _MATRIX_MULTIPLY_KERNEL_H_
#define _MATRIX_MULTIPLY_KERNEL_H_

#include "matrix.h"

/* Kernel uses global memory. It exhibits redundant loading of both rows and columns. */
__global__ void 
matrix_multiply_kernel_naive (float *P, const float *M, const float *N, int matrix_size)
{
	/* Obtain thread index within the thread block. */
	int threadX = threadIdx.x;
	int threadY = threadIdx.y;

	/* Obtain block index within the grid. */
	int blockX = blockIdx.x;
	int blockY = blockIdx.y;

	/* Find position in matrix. */
	int column = blockDim.x * blockX + threadX;
	int row = blockDim.y * blockY + threadY;

	float P_temp = 0;
	for (int k = 0; k < matrix_size; k++) {
		float M_element = M[matrix_size * row + k]; /* Row elements. */
		float N_element = N[matrix_size * k + column]; /* Column elements. */
		P_temp += M_element * N_element; 
	}

	/* Write result to P. */
	P[row * matrix_size + column] = P_temp;
    return;
}

/* Kernel uses shared memory as the mechanism to reuse data between threads. */
__global__ void 
matrix_multiply_kernel_optimized (float *P, const float *M, const float *N, int matrix_size)
{
    /* Allocate shared memory for the thread block. */
    __shared__ float Msub[TILE_SIZE][TILE_SIZE];
    __shared__ float Nsub[TILE_SIZE][TILE_SIZE];

    /* Obtain thread index within the thread block. */
    int threadX = threadIdx.x; 
    int threadY = threadIdx.y; 

    /* Obtain block index within the grid. */
	int blockX = blockIdx.x;
	int blockY = blockIdx.y;

	/* Find position in matrix; which is the thread to data mapping. */
	int column = blockDim.x * blockX + threadX;
	int row = blockDim.y * blockY + threadY;
   
    int k = 0;
    float Psub = 0.0f;
   
    while (k < matrix_size) {
        
        /* Check edge condtions for matrix M for this tile. */
        if (((k + threadX) < matrix_size) && (column < matrix_size))
            Msub[threadY][threadX] = M[row * matrix_size + k + threadX];
        else
            Msub[threadY][threadX] = 0.0f; /* Pad out the shared memory area. */ 

        /* Check edge conditions for matrix N for this tile. */
        if(((k + threadY) < matrix_size) && (row < matrix_size))
            Nsub[threadY][threadX] = N[(k + threadY) * matrix_size + column];
        else
            Nsub[threadY][threadX] = 0.0f; 

        /* Barrier for threads to wait while shared memory is populated by the thread block. */
        __syncthreads(); 
    
        /* Multiply the row and column entries corresponding to the tile just loaded. */ 
        for (int i = 0; i < TILE_SIZE; i++)
            Psub += Msub[threadY][i] * Nsub[i][threadX];

        __syncthreads();
    
        k += TILE_SIZE;
  }

    /* Write result to P. */
    if (column < matrix_size && row < matrix_size)
        P[row * matrix_size + column] = Psub;

    return;
}


#endif

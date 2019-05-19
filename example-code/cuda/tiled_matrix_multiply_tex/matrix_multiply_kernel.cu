/* Matrix multiplication: P = M * N.
 * Device code.

    Author: Naga Kandasamy
    Date: 2/16/2017
 */

#ifndef _MATRIX_MULTIPLY_KERNEL_H_
#define _MATRIX_MULTIPLY_KERNEL_H_

#include "matrix.h"

/* Declare 1D textures to hold the M and N matrices. */
texture<float> M_on_tex;
texture<float> N_on_tex;

/* Declare 2D textures to hold the M and N matrices. */
texture<float, 2> M_on_tex_2D;
texture<float, 2> N_on_tex_2D;

/* Example of a kernel that uses 2D textures. */
__global__ void 
matrix_multiply_kernel_2Dtex (float *P, const float *M, const float *N, int matrix_size)
{	
	/* Obtain thread and block indices. */
	int threadX = threadIdx.x; 
    int threadY = threadIdx.y;
	int blockX = blockIdx.x; 
    int blockY = blockIdx.y;

	/* Find position within matrix. */
	int column_number = TILE_SIZE * blockX + threadX; 
    int row_number = TILE_SIZE * blockY + threadY;

	double P_temp = 0;	
    for (int k = 0; k < matrix_size; k++) {		
        /* Scan through row elements. Texture values are indexed in (x, y), that is (col, row) form rather 
           than the (y, x) or (row, col) form. */
        double M_element = tex2D (M_on_tex_2D, k, row_number); 
        double N_element = tex2D (N_on_tex_2D, column_number, k);
        P_temp += M_element * N_element; 
	}
	
	/* Write result to P. */
	P[row_number * matrix_size + column_number] = (float)P_temp;
}

/* Example of a kernel that uses a 1D texture. */
__global__ void 
matrix_multiply_kernel_1Dtex (float *P, const float *M, const float *N, int matrix_size)
{
	/* Obtain thread and block indices. */
	int threadX = threadIdx.x; 
    int threadY = threadIdx.y;
	int blockX = blockIdx.x; 
    int blockY = blockIdx.y;

	/* Find position in matrix. */
	int column_number = TILE_SIZE * blockX + threadX; 
    int row_number = TILE_SIZE * blockY + threadY;

	double P_temp = 0;	
    for (int k = 0; k < matrix_size; k++) {		
        double M_element = tex1Dfetch (M_on_tex, (matrix_size * row_number + k)); /* Scan through row elements. */
		double N_element = tex1Dfetch (N_on_tex, (matrix_size * k + column_number)); /* Scan through column elements. */	
        P_temp += M_element * N_element; 
	}
	
	/* Write result to P. */
	P[row_number * matrix_size + column_number] = (float) P_temp;
}


/* Kernel uses shared memory as the mechanism to reuse data between threads. */
__global__ void 
matrix_multiply_kernel_shm (float *P, const float *M, const float *N, int matrix_size)
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


__global__ void 
matrix_multiply_kernel_vanilla (float *P, const float *M, const float *N, int matrix_size)
{
	/* Obtain thread index within the thread block. */
	int threadX = threadIdx.x;
	int threadY = threadIdx.y;

	/* Obtain block index within the grid. */
	int blockX = blockIdx.x;
	int blockY = blockIdx.y;

	/* Find position in matrix. */
	int column_number = blockDim.x * blockX + threadX;
	int row_number = blockDim.y * blockY + threadY;

	double P_temp = 0;
	for (int k = 0; k < matrix_size; k++) {
		double M_element = M[matrix_size * row_number + k]; /* Row elements. */
		double N_element = N[matrix_size * k + column_number]; /* Column elements. */
		P_temp += M_element * N_element; 
	}

	/* Write result to P. */
	P[row_number * matrix_size + column_number] = (float) P_temp;
}
#endif

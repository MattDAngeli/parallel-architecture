/* Matrix multiplication: P = M * N.
 * Device code.

    Author: Naga Kandasamy
    Date: 2/16/2017
 */

#ifndef _MATRIX_MULTIPLY_KERNEL_H_
#define _MATRIX_MULTIPLY_KERNEL_H_

#include "matrix.h"

__global__ void 
matrix_multiply_kernel (float *P, const float *M, const float *N, int matrix_size)
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

/* Matrix multiplication: P = M * N.
 * Device code.
 */
#ifndef _MATRIX_MULTIPLY_KERNEL_H_
#define _MATRIX_MULTIPLY_KERNEL_H_

#include "matrix.h"

__global__ void
matrix_multiply (float *P, float *M, float *N)
{
	/* Obtain thread location within the block. */
	int tx = threadIdx.x;
	int ty = threadIdx.y;

	float P_temp = 0;
	for (int k = 0; k < WM; ++k) {
		float M_element = M[ty * WM + k];
		float N_element = N[k * WN + tx];
		P_temp += M_element * N_element;
	}

	P[ty * WN + tx] = P_temp;
}
#endif // #ifndef _MATRIX_MULTIPY_KERNEL_H_

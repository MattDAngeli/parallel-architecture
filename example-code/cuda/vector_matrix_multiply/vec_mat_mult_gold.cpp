/* Reference solution for Y = A.X 
 * Author: Naga Kandasamy
 * Date modified: May 4, 2019
 * */
 
#include <stdlib.h>
#include "vec_mat_mult.h"

extern "C" void compute_gold (const matrix_t, const matrix_t, matrix_t);

void
compute_gold (const matrix_t A, const matrix_t X, matrix_t Y)
{
    for (unsigned int i = 0; i < A.num_rows; i++) {
		float sum = 0.0;
		for (unsigned int j = 0; j < X.num_rows; j++) {
			double a = A.elements[i * A.num_rows + j]; /* Pick A[i, j] */
			double x = X.elements[j]; /* Pick X[j] */
			sum += a * x;
		}

		Y.elements[i] = sum; /* Store result in Y. */
	}	

    return;
}

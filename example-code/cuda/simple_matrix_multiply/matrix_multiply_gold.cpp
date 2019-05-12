#include <stdlib.h>

extern "C"
void compute_gold (const float *, const float *, float *, unsigned int, unsigned int, unsigned int);

/* hM: Height of matrix M.
 * wM: Width of matrix M.
 * wN: Width of matrix N.
 */
void
compute_gold (const float *M, const float *N, float *P, unsigned int hM, unsigned int wM, unsigned int wN)
{
	for (unsigned int i = 0; i < hM; ++i)
		for (unsigned int j = 0; j < wN; ++j) {
			double sum = 0;
			for (unsigned int k = 0; k < wM; ++k) {
				double a = M[i * wM + k];
				double b = N[k * wN + j];
				sum += a * b;
			}

			P[i * wN + j] = (float) sum;
		}
}

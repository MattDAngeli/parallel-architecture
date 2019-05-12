// Reference code for vector addition
#include <stdio.h>
#include <math.h>
#include <float.h>

extern "C" void compute_gold( float *, float *, float *, int );

void compute_gold (float *A, float *B, float *C, int n_elements )
{
   int i; 
   for (i = 0; i < n_elements; i++)
      C[i] = A[i] + B[i];
}

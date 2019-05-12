/* Matrix multiplication: C = A * B.
 * Host code.

 * Modified: Naga Kandasamy
 * Date: April 26, 2019
 */

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <sys/time.h>
#include <math.h>

/* include the CUDA kernels during the preprocessing step. */
#include "matrix_multiply_kernel.cu"

extern "C" void compute_gold (const float *, const float *, float *, unsigned int, unsigned int, unsigned int);
matrix allocate_matrix_on_device (const matrix);
matrix allocate_matrix (int, int, int);
void copy_matrix_to_device (matrix, const matrix);
void copy_matrix_from_device (matrix, const matrix);
void matrix_multiply_on_device (const matrix, const matrix, matrix);
int check_results (float *, float *, int, float);

int 
main (int argc, char** argv) {
    
    /* Allocate and populate the matrices. Assume square matrices*/
    matrix  M, N, P;
	unsigned int num_elements = WP * HP;
	srand (time (NULL));
    M  = allocate_matrix (MATRIX_SIZE, MATRIX_SIZE, 1); 
    N  = allocate_matrix (MATRIX_SIZE, MATRIX_SIZE, 1);
    P  = allocate_matrix (MATRIX_SIZE, MATRIX_SIZE, 0);

    /* Multiply using CPU. */
    printf ("Multiplying two %d x %d matrices on CPU\n", M.height, M.width);
	matrix reference = allocate_matrix (MATRIX_SIZE, MATRIX_SIZE, 0);
	compute_gold (M.elements, N.elements, reference.elements, HM, WM, WN);

    /* Multiply matrices on the device. */
    printf ("Multiplying two %d x %d matrices on GPU\n", M.height, M.width);
    matrix_multiply_on_device(M, N, P);         	
		
	/* Check if the device result is equivalent to the expected solution. */
    printf ("Checking GPU result for correctness\n");
    float eps = 1e-4;
	int status = check_results (reference.elements, P.elements, num_elements, eps);
	printf ("TEST %s\n", (1 == status) ? "PASSED" : "FAILED");
	
    /* Free host matrices. */
	free (M.elements); 
    free (N.elements); 
	free (P.elements); 
	
    exit (EXIT_SUCCESS);
}

void 
matrix_multiply_on_device (const matrix M, const matrix N, matrix P)
{
	/* Allocate device memory. */
	matrix d_M = allocate_matrix_on_device (M);
	matrix d_N = allocate_matrix_on_device (N);
	matrix d_P = allocate_matrix_on_device (P);

	/* Copy matrices to device memory. */
	copy_matrix_to_device (d_M, M);
	copy_matrix_to_device (d_N, N);

	/* Set up execution grid. */
	dim3 threads (MATRIX_SIZE, MATRIX_SIZE);
	dim3 grid (d_M.width/threads.x, d_N.height/threads.y);

	/* Launch kernel. */
	matrix_multiply<<<grid, threads>>>(d_P.elements, d_M.elements, d_N.elements);

	/* Check if kernel execution generated an error. */
	cudaError_t err = cudaGetLastError ();
	if (cudaSuccess != err) {
		fprintf (stderr, "Kernel execution failed: %s\n", cudaGetErrorString (err));
		exit (EXIT_FAILURE);
	}

	/* Copy result from device to host. */
	copy_matrix_from_device (P, d_P);

	/* Clean up memory on the GPU. */
	cudaFree (d_M.elements);
	cudaFree (d_N.elements);
	cudaFree (d_P.elements);

    return;
}

/* Allocate matrix on device. */
matrix 
allocate_matrix_on_device (const matrix M)    
{
	matrix Mdevice = M;
	int size = M.width * M.height * sizeof (float);
	cudaMalloc ((void**) &Mdevice.elements, size);
	return Mdevice;
}

/* Allocate a matrix of dimensions height * width
   If init == 0, initialize to all zeros.  
   If init == 1, perform random initialization.
   */
matrix 
allocate_matrix (int height, int width, int init)
{
	matrix M;
	M.width = M.pitch = width; 
    M.height = height;
	int size = M.width * M.height;
	M.elements = (float *) malloc (size * sizeof (float));

	for (unsigned int i = 0; i < M.height * M.width; i++)
		M.elements[i] = (init == 0) ? (0.0f) : (rand ()/(float) RAND_MAX);
	
	return M;
}	

/* Copy from host to device. */
void 
copy_matrix_to_device (matrix Mdevice, const matrix Mhost)
{
	int size = Mhost.width * Mhost.height * sizeof (float);
	Mdevice.height = Mhost.height;
	Mdevice.width = Mhost.width;
	Mdevice.pitch = Mhost.pitch;
	cudaMemcpy (Mdevice.elements, Mhost.elements, size, cudaMemcpyHostToDevice);
}

/* Copy from device to host. */
void 
copy_matrix_from_device (matrix Mhost, const matrix Mdevice)    
{
	int size = Mdevice.width * Mdevice.height * sizeof (float);
	cudaMemcpy (Mhost.elements, Mdevice.elements, size, cudaMemcpyDeviceToHost);
}

int 
check_results (float *reference, float *gpu_result, int num_elements, float threshold)
{
    int check_mark = 1;
    for (int i = 0; i < num_elements; i++)
        if (fabsf((reference[i] - gpu_result[i])/reference[i]) > threshold) {
            check_mark = 0;
            break;
        }

    return check_mark;
}

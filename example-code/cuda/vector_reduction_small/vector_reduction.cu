/* Vector reduction example using shared memory. 
 * Works for small vectors that can be operated upon by a single thread block.
 
 * Build as follows: make clean && make
 * Execute as follows: ./vector_reduction

 * Author: Naga Kandasamy
 * Date modified: 11/10/2015
*/

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <float.h>
#include <time.h>

/* We are fixing the problem size to fit within a thread block. */
#define NUM_ELEMENTS 1024

/* Include kernels. */
#include "vector_reduction_kernel.cu"

void run_test (void);
extern "C" double compute_gold (float *, const unsigned int);
double compute_on_device (float *, int);
void check_CUDA_error (const char *);

int 
main (int argc, char** argv) 
{
	run_test ();
	exit (EXIT_SUCCESS);
}

void
run_test (void) 
{
	int num_elements = NUM_ELEMENTS;
	const unsigned int array_mem_size = sizeof (float) * num_elements;

	/* Allocate memory on host to store the input data. */
	float* h_data = (float *) malloc (array_mem_size);

	/* Initialize the input data on the host to be floating-point values
	   between [-.5, +.5]. */
	srand (time (NULL));
	for (unsigned int i = 0; i < num_elements; ++i) {
		h_data[i] = rand ()/(float) RAND_MAX - 0.5;
	}

	/* Calculate reference solution. */
	double reference = 0.0; 
    printf ("Reducing vector on CPU\n");
	reference = compute_gold (h_data, num_elements);    
    printf ("Answer = %f\n", reference);

    /* Calculate solution on the GPU. */
    printf ("Reducing vector on GPU\n");    
    float gpu_result = compute_on_device (h_data, num_elements);
    printf ("Answer = %f\n", gpu_result);

	/* Check for correctness. */
    float eps = 1e-6;
	if (fabsf ((reference - gpu_result)/reference) <= eps) 
        printf ("TEST PASSED\n");
    else
        printf ("TEST FAILED\n");

	free (h_data);
    exit (EXIT_SUCCESS);
}

/* Reduce the vector on the GPU. */
double 
compute_on_device (float* h_data, int num_elements)
{
	float *d_data; /* Pointer to device address holding array. */
    double *d_result; /* Pointer to device address holding result. */
   	int data_size = sizeof (float) * num_elements;

	/* Allocate memory on device for the array. */
	cudaMalloc ((void**) &d_data, data_size);
	check_CUDA_error ("Error allocating memory");
	/* Copy data from host memory to device memory. */
	cudaMemcpy (d_data, h_data, data_size, cudaMemcpyHostToDevice);
	check_CUDA_error ("Error copying host to device memory");

    /* Allocate memory on device to store the reduction result. */
    cudaMalloc ((void **) &d_result, sizeof (double));
    check_CUDA_error ("Error allocating memory");

	/* Set up execution grid and invoke kernel. */
	dim3 threads (NUM_ELEMENTS, 1, 1);
	dim3 grid (1, 1);

    printf ("Using kernel, version 1\n");
	vector_reduction_kernel_v1<<<grid, threads>>>(d_data, d_result, num_elements);
    check_CUDA_error ("Error in kernel");

    printf ("Using kernel, version 2\n");
    vector_reduction_kernel_v2<<<grid, threads>>>(d_data, d_result, num_elements);
	check_CUDA_error ("Error in kernel");

	/* Copy result from device to host. */
    double h_result;
	cudaMemcpy(&h_result, d_result, sizeof (double), cudaMemcpyDeviceToHost);
	check_CUDA_error ("Error copying host to device memory");

	/* Clean up device memory. */
	cudaFree (d_data);
    cudaFree (d_result);
	check_CUDA_error ("Error freeing memory");

    return h_result;
}

void 
check_CUDA_error (const char *msg)
{
	cudaError_t err = cudaGetLastError ();
	if (cudaSuccess != err) {
		printf ("CUDA ERROR: %s (%s).\n", msg, cudaGetErrorString(err));
		exit (EXIT_FAILURE);
	}						 
}

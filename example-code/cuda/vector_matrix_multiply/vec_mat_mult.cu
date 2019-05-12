/* Host code for vector-matrix multiplication: Y = A * X.
 * A is a n x m matrix, X is a m x 1 vector and Y is the n x 1 result vector.
 * 
 * Build and run as follows: 
 *  make clean && make
 * ./vec_mat_mult num-rows num-columns

 * Author: Naga Kandasamy
 * Date modified: May 3, 2019
 */

#include <stdlib.h>
#include <stdio.h>
#include <sys/time.h>
#include <string.h>
#include <math.h>

/* Include the kernel code here. */
#include "vec_mat_mult_kernel.cu"

extern "C" void compute_gold (const matrix_t, const matrix_t, matrix_t);
matrix_t allocate_matrix_on_device (const matrix_t);
matrix_t allocate_matrix_on_host (int, int, int);
void copy_matrix_to_device (matrix_t, const matrix_t);
void copy_matrix_from_device (matrix_t, const matrix_t);
void compute_on_device (const matrix_t, const matrix_t, matrix_t);
void print_matrix (const matrix_t);
float get_random_number (int, int);
void check_CUDA_error (const char *);
int check_results (const float *, const float *, int, float);

int 
main (int argc, char** argv) 
{
    if (argc < 3) {
        printf ("Usage: %s num-rows num-columns\n", argv[0]);
        printf ("num-rows: Height of the matrix\n");
        printf ("num-columns: Width of the matrix\n");
		exit (EXIT_FAILURE);
	}

    int num_rows = atoi (argv[1]);
    int num_columns = atoi (argv[2]);

    /* Allocate and initialize the matrices on the host. */
    matrix_t  A; /* n x m matrix. */
	matrix_t  X; /* m x 1 vector. */
	
	/* Initialize the random number generator with a seed value. */ 
	srand (time(NULL));
	
    printf ("Creating the %d x %d matrix\n", num_rows, num_columns);
	A  = allocate_matrix_on_host (num_rows, num_columns, 1);
    if (A.elements == NULL) {
        perror ("Malloc");
        exit (EXIT_FAILURE);
    }

    printf ("Creating the %d x 1 vector\n", num_columns);
	X  = allocate_matrix_on_host (num_columns, 1, 1);
    if (X.elements == NULL) {
        perror ("Malloc");
        exit (EXIT_FAILURE);
    }

    /* compute the vector-matrix multiplication on the CPU. */
    matrix_t Y_ref = allocate_matrix_on_host (num_rows, 1, 0);
    if (Y_ref.elements == NULL) {
        perror ("Malloc");
        exit (EXIT_FAILURE);
    }
    printf ("\nComputing the vector-matrix multiplication on the CPU\n");
	compute_gold (A, X, Y_ref);
 
	/* Perform the vector-matrix multiplication on the GPU. */
    matrix_t Y_device  = allocate_matrix_on_host (num_rows, 1, 0);
    if (Y_device.elements == NULL) {
        perror ("Malloc");
        exit (EXIT_FAILURE);
    }

	printf ("\nComputing the vector-matrix multiplication on the GPU\n");
    compute_on_device (A, X, Y_device);

	/* Check device result aganist reference. */
    float eps = 1e-4;
    int check = check_results (Y_ref.elements, Y_device.elements, Y_ref.num_rows, eps);
    if (check == 1)
        printf ("TEST PASSED\n");
    else
        printf ("TEST FAILED\n");

	free ((void *) A.elements); 
	free ((void *) X.elements); 
    free ((void *) Y_ref.elements);
	free ((void *) Y_device.elements);

	exit (EXIT_SUCCESS);
}

/* Perform the multiplication on the device. */
void 
compute_on_device (const matrix_t A, const matrix_t X, matrix_t Y) 
{	
	/* Load matrices A and X on to the device. */
	matrix_t Ad = allocate_matrix_on_device (A);
	copy_matrix_to_device (Ad, A);

	matrix_t Xd = allocate_matrix_on_device (X);
	copy_matrix_to_device (Xd, X);
	
	/* Allocate Y on the device. */
	matrix_t Yd = allocate_matrix_on_device (Y);

    struct timeval start, stop;	

    printf ("\nUsing naive verion\n");
    gettimeofday (&start, NULL);

    /* Set up the execution configuration for the naive kernel and launch it. */
	dim3 threads (1, THREAD_BLOCK_SIZE, 1);
	dim3 grid (1, (Yd.num_rows + THREAD_BLOCK_SIZE - 1)/THREAD_BLOCK_SIZE);

    multiply_kernel_naive<<<grid, threads>>>(Ad.elements, Xd.elements, Yd.elements, Ad.num_rows, Ad.num_columns);
    cudaDeviceSynchronize ();
 
    gettimeofday (&stop, NULL);
	printf ("Execution time = %fs\n", (float)(stop.tv_sec - start.tv_sec +\
                (stop.tv_usec - start.tv_usec)/(float)1000000));

    check_CUDA_error ("Error in kernel");

	/* Set up the execution configuration for the optimized kernel that uses shared memory. */
    printf ("\nUsing optimized version\n");
    gettimeofday (&start, NULL);

	threads.x = threads.y = TILE_SIZE;
	grid.x = 1;
    grid.y = (Yd.num_rows + TILE_SIZE - 1)/TILE_SIZE;
	
    multiply_kernel_optimized<<< grid, threads >>>(Ad.elements, Xd.elements, Yd.elements, Ad.num_rows, Ad.num_columns);
	cudaDeviceSynchronize();

    gettimeofday (&stop, NULL);
	printf ("Execution time = %fs\n", (float)(stop.tv_sec - start.tv_sec +\
                (stop.tv_usec - start.tv_usec)/(float)1000000));
	
	check_CUDA_error ("Error in kernel");
	
	/* Copy result from the device. */
	copy_matrix_from_device (Y, Yd); 
	
	/* Free memory on device. */
	cudaFree (&Ad);
	cudaFree (&Xd);
	cudaFree (&Yd);
}

/* Allocate memory for the matrix on device. */
matrix_t 
allocate_matrix_on_device (const matrix_t M) 
{
	matrix_t Mdevice = M;
	int size = M.num_rows * M.num_columns * sizeof (float);
	cudaMalloc ((void **) &Mdevice.elements, size);
	return Mdevice;
}

/* Allocate a matrix of dimensions height * width.
   If init == 0, initialize to all zeroes.  
   If init == 1, perform random initialization with values between [-0.5, +0.5].
*/
matrix_t 
allocate_matrix_on_host (int num_rows, int num_columns, int init) 
{
    matrix_t M;
    M.num_columns = num_columns;
    M.num_rows = num_rows;
    int size = M.num_rows * M.num_columns;
	M.elements = (float *) malloc (size * sizeof (float));
    if (M.elements == NULL) {
        return M;
    }
    
	for (unsigned int i = 0; i < size; i++) {
        if(init == 0)
            M.elements[i] = 0.0; 
		else
			M.elements[i] = rand ()/(float) RAND_MAX - 0.5;
	}
	
    return M;
}	

/* Copy matrix from host memory to device memory. */
void 
copy_matrix_to_device (matrix_t Mdevice, const matrix_t Mhost)
{
	int size = Mhost.num_rows * Mhost.num_columns * sizeof (float);
	Mdevice.num_rows = Mhost.num_rows;
	Mdevice.num_columns = Mhost.num_columns;
	cudaMemcpy (Mdevice.elements, Mhost.elements, size, cudaMemcpyHostToDevice);
    return;
}

/* Copy matrix from device memory to host memory. */
void 
copy_matrix_from_device (matrix_t Mhost, const matrix_t Mdevice) 
{
	int size = Mdevice.num_rows * Mdevice.num_columns * sizeof (float);
	cudaMemcpy (Mhost.elements, Mdevice.elements, size, cudaMemcpyDeviceToHost);
    return;
}

/* Prints the matrix out to screen. */
void 
print_matrix (const matrix_t M) 
{
	for (unsigned int i = 0; i < M.num_rows; i++) {
		for (unsigned int j = 0; j < M.num_columns; j++) {
			printf ("%f ", M.elements[i * M.num_columns + j]);
        }
	
        printf("\n");
	} 

	printf("\n");
    return;
}

void 
check_CUDA_error (const char *msg)
{
	cudaError_t err = cudaGetLastError();
	if ( cudaSuccess != err) {
		printf ("CUDA ERROR: %s (%s).\n", msg, cudaGetErrorString (err));
		exit (EXIT_FAILURE);
	}

    return;
}

/* Compare the reference and device results. */
int 
check_results (const float *val1, const float *val2, int num_elements, float eps)
{
    float max_re = 0.0; 
    float re = 0.0;
    for (int i = 0; i < num_elements; i++) {
        re = fabsf ((val1[i] - val2[i])/val1[i]);
        if (re > max_re)
            max_re = re;
    }

    printf ("Max relative error = %f\n", max_re);
    if (max_re <= eps)
        return 1;
    else
        return 0;
}

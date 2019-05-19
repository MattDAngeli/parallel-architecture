/* Matrix multiplication: C = A * B.
 * Host code.
 * 
 * Author: Naga Kandasamy
 * Date created: 02/14/2017
 * Date modified: May 18, 2019
 */

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <sys/time.h>

/* Include the kernel code. */
#include "matrix_multiply_kernel.cu"

extern "C" void compute_gold_naive (const matrix, const matrix, matrix);
extern "C" void compute_gold_blocked (const matrix, const matrix, matrix);
matrix allocate_matrix_on_device (const matrix);
matrix allocate_matrix (int, int, int);
void copy_matrix_to_device (matrix, const matrix);
void copy_matrix_from_device (matrix, const matrix);
void free_matrix_on_device (matrix *);
void free_matrix (matrix *);
void matrix_multiply_on_device (const matrix, const matrix, matrix);
void check_CUDA_error (const char *);
int check_results (float *, float *, int, float);

int 
main (int argc, char** argv) 
{
    /* Create and populate the matrices. */
    srand (time (NULL));
    matrix M  = allocate_matrix (MATRIX_SIZE, MATRIX_SIZE, 1);	
	matrix N  = allocate_matrix (MATRIX_SIZE, MATRIX_SIZE, 1);

    struct timeval start, stop;	
    printf("\nMultiplying %d x %d matrices on CPU using naive version\n", MATRIX_SIZE, MATRIX_SIZE);
	gettimeofday (&start, NULL);

    matrix P_reference_naive = allocate_matrix (MATRIX_SIZE, MATRIX_SIZE, 0);  
    compute_gold_naive (M, N, P_reference_naive);

    gettimeofday (&stop, NULL);
	printf ("Execution time = %fs\n", (float)(stop.tv_sec - start.tv_sec +\
                (stop.tv_usec - start.tv_usec)/(float)1000000));

    printf("\nMultiplying %d x %d matrices on CPU using blocked version\n", MATRIX_SIZE, MATRIX_SIZE);
	gettimeofday (&start, NULL);

    matrix P_reference_blocked = allocate_matrix (MATRIX_SIZE, MATRIX_SIZE, 0);  
    compute_gold_blocked (M, N, P_reference_blocked);

    gettimeofday (&stop, NULL);
	printf ("Execution time = %fs\n", (float)(stop.tv_sec - start.tv_sec +\
                (stop.tv_usec - start.tv_usec)/(float)1000000));


    printf ("\nMultiplying %d x %d matrices on the GPU\n", MATRIX_SIZE, MATRIX_SIZE);
    
    matrix P_device = allocate_matrix (MATRIX_SIZE, MATRIX_SIZE, 0);
    matrix_multiply_on_device (M, N, P_device); 

    /* Check if the device result matches the reference. */
    printf ("\nChecking reference and device results\n");
    int num_elements = M.height * M.width;
    float eps = 1e-3;
	int status = check_results (P_reference_blocked.elements, P_device.elements, num_elements, eps);
	printf ("TEST %s\n", (1 == status) ? "PASSED" : "FAILED");
	
	/* Free matrices on host. */
	free_matrix (&M);
	free_matrix (&N);
    free_matrix (&P_reference_naive);
    free_matrix (&P_reference_blocked);
	free_matrix (&P_device);

	exit (EXIT_SUCCESS);
}

void 
matrix_multiply_on_device (const matrix M, const matrix N, matrix P)
{
    /* Allocate memory and copy matrices to the device. */
    struct timeval start, stop;	
	gettimeofday (&start, NULL);
    
    matrix Md = allocate_matrix_on_device (M);
	copy_matrix_to_device (Md, M);
	
    matrix Nd = allocate_matrix_on_device (N);
	copy_matrix_to_device (Nd, N);

    matrix Pd = allocate_matrix_on_device (P);

    gettimeofday (&stop, NULL);
	printf ("Data transfer time = %fs\n", (float)(stop.tv_sec - start.tv_sec +\
                (stop.tv_usec - start.tv_usec)/(float)1000000));

    /* Set up the execution grid. */
    dim3 threads(TILE_SIZE, TILE_SIZE);                     
    printf ("Setting up a %d x %d grid of thread blocks\n", (Pd.width + TILE_SIZE - 1)/TILE_SIZE,\\
            (Pd.height + TILE_SIZE - 1)/TILE_SIZE);
	dim3 grid ((Pd.width + TILE_SIZE - 1)/TILE_SIZE, (Pd.height + TILE_SIZE - 1)/TILE_SIZE);

    printf ("\nUsing global memory\n");
	gettimeofday (&start, NULL);
	matrix_multiply_kernel_vanilla<<< grid, threads >>>(Pd.elements, Md.elements, Nd.elements, MATRIX_SIZE);
	cudaDeviceSynchronize ();
    gettimeofday (&stop, NULL);
	printf ("Execution time = %fs\n", (float)(stop.tv_sec - start.tv_sec +\
                (stop.tv_usec - start.tv_usec)/(float)1000000));
    check_CUDA_error ("Error in kernel");

    printf ("\nUsing shared memory\n");
	gettimeofday (&start, NULL);
	matrix_multiply_kernel_shm<<< grid, threads >>>(Pd.elements, Md.elements, Nd.elements, MATRIX_SIZE);
	cudaDeviceSynchronize ();
    gettimeofday (&stop, NULL);
	printf ("Execution time = %fs\n", (float)(stop.tv_sec - start.tv_sec +\
                (stop.tv_usec - start.tv_usec)/(float)1000000));
    check_CUDA_error ("Error in kernel");

    /* Bind Md and Nd to 1D textures. Note: the maximum width for 1D texture reference bound to linear memory
       varies with the GPU generation and compute capability. 
       Currently it is set to 2^{27} elements. 
     */
    printf ("\nUsing 1D texture memory\n");
	cudaBindTexture (NULL, M_on_tex, Md.elements, M.width * M.height * sizeof (float));
	cudaBindTexture (NULL, N_on_tex, Nd.elements, N.width * N.height * sizeof (float));

	gettimeofday (&start, NULL);
    matrix_multiply_kernel_1Dtex<<< grid, threads >>>(Pd.elements, Md.elements, Nd.elements, MATRIX_SIZE);
	cudaDeviceSynchronize ();
    gettimeofday (&stop, NULL);
	printf ("Execution time = %fs\n", (float)(stop.tv_sec - start.tv_sec +\
                (stop.tv_usec - start.tv_usec)/(float)1000000));
    check_CUDA_error ("Error in kernel");

    /* Unbind 1D texture references. */
	cudaUnbindTexture (M_on_tex);
	cudaUnbindTexture (N_on_tex);

    /* Bind Md and Nd to 2D textures. Note: as with 1D textures, there is a maximum 
       width and height for 2D texture reference bound to a CUDA array or to a linear memory. 
     */
    printf ("\nUsing 2D texture memory\n");
	cudaChannelFormatDesc desc = cudaCreateChannelDesc<float> ();
	cudaBindTexture2D (NULL, M_on_tex_2D, Md.elements, desc, M.width, M.height, M.width * sizeof (float));
	cudaBindTexture2D (NULL, N_on_tex_2D, Nd.elements, desc, N.width, N.height, N.width * sizeof (float));

    gettimeofday (&start, NULL);
    matrix_multiply_kernel_2Dtex<<< grid, threads >>>(Pd.elements, Md.elements, Nd.elements, MATRIX_SIZE);
	cudaDeviceSynchronize ();
    gettimeofday (&stop, NULL);
	printf ("Execution time = %fs\n", (float)(stop.tv_sec - start.tv_sec +\
                (stop.tv_usec - start.tv_usec)/(float)1000000));
    check_CUDA_error ("Error in kernel");

    /* Unbind 2D texture references. */
    cudaUnbindTexture (M_on_tex_2D);
	cudaUnbindTexture (N_on_tex_2D);

    /* Copy results from device to host. */
    copy_matrix_from_device (P, Pd);                        

    /* Free allocated memory on the device. */
    free_matrix_on_device (&Md);                                  
	free_matrix_on_device (&Nd);
	free_matrix_on_device (&Pd);

    return;
}

/* Allocate memory on device for matrix. */
matrix 
allocate_matrix_on_device (const matrix M)                        
{
	matrix Mdevice = M;
	int size = M.width * M.height * sizeof (float);
	
    cudaMalloc ((void**) &Mdevice.elements, size);
    if (Mdevice.elements == NULL) {
        printf ("CudaMalloc error\n");
        exit (EXIT_FAILURE);
    }

	return Mdevice;
}

/* Allocate a matrix of dimensions height * width.
   If init == 0, initialize to all zeroes.  
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
    if (M.elements == NULL) {
        perror ("Malloc");
        exit (EXIT_FAILURE);
    }

	for (unsigned int i = 0; i < M.height * M.width; i++)
		M.elements[i] = (init == 0) ? (0.0f) : floor ((3 * (rand ()/(float) RAND_MAX)));
	
	return M;
}	

/* Copy matrix from host memory to device memory. */
void 
copy_matrix_to_device (matrix Mdevice, const matrix Mhost)      
{
	int size = Mhost.width * Mhost.height * sizeof (float);
	cudaMemcpy (Mdevice.elements, Mhost.elements, size, cudaMemcpyHostToDevice);
    return;
}

/* Copy matrix from device memory to host memory. */
void 
copy_matrix_from_device (matrix Mhost, const matrix Mdevice)   
{
	int size = Mdevice.width * Mdevice.height * sizeof (float);
	cudaMemcpy (Mhost.elements, Mdevice.elements, size, cudaMemcpyDeviceToHost);
    return;
}

/* Free the matrix allocated on device. */
void 
free_matrix_on_device (matrix* M)                              
{
	cudaFree (M->elements);
	M->elements = NULL;
    return;
}

/* Free the matrix structure on the host. */
void 
free_matrix (matrix *M)
{
	free (M->elements);
	M->elements = NULL;
    return;
}

/* Check for errors during kernel execution. */
void 
check_CUDA_error (const char *msg)
{
	cudaError_t err = cudaGetLastError ();
	if (cudaSuccess != err) {
		printf ("CUDA ERROR: %s (%s).\n", msg, cudaGetErrorString(err));
		exit (EXIT_FAILURE);
	}	

    return;    
}

/* Check the correctness of the reference and device results. */
int 
check_results (float *reference, float *gpu_result, int num_elements, float threshold)
{
    int check_mark = 1;
    float max_diff = 0.0;
    int i;

    for (i = 0; i < num_elements; i++)
        if (fabsf ((reference[i] - gpu_result[i])/reference[i]) > threshold)
            check_mark = 0;
        
    for (i = 0; i < num_elements; i++)
        if (fabsf ((reference[i] - gpu_result[i])/reference[i]) > max_diff)
            max_diff = fabsf ((reference[i] - gpu_result[i])/reference[i]);
        
    printf ("Max diff = %f\n", max_diff); 

    return check_mark;
}

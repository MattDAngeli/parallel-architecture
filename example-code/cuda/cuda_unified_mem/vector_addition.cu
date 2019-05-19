/* This code illustrates the use of the GPU to perform vector addition on arbirarily large vectors. 
 * We benckmark the execution time using both cudaMemcpy as well as cudaMallocManaged. 
    Author: Naga Kandasamy
    Date modifeid: 02/18/2018
*/  
  
#include <stdlib.h>
#include <stdio.h>
#include <sys/time.h>
#include <string.h>
#include <math.h>
#include <float.h>

/* Include the kernel code during the preprocessing step. */
#include "vector_addition_kernel.cu"
  
#define THREAD_BLOCK_SIZE 1024
#define NUM_THREAD_BLOCKS 40
  
void run_test_explicit_mem_mgmt (int);
void run_test_unified_mem_mgmt (int);
void run_test_unified_mem_mgmt_optimized (int);
void check_for_error (char const *);
extern "C" void compute_gold (float *, float *, float *, int);

int 
main (int argc, char **argv) 
{
    if (argc != 2) {
        printf ("Usage: %s num-elements\n", argv[0]);
        exit (EXIT_FAILURE);
    }

    int num_elements = atoi (argv[1]);

    printf ("\nRunning test. Device memory is explicity managed using cudaMalloc and cudaMemcpy\n");
    run_test_explicit_mem_mgmt (num_elements);

    printf ("\nRunning test using unified memory\n");
    run_test_unified_mem_mgmt (num_elements);

    printf ("\nRunning test using unified memory with optimizations\n");
    run_test_unified_mem_mgmt_optimized (num_elements);

    exit (EXIT_SUCCESS);
}


/* Perform vector addition on the host and the device using explicit management of 
 * device memory. */
void 
run_test_explicit_mem_mgmt (int num_elements)                
{
    /* Allocate memory on the host for the input vectors A and B, and the output vector C. */
     int vector_length = sizeof (float) * num_elements;
     float *A_on_host = (float *) malloc (vector_length);
     float *B_on_host = (float *) malloc (vector_length);
     float *gpu_result = (float *) malloc (vector_length);	/* The result vector computed on the GPU. */
			 
     /* Randomly generate input data on the host. Initialize the input data to be integer values between 0 and 5. */ 
     for (int i = 0; i < num_elements; i++) {
         A_on_host[i] = floorf (5 * (rand () / (float) RAND_MAX));
         B_on_host[i] = floorf (5 * (rand () / (float) RAND_MAX));
     } 
				
     /* Allocate space on the GPU for vectors A and B, and copy the contents of the vectors to the GPU. */
     float *A_on_device, *B_on_device, *C_on_device;
     
     struct timeval start, stop;	
     gettimeofday (&start, NULL);

     cudaMalloc ((void **) &A_on_device, num_elements * sizeof (float));
     cudaMemcpy (A_on_device, A_on_host, num_elements * sizeof (float), cudaMemcpyHostToDevice);
	
     cudaMalloc ((void **) &B_on_device, num_elements * sizeof (float));
     cudaMemcpy (B_on_device, B_on_host, num_elements * sizeof (float), cudaMemcpyHostToDevice);
    
     /* Allocate space for the result vector on the GPU. */
     cudaMalloc ((void **) &C_on_device, num_elements * sizeof (float));
  
     /* Set up the execution grid on the GPU. */
     int num_thread_blocks = NUM_THREAD_BLOCKS;
     dim3 thread_block (THREAD_BLOCK_SIZE, 1, 1);	/* Set the number of threads in the thread block. */
     dim3 grid (num_thread_blocks, 1);
    
     vector_addition_kernel <<< grid, thread_block >>> (A_on_device, B_on_device, C_on_device, num_elements);	 
     cudaDeviceSynchronize (); 
     check_for_error ("KERNEL FAILURE");
     cudaMemcpy (gpu_result, C_on_device, num_elements * sizeof (float), cudaMemcpyDeviceToHost);
     
     gettimeofday (&stop, NULL);
     printf ("Total time elapsed = %fs\n", (float)(stop.tv_sec - start.tv_sec +\
                (stop.tv_usec - start.tv_usec)/(float)1000000));

     /* Free the data structures. */
     cudaFree (A_on_device); 
     cudaFree (B_on_device); 
     cudaFree (C_on_device);
     free (A_on_host); 
     free (B_on_host);
     free (gpu_result);
     return;
}

/* Test allows the CUDA run-time to manage memory on its behalf using 
 * using dynamic page migration between the host and device. 
 */
void 
run_test_unified_mem_mgmt (int num_elements)                
{
    struct timeval start, stop;	
    gettimeofday (&start, NULL);
    
    /* Allocate unified for the vectors. */
     float *A_unified, *B_unified, *gpu_result_unified;
     cudaMallocManaged (&A_unified, sizeof (float) * num_elements);
     cudaMallocManaged (&B_unified, sizeof (float) * num_elements);
     cudaMallocManaged (&gpu_result_unified, sizeof (float) * num_elements);
 			 
     /* Initialize the input vectors to be integer values between 0 and 5. */ 
     for (int i = 0; i < num_elements; i++) {
         A_unified[i] = floorf (5 * (rand () / (float) RAND_MAX));
         B_unified[i] = floorf (5 * (rand () / (float) RAND_MAX));
     } 
				
     /* Set up the execution grid on the GPU. */
     int num_thread_blocks = NUM_THREAD_BLOCKS;
     dim3 thread_block (THREAD_BLOCK_SIZE, 1, 1);
     dim3 grid (num_thread_blocks, 1);
    
     vector_addition_kernel <<< grid, thread_block >>> (A_unified, B_unified, gpu_result_unified, num_elements);	 
     cudaDeviceSynchronize (); 
     check_for_error ("KERNEL FAILURE");
     
     gettimeofday (&stop, NULL);
     printf ("Total time elapsed = %fs\n", (float)(stop.tv_sec - start.tv_sec +\
                (stop.tv_usec - start.tv_usec)/(float)1000000));

     /* Use gpu_result as appropriate on the host. No 
      * need to copy it back. 
      */

     /* Free the data structures. */
     cudaFree (A_unified); 
     cudaFree (B_unified); 
     cudaFree (gpu_result_unified);
     return;
}
 
/* Test allows the CUDA run-time to manage memory on its behalf using 
 * using dynamic page migration between the host and device. 
 * Implementation is optimized to reduce the number of page 
 * migrations between the host and the device by moving the 
 * initialization code to the device.
 */
void 
run_test_unified_mem_mgmt_optimized (int num_elements)                
{
    struct timeval start, stop;	
    gettimeofday (&start, NULL);
    
    /* Allocate unified for the vectors. */
     float *A_unified, *B_unified, *gpu_result_unified;
     cudaMallocManaged (&A_unified, sizeof (float) * num_elements);
     cudaMallocManaged (&B_unified, sizeof (float) * num_elements);
     cudaMallocManaged (&gpu_result_unified, sizeof (float) * num_elements);
 			 
     /* Set up the execution grid on the GPU. */
     int num_thread_blocks = NUM_THREAD_BLOCKS;
     dim3 thread_block (THREAD_BLOCK_SIZE, 1, 1);
     dim3 grid (num_thread_blocks, 1);

     /* Initialize A and B on the device rather than on the host. */
     curandState *state; /* PRNG state for threads. */
     cudaMalloc ((void **) &state, NUM_THREAD_BLOCKS * THREAD_BLOCK_SIZE * sizeof (curandState));
     setup_kernel <<<grid, thread_block>>> (state);
     cudaDeviceSynchronize ();
     init_kernel <<<grid, thread_block >>> (A_unified, B_unified, num_elements, state);
     cudaDeviceSynchronize ();

     /* Launch computation kernel. */
     vector_addition_kernel <<< grid, thread_block >>> (A_unified, B_unified, gpu_result_unified, num_elements);	 
     cudaDeviceSynchronize (); 
     check_for_error ("KERNEL FAILURE");
     
     gettimeofday (&stop, NULL);
     printf ("Total time elapsed = %fs\n", (float)(stop.tv_sec - start.tv_sec +\
                (stop.tv_usec - start.tv_usec)/(float)1000000));

     /* Use gpu_result as appropriate on the host. No 
      * need to copy it back. 
      */

     /* Free the data structures. */
     cudaFree (A_unified); 
     cudaFree (B_unified); 
     cudaFree (gpu_result_unified);
     return;
}


/* Checks for errors when executing the kernel. */
void 
check_for_error (char const *msg)                
{
    cudaError_t err = cudaGetLastError ();
    if (cudaSuccess != err) {
        printf ("CUDA ERROR: %s (%s). \n", msg, cudaGetErrorString (err));
        exit (EXIT_FAILURE);
    }
}



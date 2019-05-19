/* Example of using CUDA streams to operate the GPU in streaming mode. 
 * Author: Naga Kandasamy
 * Date created: May 18, 2019
*/
#include <stdlib.h>
#include <stdio.h>
#include <sys/time.h>
#include <string.h>
#include <math.h>
#include <float.h>

#define CHUNK_SIZE 1024*1024
#define FULL_DATA_SIZE CHUNK_SIZE*200
#define THREAD_BLOCK_SIZE 1024

/* A simple kernel that performs some computation. In this case, the kernel computes the average of three 
	values in A and three values in B and stores this average in C */
__global__ 
void 
avg_kernel (int *A, int *B, int *C, int num_elements)
{
    int idx = blockIdx.x*blockDim.x + threadIdx.x;
    int idx1, idx2;
    float avg1 = 0.0f;
    float avg2 = 0.0f; 

    if (idx < num_elements) {
        for (int i = 0; i < 10000; i++) {
            idx1 = (idx + 1) % THREAD_BLOCK_SIZE;
            idx2 = (idx + 2) % THREAD_BLOCK_SIZE;
            avg1 += (A[idx] + A[idx1] + A[idx2])/10000.0f;
            avg2 += (B[idx] + B[idx1] + B[idx2])/10000.0f;
        }
        
        C[idx] = (avg1 + avg2)/2;
    }

    return;
}

/* Time the execution using multiple, in this case two, CUDA streams. This implementation achieves 
   the most overlap between kernel execution and data transfer. */
float 
run_test_with_multiple_streams (const int *A, const int *B, int *C)
{
    int *A_on_device_0, *B_on_device_0, *C_on_device_0; /* Device pointers for stream 0. */
    int *A_on_device_1, *B_on_device_1, *C_on_device_1; /* Device pointers for stream 1. */

    /* Create and initialize the streams. */
    cudaStream_t stream_0, stream_1;
    cudaStreamCreate (&stream_0);
    cudaStreamCreate (&stream_1); 
		  
    /* Allocate memory on the GPU for each stream. Note that we only need to 
     * allocate memory of size CHUNK_SIZE for each stream. 
     */
    cudaMalloc ((void **) &A_on_device_0, CHUNK_SIZE * sizeof (int));
    cudaMalloc ((void **) &B_on_device_0, CHUNK_SIZE * sizeof (int));
    cudaMalloc ((void **) &C_on_device_0, CHUNK_SIZE * sizeof (int));

    cudaMalloc ((void **) &A_on_device_1, CHUNK_SIZE * sizeof (int));
    cudaMalloc ((void **) &B_on_device_1, CHUNK_SIZE * sizeof (int));
    cudaMalloc ((void **) &C_on_device_1, CHUNK_SIZE * sizeof (int));

    /* Set up the execution grid for the kernel. */
    dim3 threads (THREAD_BLOCK_SIZE, 1, 1);
    dim3 grid (CHUNK_SIZE/THREAD_BLOCK_SIZE, 1);

    float elapsed_time;
    struct timeval start, stop;	
	gettimeofday (&start, NULL);

    /* Process the full data payload in chunks. */
    for (int i = 0; i < FULL_DATA_SIZE; i += 2 * CHUNK_SIZE) {
        /* Copy chunks of A and B from the pinned memory on the host to the device streams 0 and 1. */
        cudaMemcpyAsync (A_on_device_0, &A[i], CHUNK_SIZE * sizeof (int), cudaMemcpyHostToDevice, stream_0);
        cudaMemcpyAsync (A_on_device_1, &A[i + CHUNK_SIZE], CHUNK_SIZE * sizeof (int), cudaMemcpyHostToDevice, stream_1);
        cudaMemcpyAsync (B_on_device_0, &B[i], CHUNK_SIZE * sizeof (int), cudaMemcpyHostToDevice, stream_0);
        cudaMemcpyAsync (B_on_device_1, &B[i + CHUNK_SIZE], CHUNK_SIZE * sizeof (int), cudaMemcpyHostToDevice, stream_1);

        avg_kernel<<<grid, threads, 0, stream_0>>>(A_on_device_0, B_on_device_0, C_on_device_0, CHUNK_SIZE);
        avg_kernel<<<grid, threads, 0, stream_1>>>(A_on_device_1, B_on_device_1, C_on_device_1, CHUNK_SIZE);
					 
        cudaMemcpyAsync (&C[i], C_on_device_0, CHUNK_SIZE * sizeof (int), cudaMemcpyHostToDevice, stream_0);
        cudaMemcpyAsync (&C[i + CHUNK_SIZE], C_on_device_1, CHUNK_SIZE * sizeof (int), cudaMemcpyHostToDevice, stream_1);
    }
		  
    /* Synchronize the CUDA streams with the host. Host waits for the GPU to finish copying the 
       final chunk to the C array. */
    cudaStreamSynchronize (stream_0);
    cudaStreamSynchronize (stream_1);

    gettimeofday (&stop, NULL);
	elapsed_time = stop.tv_sec - start.tv_sec + (stop.tv_usec - start.tv_usec)/(float) 1000000;

    cudaFree (A_on_device_0); 
    cudaFree (A_on_device_1); 
    cudaFree (B_on_device_0); 
    cudaFree (B_on_device_1); 
    cudaFree (C_on_device_0); 
    cudaFree (C_on_device_1); 
		  
    /* Finally destroy the streams used to queue GPU operations. */
    cudaStreamDestroy (stream_0);
    cudaStreamDestroy (stream_1);

    return elapsed_time;
}

		  
/* Time the execution using a cuda stream. Here, the data is transferred in smaller chunks in 
   streaming fashion to the GPU. Streaming requires that pinned memory be allocated 
   on the host side. 
*/
float 
run_test_with_single_stream (const int *A, const int *B, int *C)
{
    int *A_on_device, *B_on_device, *C_on_device;
	
    float elapsed_time;
    struct timeval start, stop;	
	gettimeofday(&start, NULL);

    /* Allocate memory on the GPU. Note that we only need to allocate memory of size CHUNK_SIZE. */
    cudaMalloc ((void **) &A_on_device, CHUNK_SIZE * sizeof (int));
    cudaMalloc ((void **) &B_on_device, CHUNK_SIZE * sizeof (int));
    cudaMalloc ((void **) &C_on_device, CHUNK_SIZE * sizeof (int));

    /* Create and initialize the CUDA stream. Set up the execution grid. */
    cudaStream_t stream;
    cudaStreamCreate (&stream);
	
    dim3 threads (THREAD_BLOCK_SIZE, 1, 1);
    dim3 grid (CHUNK_SIZE/THREAD_BLOCK_SIZE, 1);

    /* Process the full data payload in chunks. */
    for (int i = 0; i < FULL_DATA_SIZE; i += CHUNK_SIZE) {
        /* Copy chunks of A and B from the pinned memory on the host to the device. 
           This copy is done in asynchronous fashion. 
         */
        cudaMemcpyAsync (A_on_device, &A[i], CHUNK_SIZE * sizeof (int), cudaMemcpyHostToDevice, stream);
        cudaMemcpyAsync (B_on_device, &B[i], CHUNK_SIZE * sizeof (int), cudaMemcpyHostToDevice, stream);
		
        avg_kernel<<<grid, threads, 0, stream>>>(A_on_device, B_on_device, C_on_device, CHUNK_SIZE);
		
        cudaMemcpyAsync (&C[i], C_on_device, CHUNK_SIZE *sizeof (int), cudaMemcpyHostToDevice, stream);
    }
	
    /* Synchronize the CUDA stream with the host. 
       Host waits for the GPU to finish copying the final chunk to the C array. 
    */
    cudaStreamSynchronize (stream);

    gettimeofday (&stop, NULL);
	elapsed_time = stop.tv_sec - start.tv_sec + (stop.tv_usec - start.tv_usec)/(float) 1000000;
    
   /* Clean up. */
    cudaFree (A_on_device);
    cudaFree (B_on_device);
    cudaFree (C_on_device);
	
    /* Finally destroy the stream used to queue GPU operations. */
    cudaStreamDestroy (stream);
	
    return elapsed_time;
}

/* Time the execution using the cudaMalloc calls. 
   Here the entire data set is transferred to the GPU prior to invoking the kernel. 
*/
float 
run_test_with_cuda_malloc (const int *A, const int *B, int *C)
{
     /* Allocate memory on the GPU for arrays A, B, and C, 
     * and transfer over A and B. 
      */
    int *A_on_device, *B_on_device, *C_on_device;
    float elapsed_time;
    
    struct timeval start, stop;	
	gettimeofday (&start, NULL);
   
    cudaMalloc ((void **) &A_on_device, FULL_DATA_SIZE * sizeof (int));
    cudaMemcpy (A_on_device, A, FULL_DATA_SIZE * sizeof (int), cudaMemcpyHostToDevice);

    cudaMalloc ((void **) &B_on_device, FULL_DATA_SIZE * sizeof (int));
    cudaMemcpy (B_on_device, B, FULL_DATA_SIZE * sizeof (int), cudaMemcpyHostToDevice);

    cudaMalloc ((void **) &C_on_device, FULL_DATA_SIZE * sizeof (int));

    /* Set up execution grid. */
    dim3 grid (FULL_DATA_SIZE/THREAD_BLOCK_SIZE, 1);
    dim3 threads (THREAD_BLOCK_SIZE, 1, 1);

    avg_kernel<<<grid, threads>>>(A_on_device, B_on_device, C_on_device, FULL_DATA_SIZE);

    cudaMemcpy (C, C_on_device, FULL_DATA_SIZE * sizeof (int), cudaMemcpyDeviceToHost);

    gettimeofday (&stop, NULL);
	elapsed_time = stop.tv_sec - start.tv_sec + (stop.tv_usec - start.tv_usec)/(float) 1000000;

    /* Clean up. */
    cudaFree (A_on_device);
    cudaFree (B_on_device);
    cudaFree (C_on_device);
	
    return elapsed_time;
}

int 
main (int argc, char **argv)
{
    /* Check the device properties if our device supports mapping host memory */
    cudaDeviceProp properties;
    int my_device;
    cudaGetDevice (&my_device);
    cudaGetDeviceProperties (&properties, my_device);
    if (properties.canMapHostMemory != 1) {
        printf ("The device cannot map host memory\n");
        exit (EXIT_FAILURE);
    }

    /* Check to see if the device supports overlaps, that is, if it can simultaneously 
       execute a CUDA kernel while performing a copy between the device and host memory. */
    if (properties.deviceOverlap != 1) {
        printf ("The device does not support overlaps\n");
        exit (EXIT_FAILURE);
    }
    /* Place the CUDA runtime in a state which supports mapping memory on the host. */ 
    cudaSetDeviceFlags (cudaDeviceMapHost);

    /* Allocate memory on the host for the A, B, and C arrays. */
    int *A, *B, *C;
    A = (int *) malloc (FULL_DATA_SIZE * sizeof (int));
    B = (int *) malloc (FULL_DATA_SIZE * sizeof (int));
    C = (int *) malloc (FULL_DATA_SIZE * sizeof (int));
    /* Fill arrays A and B with randomly generated integers. */
    printf ("\nCreating input arrays of size %d MB each in non-pinned memory\n", FULL_DATA_SIZE/(1024 * 1024));
    for (int i = 0; i < FULL_DATA_SIZE; i++) {
        A[i] = rand ();
        B[i] = rand ();
    }

    float elapsed_time;
		  
    elapsed_time = run_test_with_cuda_malloc (A, B, C);
    printf ("Elapsed time using cudaMalloc: %3.1fs\n", elapsed_time);

    free (A);
    free (B);
    free (C);

    /* Allocate pinned or page-locked memory on the host for the entire data set. 
       This is the memory used to stream chunks of the data set. */
    cudaHostAlloc ((void **) &A, FULL_DATA_SIZE * sizeof (int), cudaHostAllocDefault);
    cudaHostAlloc ((void **) &B, FULL_DATA_SIZE * sizeof (int), cudaHostAllocDefault);
    cudaHostAlloc ((void **) &C, FULL_DATA_SIZE * sizeof (int), cudaHostAllocDefault);
    /* Fill arrays A and B with randomly generated integers. */
    printf ("\nCreating input arrays of size %d MB each in pinned memory\n", FULL_DATA_SIZE/ (1024 * 1024));
    for (int i = 0; i < FULL_DATA_SIZE; i++) {
        A[i] = rand();
        B[i] = rand();
    }

    elapsed_time = run_test_with_single_stream (A, B, C);
    printf ("Elapsed time using a single stream: %3.1fs\n", elapsed_time);

    elapsed_time = run_test_with_multiple_streams (A, B, C);
    printf ("Elapsed time using multiple streams: %3.1fs\n", elapsed_time);

    /* Free pinned memory on the host */
    cudaFreeHost (A);
    cudaFreeHost (B);
    cudaFreeHost (C);
 
    exit (EXIT_SUCCESS);
}


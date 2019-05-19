/* Bandwidth test using non-pinned and pinned memory for host/device 
   data transfer.
 * Author: Naga Kandasamy
 * Date created: May 18, 2019
*/
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <float.h>
#include <sys/time.h>

#define NUM_RUNS 100
#define SIZE 10*1024*1024


float 
cuda_malloc_test (int size, int up)
{
    int *a, *dev_a;
    struct timeval start, stop;	
    float elapsed_time;

    a = (int *) malloc (SIZE * sizeof (int));
    cudaMalloc ((void **)&dev_a, SIZE * sizeof (int));

    gettimeofday (&start, NULL);
	
    for (int i = 0; i < NUM_RUNS; i++) {
        if (up)
            cudaMemcpy (dev_a, a, SIZE * sizeof (int), cudaMemcpyHostToDevice);
        else
            cudaMemcpy (a, dev_a, SIZE * sizeof (int), cudaMemcpyDeviceToHost);
    }
	
    gettimeofday (&stop, NULL);
    elapsed_time = stop.tv_sec - start.tv_sec + (stop.tv_usec - start.tv_usec)/(float)1000000;

    free (a);
    cudaFree (dev_a);

    return elapsed_time;
}

float 
cuda_host_alloc_test (int size, int up)
{  
    int *a, *dev_a;
	struct timeval start, stop;	
    float elapsed_time;
    
    cudaHostAlloc ((void **) &a, SIZE * sizeof (int), cudaHostAllocDefault);
    cudaMalloc ((void **) &dev_a, SIZE * sizeof (int));

    gettimeofday (&start, NULL);
		  
    for(int i = 0; i < NUM_RUNS; i++) {
        if (up)
            cudaMemcpy (dev_a, a, SIZE * sizeof (int), cudaMemcpyHostToDevice);
        else
            cudaMemcpy (a, dev_a, SIZE * sizeof (int), cudaMemcpyDeviceToHost);
    }

    gettimeofday (&stop, NULL);
    elapsed_time = stop.tv_sec - start.tv_sec + (stop.tv_usec - start.tv_usec)/(float)1000000;

    cudaFreeHost (a);
    cudaFree (dev_a);
	
    return elapsed_time;
}

int 
main (int argc, char **argv)
{
    /* Check the device properties if device supports mapping host memory. */
    cudaDeviceProp properties;
    int my_device;
    cudaGetDevice (&my_device);
    cudaGetDeviceProperties (&properties, my_device);
    if (properties.canMapHostMemory != 1) {
        printf ("The device cannot map host memory. \n");
        exit (EXIT_FAILURE);
    }
		  
    /* Place the CUDA runtime in a state which supports mapping memory on the host. */ 
    cudaSetDeviceFlags(cudaDeviceMapHost);

    float elapsed_time;
    float MB = (float) NUM_RUNS * SIZE * sizeof (int)/(1024 * 1024); /* Total size of data transferred in MB. */

    /* Benchmark the transfer time when using cudaMalloc up to device, that is from host to the device. */
    elapsed_time = cuda_malloc_test (SIZE, 1);
    printf ("Elapsed time using cudaMalloc: %3.1fs\n", elapsed_time);
    printf ("MB/s during copy up: %3.1f\n", MB/elapsed_time);

    /* Benchmark the transfer time when using cudaMalloc in the opposite direction, that is from device to the host. */
    elapsed_time = cuda_malloc_test (SIZE, 0);
    printf ("Elapsed time using cudaMalloc: %3.1fs\n", elapsed_time);
    printf ("MB/s during copy down: %3.1f\n", MB/elapsed_time);
	
    /* Benchmark the transfer time when using cudaHostAlloc up to device, that is from host to the device. */
    elapsed_time = cuda_host_alloc_test (SIZE, 1);
    printf ("Elapsed time using cudaHostAlloc: %3.1fs\n", elapsed_time);
    printf ("MB/s during copy up: %3.1f\n", MB/elapsed_time);
	
    /* Benchmark the transfer time when using cudaHostAlloc in the opposite direction, that is from device to the host. */
    elapsed_time = cuda_host_alloc_test (SIZE, 0);
    printf ("Elapsed time using cudaHostAlloc: %3.1fs\n", elapsed_time);
    printf ("MB/s during copy down: %3.1f\n", MB/elapsed_time);
	
    exit (EXIT_SUCCESS);
}


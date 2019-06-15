/* Host-side code to perform counting sort 
 * Author: Naga Kandasamy
 * Date modified: May 19, 2019
 * 
 * Compile as follows: make clean && make
 */

#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <sys/time.h>
#include <string.h>
#include <math.h>
#include <limits.h>

#include "counting_sort_kernel.cu"
#include "counting_sort.h"

/* Do not change the range value. */
#define MIN_VALUE 0
#define MAX_VALUE 255
#define DEBUG 

extern "C" int counting_sort_gold (int *, int *, int, int);
int rand_int (int, int);
void print_array (int *, int);
void print_min_and_max_in_array (int *, int);
void compute_on_device (int *, int *, int, int);
int check_if_sorted (int *, int);
int compare_results (int *, int *, int);
void print_exec_time ( struct timeval, struct timeval );
void check_CUDA_error ( const char * );

int 
main (int argc, char **argv)
{
    if (argc < 2) {
        printf ("Usage: %s num-elements\n", argv[0]);
        exit (EXIT_FAILURE);
    }

    int num_elements = atoi (argv[1]);
    int range = MAX_VALUE - MIN_VALUE;
    int *input_array, *sorted_array_reference, *sorted_array_d;

    /* Populate the input array with random integers between [0, range]. */
    printf ("Generating input array with %d elements in the range 0 to %d\n", num_elements, range);
    input_array = (int *) malloc (num_elements * sizeof (int));
    if (input_array == NULL) {
        printf ("Cannot malloc memory for the input array. \n");
        exit (EXIT_FAILURE);
    }
    srand (time (NULL));
    for (int i = 0; i < num_elements; i++)
        input_array[i] = rand_int (MIN_VALUE, MAX_VALUE);

#ifdef DEBUG
    print_array (input_array, num_elements);
    print_min_and_max_in_array (input_array, num_elements);
#endif

    /* Sort the elements in the input array using the reference implementation. 
     * The result is placed in sorted_array_reference. */
    printf ("\nSorting array on CPU\n");
    int status;
    sorted_array_reference = (int *) malloc (num_elements * sizeof (int));
    if (sorted_array_reference == NULL) {
        perror ("Malloc"); 
        exit (EXIT_FAILURE);
    }
    memset (sorted_array_reference, 0, num_elements);

    struct timeval start, stop;
    gettimeofday( &start, NULL );
    status = counting_sort_gold (input_array, sorted_array_reference, num_elements, range);
    if (status == 0) {
        exit (EXIT_FAILURE);
    }
    gettimeofday( &stop, NULL );
    print_exec_time( start, stop );

    status = check_if_sorted (sorted_array_reference, num_elements);
    if (status == 0) {
        printf ("Error sorting the input array using the reference code\n");
        exit (EXIT_FAILURE);
    }

    printf ("Counting sort was successful on the CPU\n");

#ifdef DEBUG
    print_array (sorted_array_reference, num_elements);
#endif

    /* FIXME: Write function to sort the elements in the array in parallel fashion. 
     * The result should be placed in sorted_array_mt. */
    printf ("\nSorting array on GPU\n");
    sorted_array_d = (int *) malloc (num_elements * sizeof (int));
    if (sorted_array_d == NULL) {
        perror ("Malloc");
        exit (EXIT_FAILURE);
    }
    memset (sorted_array_d, 0, num_elements);
    compute_on_device (input_array, sorted_array_d, num_elements, range);

#ifdef DEBUG
    print_array( sorted_array_d, num_elements );
#endif

    /* Check the two results for correctness. */
    printf ("\nComparing CPU and GPU results\n");
    status = compare_results (sorted_array_reference, sorted_array_d, num_elements);
    if (status == 1)
        printf ("Test passed\n");
    else
        printf ("Test failed\n");

    exit(EXIT_SUCCESS);
}


/* FIXME: Write the GPU implementation of counting sort. */
void 
compute_on_device (int *input_array, int *sorted_array, int num_elements, int range )
{
   int *input_array_dev = NULL;
   int *hist_dev = NULL;
   int *sorted_array_dev = NULL;
   int *scan_dev = NULL;
   int *scan_pong_dev = NULL;

#ifdef DEBUG
   
   int *hist = (int *) malloc( range * sizeof( int ) );
   memset( hist, 0, range * sizeof( int ) );
   int *scan = (int *) malloc( range * sizeof( int ) );
   memset( scan, 0, range * sizeof( int ) );
   
#endif

   cudaMalloc( (void **) &input_array_dev, num_elements * sizeof( int ) );
   cudaMemcpy( input_array_dev, input_array, num_elements * sizeof( int ), cudaMemcpyHostToDevice );

   cudaMalloc( (void **) &hist_dev, range * sizeof( int ) );
   cudaMemset( hist_dev, 0, range * sizeof( int ) );

   cudaMalloc( (void **) &scan_dev, range * sizeof( int ) );
   cudaMemset( scan_dev, 0, sizeof( int ) );

   cudaMalloc( (void **) &scan_pong_dev, range * sizeof( int ) );

   cudaMalloc( (void **) &sorted_array_dev, num_elements * sizeof( int ) );

   dim3 thread_block( THREAD_BLOCK_SIZE );
   dim3 grid( NUM_BLOCKS );

   struct timeval start, stop;
   gettimeofday( &start, NULL );

   // Launch kernel
   counting_sort_kernel <<< grid, thread_block >>> (
         sorted_array_dev, scan_dev, scan_pong_dev, hist_dev, input_array_dev, num_elements);

   cudaDeviceSynchronize( );

   gettimeofday( &stop, NULL);

   // Check for CUDA errors
   check_CUDA_error( "Kernel execution" );

   print_exec_time( start, stop );

#ifdef DEBUG
   
   cudaMemcpy( scan, scan_dev, range * sizeof( int ), cudaMemcpyDeviceToHost );
   cudaMemcpy( hist, hist_dev, range * sizeof( int ), cudaMemcpyDeviceToHost );

   printf("Histogram:\n");
   print_array( hist, range );

   printf("Scan:\n");
   print_array( scan, range );
   
#endif

   cudaMemcpy( sorted_array, sorted_array_dev, num_elements * sizeof( int ), cudaMemcpyDeviceToHost );

#ifdef DEBUG
   free( hist );
   free( scan );
#endif

   cudaFree( input_array_dev );
   cudaFree( hist_dev );
   cudaFree( sorted_array_dev );
   cudaFree( scan_dev );
   cudaFree( scan_pong_dev );
   return;
}

/* Check if the array is sorted. */
int
check_if_sorted (int *array, int num_elements)
{
    int status = 1;
    for (int i = 1; i < num_elements; i++) {
        if (array[i - 1] > array[i]) {
            status = 0;
            break;
        }
    }

    return status;
}

/* Check if the arrays elements are identical. */ 
int 
compare_results (int *array_1, int *array_2, int num_elements)
{
    int status = 1;
    for (int i = 0; i < num_elements; i++) {
        if (array_1[i] != array_2[i]) {
            status = 0;
            break;
        }
    }

    return status;
}


/* Returns a random integer between [min, max]. */ 
int
rand_int (int min, int max)
{
    float r = rand ()/(float) RAND_MAX;
    return (int) floorf (min + (max - min) * r);
}

/* Helper function to print the given array. */
void
print_array (int *this_array, int num_elements)
{
    printf ("Array: ");
    for (int i = 0; i < num_elements; i++)
        printf ("%d ", this_array[i]);
    printf ("\n");
    return;
}

/* Helper function to return the min and max values in the given array. */
void 
print_min_and_max_in_array (int *this_array, int num_elements)
{
    int i;

    int current_min = INT_MAX;
    for (i = 0; i < num_elements; i++)
        if (this_array[i] < current_min)
            current_min = this_array[i];

    int current_max = INT_MIN;
    for (i = 0; i < num_elements; i++)
        if (this_array[i] > current_max)
            current_max = this_array[i];

    printf ("Minimum value in the array = %d\n", current_min);
    printf ("Maximum value in the array = %d\n", current_max);
    return;
}

void check_CUDA_error ( const char *msg )
{
   cudaError_t err = cudaGetLastError( );
   if ( cudaSuccess != err ) {
      printf( "[CUDA ERROR]: %s (%s).\n", msg, cudaGetErrorString(err) );
      exit( EXIT_FAILURE );
   }
   return;
}

void print_exec_time ( struct timeval start, struct timeval stop )
{
   printf( "Execution time:\t%f\n", 
         stop.tv_sec - start.tv_sec + (stop.tv_usec - start.tv_usec)/((float)1000000));
   return;
}

/* Illustrate the use of the GPU to perform vector addition on arbitrarily large vectors
 * Matt D'Angeli
 * 2019-05-12
 */

#include <stdlib.h>
#include <stdio.h>
#include <sys/time.h>
#include <string.h>
#include <math.h>
#include <float.h>

// Include CUDA kernel during preprocessing
#include "vector_addition_kernel.cu"

#define THREAD_BLOCK_SIZE 128
#define N_THREAD_BLOCKS 240

void run_test ( int );
void print_usage ( char * );
void compute_on_device ( float *, float *, float *, int );
void check_for_error ( char const * );
extern "C" void compute_gold ( float *, float *, float *, int );

int main ( int argc, char **argv )
{
   if (argc != 2) {
      print_usage( argv[0] );
      exit( EXIT_FAILURE );
   }

   int n_elements = atoi( argv[1] );
   run_test( n_elements );

   exit( EXIT_SUCCESS );
}

void run_test ( int n_elements )
{
   float diff;
   int i;

   // Allocate memory on CPU for input vectors A and B and output vector C
   int vector_length = sizeof( float ) * n_elements;
   float *A = (float *) malloc( vector_length );
   float *B = (float *) malloc( vector_length );
   float *host_result = (float *) malloc( vector_length ); // CPU result vector
   float *device_result = (float *) malloc( vector_length ); // GPU result vector

   // Randomly generate input data; populate with integer values between 0 and 5
   for (i = 0; i < n_elements; i++) {
      A[i] = floorf( 5 * (rand() / (float) RAND_MAX) );
      B[i] = floorf( 5 * (rand() / (float) RAND_MAX) );
   }

   // Compute reference solution on CPU
   printf( "Adding vectors on CPU\n" );
   compute_gold( A, B, host_result, n_elements );

   // Compute solution on GPU
   printf( "Adding vectors on GPU\n" );
   compute_on_device( A, B, device_result, n_elements );

   // Compute differences between CPU and GPU results
   diff = 0.0;
   for (i = 0; i < n_elements; i++)
      diff += abs( host_result[i] - device_result[i] );

   printf( "Difference between CPU and GPU results: %f\n", diff );

   // Free data structures
   free( (void *) A );
   free( (void *) B );
   free( (void *) host_result );
   free( (void *) device_result );

   exit( EXIT_SUCCESS );
}

/* HOST-SIDE CODE
 *    Transfer vectors A and B from CPU to GPU
 *    Set up grid and thread dimensions
 *    Execute kernel function
 *    Copy result vector back to CPU
 */
void compute_on_device( float *A_host, float *B_host, float *device_result, int n_elements )
{
   float *A_device = NULL;
   float *B_device = NULL;
   float *C_device = NULL;

   // Allocate space on GPU for vectors A and B; Copy contents of vectors to GPU
   cudaMalloc( (void **) &A_device, n_elements * sizeof( float ) );
   cudaMemcpy( A_device, A_host, n_elements * sizeof( float ), cudaMemcpyHostToDevice );

   cudaMalloc( (void **) &B_device, n_elements * sizeof( float ) );
   cudaMemcpy( B_device, B_host, n_elements * sizeof( float ), cudaMemcpyHostToDevice );

   // Allocate space for result vector on GPU
   cudaMalloc( (void **) &C_device, n_elements * sizeof( float ) );

   // Set up execution grid on GPU
   int n_thread_blocks = N_THREAD_BLOCKS;
   dim3 thread_block( THREAD_BLOCK_SIZE, 1, 1 ); // Set number of threads in block
   printf( "Setting up a (%d x 1) execution grid\n", n_thread_blocks );
   dim3 grid( n_thread_blocks, 1 );

   printf( "Adding vectors on GPU\n" );

   // Launch kernel with multiple thread blocks
   // NOTE: Kernel call is non-blocking
   vector_addition_kernel <<< grid, thread_block >>> (A_device, B_device, C_device, n_elements);
   cudaDeviceSynchronize( ); // Force CPU to wait for GPU to complete
   check_for_error( "KERNEL FAILURE" );

   // Copy result vector back from GPU and store
   cudaMemcpy( device_result, C_device, n_elements * sizeof( float ), cudaMemcpyDeviceToHost );

   // Free GPU memory
   cudaFree( A_device );
   cudaFree( B_device );
   cudaFree( C_device );
}

void check_for_error ( char const *msg )
{
   cudaError_t err = cudaGetLastError( );
   if (cudaSuccess != err) {
      printf( "CUDA ERROR: %s (%s). \n", msg, cudaGetErrorString( err ) );
      exit( EXIT_FAILURE );
   }
}

void print_usage ( char *script_name )
{
   printf( "Usage: %s n_elements\n", script_name );
}

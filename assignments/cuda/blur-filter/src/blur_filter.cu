/* Reference code implementing the box blur filter.

  Build and execute as follows: 
    make clean && make 
    ./blur_filter size

  Author: Naga Kandasamy
  Date created: May 3, 2019
*/

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <sys/time.h>

//#define DEBUG 1
#define TILE_SIZE 32
#define EPS 1e-6

/* Include the kernel code. */
#include "blur_filter_kernel.cu"

extern "C" void compute_gold (const image_t, image_t);
void compute_on_device (const image_t, image_t);
int check_results (const float *, const float *, int, float);
void print_image (const image_t);

image_t allocate_image_on_device ( const image_t );
void copy_image_to_device ( image_t, const image_t );
void copy_image_from_device ( image_t, const image_t );
void free_image_on_device ( image_t * );
void check_CUDA_error ( const char * );
void print_exec_time ( struct timeval , struct timeval );

int 
main (int argc, char **argv)
{
    if (argc < 2) {
        printf ("Usage: %s size\n", argv[0]);
        printf ("size: Height of the image. The program assumes size x size image.\n");
        exit (EXIT_FAILURE);
    }

    /* Allocate memory for the input and output images. */
    int size = atoi (argv[1]);

    printf ("Creating %d x %d images\n", size, size);
    image_t in, out_gold, out_gpu;
    in.size = out_gold.size = out_gpu.size = size;
    in.element = (float *) malloc (sizeof (float) * size * size);
    out_gold.element = (float *) malloc (sizeof (float) * size * size);
    out_gpu.element = (float *) malloc (sizeof (float) * size * size);
    if ((in.element == NULL) || (out_gold.element == NULL) || (out_gpu.element == NULL)) {
        perror ("Malloc");
        exit (EXIT_FAILURE);
    }

    /* Poplulate our image with random values between [-0.5 +0.5] */
    srand (time (NULL));
    for (int i = 0; i < size * size; i++)
        in.element[i] = rand ()/ (float) RAND_MAX -  0.5;
        // in.element[i] = 1;

   struct timeval start, stop;
  
   /* Calculate the blur on the CPU. The result is stored in out_gold. */
   printf ("Calculating blur on the CPU\n");
   gettimeofday( &start, NULL );
   compute_gold (in, out_gold); 
   gettimeofday( &stop, NULL );
   print_exec_time( start, stop );

   /* Calculate the blur on the GPU. The result is stored in out_gpu. */
   printf ("Calculating blur on the GPU\n");
   compute_on_device (in, out_gpu);

#ifdef DEBUG 
   fprintf( stderr, "Input:\n");
   print_image (in);

   fprintf( stderr, "CPU Output:\n");
   print_image (out_gold);

   fprintf( stderr, "GPU Output:\n");
   print_image( out_gpu );
#endif


   /* Check the CPU and GPU results for correctness. */
   printf ("Checking CPU and GPU results\n");
   int num_elements = out_gold.size * out_gold.size;
   float eps = EPS;
   int check = check_results (out_gold.element, out_gpu.element, num_elements, eps);
   if (check == 1) 
       printf ("TEST PASSED\n");
   else
       printf ("TEST FAILED\n");
   
   /* Free data structures on the host. */
   free ((void *) in.element);
   free ((void *) out_gold.element);
   free ((void *) out_gpu.element);

    exit (EXIT_SUCCESS);
}

/* FIXME: Complete this function to calculate the blur on the GPU. */
void compute_on_device (const image_t in, image_t out)
{
   // Allocate data structures and copy data to GPU
   image_t unfiltered_dev = allocate_image_on_device( in );
   copy_image_to_device( unfiltered_dev, in );

   image_t filtered_dev = allocate_image_on_device( out );

   // Set up execution grid and thread tiles
   dim3 threads( TILE_SIZE, TILE_SIZE );
   int grid_dimension = (filtered_dev.size + TILE_SIZE - 1) / TILE_SIZE;
   printf( "\tSetting up a %d x %d grid of thread blocks\n", grid_dimension, grid_dimension );

   dim3 grid( grid_dimension, grid_dimension );

   // Launch the kernel
   struct timeval start, stop;
   gettimeofday( &start, NULL );
   blur_filter_kernel <<< grid, threads >>> ( filtered_dev.element, unfiltered_dev.element, in.size );

   // Sync device and host
   cudaDeviceSynchronize( );

   gettimeofday( &stop, NULL );
   print_exec_time( start, stop );


   // Check for errors
   check_CUDA_error( "Error in kernel execution" );

   // Copy data from device to host
   copy_image_from_device( out, filtered_dev );

   // Free device memory
   free_image_on_device( &unfiltered_dev );
   free_image_on_device( &filtered_dev );

   return;
}

void print_exec_time ( struct timeval start, struct timeval stop )
{
   printf( "Execution time:\t%fs\n", (float) (stop.tv_sec - start.tv_sec +\
            (stop.tv_usec - start.tv_usec) / (float) 1000000) );
   return;
}

image_t allocate_image_on_device ( const image_t I )
{
   image_t I_device = I;
   int size = I.size * I.size * sizeof( float );

   cudaMalloc( (void **) &I_device.element, size );
   if (I_device.element == NULL) {
      fprintf( stderr, "[ERROR] CUDA malloc error\n" );
      exit( EXIT_FAILURE );
   }

   return I_device;
}

void copy_image_to_device ( image_t I_device, const image_t I_host )
{
   int size = I_host.size * I_host.size * sizeof( float );
   cudaMemcpy( I_device.element, I_host.element, size, cudaMemcpyHostToDevice );
   return;
}

void copy_image_from_device ( image_t I_host, const image_t I_device )
{
   int size = I_device.size * I_device.size * sizeof( float );
   cudaMemcpy( I_host.element, I_device.element, size, cudaMemcpyDeviceToHost );
   return;
}

void free_image_on_device ( image_t *I )
{
   cudaFree( I->element );
   I->element = NULL;
   return;
}

// Check for errors during kernel execution
void check_CUDA_error ( const char *msg )
{
   cudaError_t err = cudaGetLastError( );
   if (cudaSuccess != err) {
      fprintf( stderr, "[CUDA ERROR] %s (%s).\n", msg, cudaGetErrorString( err ) );
      exit( EXIT_FAILURE );
   }

   return;
}

/* Function to check correctness of the results. */
int 
check_results (const float *pix1, const float *pix2, int num_elements, float eps) 
{
    for (int i = 0; i < num_elements; i++)
        if (fabsf ((pix1[i] - pix2[i])/pix1[i]) > eps) 
            return 0;
    
    return 1;
}

/* Function to print out the image contents. */
void 
print_image (const image_t img)
{
    for (int i = 0; i < img.size; i++) {
        for (int j = 0; j < img.size; j++) {
            float val = img.element[i * img.size + j];
            printf ("%0.4f\t", val);
        }
        printf ("\n");
    }

    printf ("\n");
    return;
}

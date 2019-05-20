/* Host code for the Jacobi method of solving a system of linear equations 
 * by iteration.

 * Author: Naga Kandasamy
 * Date modified: May 13, 2019
*/

#include <stdlib.h>
#include <stdio.h>
#include <sys/time.h>
#include <string.h>
#include <math.h>
#include <cuda_runtime.h>
#include "jacobi_iteration.h"

/* Include the kernel code. */
#include "jacobi_iteration_kernel.cu"

/* Uncomment the line below if you want the code to spit out debug information. */ 
//#define DEBUG

int main ( int argc, char** argv ) 
{
   if (argc > 1) {
      printf ("This program accepts no arguments\n");
      exit (EXIT_FAILURE);
   }

   matrix_t  A;                    /* N x N constant matrix. */
   matrix_t  B;                    /* N x 1 b matrix. */
   matrix_t reference_x;           /* Reference solution. */ 
   matrix_t gpu_hybrid_solution_x;  /* Solution computed by hybrid kernel. */

   /* Initialize the random number generator. */
   srand (time (NULL));

   /* Generate diagonally dominant matrix. */ 
   A = create_diagonally_dominant_matrix (MATRIX_SIZE, MATRIX_SIZE);
   if (A.elements == NULL) {
      printf ("Error creating matrix\n");
      exit (EXIT_FAILURE);
   }
	
   /* Create the other vectors. */
   B = allocate_matrix_on_host (MATRIX_SIZE, 1, 1);
   reference_x = allocate_matrix_on_host (MATRIX_SIZE, 1, 0);
   gpu_hybrid_solution_x = allocate_matrix_on_host (MATRIX_SIZE, 1, 0);

#ifdef DEBUG
   print_matrix (A);
   print_matrix (B);
   print_matrix (reference_x);
#endif

   /* Compute the Jacobi solution on the CPU. */
   printf ("Performing Jacobi iteration on the CPU\n");
   struct timeval start, stop;
   gettimeofday( &start, NULL );
   compute_gold (A, reference_x, B);
   gettimeofday( &stop, NULL );
   print_exec_time( start, stop );
   display_jacobi_solution (A, reference_x, B); /* Display statistics. */
	
   /* Compute the Jacobi solution on the GPU. 
    * The solutions are returned in gpu_hybrid_solution_x and gpu_opt_solution_x. 
    */
   printf ("\nPerforming Jacobi iteration on device. \n");
   compute_on_device (A, gpu_hybrid_solution_x, B);
   display_jacobi_solution (A, gpu_hybrid_solution_x, B); // Display statistics.

   free (A.elements); 
   free (B.elements); 
   free (reference_x.elements); 
   free (gpu_hybrid_solution_x.elements);
	
   exit (EXIT_SUCCESS);
}


/* FIXME: Complete this function to perform the Jacobi calculation on the GPU. */
void compute_on_device (
      const matrix_t A, 
      matrix_t gpu_hybrid_sol_x, 
      const matrix_t B )
{
   // Allocate data structures and copy data to GPU
   matrix_t A_dev = allocate_matrix_on_device( A );
   copy_matrix_to_device( A_dev, A );

   matrix_t B_dev = allocate_matrix_on_device( B );
   copy_matrix_to_device( B_dev, B );

   matrix_t hybrid_soln_dev = allocate_matrix_on_device( gpu_hybrid_sol_x );

   matrix_t temp_matrix = allocate_matrix_on_host( MATRIX_SIZE, 1, 0 );
   matrix_t temp_matrix_dev = allocate_matrix_on_device( temp_matrix );

   // Naive solution
   // Set up execution grid and thread tiles for hybrid solution
   dim3 threads_hybrid( THREAD_BLOCK_SIZE);
   int grid_dimension_hybrid = (MATRIX_SIZE + THREAD_BLOCK_SIZE - 1) / THREAD_BLOCK_SIZE;
   printf("\tSetting up a %d x 1 grid of thread blocks for hybrid solution\n",
         grid_dimension_hybrid );

   dim3 grid_hybrid( grid_dimension_hybrid );

   // Launch hybrid kernel
   struct timeval start, stop;
   gettimeofday( &start, NULL );
   jacobi_iteration_kernel_hybrid <<< grid_hybrid, threads_hybrid >>> ( 
         hybrid_soln_dev.elements, temp_matrix_dev.elements, A_dev.elements, B_dev.elements );
   
   // Sync device and host
   cudaDeviceSynchronize( );

   gettimeofday( &stop, NULL );
   print_exec_time( start, stop );

   // Check for errors
   check_CUDA_error ( "[CUDA Error] Kernel execution experienced an error" );

   // Copy data from device to host
   copy_matrix_from_device( gpu_hybrid_sol_x, hybrid_soln_dev );

   // Free device memory
   free_matrix_on_device( &A_dev );
   free_matrix_on_device( &B_dev );
   free_matrix_on_device( &hybrid_soln_dev );
   free_matrix_on_device( &temp_matrix_dev );

   // Free host memory
   free( temp_matrix.elements );

   return;
}

/* Allocate matrix on the device of same size as M. */
matrix_t allocate_matrix_on_device ( const matrix_t M )
{
   matrix_t Mdevice = M;
   int size = M.num_rows * M.num_columns * sizeof(float);
   cudaMalloc ((void **) &Mdevice.elements, size);
   return Mdevice;
}

/* Allocate a matrix of dimensions height * width.
 * If init == 0, initialize to all zeroes.  
 * If init == 1, perform random initialization.
 */
matrix_t allocate_matrix_on_host ( int num_rows, int num_columns, int init )
{	
   matrix_t M;
   M.num_columns = num_columns;
   M.num_rows = num_rows;
   int size = M.num_rows * M.num_columns;
		
   M.elements = (float *) malloc (size * sizeof (float));
   for (unsigned int i = 0; i < size; i++) {
      if (init == 0) 
         M.elements[i] = 0; 
      else
         M.elements[i] = get_random_number (MIN_NUMBER, MAX_NUMBER);
   }
    
   return M;
}	

/* Copy matrix to a device. */
void copy_matrix_to_device ( matrix_t Mdevice, const matrix_t Mhost )
{
   int size = Mhost.num_rows * Mhost.num_columns * sizeof (float);
   Mdevice.num_rows = Mhost.num_rows;
   Mdevice.num_columns = Mhost.num_columns;
   cudaMemcpy (Mdevice.elements, Mhost.elements, size, cudaMemcpyHostToDevice);
   return;
}

/* Copy matrix from device to host. */
void copy_matrix_from_device ( matrix_t Mhost, const matrix_t Mdevice )
{
   int size = Mdevice.num_rows * Mdevice.num_columns * sizeof (float);
   cudaMemcpy (Mhost.elements, Mdevice.elements, size, cudaMemcpyDeviceToHost);
   return;
}

/* Prints the matrix out to screen. */
void print_matrix ( const matrix_t M )
{
   for (unsigned int i = 0; i < M.num_rows; i++) {
      for (unsigned int j = 0; j < M.num_columns; j++) {
         printf ("%f ", M.elements[i * M.num_rows + j]);
      }
		
      printf ("\n");
   } 
	
   printf ("\n");
   return;
}

/* Returns a floating-point value between min and max values. */
float get_random_number ( int min, int max )
{
   float r = rand() / (float) RAND_MAX;
   return (float) floor( (double) (min + (max - min + 1) * r) );
}

/* Check for errors in kernel execution. */
void check_CUDA_error ( const char *msg )
{
   cudaError_t err = cudaGetLastError();
   if ( cudaSuccess != err ) {
      printf ("CUDA ERROR: %s (%s).\n", msg, cudaGetErrorString(err));
      exit (EXIT_FAILURE);
   }	
   return;    
}

/* Checks the reference and GPU results. */
int check_results ( float *reference, float *gpu_result, int num_elements, float eps )
{
   int check = 1;
   float max_eps = 0.0;
    
   for (int i = 0; i < num_elements; i++) {
      if (fabsf((reference[i] - gpu_result[i])/reference[i]) > eps) {
         check = 0;
         printf("Error at index %d\n",i);
         printf("Element r %f and g %f\n", reference[i] ,gpu_result[i]);
         break;
      }
   }
	
   int maxEle;
   for (int i = 0; i < num_elements; i++) {
      if (fabsf((reference[i] - gpu_result[i])/reference[i]) > max_eps) {
         max_eps = fabsf ((reference[i] - gpu_result[i])/reference[i]);
         maxEle=i;
      }
   }

   printf ("Max epsilon = %f at i = %d value at cpu %f and gpu %f \n", 
         max_eps, maxEle, reference[maxEle], gpu_result[maxEle]); 
    
   return check;
}


/* Function checks if the matrix is diagonally dominant. */
int check_if_diagonal_dominant ( const matrix_t M )
{
   float diag_element;
   float sum;
   for (unsigned int i = 0; i < M.num_rows; i++) {
      sum = 0.0; 
      diag_element = M.elements[i * M.num_rows + i];
      for (unsigned int j = 0; j < M.num_columns; j++) {
         if (i != j)
            sum += abs(M.elements[i * M.num_rows + j]);
      }
		
      if (diag_element <= sum)
         return 0;
   }

   return 1;
}

/* Create a diagonally dominant matrix. */
matrix_t create_diagonally_dominant_matrix ( unsigned int num_rows, unsigned int num_columns )
{
   matrix_t M;
   M.num_columns = num_columns;
   M.num_rows = num_rows; 
   unsigned int size = M.num_rows * M.num_columns;
   M.elements = (float *) malloc (size * sizeof (float));

   /* Create a matrix with random numbers between [-.5 and .5]. */
   unsigned int i, j;
   printf ("Generating %d x %d matrix with numbers between [-.5, .5]\n", num_rows, num_columns);
   for (i = 0; i < size; i++)
      // M.elements[i] = ((float)rand ()/(float)RAND_MAX) - 0.5;
      M.elements[i] = get_random_number (MIN_NUMBER, MAX_NUMBER);
	
   /* Make diagonal entries large with respect to the entries on each row. */
   for (i = 0; i < num_rows; i++) {
      float row_sum = 0.0;		
      for (j = 0; j < num_columns; j++) {
         row_sum += fabs (M.elements[i * M.num_rows + j]);
      }
		
      M.elements[i * M.num_rows + i] = 0.5 + row_sum;
   }

   /* Check if matrix is diagonal dominant. */
   if (!check_if_diagonal_dominant (M)) {
      free (M.elements);
      M.elements = NULL;
   }
	
   return M;
}

void free_matrix_on_device ( matrix_t *M )
{
   cudaFree( M->elements );
   M->elements = NULL;
   return;
}

void print_exec_time ( struct timeval start, struct timeval stop )
{
   printf( "Execution time: \t%fs\n", (float) (stop.tv_sec - start.tv_sec +\
            (stop.tv_usec - start.tv_usec) / (float) 1000000) );
   return;
}

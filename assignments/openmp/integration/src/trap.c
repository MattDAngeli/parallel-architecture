/*  Purpose: Calculate definite integral using trapezoidal rule.
 *
 * Input:   a, b, n_trapezoids, n_threads
 * Output:  Estimate of integral from a to b of f(x)
 *          using n trapezoids, with n_threads.
 *
 * Compile: gcc -fopenmp -o trap trap.c -O3 -std=c99 -Wall -lm
 * Usage:   ./trap
 *
 * Note:    The function f(x) is hardwired.
 *
 * Author: Naga Kandasamy
 * Date modified: April 25, 2019
 *
 * Student: Matthew D'Angeli
 * Date completed: 30 April 2019
 *
 * Parallelized using OpenMP parallel for w/ reduction variable
 */

#ifdef _WIN32
#  define NOMINMAX 
#endif

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <float.h>
#include <sys/time.h>
#include <omp.h>

double compute_using_omp (float, float, int, float, int);
double compute_gold (float, float, int, float);

int main (int argc, char **argv) 
{
   if (argc < 5) {
      printf ("Usage: trap lower-limit upper-limit num-trapezoids num-threads\n");
      printf ("lower-limit: The lower limit for the integral\n");
      printf ("upper-limit: The upper limit for the integral\n");
      printf ("num-trapezoids: Number of trapeziods used to approximate the area under the curve\n");
      printf ("num-threads: Number of threads to use in the calculation\n");
      exit (EXIT_FAILURE);
   }

   float a = atoi (argv[1]); /* Lower limit. */
   float b = atof (argv[2]); /* Upper limit. */
   float n_trapezoids = atof (argv[3]); /* Number of trapezoids. */

   float h = (b - a)/(float) n_trapezoids; /* Base of each trapezoid. */  
   printf ("The base of the trapezoid is %f \n", h);

   struct timeval start, stop;

   gettimeofday( &start, NULL );
   double reference = compute_gold (a, b, n_trapezoids, h);
   gettimeofday( &stop, NULL );

   printf ("Reference solution computed using single-threaded version = %f \n", reference);
   printf("Execution time = %fs.\n", (float)(stop.tv_sec - start.tv_sec +\
            (stop.tv_usec - start.tv_usec) / (float) 1000000));

   /* Write this function to complete the trapezoidal rule using omp. */
   int n_threads = atoi (argv[4]); /* Number of threads. */

   gettimeofday( &start, NULL );
   double omp_result = compute_using_omp (a, b, n_trapezoids, h, n_threads);
   gettimeofday( &stop, NULL );

   printf ("Solution computed using %d threads = %f \n", n_threads, omp_result);
   printf("Execution time = %fs.\n", (float)(stop.tv_sec - start.tv_sec +\
            (stop.tv_usec - start.tv_usec) / (float) 1000000));

   exit(EXIT_SUCCESS);
} 

/*------------------------------------------------------------------
 * Function:    f
 * Purpose:     Defines the integrand
 * Input args:  x
 * Output: sqrt((1 + x^2)/(1 + x^4))

 */
float f (float x) 
{
    return sqrt ((1 + x*x)/(1 + x*x*x*x));
}

/*------------------------------------------------------------------
 * Function:    compute_gold
 * Purpose:     Estimate integral from a to b of f using trap rule and
 *              n trapezoids using a single-threaded version
 * Input args:  a, b, n_trapezoids, h
 * Return val:  Estimate of the integral 
 */
double compute_gold (float a, float b, int n_trapezoids, float h) 
{
   double integral;
   int k;

   integral = (f(a) + f(b))/2.0;

   for (k = 1; k <= n_trapezoids-1; k++) 
     integral += f(a+k*h);
   
   integral = integral*h;

   return integral;
}  

/* FIXME: Complete this function to perform the trapezoidal rule using omp. */
double compute_using_omp (float a, float b, int n_trapezoids, float h, int n_threads)
{
   double integral = ( f(a) + f(b) ) / 2.0;
   int i;

#pragma omp parallel for num_threads(n_threads) private(i) reduction(+:integral)
   for (i = 1; i <= n_trapezoids - 1; i++) {
      // printf("[THREAD %d] Adding iteration %d\n", omp_get_thread_num(), i);
      integral += f(a + i*h);
   }

   integral *= h;

   return integral;
}


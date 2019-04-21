/*  Purpose: Calculate definite integral using trapezoidal rule.
 *
 * Input:   a, b, n_trapezoids, n_threads
 * Output:  Estimate of integral from a to b of f(x)
 *          using n_trapezoids trapezoids, with n_threads.
 *
 * Compile: gcc -o trap trap.c -O3 -std=c99 -Wall -lpthread -lm
 * Usage:   ./trap
 *
 * Note:    The function f(x) is hardwired.
 *
 * Author: Naga Kandasamy
 * Date modified: 4/1/2019
 *
 * Student: Matthew D'Angeli
 * Modified for homework due 21 April 2019
 *
 */

#ifdef _WIN32
#  define NOMINMAX 
#endif

#define _REENTRANT // Ensure multi-thread safety of library functions

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <float.h>
#include <time.h>
#include <pthread.h>
#include <unistd.h>

double compute_using_pthreads (float, float, int, float, int);
double compute_gold (float, float, int, float);

void *trap_integrate( void *args );

typedef struct thread_data_t {
   int tid; // Thread ID
   int n_threads;
   int chunk_size;
   float a; // Lower bound of the integral
   float b; // Upper bound of the integral
   float offset;
   float h; // Width of the trapezoids
   float n_trapezoids; // Number of trapezoids
   double *integral;
   pthread_mutex_t *mutex_for_integral;
} thread_data_t;

int main ( int argc, char **argv ) 
{
   if (argc < 5) {
      printf ("Usage: trap lower-limit upper-limit num-trapezoids num-threads\n");
      printf ("lower-limit: The lower limit for the integral\n");
      printf ("upper-limit: The upper limit for the integral\n");
      printf ("num-trapezoids: Number of trapeziods used to approximate the area under the curve\n");
      printf ("num-threads: Number of threads to use in the calculation\n");
      exit (EXIT_FAILURE);
   }

   float a = atoi (argv[1]); /* Lower limit */
   float b = atof (argv[2]); /* Upper limit */
   float n_trapezoids = atof (argv[3]); /* Number of trapezoids */

   float h = (b - a)/(float) n_trapezoids; /* Base of each trapezoid */  
   printf ("The base of the trapezoid is %f \n", h);

   double reference = compute_gold (a, b, n_trapezoids, h);
   printf ("Reference solution computed using single-threaded version = %f \n", reference);

   /* Write this function to complete the trapezoidal rule using pthreads. */
   int n_threads = atoi (argv[4]); /* Number of threads */
   double pthread_result = compute_using_pthreads (a, b, n_trapezoids, h, n_threads);
   printf ("Solution computed using %d threads = %f \n", n_threads, pthread_result);

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
 *              n_trapezoids trapezoids using a single-threaded version
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

/* FIXME: Complete this function to perform the trapezoidal rule using pthreads. */
/*------------------------------------------------------------------
 * Function:	compute_using_pthreads
 * Purpose:	Estimate integral from a to b of f using trap rule and
 * 		n_trapezoids trapezoids using multiple threads
 * Input args:	a, b, n_trapezoids, h, n_threads
 * Return val:	Estimate of the integral
 */
double compute_using_pthreads (float a, float b, int n_trapezoids, float h, int n_threads)
{
   // Define structure where thread IDs will be stored
   pthread_t *thread_id = (pthread_t *) malloc( sizeof( pthread_t ) * n_threads );
   
   pthread_attr_t attributes; // Thread attributes
   pthread_attr_init( &attributes ); // Initialize thread attributes to default

   // Mutex setup
   pthread_mutex_t mutex_for_integral;
   pthread_mutex_init( &mutex_for_integral, NULL );

   // Allocate heap space for thread data and create worker threads
   int i;
   double integral = ( f(a) + f(b) ) / 2.0;
   thread_data_t *thread_data = (thread_data_t *) malloc( sizeof( thread_data_t ) * n_threads );
   int chunk_size = (int) floor( (float) n_trapezoids / (float) n_threads );

   for (i = 0; i < n_threads; i++) {
      thread_data[i].tid = i;
      thread_data[i].n_threads = n_threads;
      thread_data[i].n_trapezoids = n_trapezoids;
      thread_data[i].chunk_size = chunk_size;
      thread_data[i].a = a;
      thread_data[i].b = b;
      thread_data[i].h = h;
      thread_data[i].offset = i * chunk_size * h;
      thread_data[i].integral = &integral;
      thread_data[i].mutex_for_integral = &mutex_for_integral;
   }

   for (i = 0; i < n_threads; i++) 
      pthread_create( &thread_id[i], &attributes, trap_integrate, (void *) &thread_data[i] );

   for (i = 0; i < n_threads; i++)
      pthread_join( thread_id[i], NULL );

   free( (void *) thread_data );

   return integral;
}

void *trap_integrate( void *args )
{
   // Cast argument to appropriate type
   thread_data_t *td = (thread_data_t *) args;

   // Compute partial integral for which this thread is responsible
   double partial_integral = 0.0;
   if (td->tid < (td->n_threads - 1)) {
      for (float i = td->offset + td->h; i <= (i+1) * td->offset; i+=td->h) {
         partial_integral += f( i );
      }
   } else {
      for (float i = td->offset + td->h; i < td->b; i+=td->h) {
         partial_integral += f( i );
      }
   }

   pthread_mutex_lock( td->mutex_for_integral );
   *(td->integral) += partial_integral;
   pthread_mutex_unlock( td->mutex_for_integral );

   pthread_exit( NULL );
}


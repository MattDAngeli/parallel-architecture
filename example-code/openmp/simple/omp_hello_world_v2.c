/* A hello word program that uses OpenMP. 
 * This provides an example of the parallel construct or MIMD style parallelism. 
 *
 * Compile as follows: gcc -o omp_hello_world omp_hello_worid_v2.c -fopenmp -std=c99 -O3 -Wall

 * Author: Naga Kandasamy
 * Date created: April 15, 2011
 * Date modified: April 20, 2019
 *  */

#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

int
main (int argc, char **argv)
{
  
    if (argc != 2) {
        printf ("Usage: %s num-threads\n", argv[0]);
        printf ("num-threads: Number of threads to create\n");
        exit (EXIT_FAILURE);
    }
  
    int thread_count = atoi (argv[1]); /* Number of threads to be created. */

    /* OpenMP block here. */
#pragma omp parallel num_threads(thread_count)
    {
        int tid = omp_get_thread_num ();
        printf ("The parallel region executed by thread %d\n", omp_get_thread_num ());
    
        if (tid == 4) {
            printf ("Thread %d does things differently\n", omp_get_thread_num ());
        }

        if (tid == 2) {
            printf ("Thread %d does things differently\n", omp_get_thread_num ());
        }
  
    } /* Barrier sync here. */
  
    exit (EXIT_SUCCESS);
}

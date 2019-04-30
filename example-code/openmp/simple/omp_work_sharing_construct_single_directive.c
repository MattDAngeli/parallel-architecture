/* Example of the work sharing construct that shows how to use the single directive.
 *
 * Compile as follows: 
 * gcc -o omp_work_sharing_construct_with_single_directive omp_work_sharing_construct_with_single_directive.c -fopenmp -std=c99 -O3 -Wall
 *
 * Author: Naga Kandasamy, April 15, 2011
 * Date modified: April 21, 2019 
 *
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
        exit (EXIT_SUCCESS);
    }
  
    int thread_count = atoi (argv[1]);
    int a;
    int b[10];
    int i;
    int n = 10;

    /* Start of parallel region. */
#pragma omp parallel num_threads(thread_count) shared(a, b) private(i)
    {
        /* Only a single thread in the team executes this code. Useful when dealing with sections of code that are 
         * not thread safe (such as I/O). The thread does not have to be master. */
#pragma omp single
        {
            a = 10;
            printf ("Single construct executed by thread %d. \n", omp_get_thread_num ());
        } /* This is an implicit barrier sync. */

        /* We parallelize the iterations within the for loop over the available threads. */ 
#pragma omp for
        for (i = 0; i < n; i++)
            b[i] = a;
  
    } /* End of parallel region. */
  
    for (i = 0; i < n; i++)
        printf ("b[%d] = %d. \n", i, b[i]);

    exit (EXIT_SUCCESS);
}

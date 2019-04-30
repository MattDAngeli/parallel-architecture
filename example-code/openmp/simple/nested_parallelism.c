/* OpenMP program showing the use of nested parallelism. Unfortunately, the implementation on Linux does not support 
 * nested parallelism.
 *
 * Compile as follows: gcc -o nested_parallelism nested_parallelism.c -fopenmp -std=c99 -O3 -Wall
 * 
 * Author: Naga Kandasamy
 * Date created: April 25, 2011
 * Date modified: April 21, 2019
 * 
 *  */

#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

void 
report_num_threads (int level)
{
#pragma omp single 
    printf("Level %d: number of threads in the team = %d\n", level, omp_get_num_threads());
    return;
}

int
main (int argc, char **argv)
{
    if (argc != 2) {
        printf ("Usage: %s num-threads\n", argv[0]);
        printf ("num-threads: Number of threads to create\n");
        exit (EXIT_FAILURE);
    }
  
    int thread_count = atoi (argv[1]);

  
    /* Check if the OpenMP implementation supports nested parallelism. */
    printf ("Nested parallelism is %s\n", omp_get_nested ()? "supported" : "not supported");

    int tid;
#pragma omp parallel num_threads(thread_count) private(tid)
    {
        tid = omp_get_thread_num ();
        report_num_threads (1); /* Report the number of threads at this level. */  
        printf ("Thread %d executes outer parallel region\n", tid);

#pragma omp parallel num_threads(thread_count) firstprivate(tid)
    {
	    report_num_threads (2);
        printf ("Parent Thread %d: Thread %d executes inner parallel region\n", tid, omp_get_thread_num ());
    }
  }

    exit (EXIT_SUCCESS);
}

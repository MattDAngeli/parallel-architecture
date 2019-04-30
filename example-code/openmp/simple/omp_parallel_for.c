/* Parallelization of a for loop using OpenMP. 
 *
 * Compile as follows: 
 * gcc -o omp_parallel_for omp_parallel_for.c -fopenmp -std=c99 -O3 -Wall
 *
 * Author: Naga Kandasamy
 * Date created: April 21, 2019
 * Date modified: 
 *  */

#include <stdio.h>
#include <unistd.h>
#include <stdlib.h>
#include <omp.h>

void foo (int);
void bar (int);

int
main (int argc, char **argv)
{
    if (argc != 3) {
        printf ("Usage: %s num-threads num-iterations\n", argv[0]);
        printf ("num-threads: Number of threads to create\n");
        printf ("num-iterations: Number of loop iterations to execute\n");
        exit (EXIT_FAILURE);
    }
  
    int thread_count = atoi (argv[1]);	/* Number of threads to create. */
    int num_iterations = atoi (argv[2]); /* Number of loop iterations to execute. */
    
#pragma omp parallel num_threads(thread_count)
    {
        int tid = omp_get_thread_num ();	/* Obtain thread ID. */
        /* All threads execute the function foo() */
        foo (tid);

        /* The variable i, by virtue of being declared inside the omp construct, 
         * is private for each thread. That is, each thread gets a local or private copy of 
         * the variable. */
        int i;

        /* Paralellize the for loop. Note that the for pragma does not create a team of threads: it takes the team of threads 
         * that is active, and divides the loop iterations over them. */
#pragma omp for 
        for (i = 0; i < num_iterations; i++) {
            printf ("Iteration %d is executed by thread %d\n", i, tid);
            /* Loop body here. */
        }

        /* All threads execute the function bar() */
        bar (tid);

    } /* End omp construct. */

    exit (EXIT_SUCCESS);
}

void
foo (int tid)
{
    printf ("Thread %d executing foo()\n", tid);
    return;
}

void
bar (int tid)
{
    printf ("Thread %d executing bar()\n", tid);
    return;
}

/* OpenMP program showing the use of loop collapsing.
 *
 * Compile as follows: gcc -o loop_collapsing loop_collapsing.c -fopenmp -std=c99 -O3 -Wall
 * 
 * Author: Naga Kandasamy
 * Date created: April 21, 2019
 * Date modified: April 22, 2019
 * 
 *  */

#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

#define DIM 4

int
main (int argc, char **argv)
{
    if (argc != 2) {
        printf ("Usage: %s num-threads\n", argv[0]);
        printf ("num-threads: Number of threads to create\n");
        exit (EXIT_FAILURE);
    }
  
    int thread_count = atoi (argv[1]);
    int i, j;
    int A[DIM][DIM], B[DIM][DIM], C[DIM][DIM];

    /* Create some 2D matrices. */
    for (i = 0; i < DIM; i++)
        for (j = 0; j < DIM; j++) {
            B[i][j] = 10;
            C[i][j] = 5;
        }

#pragma omp parallel num_threads(thread_count) private(i, j)
    {
        /* The omp for will only parallelize the iterations of the outer loop. 
         * Each thread in the thread group will execute all iteration of the i
         * inner loop sequentially.
         * */
#pragma omp for 
        for (i = 0; i < DIM; i++) 
            for (j = 0; j < DIM; j++) {
                printf ("Thread %d calculating element A[%d][%d]\n", omp_get_thread_num (), i, j);
                A[i][j] = B[i][j] + C[i][j];
            }
    }

    printf ("\n");

#pragma omp parallel num_threads(thread_count) private(i, j) 
    {
        /* Example of directing omp to collapse the two loops into one. 
         * This has potential to improve the degree of parallelism. 
         * For example, if we create eight threads but only have four iterations of 
         * the outer loop, that is DIM = 4, then four threads are being wasted. 
         * On the other hand, if we collapse the loops into a single loop comprising of 16 
         * iterations, we can distribute those among the eight threads. 
         *
         * Important note: The loops must be independent of each other. Also, it is only possible to collapse 
         * perfectly nested loops, that is, the loop body of the outer loop can consist only of the inner loop; 
         * there can be no statements before or after the inner loop in the body of the outer loop.
         * */
#pragma omp for collapse(2)
        for (i = 0; i < DIM; i++) 
            for (j = 0; j < DIM; j++) {
                printf ("Thread %d calculating element A[%d][%d]\n", omp_get_thread_num (), i, j);
                A[i][j] = B[i][j] + C[i][j];
            }
    }

    exit (EXIT_SUCCESS);
}

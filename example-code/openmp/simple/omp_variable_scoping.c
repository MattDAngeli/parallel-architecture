/* Examples illustrating the concept of variable scoping in omp.
 * 
 * Compile as follows: gcc -o omp_variable_scoping omp_variable_scoping.c -fopenmp -std=c99 -O3 -Wall 
 *
 * Author: Naga Kandasamy
 * Date created: April 15, 2011 
 * Date modified: April 21, 2019
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
  
    int thread_count = atoi (argv[1]);	
    int i, a; 
    int n = 10;

    /* Start of parallel region. Declare i and a as private variable for each thread. Copies of these 
     * variables will be made for each thread within the parallel region. The shared variable is n. 
     * We parallelize the iterations of the for loop over each of the available threads. 
     * */
#pragma omp parallel num_threads(thread_count) shared(n) private(i, a)
    {
#pragma omp for
        for (i = 0; i < n; i++) {
            a = i + 1;
            printf ("Thread %d executes loop iteration %d with value for a = %d\n", omp_get_thread_num (), i, a);
        }
    } /* End of omp parallel construct. */
  
    /* Value is of a is undefined after threads exit the parallel region. */
    printf ("Value of a after the parallel for: a = %d\n", a); 
  
    /* Use of the lastprivate clause. The value copied back into the original variable a is obtained 
     * from the last (sequentially) iteration or section of the enclosing construct. */
#pragma omp parallel for num_threads(thread_count) private(i) lastprivate(a) shared(n)
    for (i = 0; i < n; i++) {
        a = i + 1;
        printf ("Thread %d executes loop iteration %d with value for a = %d \n", omp_get_thread_num (), i, a);
    }
    
    printf ("Value of a after the parallel for: a = %d \n", a);

  /* Use of the first private clause. Listed variables are initialized according to the 
   * value of their original objects prior to entry into the parallel or work-sharing construct. */
    int offset = 2;
#pragma omp parallel for num_threads(thread_count) private(i) lastprivate(a) shared(n) firstprivate(offset)
    for (i = 0; i < n; i++) {
        a = offset + i + 1;
        printf ("Thread %d executes loop iteration %d with value for a = %d \n", omp_get_thread_num (), i, a);
    }
    
    printf ("Value of a after the parallel for: a = %d \n", a);

    exit (EXIT_SUCCESS);
}

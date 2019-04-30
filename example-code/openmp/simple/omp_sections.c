/* OpenMP example of the section construct. 
 *
 * Compile as follows: gcc -o omp_sections omp_sections.c -fopenmp -std=c99 -O3 -Wall 
 *
 * Author: Naga Kandasamy
 * Date created: April 15, 2011
 * Date modified: April 21, 2019
 *  */

#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

void function_a (void);
void function_b (void);
void function_c (void);

int
main (int argc, char **argv)
{
  if (argc != 2) {
      printf ("Usage: %s num-threads\n", argv[0]);
      printf ("num-threads: Number of threads to create\n");
      exit (EXIT_FAILURE);
  }
  
  int thread_count = atoi (argv[1]);

  /* Start of parallel region. The SECTIONS directive is a non-iterative work-sharing construct, specifying that the enclosed 
   * section(s) of code are to be divided among the threads in the team. Independent SECTION directives are nested within a SECTIONS 
   * directive and each SECTION is executed once by a thread in the team. Different sections may be executed by different threads. 
   * It is possible for a thead to execute more than one section if it is quick enough and the implementation permits such. */
#pragma omp parallel num_threads(thread_count)
  {
#pragma omp sections
    {
#pragma omp section
      function_a ();

#pragma omp section
      function_b ();

#pragma omp section
      function_c();
    }
  } /* End of parallel region. */               
  
  exit (EXIT_SUCCESS);
}

void
function_a (void)
{
  printf ("Thread %d is executing function A\n", omp_get_thread_num ());
  return;
}

void
function_b (void)
{
  printf ("Thread %d is executing function B\n", omp_get_thread_num ());
  return;
}

void 
function_c (void)
{							
    printf ("Thread %d is executing function C\n", omp_get_thread_num());
    return;
}

/* Program to illustrate the creation of multiple threads. 
 *
 * Compile as follows: gcc -o multiple_threads multiple_threads.c -O3 -std=c99 -lpthread
 * Execute as follows: ./multiple_threads num-threads
 *
 * Author: Naga Kandasamy
 * Date created; January 21, 2019
 * Date modified: April 04, 2019 
*/

#define _REENTRANT /* Make sure the library functions are MT (muti-thread) safe. */
#include <stdio.h>
#include <pthread.h>
#include <unistd.h>
#include <stdlib.h>

/* Function prototype for the thread routines. */
void *my_func (void *);
float uniform_rand (int min, int max);

/* Declare a structure data type that will be used to pass arguments to the worker threads. */
typedef struct args_for_thread_t {
    int threadID;           /* thread ID */
    int arg1;               /* First argument */
    float arg2;             /* Second argument */
    int processing_time;    /* Third argument */
} ARGS_FOR_THREAD; 

int 
main (int argc, char **argv)
{
    pthread_t main_thread;
    pthread_t *worker_thread;
    ARGS_FOR_THREAD *args_for_thread;
    int i;
   
    if (argc < 2) {
        printf ("Usage: %s num-threads\n", argv[0]);
        printf ("num-threads: Number of worker threads to create\n");
        exit (EXIT_FAILURE);
    }
    
    int num_threads = atoi (argv[1]); /* Number of threads to create. */

    main_thread = pthread_self (); 
    printf ("Main thread = %u is creating %d worker threads \n", (int) main_thread, num_threads);

    /* Create the worker threads to execute my_func that takes a structure as an argument. */
    worker_thread = (pthread_t *) malloc (num_threads * sizeof (pthread_t));
    for (i = 0; i < num_threads; i++) {
        /* Allocate memory for the structure that will be used to pack the arguments. */
        args_for_thread = (ARGS_FOR_THREAD *) malloc (sizeof (ARGS_FOR_THREAD)); 
        args_for_thread->threadID = i; /* Fill the structure with some dummy arguments. */
        args_for_thread->arg1 = uniform_rand (0, 5); 
        args_for_thread->arg2 = uniform_rand (5, 10);
        args_for_thread->processing_time = 2; 
              
        if ((pthread_create (&worker_thread[i], NULL, my_func, (void *) args_for_thread)) != 0) {
            printf ("Cannot create worker thread \n");
            exit (EXIT_FAILURE);
        }	  
    }
	  
    /* Wait for all the worker threads to finish. */	  
    for (i = 0; i < num_threads; i++)
        pthread_join (worker_thread[i], NULL);

    printf ("Main thread exiting \n");
    pthread_exit ((void *) main_thread);
}


/* Function that will be executed by all the worker threads. */
void *
my_func (void *this_arg)
{
    ARGS_FOR_THREAD *args_for_me = (ARGS_FOR_THREAD *) this_arg; /* Typecast to appropriate type. */

    /* Simulate some processing. */
    printf ("Thread %d is using args %d and %f \n", args_for_me->threadID, \
            args_for_me->arg1, args_for_me->arg2);

    int processing_time = args_for_me->processing_time;
    sleep (processing_time);
    printf ("Thread %d is done \n", args_for_me->threadID);

    free ((void *) args_for_me); /* Free up the structure. */
    pthread_exit (NULL);
}

/* Function to return a random number uniformly distributed between min and max */
float 
uniform_rand (int min, int max)
{
    return min + (max - min) * rand()/(float)RAND_MAX;
}






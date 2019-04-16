/* Example code that shows how to use semaphores to implement a barrier.
 * 
 * Compile as follows: gcc -o barrier_synchronization_with_semaphores barrier_synchronization_with_semaphores.c -std=c99 -Wall -lpthread -lm
 * 
 * Author: Naga Kandasamy
 * Date created: April 5, 2011 
 * Date modified: April 8, 2019
 *
 *  */

#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <math.h>
#include <semaphore.h>
#include <pthread.h>

/* Structure that defines the barrier. */
typedef struct barrier_s {
    sem_t counter_sem;          /* Protects access to the counter. */
    sem_t barrier_sem;          /* Signals that barrier is safe to cross. */
    int counter;                /* The value itself. */
} barrier_t;

/* Structure that defines the data passed to each thread. */
typedef struct thread_data_s {
    int tid;                    /* Thread identifier. */
    int num_threads;            /* Number of threads in the pool. */
    int num_iterations;         /* Number of iterations executed by each thread. */
} thread_data_t;

barrier_t barrier;  

void *my_thread (void *);
void barrier_sync (barrier_t *, int, int);

int 
main (int argc, char **argv)
{   

    if (argc < 3) {
        printf ("Usage: %s num-threads num-iterations\n", argv[0]);
        printf ("num-threads: Number of threads\n");
        printf ("num-iterations: Number of iterations executed by each thread\n");
        exit (EXIT_FAILURE);
    }
		  
    int num_threads = atoi (argv[1]);
    int num_iterations = atoi (argv[2]);

    /* Initialize the barrier data structure. */
    barrier.counter = 0;
    sem_init (&barrier.counter_sem, 0, 1); /* Initialize the semaphore protecting the counter to unlocked. */
    sem_init (&barrier.barrier_sem, 0, 0); /* Initialize the semaphore protecting the barrier to locked. */

    /* Create the threads. */
    int i;
    pthread_t *tid = (pthread_t *) malloc (sizeof (pthread_t) * num_threads);
    thread_data_t *thread_data = (thread_data_t *) malloc (sizeof (thread_data_t) * num_threads);

    for (i = 0; i < num_threads; i++) {
        thread_data[i].tid = i;
        thread_data[i].num_threads = num_threads;
        thread_data[i].num_iterations = num_iterations;
        pthread_create (&tid[i], NULL, my_thread, (void *) &thread_data[i]);
    }

    /* Wait to reap the threads that we have created. */
    for (i = 0; i < num_threads; i++)
        pthread_join (tid[i], NULL);
	  
    pthread_exit (NULL);
}

/* The function executed by the threads. */
void *
my_thread (void *args)
{
    thread_data_t *thread_data = (thread_data_t *) args;
    
    for (int i = 0; i < thread_data->num_iterations; i++) {
        printf ("Thread number %d is processing for iteration %d\n", thread_data->tid, i);
        sleep (ceil (rand()/(float) RAND_MAX * 10)); /* Simulate some processing. */

        printf ("Thread %d is at the barrier\n", thread_data->tid);
        barrier_sync (&barrier, thread_data->tid, thread_data->num_threads); /* Wait here for all threads to catch up before starting the next iteration. */
    }
    
    pthread_exit (NULL);
}

/* The function that implements the barrier synchronization. */
void 
barrier_sync(barrier_t *barrier, int tid, int num_threads)
{
    sem_wait (&(barrier->counter_sem));

    /* Check if all threads before us, that is num_threads - 1 threads have reached this point. */	  
    if (barrier->counter == (num_threads - 1)) {
        barrier->counter = 0; /* Reset the counter. */
        sem_post (&(barrier->counter_sem)); 
					 
        /* Signal the blocked threads that it is now safe to cross the barrier. */
        printf ("Thread number %d is signalling other threads to proceed\n", tid); 
        for (int i = 0; i < (num_threads - 1); i++)
            sem_post (&(barrier->barrier_sem));
    } 
    else {
        barrier->counter++;
        sem_post (&(barrier->counter_sem));
        sem_wait (&(barrier->barrier_sem)); /* Block on the barrier semaphore. */
    }
}




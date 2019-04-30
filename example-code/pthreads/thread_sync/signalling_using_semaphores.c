/* Example code that shows how to use semaphores to signal between threads. 
 * 
 * Compile as follows: gcc -o signalling_using_semaphores signalling_using_semaphores.c -std=c99 -Wall -lpthread
 *
 * Author: Naga Kandasamy
 * Date created: April 5, 2011
 * Date modified: April 8, 2019
 *
 *  */

#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <semaphore.h>
#include <pthread.h>

typedef struct critical_section_s {
    sem_t semaphore;                    /* Signals a change to the value. */
    pthread_mutex_t mutex;              /* Protects access to the value. */
    int value;                          /* The value itself. */
} critical_section_t;

critical_section_t data; 

/* Function prototypes. */
void *my_thread(void *);

int 
main (int argc, char **argv)
{
    pthread_t thread_id;
		 
    /* Initialize the semaphore and value within data. */
    data.value = 0;
    sem_init (&data.semaphore, 0, 0); /* Semaphore is not shared among processes and is initialized to 0. */
    pthread_mutex_init (&data.mutex, NULL); /* Initialize the mutex. */

    /* Create a thread. */
    pthread_create (&thread_id, NULL, my_thread, NULL);
		 		  	  
    pthread_mutex_lock (&data.mutex);
    /* Test the predicate, that is the data and wait for a condition to become true if neccessary. */	  
    if (data.value == 0) {
        printf ("Value of data = %d. Waiting for someone to change it and signal me.\n", data.value);
        pthread_mutex_unlock (&data.mutex);
        sem_wait (&data.semaphore); /* Probe the semaphore, P(). */
    }

    /* Someone changed the value of data to one. */
    pthread_mutex_lock (&data.mutex);
    if (data.value != 0) {
        printf ("Change in variable state was signalled.\n");
        printf ("Value of data = %d.\n", data.value);
    }
    pthread_mutex_unlock (&data.mutex);

    pthread_join (thread_id, NULL);
    pthread_exit (NULL);
}

/* The function executed by the thread. */
void 
*my_thread (void *args)
{
    sleep (5);
		  
    pthread_mutex_lock (&data.mutex);	  
    data.value = 1;
    pthread_mutex_unlock (&data.mutex);
		  	  
    sem_post (&data.semaphore); /* Signal the change to the blocked thread. */

    pthread_exit (NULL);
}

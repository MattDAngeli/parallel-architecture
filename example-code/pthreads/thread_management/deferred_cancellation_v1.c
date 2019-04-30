/* Example code to illustrate the use of deferred thread cancellation. 
 * 
 * Compile as follows: gcc -o deferred_cancellation_v1 deferred_cancellation_v1.c -O3 -std=c99 -lpthread 
 * Execute as follows: ./deferred_cancellation_v1
 *
 * Author: Naga Kandasamy
 * Date created: April 10, 2011
 * Date modified: April 6, 2019
 *
 */
 
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <pthread.h>

static int counter;

void *
my_thread (void *args)
{
    /* Stay in an infinite loop and check for a cancellation request from the creator 
     * every 1000 iterations. */	  
    for (counter = 0; ; counter++){
        if ((counter % 1000) == 0){
            pthread_testcancel ();
        }
    }
} 

int 
main (int argc, char **argv)
{
    pthread_t thread_id;
    void *result;

    /* Create the thread. The thread is always created with its cancellation flag set to 
     * DEFERRED as the default setting. */
    printf ("Creating the worker thread. \n");
    pthread_create (&thread_id, NULL, my_thread, NULL);

    /* Simulate some processing in the main thread. */
    sleep (2);

    printf ("Cancelling the thread. \n");
    pthread_cancel (thread_id);

    /* Wait for my_thread to terminate. */
    pthread_join (thread_id, &result);
    
    if (result == PTHREAD_CANCELED)
        printf ("Thread was cancelled at iteration %d. \n", counter);
    else
        printf ("Thread was not cancelled. \n");

    pthread_exit (NULL);
}


/* Example code to illustrate the use of deferred thread cancellation. 
 * This code illustrates how to enable/disable cancellation at certain points in the code. 
 * 
 * Compile as follows: gcc -o deferred_cancellation_v2 deferred_cancellation_v2.c -lpthread -std=c99 
 *
 * Author: Naga Kandasamy
 * Date created: April 10, 2011
 * Date modified; April 6, 2019
 *  */
 
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <pthread.h>

static int counter;

/* Disable the cancellation after every 1000 iterations and sleep for 1 second. 
 * Check for a cancellation request every 2000 iterations. */
void *
my_thread(void *args)
{
    int state; 
    for (counter = 0; ; counter++) {
        if ((counter % 1000) == 0) {
            /* Disable cancellation. Store the old state (CANCEL_DEFERRED). */
            pthread_setcancelstate (PTHREAD_CANCEL_DISABLE, &state); 								
            sleep (1); /* Perhaps you do not wish to be cancelled when executuing this code portion. */ 
            pthread_setcancelstate (state, &state); /* Re-enable cancellation. */
        }
					 
        if ((counter % 1500) == 0){					
            pthread_testcancel (); /* Test for cancellation request. */
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
    printf ("Creating the thread. \n");
    pthread_create (&thread_id, NULL, my_thread, NULL);

    /* Simulate some processing. */
    sleep (2);

    printf ("Cancelling the thread. \n");
    pthread_cancel (thread_id);

    pthread_join (thread_id, &result);
    if (result == PTHREAD_CANCELED)
        printf ("Thread was cancelled at iteration %d. \n", counter);
    else		
        printf ("Thread was not cancelled. \n");
	
    pthread_exit (NULL);
}


/* Example code to illustrate the use of asynchronous thread cancellation.
 * 
 * Compile as follows: gcc -o async_cancellation async_cancellation.c -O3 -std=c99 -lpthread 
 *
 * Author: Naga Kandasamy
 * Date created: April 10, 2011
 * Date modified: April 6, 2019
 * */
 
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <pthread.h>

#define SIZE 25

static int matrix_a[SIZE][SIZE];
static int matrix_b[SIZE][SIZE];
static int matrix_c[SIZE][SIZE];

/* Perform the matrix multiplication operation until we get a cancellation request. */
void *
my_thread (void *args)
{	  
    int i, j, k;
    int cancel_type;
		  
    /* Initialize the matrices with some numbers. */
    for (i = 0; i < SIZE; i++)         
        for (j = 0; j < SIZE; j++) {
            matrix_a[i][j] = i;
            matrix_b[i][j] = j;	
        }
		
    while (1) {
        printf ("Multiplying matrices. \n");

        /* Set the cacellation type to asynchronous and store the old cancellation type. */
        pthread_setcanceltype (PTHREAD_CANCEL_ASYNCHRONOUS, &cancel_type); 
        for (i = 0; i < SIZE; i++)
            for (j = 0; j < SIZE; j++) {
                matrix_c[i][j] = 0;
                for (k = 0; k < SIZE; k++)
                    matrix_c[i][j] += matrix_a[i][k] * matrix_b[k][j];
            }
        
        pthread_setcanceltype (cancel_type, &cancel_type); /* Reset cancellation type. */
							  
        /* Copy the result matrix into matrix_a to start over again. */		  
        for (i = 0; i < SIZE; i++)			 
            for (j = 0 ; j < SIZE; j++)			
                matrix_a[i][j] = matrix_c[i][j];
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
        printf ("Thread was cancelled. \n");	  
    else
        printf ("Thread was not cancelled. \n");

    pthread_exit (NULL);
}


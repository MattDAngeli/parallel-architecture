/* Example code to illustrate the use of a cleanup handler that is invoked upon thread cancellation.
 * 
 * Compile as follows: gcc -o cleanup_v2 cleanup_v2.c -lpthread -std=c99 
 *
 * Author: Naga Kandasamy
 * Date created: April 10, 2011
 * Date modified: April 6, 2019 
 */
 
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <pthread.h>

/* Structure defining the team of sub-contractor workers. */
typedef struct team_s {
    int num_workers;
    pthread_t *worker;
} team_t;

/* The cleanup handler associated with the contractor thread. */
void 
cleanup_handler (void *args)
{
    team_t *team = (team_t *) args;
	
    for (int i = 0; i < team->num_workers; i++) {
        pthread_cancel (team->worker[i]);
        pthread_detach (team->worker[i]); /* We don't wait for the sub-contractor thread to join us. */
        printf ("Cleanup: Cancelled sub-contractor thread %d in the team. \n", i);
    }
}

/* Simulate some processing while checking for cancellation requests every 1000 iterations. */
void *
sub_contractor (void *args)
{
    int counter;
    for (counter = 0; ; counter++)
        if ((counter % 1000))
            pthread_testcancel ();
}

/* The contractor creates a team of threads that does some processing on its behalf. */
void 
*contractor (void *args)
{
    /* Create and populate the team of workers. */
    team_t *team = (team_t *) malloc (sizeof (team_t));
    team->num_workers = (int) args;
    team->worker = (pthread_t *) malloc (team->num_workers * sizeof (pthread_t));

    int i;
    for (i = 0; i < team->num_workers; i++) {
        printf ("Contractor: Creating sub-contractor thread %d. \n", i);
        pthread_create (&team->worker[i], NULL, sub_contractor, NULL);
    }

    pthread_cleanup_push (cleanup_handler, (void *) team);
	
    for (i = 0; i < team->num_workers; i++)
        pthread_join (team->worker[i], NULL);

    pthread_cleanup_pop (0);
} 

int 
main (int argc, char **argv)
{
    pthread_t thread_id;
    void *result;

    if (argc < 2) {
        printf ("Usage: %s num-threads\n", argv[0]);
        printf ("num-threads: Number of worker threads to create\n");
        exit (EXIT_FAILURE);
    }
    /* Create the contractor thread. The thread is created with its cancellation flag set to 
     * DEFERRED as the default setting. */
    printf ("Main: Creating the contractor thread. \n");
    int num_threads = atoi (argv[1]);
    pthread_create (&thread_id, NULL, contractor, (void *) num_threads);

    /* Simulate some processing. */
    sleep (5);

    printf ("Main: Cancelling the contractor thread. \n");
    pthread_cancel (thread_id);

    pthread_join (thread_id, &result);
    if (result == PTHREAD_CANCELED)
        printf ("Main: Contractor thread was cancelled. \n");
    else
        printf ("Contractor thread was not cancelled. \n");

    pthread_exit (NULL);
}


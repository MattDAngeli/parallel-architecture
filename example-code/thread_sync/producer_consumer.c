/* Example code illustrating the producer consumer synchronization problem. 
 * 
 * Compile as follows: gcc -o producer_consumer producer_consumer.c -std=c99 -Wall -lpthread -lm
 *  
 * Author: Naga Kandasamy
 * Date created: July 15, 2011 
 * Date modified: April 8, 2019
 * 
 */
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <semaphore.h>
#include <pthread.h>
#include <math.h>
#include <sys/time.h>

int all_done = 0;

/* Define the queue data structure type. */
typedef struct queue_s {
    int size;               /* Size of the queue. */
    int *buffer;            /* Buffer storing items. */
    int counter;            /* Number of items currently in queue. */
    int in, out;         /* Indices for the producer and the consumer. */
    pthread_mutex_t lock;   /* Lock to protect the shared buffer. */
    sem_t full, empty;      /* Semaphores to signal full/empty queue conditions. */ 
} queue_t;


/* Function prototypes for the functions to be executed by the threads. */
void *producer (void *args);
void *consumer (void *args);

/* Fuction prototypes for queue operations. */
queue_t *create_queue (int);
void delete_queue (queue_t *);
void add_to_queue (queue_t *, int);
int remove_from_queue (queue_t *);

/* Other helper functions. */
int UD (int, int); /* Return a random integer from a uniform distribution. */

int 
main (int argc, char **argv)
{
    if (argc < 2) {
        printf ("Usage: %s queue-size\n", argv[0]);
        printf ("queue-size: Number of queue entries\n");
        exit (EXIT_FAILURE);
    }
    
    /* Create and initialize the queue data structure. */
    int size = atoi (argv[1]);
    queue_t *queue = create_queue (size);
    if (queue == NULL) {
        printf ("Error creating the queue data structure. Exiting.\n");
        exit (EXIT_FAILURE);
    }
		  
    /* Create the producer and consumer threads. */
    srand (time(NULL));
    pthread_t producer_id, consumer_id;
    pthread_create (&producer_id, NULL, producer, (void *) queue);
    pthread_create (&consumer_id, NULL, consumer, (void *) queue);
		  
    /* Wait for the producer and consumer threads to finish and join the main thread. */
    pthread_join (producer_id, NULL);
    pthread_join (consumer_id, NULL);

    delete_queue (queue);
    pthread_exit (NULL);
}


/* Function executed by the producer thread. */
void *
producer (void *args)
{
    int i = 0;
    queue_t *queue = (queue_t *) args; /* Typecast to the queue data structure. */
    
    int num_items_to_be_produced = UD (10, 20); 
    printf ("Producer: I will produce %d items\n", num_items_to_be_produced);

    while (num_items_to_be_produced-- > 0) {
        int item = UD (5, 10); /* Produce an item which simulates some processing time. */
        i++;

        /* Add the item to queue. */
        printf ("Producer: Checking if queue is full\n");
        sem_wait (&(queue->full)); /* Block here if the queue is full. */
        pthread_mutex_lock (&(queue->lock));         					 
        printf ("Producer: adding item %d with processing time of %d. \n", i, item);
        add_to_queue (queue, item);
        pthread_mutex_unlock (&(queue->lock));
        sem_post (&(queue->empty)); /* Signal consumer in case it's blocked on an empty queue. */

        sleep (UD (2, 5)); /* The producer sleeps for some random time. */
    }

	all_done = 1; /* Producer is all done. Going home. */
    sem_post (&(queue->empty)); /* One last post to unblock consumer. */

    pthread_exit (NULL);
}

/* Function executed by the consumer thread. */
void *
consumer (void *args)
{
    queue_t *queue = (queue_t *) args;
    int num_items_consumed = 0;

    while (1) {
        printf ("Consumer: Checking if queue has items in it\n");
        sem_wait (&(queue->empty)); /* Block if queue is empty. */
        if ((all_done == 1) && (queue->counter == 0)) {
            printf ("Consumer: All done. Comsumed %d items.\n", num_items_consumed);
            pthread_exit (NULL);
        }
        
        pthread_mutex_lock (&(queue->lock));
        int item = remove_from_queue (queue);
        pthread_mutex_unlock (&(queue->lock));

        sem_post (&(queue->full)); /* Signal producer in case it's blockedon a full queue. */

        num_items_consumed++;
        printf ("Consumer: processing item %d from the queue with a processing time of %d\n", num_items_consumed, item);
        sleep (item); /* Simulate some processing. */
    }
		  
    pthread_exit (NULL);
}


/* Function adds an item to the queue. */
void 
add_to_queue (queue_t *queue, int item)
{
    queue->buffer[queue->in] = item; 
    queue->in++;
    queue->counter++;
    
    if (queue->in == queue->size) 
        queue->in = 0; /* Wrap around the circular buffer. */
		  
    return;
}

/* Function removes an item from the queue. */
int 
remove_from_queue (queue_t *queue)
{
    int item = queue->buffer[queue->out];
    queue->out++;
    queue->counter--;
		  
    if(queue->out == queue->size)
        queue->out = 0; /* Warp around the circular buffer. */
		  
    return item;
}

/* Function to create and initialize the queue. */
queue_t *
create_queue (int size)
{
    queue_t *queue = (queue_t *)malloc(sizeof(queue_t));
    if(queue == NULL) return NULL;

    queue->size = size;
    queue->buffer = (int *) malloc (sizeof (int) * size);
    if (queue->buffer == NULL) return NULL;
	  
    /* Initialize the other members of the structure. */
    queue->counter = 0;
    queue->in = queue->out = 0;
    pthread_mutex_init (&(queue->lock), NULL);
    sem_init (&(queue->full), 0, size);
    sem_init (&(queue->empty), 0, 0);

    return queue;
}

/* Function to delete the queue struture. */
void 
delete_queue (queue_t *queue)
{
    pthread_mutex_destroy (&(queue->lock));
    sem_destroy (&(queue->full));
    sem_destroy (&(queue->empty));
  
    free ((void *) queue->buffer);
    free ((void *) queue);

    return;
}

/* Returns a random number uniformly distributed between min and max. */
int 
UD (int min, int max)
{
    return (int) floor (min + (max - min + 1)*(rand ()/(float) RAND_MAX));
}


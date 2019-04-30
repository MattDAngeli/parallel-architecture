/* Program calculates the vector dot product of two vectors A and B, P = A.B, using pthreads. 
 *
 * This version shows how to use the concept of striding to process very 
 * large arrays.This code also illustrates the increased cache misses affecting overall performance on a shared 
 * memory CPU. This technique works fine on the GPU, however. So, it is important to understand the underlying 
 * machine architecture well in order to write high-performance code.  

 *  
 * Compile as follows: gcc -o vector_dot_product_v1 vector_dot_product_v1.c -O3 -Wall -std=c99 -lpthread -lm
 *
 * Author: Naga Kandasamy
 * Date created: September 30, 2014
 * Date modified: April 6, 2019 
 *  */


#define _REENTRANT /* Make sure the library functions are MT (muti-thread) safe. */
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <math.h>
#include <pthread.h>

/* Data structure defining what to pass to each worker thread. */
typedef struct thread_data_s {
    int tid;                        /* The thread ID. */
    int num_threads;                /* Number of threads in the pool. */
    int num_elements;               /* Number of elements in the vector. */
    float *vector_a;                /* Pointer to vector_a. */
    float *vector_b;                /* Pointer to vector_b. */
    double *sum;                    /* Pointer to shared variable sum. */
    pthread_mutex_t *mutex_for_sum; /* Lock for the shared sum variable. */
} thread_data_t;


/* Function prototypes. */
double compute_gold (float *, float *, int);
double compute_using_pthreads (float *, float *, int, int);
void *dot_product (void *);
void print_args (thread_data_t *);


int 
main (int argc, char **argv)
{
	if (argc < 3) {
		printf ("Usage: %s num-elements num-threads\n", argv[0]);
        printf ("num-elements: The number of elements in the input vectors\n");
        printf ("num-threads: The number of threads\n");
		exit(EXIT_FAILURE);
	}
	
    int num_elements = atoi (argv[1]); 
    int num_threads = atoi (argv[2]);

	/* Create the vectors A and B and fill them with random numbers between [-.5, .5]. */
	float *vector_a = (float *) malloc (sizeof (float) * num_elements);
	float *vector_b = (float *) malloc (sizeof (float) * num_elements); 
	srand (time(NULL)); /* Seed the random number generator with the current system time as the seed. */
	for (int i = 0; i < num_elements; i++) {
		vector_a[i] = rand()/(float) RAND_MAX - 0.5;
		vector_b[i] = rand()/(float) RAND_MAX - 0.5;
	}

	/* Compute the dot product using the reference, single-threaded solution. */
	struct timeval start, stop;	
	gettimeofday (&start, NULL);
	
    double reference = compute_gold (vector_a, vector_b, num_elements); 
	
    gettimeofday (&stop, NULL);
	printf ("Reference solution = %f. \n", reference);
	printf ("Execution time = %fs. \n", (float)(stop.tv_sec - start.tv_sec +\
                (stop.tv_usec - start.tv_usec)/(float)1000000));
	printf ("\n");

	/* Compute the dot product using the multi-threaded solution. */
	gettimeofday (&start, NULL);

	double mt_result = compute_using_pthreads (vector_a, vector_b, num_elements, num_threads);
	
    gettimeofday (&stop, NULL);
	printf ("Pthread solution = %f. \n", mt_result);
	printf ("Execution time = %fs. \n", (float)(stop.tv_sec - start.tv_sec +\
                (stop.tv_usec - start.tv_usec)/(float)1000000));
	printf ("\n");

	/* Free memory here. */ 
	free ((void *) vector_a);
	free ((void *) vector_b);

	pthread_exit (NULL);
}

/* This function computes the reference soution using a single thread. The sum is computed in double precision. */
double 
compute_gold (float *vector_a, float *vector_b, int num_elements)
{
	double sum = 0.0;
	for (int i = 0; i < num_elements; i++)
        sum += vector_a[i] * vector_b[i];
	
	return sum;
}


/* This function computes the dot product using pthreads. */
double 
compute_using_pthreads (float *vector_a, float *vector_b, int num_elements, int num_threads)
{
    pthread_t *thread_id = (pthread_t *) malloc (num_threads * sizeof (pthread_t)); /* Data structure to store the thread IDs. */
    pthread_attr_t attributes; /* Thread attributes. */
    pthread_attr_init (&attributes); /* Initialize the thread attributes to the default values. */
		  
    /* Allocate memory for the required data structures and create the worker threads. */
    int i;
    double sum = 0;
    pthread_mutex_t mutex_for_sum;
    pthread_mutex_init (&mutex_for_sum, NULL);

    thread_data_t *thread_data = (thread_data_t *) malloc (sizeof (thread_data_t) * num_threads);
    for (i = 0; i < num_threads; i++) {
        thread_data[i].tid = i; 
        thread_data[i].num_threads = num_threads;
        thread_data[i].num_elements = num_elements; 
        thread_data[i].vector_a = vector_a; 
        thread_data[i].vector_b = vector_b;
        thread_data[i].sum = &sum;
        thread_data[i].mutex_for_sum = &mutex_for_sum;
    }

    for (i = 0; i < num_threads; i++)
        pthread_create (&thread_id[i], &attributes, dot_product, (void *) &thread_data[i]);
					 
    /* Wait for the workers to finish. */
    for (i = 0; i < num_threads; i++)
        pthread_join (thread_id[i], NULL);
		 
    /* Free dynamically allocated data structures. */
    free ((void *) thread_data);

    return sum;
}

/* This function is executed by each thread to compute the overall dot product. */
void *
dot_product (void *args)
{
    /* Typecast the argument to a pointer to the thread_data_t structure. */
    thread_data_t *thread_data = (thread_data_t *) args;

    int offset = thread_data->tid;
    int stride = thread_data->num_threads;
    double partial_sum = 0.0;

    while (offset < thread_data->num_elements) {
        partial_sum += thread_data->vector_a[offset] * thread_data->vector_b[offset];
        offset += stride;
    }

    /* Accumulate partial sums into the shared sum variable. */ 
    pthread_mutex_lock (thread_data->mutex_for_sum);
    *(thread_data->sum) += partial_sum;
    pthread_mutex_unlock (thread_data->mutex_for_sum);
		  
    pthread_exit (NULL);
}

/* Helper function. */
void 
print_args (thread_data_t *thread_data)
{
    printf ("Thread ID: %d\n", thread_data->tid);
    printf ("Numer of threads: %d\n", thread_data->num_threads);
    printf ("Num elements: %d\n", thread_data->num_elements); 
    printf ("Address of vector A on heap: %p\n", &(thread_data->vector_a));
    printf ("Address of vector B on heap: %p\n", &(thread_data->vector_b));
    printf ("\n");
}


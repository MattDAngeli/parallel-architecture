/* Vector matrix multiplication AX = Y.
 * Here A is an m x n matrix, A is a n x 1 vector, and Y is the m x 1 resulting vector.
 *
 * Compile as follows: gcc -o mult mult.c -O3 -Wall -std=c99 -lpthread -lm
 *
 * Naga Kandasamy
 * Date created April 6, 2019
 *
 */

#define _REENTRANT /* Make sure the library functions are MT (muti-thread) safe. */
#include <stdio.h>
#include <pthread.h>
#include <unistd.h>
#include <stdlib.h>
#include <sys/time.h>
#include <math.h>
#include <errno.h>

/* Define the per-thread data structures. */
typedef struct matrix_s {
    int num_rows; /* Number of rows. */
    int num_cols; /* Number of columns. */
    float *val;
} matrix_t;

typedef struct thread_data_s {
    int tid;            /* Thread identifier. */
    int num_threads;    /* Number of threads in the worker pool. */
    int chunk_size;     /* Size of data to be processed by thread. */
    matrix_t *A;       /* The A matrix. */
    matrix_t *X;       /* The x matrix. */
    matrix_t *Y;       /* The result matrix, y */
} thread_data_t;

/* Function prototypes. */
void print_matrix (matrix_t *);
void compute_gold (matrix_t *, matrix_t *, matrix_t *);
int compute_using_pthreads_v1 (matrix_t *, matrix_t *, matrix_t *, int);
int compute_using_pthreads_v2 (matrix_t *, matrix_t *, matrix_t *, int);
void *mt_mult_v1 (void *);
void *mt_mult_v2 (void *);
int check_results (float *, float *, int, float);


int 
main (int argc, char **argv)
{
    struct timeval start, stop;	
    int i;
    
    srand(time(NULL)); /* Seed the random number generator. */

    if (argc < 4) {
        printf ("Usage: %s num-rows num-columns num-threads\n", argv[0]);
        printf ("num-rows: Number of rows in matrix\n");
        printf ("num-columns: Number of columns in matrix\n");
        printf ("num-threads: The number of threads\n");
        exit (EXIT_FAILURE);
    }

    /* Create the A and x as matrices containing random FP values between -0.5 and 0.5. */
    matrix_t *A = (matrix_t *) malloc (sizeof (matrix_t));
    A->num_rows = atoi (argv[1]);
    A->num_cols = atoi (argv[2]);
    int num_elements = A->num_rows * A->num_cols;
    A->val = (float *) malloc (num_elements * sizeof (float));
    for (i = 0; i < num_elements; i++)
        A->val[i] = -0.5 + rand()/(float) RAND_MAX;

    matrix_t *X = (matrix_t *) malloc (sizeof (matrix_t));
    X->num_rows = atoi (argv[2]);
    X->num_cols = 1;
    X->val = (float *) malloc (X->num_rows * sizeof (float));
    for (i = 0; i < X->num_rows; i++) {
        X->val[i] = -0.5 + rand ()/(float) RAND_MAX; 
    }

    /* Create the output matrix to store the reference result. */
    matrix_t *Y_ref = (matrix_t *) malloc (sizeof (matrix_t));
    Y_ref->num_rows = atoi (argv[1]);
    Y_ref->num_cols = 1;
    Y_ref->val = (float *) malloc (Y_ref->num_rows * sizeof (float));

    /* Calculate the reference result using a single-threaded version. */
    printf("Performing AX = Y using the single-threaded version.\n");
	gettimeofday (&start, NULL);
    
    compute_gold (A, X, Y_ref);

    gettimeofday (&stop, NULL);
	printf ("Execution time = %fs. \n", (float)(stop.tv_sec - start.tv_sec +\
                (stop.tv_usec - start.tv_usec)/(float)1000000));

    /* Calculate the result using pthreads. Version 1. */
    int num_threads = atoi (argv[3]);

    matrix_t *Y_mt_1 = (matrix_t *) malloc (sizeof (matrix_t));
    Y_mt_1->num_rows = atoi (argv[1]);
    Y_mt_1->num_cols = 1;
    Y_mt_1->val = (float *) malloc (Y_mt_1->num_rows * sizeof (float));
    
    printf("Performing AX = Y using pthreads. Version 1.\n");
	gettimeofday (&start, NULL);

    if (compute_using_pthreads_v1 (A, X, Y_mt_1, num_threads) != 0) {
        exit (EXIT_FAILURE);
    }

    gettimeofday (&stop, NULL);
    printf ("Execution time = %fs. \n", (float)(stop.tv_sec - start.tv_sec +\
                (stop.tv_usec - start.tv_usec)/(float)1000000));

    /* Check the results for correctness. */
    float eps = 1e-6;
    if (check_results (Y_ref->val, Y_mt_1->val, Y_ref->num_rows, eps) == 0)
        printf ("TEST PASSED\n");
    else 
        printf ("TEST FAILED\n");

    /* Calculate the result using pthreads. Version 2. */
    matrix_t *Y_mt_2 = (matrix_t *) malloc (sizeof (matrix_t));
    Y_mt_2->num_rows = atoi (argv[1]);
    Y_mt_2->num_cols = 1;
    Y_mt_2->val = (float *) malloc (Y_mt_2->num_rows * sizeof (float));
    
    printf("Performing AX = Y using pthreads. Version 2.\n");
    gettimeofday (&start, NULL);
    
    if (compute_using_pthreads_v2 (A, X, Y_mt_2, num_threads) != 0) {
        exit (EXIT_FAILURE);
    }

    gettimeofday (&stop, NULL);
    printf ("Execution time = %fs. \n", (float)(stop.tv_sec - start.tv_sec +\
                (stop.tv_usec - start.tv_usec)/(float)1000000));

    if (check_results (Y_ref->val, Y_mt_2->val, Y_ref->num_rows, eps) == 0)
        printf ("TEST PASSED\n");
    else 
        printf ("TEST FAILED\n");

    /* Free up data structures and exit. */
    free ((void *) A->val);
    free ((void *) X->val);
    free ((void *) Y_ref->val);
    free ((void *) Y_mt_1->val);
    free ((void *) Y_mt_2->val);
    exit (EXIT_SUCCESS);
}

/* The reference implementation of AX = Y. */
void 
compute_gold (matrix_t *A, matrix_t *X, matrix_t *Y){
    int i, j;

    double sum;
    for (i = 0; i < A->num_rows; i++) {
        sum = 0.0;
        for (j = 0; j < A->num_cols; j++) {
            sum += A->val[i * A->num_cols + j] * X->val[j];
        }
        Y->val[i] = sum;
    }
}

   
/* Multi-threaded implementation of AX = Y. This version chunks up the output elements 
 * for each thread to process.
 */

void *
mt_mult_v1 (void *args)
{
    thread_data_t *thread_data = (thread_data_t *) args;

    int i, j;
    double sum;

    if (thread_data->tid < (thread_data->num_threads - 1)) { /* Threads 0 through n - 2 process chunk_size output elements. */
        for (i = thread_data->tid * thread_data->chunk_size; i < (thread_data->tid + 1) * thread_data->chunk_size; i++) {
            sum = 0.0;
            for (j = 0; j < thread_data->A->num_cols; j++) {
                sum += thread_data->A->val[i * thread_data->A->num_cols + j] * thread_data->X->val[j]; 
            }
            thread_data->Y->val[i] = sum;
        } 
    }
    else { /* The last thread may have to process more than chunk_size output elements. */
        for (i = thread_data->tid * thread_data->chunk_size; i < thread_data->Y->num_rows; i++) {
            sum = 0.0;
            for (j = 0; j < thread_data->A->num_cols; j++) {
                sum += thread_data->A->val[i * thread_data->A->num_cols + j] * thread_data->X->val[j]; 
            }
            thread_data->Y->val[i] = sum;
        } 
    }

    free ((void *) thread_data);
    pthread_exit (NULL);
}
    
int 
compute_using_pthreads_v1 (matrix_t *A, matrix_t *X, matrix_t *Y, int num_threads)
{
    int i;
    pthread_t *worker = (pthread_t *) malloc (num_threads * sizeof (pthread_t));
    thread_data_t *thread_data;
    
    int chunk_size = (int)floor(Y->num_rows/(float)num_threads);
    for (i = 0; i < num_threads; i++) {
        thread_data = (thread_data_t *) malloc (sizeof (thread_data_t));
        thread_data->tid = i;
        thread_data->num_threads = num_threads;
        thread_data->chunk_size = chunk_size;
        thread_data->A = A;
        thread_data->X = X;
        thread_data->Y = Y;

        if ((pthread_create (&worker[i], NULL, mt_mult_v1, (void *) thread_data)) != 0) {
            perror ("pthread_create");
            return -1;
        } 
    }

    /* Wait for all the worker threads to finish. */	  
    for (i = 0; i < num_threads; i++)
        pthread_join (worker[i], NULL);

    return 0;
}

/* Multi-threaded implementation of AX = Y. This version uses the concept of striding. 
 */

void *
mt_mult_v2 (void *args)
{
    thread_data_t *thread_data = (thread_data_t *) args;
    int tid = thread_data->tid;
    int stride = thread_data->num_threads;
    int i;
    double sum;

    while (tid < thread_data->Y->num_rows) {
        sum = 0.0;
        for (i = 0; i < thread_data->A->num_cols; i++) {
            sum += thread_data->A->val[tid * thread_data->A->num_cols + i] * thread_data->X->val[i];
        }

        thread_data->Y->val[tid] = sum;
        tid += stride;
    }

    free ((void *) thread_data);
    pthread_exit (NULL);
}

int 
compute_using_pthreads_v2 (matrix_t *A, matrix_t *X, matrix_t *Y, int num_threads)
{
    int i;
    pthread_t *worker = (pthread_t *) malloc (num_threads * sizeof (pthread_t));
    thread_data_t *thread_data;
    
    for (i = 0; i < num_threads; i++) {
        thread_data = (thread_data_t *) malloc (sizeof (thread_data_t));
        thread_data->tid = i;
        thread_data->num_threads = num_threads;
        thread_data->chunk_size = 1;
        thread_data->A = A;
        thread_data->X = X;
        thread_data->Y = Y;

        if ((pthread_create (&worker[i], NULL, mt_mult_v2, (void *) thread_data)) != 0) {
            perror ("pthread_create");
            return -1;
        }
    }

    /* Wait for all the worker threads to finish. */	  
    for (i = 0; i < num_threads; i++)
        pthread_join (worker[i], NULL);
 
    return 0;
}

/* Performs an element-by-element check of the the provided vectors 
 * A and B to test if their values are within the specified threshold. 
 */
int 
check_results (float *A, float *B, int num_elements, float threshold)
{
    int i;

    for (i = 0; i < num_elements; i++){
        if (fabsf((A[i] - B[i])/A[i]) > threshold)
            return -1;
    }

    return 0;
}

/* Function to print out the contents of the given matrix. */
void 
print_matrix (matrix_t *mat)
{
    int i, j;

    for (i = 0; i < mat->num_rows; i++) {
        for (j = 0; j < mat->num_cols; j++) {
            printf ("%f ", mat->val[mat->num_cols * i + j]);
        }
        printf ("\n");
    }
}

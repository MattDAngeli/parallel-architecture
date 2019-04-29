/* Code for the Jacbi equation solver. 
 * Author: Naga Kandasamy
 * Date created: April 19, 2019
 * Date modified: April 20, 2019
 *
 * Compile as follows:
 * gcc -o solver solver.c solver_gold.c -O3 -Wall -std=c99 -lm -lpthread
 *
 * If you wish to see debug info add the -D DEBUG option when compiling the code.
 */

#include <stdio.h>
#include <string.h>
#include <malloc.h>
#include <sys/time.h>
#include <stdlib.h>
#include <pthread.h>
#include <math.h>
#include "grid.h" 

extern int compute_gold (grid_t *);
int compute_using_pthreads_jacobi (grid_t *, int);
void compute_grid_differences(grid_t *, grid_t *);
grid_t *create_grid (int, float, float);
grid_t *copy_grid (grid_t *);
void print_grid (grid_t *);
void print_stats (grid_t *);
double grid_mse (grid_t *, grid_t *);

void *compute_jacobi ( void *args );

pthread_barrier_t sync_barrier;
pthread_barrier_t sync_barrier_2;

float eps = 1e-2; // Convergence criteria

typedef struct thread_data_t {
  int tid; // Thread ID
  int n_threads;
  int chunk_size;
  grid_t *new_grid; // Pointer to new grid
  grid_t *old_grid; // Pointer to old grid

  int *done;
  pthread_mutex_t *mutex_for_done;

  int *iterations;
  pthread_mutex_t *mutex_for_iterations;

  double *diff;
  pthread_mutex_t *mutex_for_diff;

  int *n_elements;
  pthread_mutex_t *mutex_for_n_elements;

} thread_data_t;

int main (int argc, char **argv)
{	
	if (argc < 5) {
    printf ("Usage: %s grid-dimension num-threads min-temp max-temp\n", argv[0]);
    printf ("grid-dimension: The dimension of the grid\n");
    printf ("num-threads: Number of threads\n"); 
    printf ("min-temp, max-temp: Heat applied to the north side of the plate is uniformly distributed between min-temp and max-temp\n");
    exit (EXIT_FAILURE);
  }
    
  /* Parse command-line arguments. */
  int dim = atoi (argv[1]);
  int n_threads = atoi (argv[2]);
  float min_temp = atof (argv[3]);
  float max_temp = atof (argv[4]);
    
  /* Generate the grids and populate them with initial conditions. */
 	grid_t *grid_1 = create_grid (dim, min_temp, max_temp);

  /* Grid 2 should have the same initial conditions as Grid 1. */
  grid_t *grid_2 = copy_grid (grid_1); 
  
  /* Initialize the pthread_barrier */
  pthread_barrier_init( &sync_barrier, NULL, n_threads );
  pthread_barrier_init( &sync_barrier_2, NULL, n_threads );

  struct timeval start, stop;

	/* Compute the reference solution using the single-threaded version. */
	printf ("\nUsing the single threaded version to solve the grid\n");
  gettimeofday( &start, NULL );
	int num_iter = compute_gold (grid_1);
  gettimeofday( &stop, NULL );
	printf ("Convergence achieved after %d iterations\n", num_iter);
  /* Print key statistics for the converged values. */
	printf ("Printing statistics for the interior grid points\n");
  print_stats (grid_1);
  #ifdef DEBUG
    print_grid (grid_1);
  #endif
  printf("Execution time = %fs.\n", (float)(stop.tv_sec - start.tv_sec +\
    (stop.tv_usec - start.tv_usec) / (float)1000000));
	
	/* Use pthreads to solve the equation using the jacobi method. */
	printf ("\nUsing pthreads to solve the grid using the jacobi method\n");
  gettimeofday( &start, NULL );
	num_iter = compute_using_pthreads_jacobi (grid_2, n_threads);
  gettimeofday( &stop, NULL );
	printf ("Convergence achieved after %d iterations\n", num_iter);			
  printf ("Printing statistics for the interior grid points\n");
	print_stats (grid_2);
#ifdef DEBUG
    print_grid (grid_2);
#endif
  printf("Execution time = %fs.\n", (float)(stop.tv_sec - start.tv_sec +\
    (stop.tv_usec - start.tv_usec) / (float)1000000));
    
  /* Compute grid differences. */
  double mse = grid_mse (grid_1, grid_2);
  printf ("MSE between the two grids: %f\n", mse);

	/* Free up the grid data structures. */
	free ((void *) grid_1->element);	
	free ((void *) grid_1); 
	free ((void *) grid_2->element);	
	free ((void *) grid_2);

	exit (EXIT_SUCCESS);
}

/* FIXME: Edit this function to use the jacobi method of solving the equation. The final result should be placed in the grid data structure. */
int compute_using_pthreads_jacobi (grid_t *grid, int n_threads)
{		
  grid_t *old_grid = grid;
  grid_t *new_grid = grid;

  // Define structure where thread IDs will be stored
  pthread_t *thread_id = (pthread_t *) malloc( sizeof( pthread_t ) * n_threads );

  pthread_attr_t attributes; // Thread attributes
  pthread_attr_init( &attributes ); // Initialize thread attributes to default

  // Mutex setup
  pthread_mutex_t mutex_for_iterations;
  pthread_mutex_init( &mutex_for_iterations, NULL );

  pthread_mutex_t mutex_for_done;
  pthread_mutex_init( &mutex_for_done, NULL );

  pthread_mutex_t mutex_for_diff;
  pthread_mutex_init( &mutex_for_diff, NULL);

  pthread_mutex_t mutex_for_n_elements;
  pthread_mutex_init( &mutex_for_n_elements, NULL);

  // Allocate heap space for thread data and create worker threads
  int i; 
  int done = 0;
  int iterations = 0;
  int n_elements = 0;
  double diff = 0.0;

  thread_data_t *thread_data = (thread_data_t *) malloc( sizeof( thread_data_t ) * n_threads );

  int chunk_size = (int) floor( (int) grid->dim / (int) n_threads );

  for (i = 0; i < n_threads; i++) {
    thread_data[i].tid = i;
    thread_data[i].n_threads = n_threads;
    thread_data[i].chunk_size = chunk_size;
    thread_data[i].old_grid = old_grid;
    thread_data[i].new_grid = new_grid;

    thread_data[i].done = &done;
    thread_data[i].mutex_for_done = &mutex_for_done;

    thread_data[i].iterations = &iterations;
    thread_data[i].mutex_for_iterations = &mutex_for_iterations;

    thread_data[i].diff = &diff;
    thread_data[i].mutex_for_diff = &mutex_for_diff;

    thread_data[i].n_elements = &n_elements;
    thread_data[i].mutex_for_n_elements = &mutex_for_n_elements;
  }

  for (i = 0; i < n_threads; i++)
    pthread_create( &thread_id[i], &attributes, compute_jacobi, (void *) &thread_data[i] );

  for (i = 0; i < n_threads; i++)
    pthread_join( thread_id[i], NULL );

  free( (void *) thread_data );

  return iterations;
}

// Jacobi worker function
void *compute_jacobi( void *args )
{
  thread_data_t *td = (thread_data_t *) args;

  double new, old;
  grid_t *ref_grid;
  int i,j;

  while (!*(td->done)) {
    ref_grid = td->old_grid; // Create a backup pointer to the old grid

    if (td->tid < (td->n_threads - 1)) {
      // For the first threads
      for (i = 1 + td->tid * td->chunk_size; i <= (td->tid + 1) * td->chunk_size; i++) {
        for (j = 1; j < (td->old_grid->dim - 1); j++) {

          // printf("[THREAD %d] I am getting OLD\n", td->tid);
          old = td->old_grid->element[i * td->old_grid->dim + j];

          // printf("[THREAD %d] I am getting NEW\n", td->tid);
          new = 0.25 * (td->old_grid->element[(i-1) * td->old_grid->dim + j] +\
                        td->old_grid->element[(i+1) * td->old_grid->dim + j] +\
                        td->old_grid->element[i * td->old_grid->dim + (j+1)] +\
                        td->old_grid->element[i * td->old_grid->dim + (j-1)]);

          // printf("[THREAD %d] OLD: %f\tNEW: %f\n", td->tid, old, new);

          // printf("[THREAD %d] I am assigning NEW value to grid\n", td->tid);
          td->new_grid->element[i * td->old_grid->dim + j] = new;

          pthread_mutex_lock( td->mutex_for_diff );
            // printf("[THREAD %d] I locked diff\n", td->tid);
            *(td->diff) += fabs(new - old);
            // printf("[THREAD %d] I am unlocking diff\n", td->tid);
          pthread_mutex_unlock( td->mutex_for_diff );

          pthread_mutex_lock( td->mutex_for_n_elements );
            // printf("[THREAD %d] I locked n_elements\n", td->tid);
            *(td->n_elements) = *(td->n_elements) + 1;
            // printf("[THREAD %d] I am unlocking n_elements\n", td->tid);
          pthread_mutex_unlock( td->mutex_for_n_elements );
        }
      }
    } else {
      // For the last thread
      for (i = 1 + td->tid * td->chunk_size; i < (td->old_grid->dim - 1); i++) {
        for (j = 1; j < (td->old_grid->dim - 1); j++) {

          // printf("[THREAD %d] I am getting OLD\n", td->tid);
          old = td->old_grid->element[i * td->old_grid->dim + j];

          // printf("[THREAD %d] I am getting NEW\n", td->tid);
          new = 0.25 * (td->old_grid->element[(i-1) * td->old_grid->dim + j] +\
                        td->old_grid->element[(i+1) * td->old_grid->dim + j] +\
                        td->old_grid->element[i * td->old_grid->dim + (j+1)] +\
                        td->old_grid->element[i * td->old_grid->dim + (j-1)]);

          // printf("[THREAD %d] OLD: %f\tNEW: %f\n", td->tid, old, new);

          // printf("[THREAD %d] I am assigning NEW value to grid\n", td->tid);
          td->new_grid->element[i * td->old_grid->dim + j] = new;

          pthread_mutex_lock( td->mutex_for_diff );
            // printf("[THREAD %d] I locked diff\n", td->tid);
            *(td->diff) += fabs(new - old);
            // printf("[THREAD %d] I am unlocking diff\n", td->tid);
          pthread_mutex_unlock( td->mutex_for_diff );

          pthread_mutex_lock( td->mutex_for_n_elements );
            // printf("[THREAD %d] I locked n_elements\n", td->tid);
            *(td->n_elements) = *(td->n_elements) + 1;
            // printf("[THREAD %d] I am unlocking n_elements\n", td->tid);
          pthread_mutex_unlock( td->mutex_for_n_elements );
        }
      }
    }

    // printf("[THREAD %d] I got to the barrier!\n", td->tid);

    int barrier_pt = pthread_barrier_wait( &sync_barrier );

    // printf("[THREAD %d] I got past the barrier...\n", td->tid);

    if (barrier_pt == PTHREAD_BARRIER_SERIAL_THREAD) {
      // printf("[THREAD %d] I am handling syncing, apparently...\n", td->tid);
      pthread_mutex_lock( td->mutex_for_diff );
        // printf("[THREAD %d] Locking diff with value %f\n", td->tid, *(td->diff));
        *(td->diff) = *(td->diff) / (float) *(td->n_elements);
        // printf("[THREAD %d] Unlocking diff with value %f\n", td->tid, *(td->diff));
      pthread_mutex_unlock( td->mutex_for_diff );

      pthread_mutex_lock( td->mutex_for_n_elements );
        *(td->n_elements) = 0;
      pthread_mutex_unlock( td->mutex_for_n_elements);

      printf("Iteration %d. DIFF: %f.\n", *(td->iterations), *(td->diff));

      pthread_mutex_lock( td->mutex_for_iterations );
        *(td->iterations) = *(td->iterations) + 1;
        // printf("Iterations after incrementing: %d\n", *(td->iterations));
      pthread_mutex_unlock( td->mutex_for_iterations);

      if (*(td->diff) < eps) {
        pthread_mutex_lock( td->mutex_for_done );
          *(td->done) = 1;
        pthread_mutex_unlock( td->mutex_for_done );
      }

      pthread_mutex_lock( td->mutex_for_diff );
        // printf("[THREAD %d] Resetting diff\n", td->tid);
        *(td->diff) = 0.0;
      pthread_mutex_unlock( td->mutex_for_diff );

      // printf("[THREAD %d] Okay, I'm done!", td->tid);
      pthread_barrier_wait( &sync_barrier_2 );
    } else {
      // printf("[THREAD %d] Waiting for sync...\n", td->tid);
      pthread_barrier_wait( &sync_barrier_2 );
      // printf("[THREAD %d] Okay!  Proceeding\n", td->tid);
    }

    td->old_grid = td->new_grid;
    td->new_grid = ref_grid;
  }
  pthread_exit( NULL );
}

/* Create a grid with the specified initial conditions. */
grid_t *create_grid (int dim, float min, float max)
{
  grid_t *grid = (grid_t *) malloc (sizeof (grid_t));
  if (grid == NULL)
    return NULL;

  grid->dim = dim;
	printf("Creating a grid of dimension %d x %d\n", grid->dim, grid->dim);
	grid->element = (float *) malloc (sizeof (float) * grid->dim * grid->dim);
  if (grid->element == NULL)
    return NULL;

  int i, j;
	for (i = 0; i < grid->dim; i++) {
		for (j = 0; j < grid->dim; j++) {
      grid->element[i * grid->dim + j] = 0.0; 			
		}
  }

  /* Initialize the north side, that is row 0, with temperature values. */ 
  srand ((unsigned) time (NULL));
	float val;		
  for (j = 1; j < (grid->dim - 1); j++) {
    val =  min + (max - min) * rand ()/(float)RAND_MAX;
    grid->element[j] = val; 	
  }

  return grid;
}

/* Creates a new grid and copies over the contents of an existing grid into it. */
grid_t *copy_grid (grid_t *grid) 
{
  grid_t *new_grid = (grid_t *) malloc (sizeof (grid_t));
  if (new_grid == NULL)
    return NULL;

  new_grid->dim = grid->dim;
	new_grid->element = (float *) malloc (sizeof (float) * new_grid->dim * new_grid->dim);
  if (new_grid->element == NULL)
    return NULL;

  int i, j;
	for (i = 0; i < new_grid->dim; i++) {
		for (j = 0; j < new_grid->dim; j++) {
      new_grid->element[i * new_grid->dim + j] = grid->element[i * new_grid->dim + j] ; 			
		}
  }

  return new_grid;
}

/* This function prints the grid on the screen. */
void print_grid (grid_t *grid)
{
  int i, j;
  for (i = 0; i < grid->dim; i++) {
    for (j = 0; j < grid->dim; j++) {
      printf ("%f\t", grid->element[i * grid->dim + j]);
    }
    printf ("\n");
  }
  printf ("\n");
}


/* Print out statistics for the converged values of the interior grid points, including min, max, and average. */
void print_stats (grid_t *grid)
{
  float min = INFINITY;
  float max = 0.0;
  double sum = 0.0;
  int num_elem = 0;
  int i, j;

  for (i = 1; i < (grid->dim - 1); i++) {
    for (j = 1; j < (grid->dim - 1); j++) {
      sum += grid->element[i * grid->dim + j];

      if (grid->element[i * grid->dim + j] > max) 
        max = grid->element[i * grid->dim + j];

      if(grid->element[i * grid->dim + j] < min) 
        min = grid->element[i * grid->dim + j];
             
      num_elem++;
    }
  }
                    
  printf("AVG: %f\n", sum/num_elem);
	printf("MIN: %f\n", min);
	printf("MAX: %f\n", max);
	printf("\n");
}

/* Calculate the mean squared error between elements of two grids. */
double grid_mse (grid_t *grid_1, grid_t *grid_2)
{
  double mse = 0.0;
  int num_elem = grid_1->dim * grid_1->dim;
  int i;

  for (i = 0; i < num_elem; i++) 
    mse += (grid_1->element[i] - grid_2->element[i]) * (grid_1->element[i] - grid_2->element[i]);
                   
  return mse/num_elem; 
}


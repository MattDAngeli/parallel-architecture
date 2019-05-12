#ifndef _MATRIXMUL_H_
#define _MATRIXMUL_H_

/* Thread block dimensions. */
#define TILE_SIZE 32 
#define THREAD_BLOCK_SIZE 1024

/* Matrix structure declaration. */
typedef struct matrix_s {
    unsigned int num_columns;   /* Width of the matrix. */
	unsigned int num_rows;      /* Height of the matrix. */
	float* elements;            /* Pointer to the first element. */
} matrix_t;

#endif // _MATRIXMUL_H_


#ifndef _MATRIXMUL_H_
#define _MATRIXMUL_H_

/* Thread block size. */
#define MATRIX_SIZE 32

/* Matrix dimensions chosen as multiples of the thread block size for simplicity. */
#define WM MATRIX_SIZE /* Matrix M width. */
#define HM MATRIX_SIZE /* Matrix M height. */
#define WN MATRIX_SIZE /* Matrix N width. */
#define HN WM  /* Matrix N height. */
#define WP WN  /* Matrix P width. */
#define HP HM  /* Matrix P height. */

/* Matrix Structure declaration. */
typedef struct {
	/* Width of the matrix represented. */
	unsigned int width;
	/* Height of the matrix represented. */
	unsigned int height;
	/* Number of elements between the beginnings of adjacent rows in 
     * the memory layout (useful for representing sub-matrices). */
	unsigned int pitch;
	/* Pointer to the first element of the matrix represented. */
	float *elements;
} matrix;

#endif // _MATRIXMUL_H_


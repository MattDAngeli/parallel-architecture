#ifndef _MATRIX_H_
#define _MATRIX_H_

/* Matrix Structure declaration. */
#define MATRIX_SIZE 2048
typedef struct {
    unsigned int width;
    unsigned int height;
    unsigned int pitch;
    float* elements;
} matrix;

/* Define the thread block size on the GPU. */
#define TILE_SIZE 32
/* Define the BLOCK_SIZE for the blocked multiplication on the CPU. */
#define BLOCK_SIZE 32

#endif // _MATRIX_H_


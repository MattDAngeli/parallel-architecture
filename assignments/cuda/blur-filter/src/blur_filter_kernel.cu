/* Blur filter. Device code. */

#ifndef _BLUR_FILTER_KERNEL_H_
#define _BLUR_FILTER_KERNEL_H_

#include "blur_filter.h"

__global__ void blur_filter_kernel ( float *out, const float *in, int size)
{
   // Find position in the image (thread -> data mapping)
   int row = blockDim.y * blockIdx.y + threadIdx.y;
   int col = blockDim.x * blockIdx.x + threadIdx.x;

   double blur_pixel = 0.0;
   int n_neighbors = 0;

   int curr_row, curr_col;

   for (int i = -BLUR_SIZE; i < (BLUR_SIZE + 1); i++) {
      for (int j = -BLUR_SIZE; j < (BLUR_SIZE + 1); j++) {
         curr_row = row + i;
         curr_col = col + j;
         if ((curr_row > -1) && (curr_row < size) &&
               (curr_col > -1) && (curr_col < size)){
            blur_pixel += (double) in[curr_row * size + curr_col];
            n_neighbors += 1;
         }
      }
   }

   if (row >= 0 && row < size && col >= 0 && col < size)
      out[row * size + col] = (float) ((double) blur_pixel / (double) n_neighbors);

   return;

}

#endif /* _BLUR_FILTER_KERNEL_H_ */

#ifndef _CONVOLUTION_KERNEL_H_
#define _CONVOLUTION_KERNEL_H_

/* The kernel is stored in GPU  global memory in this kernel implementation. 
 * The input vector is also stored in global memory. 
 */ 
__global__ 
void convolution_kernel_v1 (float *N, float *result, float *kernel, int num_elements, int kernel_width)
{
    /* Obtain the index of the thread within the execution grid. */ 
    int i = blockIdx.x * blockDim.x + threadIdx.x; 
    float sum = 0.0;
    
    if(i >= num_elements)
        return;
    
    int N_start_point = i - (kernel_width/2); 
    for (int j = 0; j < kernel_width; j++) {
        if ((j + N_start_point >= 0) && (j + N_start_point < num_elements))
            sum += N[j + N_start_point] * kernel[j];
    }
    
    result[i] = sum;
    return;
}

/* The kernel is stored in GPU constant memory in kernel_c. 
 * The input vector is stored in global memory. 
 */
__global__ 
void convolution_kernel_v2 (float *N, float *result, int num_elements, int kernel_width)
{
    /* Obtain the index of the thread within the grid. */
    int i = blockIdx.x * blockDim.x + threadIdx.x;     
    float sum = 0.0;

    if(i >= num_elements)
        return;

    int N_start_point = i - (kernel_width/2); 
    for (int j = 0; j < kernel_width; j++) {
        if ((j + N_start_point >= 0) && (j + N_start_point < num_elements))
            sum += N[j + N_start_point] * kernel_c[j]; 
    }
    
    result[i] = sum;
    return;
}

/* Tiled convolution kernel using shared memory.
 * The kernel is stored in GPU constant memory in kernel_c.
*/
__global__ void 
convolution_kernel_tiled (float *N, float *result, int num_elements, int kernel_width)
{
    __shared__ float N_s[THREAD_BLOCK_SIZE + MAX_KERNEL_WIDTH - 1];
    
    /* Obtain the index of the thread within the grid. */   
    int i = blockIdx.x * blockDim.x + threadIdx.x; 
    
    int half_width = kernel_width/2;
    /* Load the left halo elements from the previous tile. 
     * The number of halo elements will be half_width.
     */
    int left_halo_index = (blockIdx.x - 1) * blockDim.x + threadIdx.x;
    if (threadIdx.x >= (blockDim.x - half_width)) {
        if (left_halo_index < 0) 
            N_s[threadIdx.x - (blockDim.x - half_width)] = 0.0;
        else
            N_s[threadIdx.x - (blockDim.x - half_width)] = N[left_halo_index];
    }
		  
    /* Load the center elements for the tile. */
    if (i < num_elements)			 
        N_s[half_width + threadIdx.x] = N[i];
    else 
        N_s[half_width + threadIdx.x] = 0.0;

    /* Load the right halo elements from the next tile. 
     * The number of halo elements will again be half_width.
     */
    int right_halo_index = (blockIdx.x + 1) * blockDim.x + threadIdx.x;
    if (threadIdx.x < half_width) {
        if (right_halo_index >= num_elements) 
            N_s[threadIdx.x + (blockDim.x + half_width)] = 0.0;
        else
            N_s[threadIdx.x + (blockDim.x + half_width)] = N[right_halo_index];
    }
		  
    __syncthreads();

    /* Convolve the kernel over the input elements. */
    float sum = 0.0;
    for (int j = 0; j < kernel_width; j++)
        sum += N_s[j + threadIdx.x] * kernel_c[j];
		  
    result[i] = sum;
    return;
}
#endif // #ifndef _CONVOLUTION_KERNEL_H_

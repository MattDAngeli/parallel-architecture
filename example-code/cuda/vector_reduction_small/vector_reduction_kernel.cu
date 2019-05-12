/* Device code for vector reduction. 

 * Author: Naga Kandasamy
 * Date modified: may 5, 2019
 */

#ifndef _REDUCTION_KERNEL_H_
#define _REDUCTION_KERNEL_H_

/* This kernel performs reduction using a tree-style reduction technique 
    that increases divergent branching between threads in a warp. 
*/
__global__ 
void vector_reduction_kernel_v1 (float *d_data, double *d_result, int n)
{
	__shared__ double partial_sum[NUM_ELEMENTS];

	/* Find our place in thread block/grid. */
	unsigned int threadID = threadIdx.x;
	unsigned int dataID = blockIdx.x * blockDim.x + threadIdx.x;
	
	/* Populate shared memory with data from global memory. */
	if(dataID < n) 
		partial_sum[threadID] = d_data[dataID];
	else
		partial_sum[threadID] = 0.0;
 
	__syncthreads();

	/* Calculate sum. */
	for (unsigned int stride = 1; stride < blockDim.x; stride *= 2) {
		if (threadID % (2 * stride) == 0)
			partial_sum[threadID] += partial_sum[threadID + stride];
		__syncthreads();
	}

	/* Store result. */
	if (threadID == 0)
		*d_result = partial_sum[0];
}

/* This kernel performs reduction in a fashion that reduces divergent 
    branching between threads in a warp.
*/
__global__ 
void vector_reduction_kernel_v2 (float *d_data, double *d_result, int n)
{
	__shared__ double partial_sum[NUM_ELEMENTS];
	
    /* Find our place in thread block/grid. */ 
	unsigned int threadID = threadIdx.x; 
	unsigned int dataID = blockIdx.x * blockDim.x + threadIdx.x; 

	/* Copy data to shared memory from global memory. */ 
	if (dataID < n) 
		partial_sum[threadID] = d_data[dataID];
	else
		partial_sum[threadID] = 0.0;
	
	__syncthreads();

	for (unsigned int stride = blockDim.x >> 1; stride > 0; stride = stride >> 1) {
		if(threadID < stride)
			partial_sum[threadID] += partial_sum[threadID + stride];
	
		__syncthreads();
	}

	/* Store result. */
	if(threadID == 0)
		*d_result = partial_sum[0];
}

#endif // #ifndef _REDUCTION_KERNEL_H_

#include <cublas_v2.h>

#include "cuda_runtime.h"  
#include "device_launch_parameters.h"  

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/generate.h>
#include <thrust/reduce.h>
#include <thrust/functional.h>
#include <thrust/random.h>
#include <thrust/sequence.h>
#include <thrust/iterator/constant_iterator.h>
#include <thrust/count.h>

#include <stdio.h>
#include <iostream>

//__global__ void CCS1(int n, float *d_ClassPenalty, float *d_result) {
//	cublasHandle_t handle;
//	cublasCreate(&handle);
//	cublasSasum(handle, n, d_ClassPenalty, 1, d_result);
//	cublasDestroy(handle);
//}

//https://stackoverflow.com/questions/38654754/how-to-find-the-sum-of-array-in-cuda-by-reduction
// the length is MaxPossibleClusters and d_penalty[0] = 0
//dim3 blocksize(256); // create 1D threadblock
//dim3 gridsize(N / blocksize.x);  //create 1D grid

//reduce << <gridsize, blocksize >> >(dev_a, dev_b);

__global__ void reduce1(int MaxPossibleClusters, int *d_ClassPenalty, int *d_penalty) {

	__shared__ int sdata[256];

	// each thread loads one element from global to shared mem
	// note use of 1D thread indices (only) in this kernel
	int i = blockIdx.x*blockDim.x + threadIdx.x;
	if(i < MaxPossibleClusters)
	sdata[threadIdx.x] = d_ClassPenalty[i];

	__syncthreads();
	// do reduction in shared mem
	for (int s = 1; s < blockDim.x; s *= 2)
	{
		int index = 2 * s * threadIdx.x;;

		if (index < blockDim.x)
		{
			sdata[index] += sdata[index + s];
		}
		__syncthreads();
	}

	// write result for this block to global mem
	if (threadIdx.x == 0)
		atomicAdd(d_penalty, sdata[0]);
}

__global__ void reduce2(int nPoints, int MaxPossibleClusters, int *d_LogP, int *d_Class, int *d_penalty) {

	__shared__ int sdata[256];

	// each thread loads one element from global to shared mem
	// note use of 1D thread indices (only) in this kernel
	int i = blockIdx.x*blockDim.x + threadIdx.x;
	if (i < nPoints)
	{
	    sdata[threadIdx.x] = d_LogP[i*MaxPossibleClusters + d_Class[i]];
	}
	__syncthreads();
	// do reduction in shared mem
	for (int s = 1; s < blockDim.x; s *= 2)
	{
		int index = 2 * s * threadIdx.x;;

		if (index < blockDim.x)
		{
			sdata[index] += sdata[index + s];
		}
		__syncthreads();
	}

	// write result for this block to global mem
	if (threadIdx.x == 0)
		atomicAdd(d_penalty, sdata[0]);
}
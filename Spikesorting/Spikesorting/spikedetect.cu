#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>

#include "parameters.h"
/****************************************************abs 操作*************************************************************/
__global__ void flit(float *sort_ary, size_t N)
{
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	if (tid < N)
	{
		if (sort_ary[tid] < 0)
			sort_ary[tid] = -sort_ary[tid];
	}
}
/*************************************************排序操作*********************************************************/
__device__ void swap(float &a, float &b){
	float t = a;
	a = b;
	b = t;
}
__global__ void even_sort(float *ary, int size, int *mark)
{
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	if ((tid + 1) % 2 == 1 && tid + 1 < size)
	{
		if (ary[tid] > ary[tid + 1]){
			swap(ary[tid], ary[tid + 1]);
			mark[0] = 1;
		}
	}
	__syncthreads();
}
__global__ void odd_sort(float *ary, int size, int *mark)
{
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	if ((tid + 1) % 2 == 0 && tid + 1 < size)
	{
		if (ary[tid] > ary[tid + 1]){
			swap(ary[tid], ary[tid + 1]);
			mark[1] = 1;
		}

	}
	__syncthreads();
}
/************************************************阈值操作**************************************************************/
__global__ void Crossing(float *ary, float *sort_ary, size_t N, int *crossing)
{
	float mid = (sort_ary[N / 2 - 1] + sort_ary[N / 2]) / 2.0/0.6745;
	float high = mid * 4.5;
	float low = mid * 2.0;
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	if (tid == 1) printf("%f  %f\n", high, low);
	if (tid < N)
	{
		if (ary[tid] < high && ary[tid] >= low)  crossing[tid] = 1;
		else if (ary[tid] >= high)  crossing[tid] = 2;
	}
}
/****************************************************调用kernel*****************************************************************/
extern "C"
void mixGPU(int *crossing, float *ary, int *mark, size_t N)
{
	int numThreads = 1024;
	int numBlocks = (N + numThreads - 1) / numThreads;
	
	float *dev_ary = 0;
	float *dev_sort_ary = 0;
	int *dev_mark = 0;

	int *dev_crossing = 0;
	//unsigned int sharedSize = numThreads * sizeof(float);

	//cudaMalloc((void**)&dev_prms, sizeof(spikedetekt_prm));

	cudaMalloc((void**)&dev_ary, N*sizeof(float));
	cudaMalloc((void**)&dev_sort_ary,N * sizeof(float));
	cudaMalloc((void**)&dev_mark, 2 * sizeof(int));

	cudaMalloc((void**)&dev_crossing, N * sizeof(int));
	cudaMemset(dev_crossing, 0, N * sizeof(int));

	cudaMemcpy(dev_ary, ary, sizeof(float)*N, cudaMemcpyHostToDevice);
	cudaMemcpy(dev_sort_ary, ary, sizeof(float)*N, cudaMemcpyHostToDevice);

	// kernel execution
	flit << <numBlocks, numThreads >> >(dev_sort_ary, N);
	while (mark[0] + mark[1] > 0)
	{
		mark[0] = 0;
		mark[1] = 0;
		cudaMemcpy(dev_mark, mark, sizeof(int) * 2, cudaMemcpyHostToDevice);
		even_sort << <numBlocks, numThreads >> >(dev_sort_ary, N, dev_mark);
		odd_sort << <numBlocks, numThreads >> >(dev_sort_ary, N, dev_mark);
		cudaMemcpy(mark, dev_mark, 2 * sizeof(int), cudaMemcpyDeviceToHost);
	}
	Crossing <<<numBlocks, numThreads >>>(dev_ary, dev_sort_ary, N, dev_crossing);


	/************************************************************CPU test******************************************************************/
	//cudaMemcpy(answer, dev_answer, 2 * sizeof(float), cudaMemcpyDeviceToHost);
	cudaMemcpy(crossing, dev_crossing, N * sizeof(int), cudaMemcpyDeviceToHost);

	cudaFree(dev_ary);
	cudaFree(dev_sort_ary);
	cudaFree(dev_mark);
	cudaFree(dev_crossing);
}

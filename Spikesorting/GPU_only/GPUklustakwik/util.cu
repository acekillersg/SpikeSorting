// Disable some Visual Studio warnings
#define _CRT_SECURE_NO_WARNINGS

#include "util.h"
#include "log.h"
#include<stdlib.h>
#include<stdarg.h>
#include<math.h>
#include<iostream>

#include "cuda_runtime.h"  
#include "device_launch_parameters.h"

const float HugeScore = (float)1e32;

/* integer random number between min and max*/
int irand(int min, int max)
{
	return (rand() % (max - min + 1) + min);
}

FILE *fopen_safe(char *fname, char *mode) {
	FILE *fp;
	fp = fopen(fname, mode);
	if (!fp) {
		fprintf(stderr, "Could not open file %s\n", fname);
		abort();
	}
	return fp;
}

// Print a matrix
void MatPrint(FILE *fp, float *Mat, int nRows, int nCols) {
	int i, j;

	for (i = 0; i<nRows; i++) {
		for (j = 0; j<nCols; j++) {
			fprintf(fp, "%.5g ", Mat[i*nCols + j]);
			Output("%.5g ", Mat[i*nCols + j]);
		}
		fprintf(fp, "\n");
		Output("\n");
	}
}

void CompareVectors(float *A, float *B, int N)
{
	float meanerr = 0.0;
	float maxerr = 0.0;
	int nerr = 0;
	int ntotal = 0;
	for (int i = 0; i < N; i++)
	{
		float err = fabs(A[i] - B[i]);
		ntotal++;
		if (err > 0)
		{
			nerr++;
			meanerr += err;
			if (err > maxerr)
				maxerr = err;
		}
	}
	if (nerr)
	{
		meanerr /= nerr;
		std::cout << "Comparison error n=" << nerr << " (" << (100.0*nerr) / ntotal << "%), mean=" << meanerr << ", max=" << maxerr << std::endl;
	}
	else
		std::cout << "No comparison error." << std::endl;
}

//https://github.com/OrangeOwlSolutions/CUDA-Utilities/blob/master/Utilities.cu
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort = false) {
	if (code != cudaSuccess) {
		printf("GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
		//fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
		if (abort) exit(code);
	}
	else {
		//printf("cuda returned code == cudaSuccess\n");
	}
}
void gpuErrchk(cudaError_t ans) { gpuAssert((ans), __FILE__, __LINE__); }

//============================================reduce============================================
//template <unsigned int blockSize>
//__global__ void reduceFlt1(float *g_idata, float *g_odata, unsigned int n)
//{
//	extern __shared__ float sdata[];
//	unsigned int tid = threadIdx.x;
//	unsigned int i = blockIdx.x*(blockSize * 2) + tid;
//	unsigned int gridSize = blockSize * 2 * gridDim.x;
//	sdata[tid] = 0;
//	while (i < n) { sdata[tid] += g_idata[i] + g_idata[i + blockSize]; i += gridSize; }
//	__syncthreads();
//	if (blockSize >= 512) { if (tid < 256) { sdata[tid] += sdata[tid + 256]; } __syncthreads(); }
//	if (blockSize >= 256) { if (tid < 128) { sdata[tid] += sdata[tid + 128]; } __syncthreads(); }
//	if (blockSize >= 128) { if (tid < 64) { sdata[tid] += sdata[tid + 64]; } __syncthreads(); }
//	if (tid < 32) {
//		if (blockSize >= 64) sdata[tid] += sdata[tid + 32];
//		if (blockSize >= 32) sdata[tid] += sdata[tid + 16];
//		if (blockSize >= 16) sdata[tid] += sdata[tid + 8];
//		if (blockSize >= 8) sdata[tid] += sdata[tid + 4];
//		if (blockSize >= 4) sdata[tid] += sdata[tid + 2];
//		if (blockSize >= 2) sdata[tid] += sdata[tid + 1];
//	}
//	if (tid == 0) g_odata[blockIdx.x] = sdata[0];
//}
//template <unsigned int blockSize>
//__global__ void reduceFlt2(float *g_idata, float *g_odata, unsigned int n)
//{
//	extern __shared__ float sdata[];
//	unsigned int tid = threadIdx.x;
//	if (tid < n) sdata[tid] = g_idata[tid];
//	else sdata[tid] = 0;
//	for (unsigned int s = blockSize / 2; s>0; s >>= 1) {
//		if (tid < s) {
//			sdata[tid] += sdata[tid + s];
//		}
//		__syncthreads();
//	}
//	if (tid == 0) g_odata[blockIdx.x] = sdata[0];
//}
//
//template <unsigned int blockSize>
//float reduceFlt(float *d_idata, unsigned int n) {
//	int blocksize = 128;
//	size_t size = sizeof(float);
//	dim3 block(blocksize, 1);
//	dim3 grid((n + block.x - 1) / block.x, 1);
//	float *d_odata = NULL; gpuErrchk(cudaMalloc((void **)&d_odata, grid.x*size));
//	float *d_result = NULL; gpuErrchk(cudaMalloc((void **)&d_result, size));
//
//	reduceFlt1<128> << <grid, block, blockSize * size >> >(d_idata, d_odata, n);
//	reduceFlt2<128> << <1, block, blockSize * size >> >(d_odata, d_result, grid.x);
//
//	float *h_result = (float *)malloc(size);
//	gpuErrchk(cudaMemcpy(h_result, d_result, size, cudaMemcpyDeviceToHost));
//	return *h_result;
//}
//
//template <unsigned int blockSize>
//__global__ void reduceInt1(int *g_idata, int *g_odata, unsigned int n)
//{
//	extern __shared__ int sdata1[];
//	unsigned int tid = threadIdx.x;
//	unsigned int i = blockIdx.x*(blockSize * 2) + tid;
//	unsigned int gridSize = blockSize * 2 * gridDim.x;
//	sdata1[tid] = 0;
//	while (i < n) { sdata1[tid] += g_idata[i] + g_idata[i + blockSize]; i += gridSize; }
//	__syncthreads();
//	if (blockSize >= 512) { if (tid < 256) { sdata1[tid] += sdata1[tid + 256]; } __syncthreads(); }
//	if (blockSize >= 256) { if (tid < 128) { sdata1[tid] += sdata1[tid + 128]; } __syncthreads(); }
//	if (blockSize >= 128) { if (tid < 64) { sdata1[tid] += sdata1[tid + 64]; } __syncthreads(); }
//	if (tid < 32) {
//		if (blockSize >= 64) sdata1[tid] += sdata1[tid + 32];
//		if (blockSize >= 32) sdata1[tid] += sdata1[tid + 16];
//		if (blockSize >= 16) sdata1[tid] += sdata1[tid + 8];
//		if (blockSize >= 8) sdata1[tid] += sdata1[tid + 4];
//		if (blockSize >= 4) sdata1[tid] += sdata1[tid + 2];
//		if (blockSize >= 2) sdata1[tid] += sdata1[tid + 1];
//	}
//	if (tid == 0) g_odata[blockIdx.x] = sdata1[0];
//}
//template <unsigned int blockSize>
//__global__ void reduceInt2(int *g_idata, int *g_odata, unsigned int n)
//{
//	extern __shared__ int sdata1[];
//	unsigned int tid = threadIdx.x;
//	if (tid < n) sdata1[tid] = g_idata[tid];
//	else sdata1[tid] = 0;
//	for (unsigned int s = blockSize / 2; s>0; s >>= 1) {
//		if (tid < s) {
//			sdata1[tid] += sdata1[tid + s];
//		}
//		__syncthreads();
//	}
//	if (tid == 0) g_odata[blockIdx.x] = sdata1[0];
//}
//
//template <unsigned int blockSize>
//int reduceInt(int *d_idata, unsigned int n) {
//	int blocksize = 128;
//	size_t size = sizeof(int);
//	dim3 block(blocksize, 1);
//	dim3 grid((n + block.x - 1) / block.x, 1);
//	int *d_odata = NULL; gpuErrchk(cudaMalloc((void **)&d_odata, grid.x*size));
//	int *d_result = NULL; gpuErrchk(cudaMalloc((void **)&d_result, size));
//
//	reduceInt1<128> << <grid, block, blockSize * size >> >(d_idata, d_odata, n);
//	reduceInt2<128> << <1, block, blockSize * size >> >(d_odata, d_result, grid.x);
//
//	int *h_result = (int *)malloc(size);
//	gpuErrchk(cudaMemcpy(h_result, d_result, size, cudaMemcpyDeviceToHost));
//	return *h_result;
//}



// Does a memory check (should only be called for first instance of KK)
/*void KK::MemoryCheck()
{
long long NP = (long long)nPoints;
long long MPC = (long long)MaxPossibleClusters;
long long ND = (long long)nDims;
vector<MemoryUsage> usages;

usages.push_back(MemoryUsage("Data", "float", sizeof(float), NP*ND, "nPoints*nDims", 2, 3));

usages.push_back(MemoryUsage("Masks", "char", sizeof(char), NP*ND, "nPoints*nDims", 2, 3));

usages.push_back(MemoryUsage("FloatMasks", "float", sizeof(float), NP*ND, "nPoints*nDims", 2, 3));
usages.push_back(MemoryUsage("Cov", "float", sizeof(float), MPC*ND*ND, "MaxPossibleClusters*nDims*nDims", 0, 3));
//usages.push_back(MemoryUsage("Cov", "float", sizeof(float), MPC*ND*ND, "MaxPossibleClusters*nDims*nDims", 2, 3));
usages.push_back(MemoryUsage("LogP", "float", sizeof(float), MPC*NP, "MaxPossibleClusters*nPoints", 2, 3));
usages.push_back(MemoryUsage("AllVector2Mean", "float", sizeof(float), NP*ND, "nPoints*nDims", 2, 3));
usages.push_back(MemoryUsage("CorrectionTerm", "float", sizeof(float), NP*ND, "nPoints*nDims", 2, 3));

check_memory_usage(usages, RamLimitGB, nPoints, nDims, MaxPossibleClusters);
}*/

////==========================================reduce======================================================
//template <unsigned int blockSize>
//__global__ void reduceFlt1(float *g_idata, float *g_odata, unsigned int n)
//{
//	extern __shared__ float sdata[];
//	unsigned int tid = threadIdx.x;
//	unsigned int i = blockIdx.x*(blockSize * 2) + tid;
//	unsigned int gridSize = blockSize * 2 * gridDim.x;
//	sdata[tid] = 0;
//	//while (i < n) { sdata[tid] += g_idata[i] + g_idata[i + blockSize]; i += gridSize; }
//	while (i < n) {
//		if (i + blockSize < n)  sdata[tid] += g_idata[i] + g_idata[i + blockSize];
//		else  sdata[tid] += g_idata[i];
//		i += gridSize;
//		//if (blockIdx.x == 6) printf("sdata1[%d] == %d\n", threadIdx.x, sdata1[threadIdx.x]);
//	}
//	__syncthreads();
//	if (blockSize >= 512) { if (tid < 256) { sdata[tid] += sdata[tid + 256]; } __syncthreads(); }
//	if (blockSize >= 256) { if (tid < 128) { sdata[tid] += sdata[tid + 128]; } __syncthreads(); }
//	if (blockSize >= 128) { if (tid < 64) { sdata[tid] += sdata[tid + 64]; } __syncthreads(); }
//	if (tid < 32) {
//		if (blockSize >= 64) sdata[tid] += sdata[tid + 32];
//		if (blockSize >= 32) sdata[tid] += sdata[tid + 16];
//		if (blockSize >= 16) sdata[tid] += sdata[tid + 8];
//		if (blockSize >= 8) sdata[tid] += sdata[tid + 4];
//		if (blockSize >= 4) sdata[tid] += sdata[tid + 2];
//		if (blockSize >= 2) sdata[tid] += sdata[tid + 1];
//	}
//	if (tid == 0) g_odata[blockIdx.x] = sdata[0];
//}
//template <unsigned int blockSize>
//__global__ void reduceFlt2(float *g_idata, float *g_odata, unsigned int n)
//{
//	extern __shared__ float sdata[];
//	unsigned int tid = threadIdx.x;
//	if (tid < n) sdata[tid] = g_idata[tid];
//	else sdata[tid] = 0;
//	for (unsigned int s = blockSize / 2; s>0; s >>= 1) {
//		if (tid < s) {
//			sdata[tid] += sdata[tid + s];
//		}
//		__syncthreads();
//	}
//	if (tid == 0) g_odata[blockIdx.x] = sdata[0];
//}
//
//template <unsigned int blockSize>
//float reduceFlt(float *d_idata, unsigned int n) {
//	int blocksize = 128;
//	size_t size = sizeof(float);
//	dim3 block(blocksize, 1);
//	dim3 grid((n + 2*block.x - 1) / (2*block.x), 1);
//	float *d_odata = NULL; gpuErrchk(cudaMalloc((void **)&d_odata, grid.x*size));
//	float *d_result = NULL; gpuErrchk(cudaMalloc((void **)&d_result, size));
//
//	reduceFlt1<128> << <grid, block, blockSize * size >> >(d_idata, d_odata, n);
//	reduceFlt2<128> << <1, block, blockSize * size >> >(d_odata, d_result, grid.x);
//
//	float h_result;
//	gpuErrchk(cudaMemcpy(&h_result, d_result, size, cudaMemcpyDeviceToHost));
//
//	cudaFree(d_odata);cudaFree(d_result);
//
//	return h_result;
//}
//
//template <unsigned int blockSize>
//__global__ void reduceInt1(int *g_idata, int *g_odata, unsigned int n)
//{
//	extern __shared__ int sdata1[];
//	unsigned int tid = threadIdx.x;
//	unsigned int i = blockIdx.x*(blockSize * 2) + tid;
//	unsigned int gridSize = blockSize * 2 * gridDim.x;
//	sdata1[tid] = 0;
//	//while (i < n) { sdata1[tid] += g_idata[i] + g_idata[i + blockSize]; i += gridSize; }
//	while (i < n) {
//		if (i + blockSize < n)
//			sdata1[tid] += g_idata[i] + g_idata[i + blockSize];
//		else
//			sdata1[tid] += g_idata[i];
//		i += gridSize;
//		//if (blockIdx.x == 6) printf("sdata1[%d] == %d\n", threadIdx.x, sdata1[threadIdx.x]);
//	}
//	__syncthreads();
//	if (blockSize >= 512) { if (tid < 256) { sdata1[tid] += sdata1[tid + 256]; } __syncthreads(); }
//	if (blockSize >= 256) { if (tid < 128) { sdata1[tid] += sdata1[tid + 128]; } __syncthreads(); }
//	if (blockSize >= 128) { if (tid < 64) { sdata1[tid] += sdata1[tid + 64]; } __syncthreads(); }
//	if (tid < 32) {
//		if (blockSize >= 64) sdata1[tid] += sdata1[tid + 32];
//		if (blockSize >= 32) sdata1[tid] += sdata1[tid + 16];
//		if (blockSize >= 16) sdata1[tid] += sdata1[tid + 8];
//		if (blockSize >= 8) sdata1[tid] += sdata1[tid + 4];
//		if (blockSize >= 4) sdata1[tid] += sdata1[tid + 2];
//		if (blockSize >= 2) sdata1[tid] += sdata1[tid + 1];
//	}
//	if (tid == 0) g_odata[blockIdx.x] = sdata1[0];
//}
//template <unsigned int blockSize>
//__global__ void reduceInt2(int *g_idata, int *g_odata, unsigned int n)
//{
//	extern __shared__ int sdata1[];
//	unsigned int tid = threadIdx.x;
//	if (tid < n) sdata1[tid] = g_idata[tid];
//	else sdata1[tid] = 0;
//	for (unsigned int s = blockSize / 2; s>0; s >>= 1) {
//		if (tid < s) {
//			sdata1[tid] += sdata1[tid + s];
//		}
//		__syncthreads();
//	}
//	if (tid == 0) g_odata[blockIdx.x] = sdata1[0];
//}
//
//template <unsigned int blockSize>
//int reduceInt(int *d_idata, unsigned int n) {
//	int blocksize = 128;
//	size_t size = sizeof(int);
//	dim3 block(blocksize, 1);
//	dim3 grid((n + block.x * 2 - 1) / (2 * block.x), 1);
//	int *d_odata = NULL; gpuErrchk(cudaMalloc((void **)&d_odata, grid.x*size));
//	int *d_result = NULL; gpuErrchk(cudaMalloc((void **)&d_result, size));
//
//	reduceInt1<128> << <grid, block, blockSize * size >> >(d_idata, d_odata, n);
//	reduceInt2<128> << <1, block, blockSize * size >> >(d_odata, d_result, grid.x);
//
//	int h_result;
//	gpuErrchk(cudaMemcpy(&h_result, d_result, size, cudaMemcpyDeviceToHost));
//
//	cudaFree(d_odata);cudaFree(d_result);
//	return h_result;
//}
//======================================================================================================================
//=====================================================prefix sum=======================================================
//#define BLOCK_SIZE 128
//__global__ void inclusive_scan(int *d_in, int *d_blocksum, int InputSize) {
//
//	// XY[2*BLOCK_SIZE] is in shared memory
//	__shared__ int temp[BLOCK_SIZE * 3];
//	int i = blockIdx.x*blockDim.x + threadIdx.x;
//	if (i < InputSize) { temp[threadIdx.x] = d_in[i]; }
//
//	//// the code below performs iterative scan on XY¡¡¡¡
//	for (unsigned int stride = 1; stride <= BLOCK_SIZE; stride *= 2) {
//		__syncthreads();
//		int index = (threadIdx.x + 1)*stride * 2 - 1;
//		if (index < 2 * BLOCK_SIZE)
//			temp[index] += temp[index - stride];//index is alway bigger than stride
//		__syncthreads();
//	}
//	//// threadIdx.x+1 = 1,2,3,4....
//	//// stridek index = 1,3,5,7...
//
//
//	for (unsigned int stride = BLOCK_SIZE / 2; stride > 0; stride /= 2) {
//		__syncthreads();
//		int index = (threadIdx.x + 1)*stride * 2 - 1;
//		if (index < 2 * BLOCK_SIZE)
//			temp[index + stride] += temp[index];
//		__syncthreads();
//	}
//	__syncthreads();
//	if (i < InputSize) {
//		d_in[i] = temp[threadIdx.x];
//		if ((threadIdx.x == BLOCK_SIZE - 1) || (i == InputSize - 1))
//			d_blocksum[blockIdx.x] = temp[threadIdx.x];
//	}
//}
//
//__global__ void inclusive_scan1(int *d_in, int *d_add, int InputSize) {
//	__shared__ int temp[BLOCK_SIZE];
//	int i = blockIdx.x*blockDim.x + threadIdx.x;
//	if (i < InputSize) { temp[threadIdx.x] = d_in[i]; }
//	if (i < InputSize) {
//		if (blockIdx.x > 0) d_in[i] = temp[threadIdx.x] + d_add[blockIdx.x - 1];
//	}
//}
//
//void scanKernel(int *d_idata, int n) {
//
//	//int *h_idata = (int *)malloc(n * sizeof(int));
//	int blocksize = 128;
//	size_t sizei = n * sizeof(int);
//
//	int grid1 = (n + blocksize - 1) / blocksize;
//	int *d_blocksum = NULL; gpuErrchk(cudaMalloc((void **)&d_blocksum, grid1 * sizeof(int)));
//	inclusive_scan << <grid1, blocksize >> > (d_idata, d_blocksum, n);
//
//	int grid2 = (grid1 + blocksize - 1) / blocksize;
//	int *d_tt; gpuErrchk(cudaMalloc((void **)&d_tt, grid2 * sizeof(int)));
//	inclusive_scan << <grid2, blocksize >> > (d_blocksum, d_tt, grid1);
//
//	inclusive_scan1 << <grid1, blocksize >> >(d_idata, d_blocksum, n);
//
//	//printf("\nin kernel:\n");
//	//gpuErrchk(cudaMemcpy(h_idata, d_idata, sizei, cudaMemcpyDeviceToHost));
//	//for (int i = 0;i < n;i++) {
//	//	printf("%d  ", h_idata[i]);
//	//	if ((i + 1) % blocksize == 0)
//	//		printf("\n");
//	//}
//	//printf("\n");
//
//	cudaFree(d_blocksum);
//	cudaFree(d_tt);
//}
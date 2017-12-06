#include "dlinalg.cuh"

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

#include<math.h>
#include<iostream>

#include <cublas_v2.h>
#include <cusolverDn.h>

#include "cuda_runtime.h"  
#include "device_launch_parameters.h"  

using namespace std;

d_BlockPlusDiagonalMatrix::d_BlockPlusDiagonalMatrix(thrust::device_vector<int> &_Masked, thrust::device_vector<int> &_Unmasked)
{
	Masked = &_Masked;
	Unmasked = &_Unmasked;
	NumUnmasked = Unmasked->size();
	NumMasked = Masked->size();
	Block.resize(NumUnmasked*NumUnmasked);
	Diagonal.resize(NumMasked);
}

/*void d_BlockPlusDiagonalMatrix::compare(float *Flat)
{
	int nDims = NumUnmasked + NumMasked;
	float meanerr = 0.0;
	float maxerr = 0.0;
	int nerr = 0;
	int ntotal = 0;
	for (int ii = 0; ii < NumUnmasked; ii++)
	{
		int i = (*Unmasked)[ii];
		for (int jj = 0; jj < NumUnmasked; jj++)
		{
			int j = (*Unmasked)[jj];
			float x = Block[ii*NumUnmasked + jj];
			float y = Flat[i*nDims + j];
			float err = fabs(x - y);
			ntotal++;
			if (err > 0)
			{
				nerr++;
				meanerr += err;
				if (err > maxerr)
					maxerr = err;
			}
		}
	}
	for (int ii = 0; ii < NumMasked; ii++)
	{
		int i = (*Masked)[ii];
		float x = Diagonal[ii];
		float y = Flat[i*nDims + i];
		float err = fabs(x - y);
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
		cout << "Comparison error n=" << nerr << " (" << (100.0*nerr) / ntotal << "%), mean=" << meanerr << ", max=" << maxerr << endl;
	}
	else
		cout << "No comparison error." << endl;
}
*/

// Cholesky Decomposition
// In provides upper triangle of input matrix (In[i*D + j] >0 if j>=i);
// which is the top half of a symmetric matrix
// Out provides lower triange of output matrix (Out[i*D + j] >0 if j<=i);
// such that Out' * Out = In.
// D is number of dimensions
//
// returns 0 if OK, returns 1 if matrix is not positive definite
//need 96 threads
__global__ void digUpdate(int num, float *in, float *out) {
	int tid = threadIdx.x;
	if (tid < num) {
		if (in[tid] <= 0) {
			printf("Cholesky failed!!!\n");
			return ;
		}
		out[tid] = (float)sqrt(in[tid]);
	}
}
//need 96 threads
__global__ void copy1(int m, float *A,int *A_id, float *C) {
	int tid = threadIdx.x;
	if (tid < m) {
		C[tid] = A[ A_id[tid] ];
	}
}
//need 96 threads
__global__ void copy2(int m, int n, float *A, int *A_id, float *B, int *B_id, float *C) {
	int tid = threadIdx.x;
	if (tid < m) {
		C[ A_id[tid] ] = A[tid];
	}
	syncthreads();
	if (tid < n) {
		C[ B_id[tid] ] = -C[B_id[tid]] / B[tid];
	}
}

int CusolverCholesky(d_BlockPlusDiagonalMatrix &In, d_BlockPlusDiagonalMatrix &Out) {
	// --- cuSOLVE input/output parameters/arrays
	int Nrows = In.NumUnmasked;
	int work_size = 0;
	int *devInfo;           
	cudaMalloc(&devInfo, sizeof(int));

	// --- CUDA solver initialization
	cusolverDnHandle_t solver_handle;
	cusolverDnCreate(&solver_handle);

	// --- CUDA CHOLESKY initialization
	cusolverDnSpotrf_bufferSize(solver_handle, CUBLAS_FILL_MODE_LOWER, Nrows, thrust::raw_pointer_cast(&In.Block[0]), Nrows, &work_size);

	// --- CUDA POTRF execution
	float *work;   
	cudaMalloc(&work, work_size * sizeof(float));
	cusolverDnSpotrf(solver_handle, CUBLAS_FILL_MODE_LOWER, Nrows, thrust::raw_pointer_cast(&In.Block[0]), Nrows, work, work_size, devInfo);
	int devInfo_h = 0;  
	cudaMemcpy(&devInfo_h, devInfo, sizeof(int), cudaMemcpyDeviceToHost);
	if (devInfo_h != 0) std::cout << "Unsuccessful potrf execution\n\n";

	// --- At this point, the upper triangular part of A contains the elements of L. Showing this.
	
	cudaMemcpy(thrust::raw_pointer_cast(&Out.Block[0]), thrust::raw_pointer_cast(&In.Block[0]), Nrows * Nrows * sizeof(double), cudaMemcpyDeviceToHost);
	/*
	printf("\nFactorized matrix\n");
	for (int i = 0; i < Nrows; i++)
		for (int j = 0; j < Nrows; j++)
			if (i <= j) printf("L[%i, %i] = %f\n", i, j, Out.Block[i * Nrows + j]);
	*/
	cusolverDnDestroy(solver_handle);

	digUpdate << <1, 128 >> > (In.NumMasked, 
		thrust::raw_pointer_cast(&In.Diagonal[0]), 
		thrust::raw_pointer_cast(&Out.Diagonal[0]));

	return 0;
}
/*
int BPDCholesky(BlockPlusDiagonalMatrix &In, BlockPlusDiagonalMatrix &Out)
{
	int ii, jj, kk;
	float sum;
	int NumUnmasked = (int)In.NumUnmasked;

	// main bit for unmasked features
	for (ii = 0; ii < NumUnmasked; ii++)
	{
		for (jj = ii; jj < NumUnmasked; jj++)
		{
			sum = In.Block[ii*NumUnmasked + jj];

			for (kk = ii - 1; kk >= 0; kk--)
			{
				sum -= Out.Block[ii*NumUnmasked + kk] * Out.Block[jj*NumUnmasked + kk];
			}
			if (ii == jj) {
				if (sum <= 0) return(1); // Cholesky decomposition has failed
				Out.Block[ii*NumUnmasked + ii] = (float)sqrt(sum);
			}
			else {
				Out.Block[jj*NumUnmasked + ii] = sum / Out.Block[ii*NumUnmasked + ii];
			}
		}
	}
	// main bit for masked features
	for (ii = 0; ii < (int)In.NumMasked; ii++)
	{
		float sum = In.Diagonal[ii];
		if (sum <= 0)
			return 1; // Cholesky failed
		Out.Diagonal[ii] = (float)sqrt(sum);
	}

	return 0; // for success
}
*/

// Solve a set of linear equations M*Out = x.
// Where M is lower triangular (M[i*D + j] >0 if j>=i);
// D is number of dimensions


void CublasTriSolve(d_BlockPlusDiagonalMatrix &M, thrust::device_vector<float> &x,
	thrust::device_vector<float> &Out)
{
	const int NumUnmasked = M.NumUnmasked;
	const int NumMasked = M.NumMasked;

	const float * __restrict ptr_x = thrust::raw_pointer_cast(&x[0]);
	float * __restrict ptr_Out = thrust::raw_pointer_cast(&Out[0]);

	const int * __restrict d_Unmasked = thrust::raw_pointer_cast(&((*M.Unmasked)[0]));
	thrust::device_vector<float> C(NumUnmasked);

	if (NumUnmasked)
	{
		copy1 << <1, 128 >> > (d_NumUnmasked, ptr_x, d_Unmasked, thrust::raw_pointer_cast(&C[0]));

		cublasHandle_t handle;
		cublasCreate(&handle);
		float alpha = 1.0;
		cublasStrsm(handle, CUBLAS_SIDE_LEFT,
			CUBLAS_FILL_MODE_LOWER, CUBLAS_OP_N,
			CUBLAS_DIAG_UNIT,
			NumUnmasked, 1,&alpha,
			thrust::raw_pointer_cast(&M.Block[0]), NumUnmasked, thrust::raw_pointer_cast(&C[0]),1);

		/*for (int ii = 0; ii < NumUnmasked; ii++)
		{
			const int i = Unmasked[ii];
			float sum = ptr_x[i];
			const float * __restrict row = &(M.Block[ii*M.NumUnmasked]);
			for (int jj = 0; jj < ii; jj++) // j<i
			{
				const int j = Unmasked[jj];
				sum += row[jj] * ptr_Out[j];
			}
			ptr_Out[i] = -sum / row[ii];
		}
		*/
	}
	const int * __restrict d_Masked = thrust::raw_pointer_cast( &((*M.Masked)[0]) );
	const float * __restrict d_Diagonal = thrust::raw_pointer_cast( &(M.Diagonal[0]));
	copy2 << <1, 128 >> > (NumUnmasked, NumMasked, thrust::raw_pointer_cast(&C[0]), 
		                   d_Unmasked, d_Diagonal, d_Masked, ptr_x);

	/*if (NumMasked)
	{
		const int * __restrict Masked = &((*M.Masked)[0]);
		const float * __restrict Diagonal = &(M.Diagonal[0]);
		for (int ii = 0; ii < NumMasked; ii++)
		{
			const int i = Masked[ii];
			Out[i] = -ptr_x[i] / Diagonal[ii];
		}
	}*/
}
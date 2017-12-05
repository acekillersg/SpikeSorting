//#include <cublas_v2.h>
//
//#include "cuda_runtime.h"  
//#include "device_launch_parameters.h"  
//
//#include <thrust/host_vector.h>
//#include <thrust/device_vector.h>
//#include <thrust/generate.h>
//#include <thrust/reduce.h>
//#include <thrust/functional.h>
//#include <thrust/random.h>
//#include <thrust/sequence.h>
//#include <thrust/iterator/constant_iterator.h>
//#include <thrust/count.h>
//
//#include <stdio.h>
//#include <iostream>
//
//__global__ void c_considerDel1(int MaxPossibleClusters, int HugeScore, int nPoints,
//	                          int *d_ClassAlive, int *d_Class, int *d_Class2, float *d_LogP, int *d_NumberInClass, float *d_ClassPenalty, float *d_DiffLoss) {
//
//	__shared__ float d_DeletionLoss[512];
//	int tid = blockDim.x * blockIdx.x + threadIdx.x;
//	int a = tid + 1;
//	if (a < MaxPossibleClusters)
//	{
//		if (d_ClassAlive[a]) d_DeletionLoss[a] = 0;
//		else d_DeletionLoss[a] = HugeScore; // don't delete classes that are already there
//	}
//	__syncthreads();
//	if (tid < nPoints)
//	{
//		int c = d_Class[tid];
//		atomicAdd(&d_DeletionLoss[c],d_LogP[tid * MaxPossibleClusters + d_Class2[tid]] - d_LogP[tid*MaxPossibleClusters + d_Class[tid]]);
//		atomicAdd(&d_NumberInClass[c], 1);
//	}
//	// find class with smallest increase in total score
//	__syncthreads();
//	if (a < MaxPossibleClusters) {
//		d_DiffLoss[a] = d_DeletionLoss[a] - d_ClassPenalty[a];
//	}
//}
//
//__global__ void c_considerDel2(int MaxPossibleClusters, int nPoints, int CandidateId,
//	                           int *d_ClassAlive,int *d_Class, int *d_Class2, float *d_DiffLoss) {
//
//	int tid = blockDim.x * blockIdx.x + threadIdx.x;
//	if (tid == 0) {
//		cublasHandle_t handle;
//		//1 index
//		cublasIsamin(handle, MaxPossibleClusters - 2, d_DiffLoss, 1, &CandidateId);
//		cublasDestroy(handle);
//	}
//	__syncthreads();
//	if (d_DiffLoss[CandidateId] >= 0) return;
//	if (tid < nPoints) {
//		if(tid == 0) d_ClassAlive[CandidateId] = 0;
//		if (d_Class[tid] == CandidateId) 
//			d_Class[tid] = d_Class2[tid];
//	}
//	
//	//c_ComputeClassPenalties();...
//
//}
//

//#include "klustakwik.h"
//#include "params.h"
//
//#include "cuda_runtime.h"  
//#include "device_launch_parameters.h"
//
//#include <cublas_v2.h>
//#include <cusolverDn.h>
//#include <algorithm>
//
//#include <stdio.h>
//#include <iostream>
//
//#define _USE_MATH_DEFINES
//#include <math.h>
//
//#define BLOCKDIM 128
//#define Bd128 128
//#define Bd64 64
//#define Bd32 32
//
//#define Bd2D32 dim(32,32)
//#define Bd2D64 dim(64,64)
//#define Bd2D128 dim(128,128)
//
////=========================================prefix sum==========================================================//
//#define BLOCK_SIZE 128
//__global__ void inclusive_scan(int *d_in, int *d_blocksum, int InputSize) {
//
//	__shared__ int temp[BLOCK_SIZE * 3];
//	int i = blockIdx.x*blockDim.x + threadIdx.x;
//	if (i < InputSize) { temp[threadIdx.x] = d_in[i]; }
//
//	//// the code below performs iterative scan on XY　　
//	for (unsigned int stride = 1; stride <= BLOCK_SIZE; stride *= 2) {
//		__syncthreads();
//		int index = (threadIdx.x + 1)*stride * 2 - 1;
//		if (index < 2 * BLOCK_SIZE)
//			temp[index] += temp[index - stride];//index is alway bigger than stride
//		__syncthreads();
//	}
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
//	//__shared__ int temp[BLOCK_SIZE];
//	int i = blockIdx.x*blockDim.x + threadIdx.x;
//	//if (i < InputSize) { temp[threadIdx.x] = d_in[i]; }
//	if (i < InputSize) {
//		if (blockIdx.x > 0) d_in[i] += d_add[blockIdx.x - 1];
//	}
//}
//
//int scanKernel(int *d_idata, int n) {
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
//	int x;
//	gpuErrchk(cudaMemcpy(&x, d_idata + n - 1, sizeof(int), cudaMemcpyDeviceToHost));
//	cudaFree(d_blocksum);
//	cudaFree(d_tt);
//
//	return x;
//}
////global kernel
////===========================================compute weight=======================================================//
//__global__ void c_Weight(int nClustersAlive, int priorPoint, int nPoints, int NoisePoint, int *d_AliveIndex, int *d_nClassMembers, float *d_Weight) {
//	int tid = blockDim.x * blockIdx.x + threadIdx.x;
//	if (tid < nClustersAlive)
//	{
//		int c = d_AliveIndex[tid];
//		if (c>0) d_Weight[c] = ((float)d_nClassMembers[c] + priorPoint) / (nPoints + NoisePoint + priorPoint*(nClustersAlive - 1));
//		else d_Weight[c] = ((float)d_nClassMembers[c] + NoisePoint) / (nPoints + NoisePoint + priorPoint*(nClustersAlive - 1));
//	}
//}
////===========================================compute mean=======================================================//
//
//__global__ void c_nClassMembers(int nPoints, int *d_Class, int *d_nClassMembers) {
//	int tidx = blockDim.x * blockIdx.x + threadIdx.x;
//	if (tidx < nPoints) {
//		atomicAdd(&d_nClassMembers[d_Class[tidx]], 1);
//	}
//}
//
//__global__ void matCopy(int npoints, int ndims, int nDims, int *d_rowId, int *d_colId, float *d_sourceMat, float *d_copyMat) {
//	int tidx = blockDim.x * blockIdx.x + threadIdx.x;
//	int tidy = blockDim.y * blockIdx.y + threadIdx.y;
//	if (tidx < ndims && tidy < npoints) {
//		int p = d_rowId[tidy];
//		int nd = d_colId[tidx];
//		//printf("in the kernel,this is the x: %d,y: %d,xx: %d, yy:%d\n", tidx, tidy, p, nd);
//		d_copyMat[tidy * ndims + tidx] = d_sourceMat[p * nDims + nd];
//	}
//
//}
//
//__global__ void c_CheckDead(int nClustersAlive, int *d_AliveIndex, int *d_nClassMembers, int *d_ClassAlive) {
//	int tid = blockDim.x * blockIdx.x + threadIdx.x;
//	if (tid <= nClustersAlive)
//	{
//		int c = d_AliveIndex[tid];
//		if (c>0 && d_nClassMembers[c]<1)
//			d_ClassAlive[c] = 0;
//	}
//}
//
//__global__ void c_FeatureSum(int nPoints, int nDims, int *d_Class, float *d_Mean, float *d_Data) {
//	int tidx = blockDim.x * blockIdx.x + threadIdx.x;
//	int tidy = blockDim.y * blockIdx.y + threadIdx.y;
//	if (tidx < nPoints && tidy < nDims) {
//		int c = d_Class[tidx];
//		atomicAdd(&d_Mean[c*nDims + tidy], d_Data[tidx*nDims + tidy]);
//	}
//}
//
//__global__ void c_FeatureMean(int nClustersAlive, int nDims, int *d_AliveIndex, float *d_Mean, int *d_nClassMembers) {
//	int tidx = blockDim.x * blockIdx.x + threadIdx.x;
//	int tidy = blockDim.y * blockIdx.y + threadIdx.y;
//	if (tidx < nClustersAlive && tidy < nDims) {
//		int c = d_AliveIndex[tidx];
//		d_Mean[c*nDims + tidy] /= d_nClassMembers[c];
//	}
//}
//
//__global__ void c_AllVector2Mean(int nPoints, int nDims, float *d_AllVector2Mean, float* d_Mean, float *d_Data, int *d_Class) {
//	int tidx = blockDim.x * blockIdx.x + threadIdx.x;
//	int tidy = blockDim.y * blockIdx.y + threadIdx.y;
//	if (tidx < nPoints && tidy < nDims) {
//		int c = d_Class[tidx];
//		d_AllVector2Mean[tidx *nDims + tidy] = d_Data[tidx*nDims + tidy] - d_Mean[c*nDims + tidy];
//	}
//}
//__global__ void c_CorrectionTerm(int nDims, int NumUnmasked, int NumMasked, int NumPointsInThisClass,
//	int *d_CurrentUnmasked, int *d_CurrentMasked, float *d_cov, float *d_dig,
//	int *d_PointsInThisClass, float *d_CorrectionTerm,
//	int priorPoint, float *d_NoiseVariance,float factor
//	) {
//	int tid = blockDim.x * blockIdx.x + threadIdx.x;
//	if (tid < NumUnmasked)
//	{
//		float ccf = 0.0;
//		int i = d_CurrentUnmasked[tid];
//		for (int q = 0; q<NumPointsInThisClass; q++)
//		{
//			const int p = d_PointsInThisClass[q];
//			ccf += d_CorrectionTerm[p*nDims + i];
//		}
//		d_cov[tid*NumUnmasked + tid] += (ccf +priorPoint * d_NoiseVariance[i]);
//		for (int ii = 0;ii < NumUnmasked;ii++)
//			d_cov[tid*NumUnmasked + ii] *= factor;
//	}
//	__syncthreads();
//	if (tid < NumMasked)
//	{
//		float ccf = 0.0;
//		int i = d_CurrentMasked[tid];
//		for (int q = 0; q<NumPointsInThisClass; q++)
//		{
//			const int p = d_PointsInThisClass[q];
//			ccf += d_CorrectionTerm[p*nDims + i];
//		}
//		d_dig[tid] = (ccf + priorPoint * d_NoiseVariance[i]);
//		d_dig[tid] *= factor;
//	}
//}
//
//
////==================================E step kernel===========================================//
//__global__ void initLogP(int nPoints, int MaxPossibleClusters, float *d_Weight, float *d_LogP) {
//	int tidx = blockDim.x * blockIdx.x + threadIdx.x;
//	if (tidx < nPoints)
//		d_LogP[tidx * MaxPossibleClusters + 0] = (float)-log(d_Weight[0]);
//}
//
//__global__ void makeTriCov(int nunmasked, float *d_cov) {
//	int tidx = blockDim.x * blockIdx.x + threadIdx.x;
//	int tidy = blockDim.y * blockIdx.y + threadIdx.y;
//	if (tidx < nunmasked&&tidy < nunmasked && tidy > tidx)
//		d_cov[tidx * nunmasked + tidy] = 0;
//}
//
//__global__ void checkDigSingular(int nmasked, int *devInfo1, float *d_dig, float *d_invdig) {
//	int tid = blockDim.x * blockIdx.x + threadIdx.x;
//	if (tid < nmasked) {
//		if (d_dig[tid] <= 0) atomicAdd(&devInfo1[0], 1);
//		else d_invdig[tid] = (float)sqrt(d_dig[tid]);
//	}
//}
//
//__global__ void makeUnitMat(int nunmasked, float *d_unitMat) {
//	int tid = blockDim.x * blockIdx.x + threadIdx.x;
//	if (tid < nunmasked) {
//		for(int i =0;i < nunmasked;i++)
//			if(tid == i) d_unitMat[tid * nunmasked + i] = -1.0;
//			else d_unitMat[tid * nunmasked + i] = 0.0;
//	}
//}
//
//__global__ void makeCov(int nunmasked,float *d_cov,float *d_temp) {
//	int tid = blockDim.x * blockIdx.x + threadIdx.x;
//	if (tid < nunmasked)
//		d_temp[tid] = log(d_cov[tid * nunmasked + tid]);
//}
//
//__global__ void makeDig(int nunmasked,int nmasked, float *d_choleskyDig, float *d_temp) {
//	int tid = blockDim.x * blockIdx.x + threadIdx.x;
//	if (tid < nmasked)
//		d_temp[tid + nunmasked] = log(d_choleskyDig[tid]);
//}
//
//__global__ void transInvCov(int nunmasked,
//	float *d_solver, float *d_InvCovDiag,int *d_CurrentUnmasked) {
//	int tid = blockDim.x * blockIdx.x + threadIdx.x;
//	if (tid < nunmasked)
//		d_InvCovDiag[d_CurrentUnmasked[tid]] = d_solver[tid];
//}
//
//__global__ void transInvDig(int nmasked,
//	float *d_choleskyDig, float *d_InvCovDiag,int *d_CurrentMasked) {
//	int tid = blockDim.x * blockIdx.x + threadIdx.x;
//	if (tid < nmasked)
//		d_InvCovDiag[d_CurrentMasked[tid]] = 1.0 / (d_choleskyDig[tid]*d_choleskyDig[tid]);
//}
//
//__global__ void pow2(int rows, int cols, float *d_unitMat) {
//	int tid = blockDim.x * blockIdx.x + threadIdx.x;
//	if (tid < rows * cols)
//		d_unitMat[tid] *= d_unitMat[tid];
//}
//
//__global__ void extractPoints2Mean(int nPoints, int nDims, int clusterId,
//	float *d_Data,float *d_Mean,float *d_points2Mean) {
//	//extern __shared__ float s_Mean[];
//	int tidx = blockDim.x * blockIdx.x + threadIdx.x;
//	int tidy = blockDim.y * blockIdx.y + threadIdx.y;
//	//int tid = (gridDim.x*blockDim.x)*tidy + tidx;
//	//if (tid < nDims) s_Mean[threadIdx.x] = d_Mean[clusterId * nDims + tid];
//	__syncthreads();
//	if (tidx < nDims && tidy < nPoints) {
//		d_points2Mean[tidy * nDims + tidx] =
//			//-(d_Data[tidy * nDims + tidx] - s_Mean[tidx]);
//			-(d_Data[tidy * nDims + tidx] - d_Mean[clusterId * nDims + tidx]);
//	}
//}
//
//
//__global__ void extractUnmaskedPoints2Mean(int nUpdatePoints, int nunmasked, int nDims,
//	int *d_updatePointsList,int *d_CurrentUnmasked,float *d_points2Mean,float *d_unMaskedPoints2Mean) {
//	int tidx = blockDim.x * blockIdx.x + threadIdx.x;
//	int tidy = blockDim.y * blockIdx.y + threadIdx.y;
//	if (tidx < nunmasked && tidy < nUpdatePoints) {
//		int nx = d_CurrentUnmasked[tidx];
//		int ny = d_updatePointsList[tidy];
//		d_unMaskedPoints2Mean[tidy * nunmasked + tidx] = -d_points2Mean[ny * nDims + nx];
//	}
//}
//
//__global__ void maskedSolver(int nUpdatePoints, int nDims, int nmasked, int* d_updatePointsList,int *d_CurrentMasked, float *d_choleskyDig, float *d_points2Mean, float *d_maskedSolver)
//{
//	int tidx = blockDim.x * blockIdx.x + threadIdx.x;
//	int tidy = blockDim.y * blockIdx.y + threadIdx.y;
//	if (tidx < nmasked && tidy < nUpdatePoints) {
//		int nx = d_CurrentMasked[tidx];
//		int ny = d_updatePointsList[tidy];
//		d_maskedSolver[tidy * nmasked + tidx] =
//			(d_points2Mean[ny * nDims + nx] * d_points2Mean[ny * nDims + nx]) / (d_choleskyDig[tidx] * d_choleskyDig[tidx]);
//	}
//}
//
//__global__ void calSubMahal(int nUpdatePoints, int nDims, int *d_updatePointsList,float *d_InvCovDiag, float *d_CorrectionTerm, float *d_subMahal) {
//	//__shared__ float sharemem[1654 * 96];
//	int tidx = blockDim.x * blockIdx.x + threadIdx.x;
//	int tidy = blockDim.y * blockIdx.y + threadIdx.y;
//
//	if (tidx < nDims && tidy < nUpdatePoints) {
//		int ny = d_updatePointsList[tidy];
//		d_subMahal[tidy * nDims + tidx] = d_CorrectionTerm[ny * nDims + tidx] * d_InvCovDiag[tidx];
//	}
//		
//}
//
////unmasked > 0
//__global__ void calLogP1(int MaxPossibleClusters, int clusterId, int nUpdatePoints, int nDims, float LogRootDet, 
//	float correction_factor, float *d_unmaskedSolver, float *d_maskedSolver, float *d_subMahalSolver,
//	int *d_updatePointsList, float *d_Weight, float *d_LogP) {
//	int tidx = blockDim.x * blockIdx.x + threadIdx.x;
//	if (tidx < nUpdatePoints) {
//		int nx = d_updatePointsList[tidx];
//		d_LogP[nx*MaxPossibleClusters + clusterId] = 
//			(d_unmaskedSolver[tidx] + d_maskedSolver[tidx] + d_subMahalSolver[tidx] * correction_factor)/ 2.0
//			+ LogRootDet
//			- log(d_Weight[clusterId])
//			+ (float)(0.5*log(2 * M_PI))*nDims;
//	}
//}
////unmasked == 0
//__global__ void calLogP2(int MaxPossibleClusters, int clusterId, int nUpdatePoints, int nDims, float LogRootDet,
//	float correction_factor, float *d_maskedSolver, float *d_subMahalSolver,
//	int *d_updatePointsList, float *d_Weight, float *d_LogP) {
//	int tidx = blockDim.x * blockIdx.x + threadIdx.x;
//	if (tidx < nUpdatePoints) {
//		int nx = d_updatePointsList[tidx];
//		d_LogP[nx*MaxPossibleClusters + clusterId] =
//			(d_maskedSolver[tidx] + d_subMahalSolver[tidx] * correction_factor) / 2.0
//			+ LogRootDet
//			- log(d_Weight[clusterId])
//			+ (float)(0.5*log(2 * M_PI))*nDims;
//	}
//}
//
//
//
//__global__ void initMarkClass(int mark, int nPoints, int *d_Class, int *d_markClass) {
//	int tidx = blockDim.x * blockIdx.x + threadIdx.x;
//	if (tidx < nPoints) {
//		if (d_Class[tidx] == mark)
//			d_markClass[tidx] = 1;
//		else d_markClass[tidx] = 0;
//	}
//}
//
//
//__global__ void replace(int nPoints, int *d_sourceTemp, int *d_dist) {
//	int tidx = blockDim.x * blockIdx.x + threadIdx.x;
//	if ((tidx == 0) && (d_sourceTemp[0] == 1)) d_dist[0] = 0;
//	if (tidx < nPoints && tidx > 0) {
//		if (d_sourceTemp[tidx] - d_sourceTemp[tidx - 1] == 1) {
//			d_dist[d_sourceTemp[tidx] - 1] = tidx;
//		}
//	}
//}
//
//__global__ void calPointsList(int nPoints, int *d_sourceTemp, int *d_dist) {
//	int tidx = blockDim.x * blockIdx.x + threadIdx.x;
//	if (tidx < nPoints) {
//		if ((tidx == 0) && d_sourceTemp[0]) d_dist[0] = 0;
//		else if (d_sourceTemp[tidx] > d_sourceTemp[tidx - 1]) d_dist[d_sourceTemp[tidx] - 1] = tidx;
//	}
//}
//
//__global__ void UnSkipeednPoints(int nPoints,int clusterId,int FullStep, int MaxPossibleClusters, float DistThresh, int *d_Class, int *d_OldClass, float *d_LogP, int *d_pIndex) {
//	int tidx = blockDim.x * blockIdx.x + threadIdx.x;
//	if (tidx < nPoints) {
//		if (!FullStep && (d_Class[tidx] == d_OldClass[tidx]) && (d_LogP[tidx*MaxPossibleClusters + clusterId] - d_LogP[tidx*MaxPossibleClusters + d_Class[tidx]] > DistThresh))
//			d_pIndex[tidx] = 0;
//		else d_pIndex[tidx] = 1;
//	}
//}
//
////==============================copy in Host and Device=========================
//__global__ void repla(int MaxPossibleClusters, int *d_sourceTemp, int *d_dist) {
//	int tidx = blockDim.x * blockIdx.x + threadIdx.x;
//	if (tidx < MaxPossibleClusters && tidx > 0) {
//		if (d_sourceTemp[tidx] - d_sourceTemp[tidx - 1] == 1) {
//			d_dist[d_sourceTemp[tidx] - 1] = tidx;
//		}
//	}
//}
//void KK::Reindex()
//{
//	if (useCpu) {
//		AliveIndex[0] = 0;
//		nClustersAlive = 1;
//		for (int c = 1; c < MaxPossibleClusters; c++)
//			if (ClassAlive[c]) {
//				AliveIndex[nClustersAlive] = c;
//				nClustersAlive++;
//			}
//	}
//	else {
//		gpuErrchk(cudaMemcpy(d_ClassAliveTemp, d_ClassAlive, MaxPossibleClusters * sizeof(int), cudaMemcpyDeviceToDevice));
//		nClustersAlive = scanKernel(d_ClassAliveTemp, MaxPossibleClusters);
//		repla << <(MaxPossibleClusters + BLOCKDIM - 1)/ BLOCKDIM, BLOCKDIM >> >(MaxPossibleClusters, d_ClassAliveTemp, d_AliveIndex);
//	}
//}
//
////__global__ void init_dones(int nDims, float *d_ones) {
////	int tid = blockDim.x * blockIdx.x + threadIdx.x;
////	if (tid < 2 * nDims) d_ones[tid] = 1.0;
////}
//void KK::CopyHostToDevice() {
//	int sizeI = sizeof(int);
//	int sizeF = sizeof(float);
//
//	gpuErrchk(cudaMalloc((void **)&d_Class, nPoints*sizeI));//
//	gpuErrchk(cudaMalloc((void **)&d_Data, nPoints*nDims*sizeF));//
//	gpuErrchk(cudaMalloc((void **)&d_nClassMembers, MaxPossibleClusters*sizeI));//
//
//	gpuErrchk(cudaMalloc((void **)&d_ClassAlive, MaxPossibleClusters*sizeI));//
//	gpuErrchk(cudaMalloc((void **)&d_AliveIndex, MaxPossibleClusters*sizeI));
//	gpuErrchk(cudaMalloc((void **)&d_NoiseMean, nDims*sizeF));//
//	gpuErrchk(cudaMalloc((void **)&d_NoiseVariance, nDims*sizeF));//
//	gpuErrchk(cudaMalloc((void **)&d_CorrectionTerm, nPoints*nDims*sizeF));//
//	gpuErrchk(cudaMalloc((void **)&d_FloatMasks, nPoints*nDims*sizeF));//
//	gpuErrchk(cudaMalloc((void **)&d_UnMaskDims, nPoints*sizeF));//
//	gpuErrchk(cudaMalloc((void **)&d_ClusterMask, MaxPossibleClusters*nDims*sizeF));
//
//	gpuErrchk(cudaMalloc((void **)&d_Mean, MaxPossibleClusters*nDims*sizeF));//
//	gpuErrchk(cudaMalloc((void **)&d_Weight, MaxPossibleClusters*sizeF));//
//	gpuErrchk(cudaMalloc((void **)&d_LogP, MaxPossibleClusters*nPoints*sizeF));
//	gpuErrchk(cudaMalloc((void **)&d_OldClass, nPoints*sizeI));//
//	gpuErrchk(cudaMalloc((void **)&d_ClassPenalty, MaxPossibleClusters*sizeF));
//	gpuErrchk(cudaMalloc((void **)&d_Class2, nPoints*sizeI));
//	gpuErrchk(cudaMalloc((void **)&d_BestClass, nPoints*sizeI));
//
//	//temp variables
//	gpuErrchk(cudaMalloc((void **)&d_ClassAliveTemp, MaxPossibleClusters*sizeI));
//	gpuErrchk(cudaMalloc((void **)&d_DeletionLoss, MaxPossibleClusters*sizeF));
//	gpuErrchk(cudaMalloc((void **)&d_tempSubtraction, MaxPossibleClusters*sizeF));
//	gpuErrchk(cudaMalloc((void **)&d_tempLogP, nPoints*sizeF));
//	gpuErrchk(cudaMalloc((void **)&d_tempOldClass, nPoints*sizeF));
//	//MEstep
//	gpuErrchk(cudaMalloc((void **)&d_unmaskedSolver, nPoints*sizeF));
//	gpuErrchk(cudaMalloc((void **)&d_AllVector2Mean, nPoints*nDims * sizeof(float)));
//	gpuErrchk(cudaMalloc((void **)&d_Current, MaxPossibleClusters * nDims * sizeof(int)));
//	gpuErrchk(cudaMalloc((void **)&d_PointsInThisClass, nPoints * sizeof(int)));
//	gpuErrchk(cudaMalloc((void **)&d_MarkClass, nPoints * sizeof(int)));
//	gpuErrchk(cudaMalloc((void **)&d_Offset, MaxPossibleClusters * sizeof(int)));
//	//for loop E step
//	gpuErrchk(cudaMalloc((void **)&d_pIndex, nPoints * sizeof(int)));
//	gpuErrchk(cudaMalloc((void **)&d_points2Mean, nPoints*nDims * sizeof(float)));
//	gpuErrchk(cudaMalloc((void **)&d_InvCovDiag, nDims * sizeof(float)));
//	gpuErrchk(cudaMalloc((void **)&d_temp, nDims * sizeof(float)));
//	gpuErrchk(cudaMalloc((void **)&d_updatePointsList, nPoints * sizeof(int)));
//	////
//	//gpuErrchk(cudaMalloc((void **)&d_ones, 2000 * sizeof(float)));
//	//===============================================
//	//======================================init array===============================================
//	gpuErrchk(cudaMemcpy(d_Class , &Class[0], nPoints*sizeI, cudaMemcpyHostToDevice));
//	gpuErrchk(cudaMemcpy(d_Data, &Data[0], nPoints*nDims*sizeF, cudaMemcpyHostToDevice));
//	gpuErrchk(cudaMemcpy(d_nClassMembers, &nClassMembers[0], MaxPossibleClusters*sizeI, cudaMemcpyHostToDevice));//==
//
//	gpuErrchk(cudaMemcpy(d_ClassAlive, &ClassAlive[0], MaxPossibleClusters*sizeI, cudaMemcpyHostToDevice));//==
//	gpuErrchk(cudaMemcpy(d_AliveIndex, &AliveIndex[0], MaxPossibleClusters*sizeI, cudaMemcpyHostToDevice));//==
//	gpuErrchk(cudaMemcpy(d_NoiseMean, &NoiseMean[0], nDims*sizeF, cudaMemcpyHostToDevice));
//	gpuErrchk(cudaMemcpy(d_NoiseVariance, &NoiseVariance[0], nDims*sizeF, cudaMemcpyHostToDevice));
//	gpuErrchk(cudaMemcpy(d_CorrectionTerm, &CorrectionTerm[0], nPoints*nDims*sizeF, cudaMemcpyHostToDevice));
//	gpuErrchk(cudaMemcpy(d_FloatMasks, &FloatMasks[0], nPoints*nDims*sizeF, cudaMemcpyHostToDevice));
//	gpuErrchk(cudaMemcpy(d_UnMaskDims, &UnMaskDims[0], nPoints*sizeF, cudaMemcpyHostToDevice));
//	gpuErrchk(cudaMemcpy(d_ClusterMask, &ClusterMask[0], MaxPossibleClusters*nDims*sizeF, cudaMemcpyHostToDevice));
//
//	//gpuErrchk(cudaMemcpy(d_Mean, &Mean[0], MaxPossibleClusters*nDims*sizeF, cudaMemcpyHostToDevice));//==
//	//gpuErrchk(cudaMemcpy(d_Weight, &Weight[0], MaxPossibleClusters*sizeF, cudaMemcpyHostToDevice));//==
//	//gpuErrchk(cudaMemcpy(d_LogP, &LogP[0], MaxPossibleClusters*nPoints*sizeF, cudaMemcpyHostToDevice));//==
//	//gpuErrchk(cudaMemcpy(d_OldClass, &OldClass[0], nPoints*sizeI, cudaMemcpyHostToDevice));//==
//	//gpuErrchk(cudaMemcpy(d_ClassPenalty, &ClassPenalty[0], MaxPossibleClusters*sizeF, cudaMemcpyHostToDevice));//==
//	//gpuErrchk(cudaMemcpy(d_Class2, &Class2[0], nPoints*sizeI, cudaMemcpyHostToDevice));//==
//	//gpuErrchk(cudaMemcpy(d_BestClass, &BestClass[0], nPoints*sizeI, cudaMemcpyHostToDevice));//==
//
//	//init_dones << <((2 * nDims) + BLOCKDIM - 1) / BLOCKDIM, BLOCKDIM >> > (nDims, d_ones);
//
//}
//
//void KK::FreeArray() {
//	cudaFree(d_Class);               cudaFree(d_nClassMembers);
//    cudaFree(d_Data);
//
//	cudaFree(d_ClassAlive);          cudaFree(d_AliveIndex);
//	cudaFree(d_NoiseMean);           cudaFree(d_NoiseVariance);
//	cudaFree(d_CorrectionTerm);      cudaFree(d_FloatMasks);
//	cudaFree(d_UnMaskDims);          cudaFree(d_ClusterMask);
//
//	cudaFree(d_Mean);                cudaFree(d_Weight);
//	cudaFree(d_LogP);                cudaFree(d_OldClass);
//	cudaFree(d_ClassPenalty);        cudaFree(d_Class2);
//	cudaFree(d_BestClass);           cudaFree(d_ClassAliveTemp);
//
//	cudaFree(d_DeletionLoss);        cudaFree(d_tempLogP);                  cudaFree(d_tempSubtraction);
//	cudaFree(d_tempOldClass);        cudaFree(d_unmaskedSolver);
//
//	cudaFree(d_Current);             cudaFree(d_Offset);
//	cudaFree(d_PointsInThisClass);   cudaFree(d_MarkClass);
//	cudaFree(d_pIndex);              cudaFree(d_points2Mean);
//	cudaFree(d_InvCovDiag);          cudaFree(d_temp);
//	cudaFree(d_updatePointsList);    cudaFree(d_AllVector2Mean);
//
//	cudaFree(d_ones);
//	//cudaDeviceReset();
//}
//
////=====================================ComputeClusterMasks======================================
//__global__ void initClusterMask(int nDims, int MaxPossibleClusters, float *d_ClusterMask) {
//	int tidx = blockDim.x * blockIdx.x + threadIdx.x;
//	if (tidx < nDims*MaxPossibleClusters) d_ClusterMask[tidx] = 0.0;
//}
//
//__global__ void ComputeClusterMask(int nPoints, int nDims, int *d_Class, float *d_ClusterMask, float *d_FloatMasks) {
//	int tidx = blockDim.x * blockIdx.x + threadIdx.x;
//	if (tidx < nPoints) {
//		int c = d_Class[tidx];
//		for (int i = 0; i < nDims; i++) atomicAdd(&d_ClusterMask[c*nDims + i], d_FloatMasks[tidx*nDims + i]);
//	}
//}
////================================================ComputeClusterMasks()===============================================================
////blocksize > nDims  && share memory = blocksize * 6;
//__global__ void updateCurrent(int nClustersAlive, int nDims, float PointsForClusterMask, float *d_ClusterMask, int *d_AliveIndex,
//	int *d_offset, int *d_Current) {
//	__shared__ int temp[BLOCK_SIZE * 6];
//	int* temp1 = (int*)temp;
//	int* temp2 = (int*)&temp[BLOCK_SIZE * 3];
//	int tid = threadIdx.x;
//	temp1[tid] = 0;
//	temp2[tid] = 0;
//
//	if (tid < nDims && blockIdx.x < nClustersAlive) {
//		if (d_ClusterMask[d_AliveIndex[blockIdx.x] * nDims + tid] >= PointsForClusterMask)
//			temp1[tid] = 1;
//		else temp2[tid] = 1;
//	}
//	__syncthreads();
//	//// the code below performs iterative scan on XY　　
//	for (unsigned int stride = 1; stride <= BLOCK_SIZE; stride *= 2) {
//		int index = (threadIdx.x + 1)*stride * 2 - 1;
//		if (index < 2 * BLOCK_SIZE)
//		{
//			temp1[index] += temp1[index - stride];//index is alway bigger than stride
//			temp2[index] += temp2[index - stride];
//		}
//		__syncthreads();
//	}
//	__syncthreads();
//	for (unsigned int stride = BLOCK_SIZE / 2; stride > 0; stride /= 2) {
//		int index = (threadIdx.x + 1)*stride * 2 - 1;
//		if (index < 2 * BLOCK_SIZE)
//		{
//			temp1[index + stride] += temp1[index];
//			temp2[index + stride] += temp2[index];
//		}
//		__syncthreads();
//	}
//	__syncthreads();
//	if (tid == 0) {
//		d_offset[blockIdx.x] = temp[nDims - 1];
//		if (temp1[0]) d_Current[blockIdx.x * nDims] = 0;
//		else d_Current[blockIdx.x * nDims + d_offset[blockIdx.x]] = 0;
//	}
//	__syncthreads();
//	if (tid < nDims && tid > 0) {
//		if (temp1[tid] > temp1[tid - 1]) d_Current[blockIdx.x * nDims + temp1[tid] - 1] = tid;
//		if (temp2[tid] > temp2[tid - 1]) d_Current[blockIdx.x * nDims + temp2[tid] - 1 + d_offset[blockIdx.x]] = tid;
//	}
//}
//void KK::ComputeClusterMasks()
//{
//	Reindex();
//	int grid = (nDims*MaxPossibleClusters + BLOCKDIM - 1)/ BLOCKDIM;
//	initClusterMask << < grid, BLOCKDIM >> > (nDims, MaxPossibleClusters, d_ClusterMask);
//	ComputeClusterMask << <(nPoints+BLOCKDIM - 1)/BLOCKDIM, BLOCKDIM >> > (nPoints,nDims,d_Class,d_ClusterMask,d_FloatMasks);
//	updateCurrent<<<nClustersAlive ,128>>>(nClustersAlive, nDims, PointsForClusterMask, d_ClusterMask, d_AliveIndex,d_Offset, d_Current);
//	cudaMemcpy(&Offset[0], d_Offset, MaxPossibleClusters * sizeof(int), cudaMemcpyDeviceToHost);
//}
////============================================================================================
//__global__ void initMeanAndClassMember(int MaxPossibleClusters, int nDims, int *d_nClassMembers, float *d_Mean) {
//	int tidx = blockDim.x * blockIdx.x + threadIdx.x;
//	if (tidx < MaxPossibleClusters) {
//		d_nClassMembers[tidx] = 0;
//		for (int i = 0;i < nDims;i++)
//			d_Mean[tidx * nDims + i] = 0.0;
//	}
//}
//
//void KK::MEstep()
//{
//	clock_t clock1 = clock();
//	if (Debug) { Output("Entering Unmasked Mstep \n"); }
//	// clear arrays
//	initMeanAndClassMember << <(MaxPossibleClusters + BLOCKDIM - 1) / BLOCKDIM, BLOCKDIM >> > (MaxPossibleClusters, nDims,d_nClassMembers, d_Mean);
//
//
//	// Accumulate total number of points in each class
//	c_nClassMembers << <(nPoints + BLOCKDIM - 1) / BLOCKDIM, BLOCKDIM >> > (nPoints,d_Class,d_nClassMembers);
//
//	// check for any dead classes
//	c_CheckDead << <(nClustersAlive + BLOCKDIM - 1) / BLOCKDIM, BLOCKDIM >> > (nClustersAlive,d_AliveIndex,d_nClassMembers,d_ClassAlive);
//
//	Reindex();
//
//	c_Weight << <(nClustersAlive + BLOCKDIM - 1)/BLOCKDIM, BLOCKDIM >> > (nClustersAlive, priorPoint, nPoints, NoisePoint,
//		d_AliveIndex,d_nClassMembers,d_Weight);
//
//
//	//================================================compute Cov=======================================//  
//	dim3 block(Bd32, Bd32);
//
//	dim3 grid1((nPoints + Bd32 - 1)/Bd32, (nDims+Bd32 - 1)/Bd32);
//	c_FeatureSum << <grid1, block >> > (nPoints, nDims, d_Class,d_Mean,d_Data);
//
//	dim3 grid2((nClustersAlive + Bd32 - 1) / Bd32, (nDims + Bd32 - 1) / Bd32);
//	c_FeatureMean << <grid2, block >> >(nClustersAlive, nDims, d_AliveIndex,d_Mean,d_nClassMembers);
//
//	c_AllVector2Mean << <grid1, block >> >(nPoints, nDims, d_AllVector2Mean,d_Mean,d_Data,d_Class);
//	// Compute the cluster masks, used below to optimise the computation
//	ComputeClusterMasks();
//	clock_t clock2 = clock();
//	//================================================
//	/*
//	Output("d_AliveIndex:  %d\n", d_AliveIndex.size());
//	for (int i = 0;i < d_AliveIndex.size();i++) cout << d_AliveIndex[i]<< "  ";
//	Output("\n");
//	Output("d_ClassAlive:  %d\n", d_ClassAlive.size());
//	for (int i = 0;i < d_ClassAlive.size();i++) cout << d_ClassAlive[i] << "  ";
//	Output("\n");
//	*/
//
//	gpuErrchk(cudaMemcpy(&AliveIndex[0], d_AliveIndex, MaxPossibleClusters * sizeof(int), cudaMemcpyDeviceToHost));
//	gpuErrchk(cudaMemcpy(&nClassMembers[0], d_nClassMembers, MaxPossibleClusters * sizeof(int), cudaMemcpyDeviceToHost));
//	gpuErrchk(cudaMemcpy(&ClassAlive[0], d_ClassAlive, MaxPossibleClusters * sizeof(int), cudaMemcpyDeviceToHost));
//
//	//=================================================
//	float alpha = 1.f;
//	float beta = 0.0f;
//	//cublasSideMode_t side = CUBLAS_SIDE_LEFT;
//	//cublasFillMode_t uplo = CUBLAS_FILL_MODE_UPPER;
//	//cublasOperation_t trans = CUBLAS_OP_T;
//	//cublasDiagType_t diag = CUBLAS_DIAG_NON_UNIT;
//
//	Output("this is the ME step ,there have %d clusters...\n", nClustersAlive);
//	for (int cc = 0; cc < nClustersAlive; cc++)
//	{
//		const int c = AliveIndex[cc];
//
//		///thrust::device_vector<int> d_PointsInThisClass(nPoints);
//		///thrust::device_vector<int> d_CurrentUnmasked(nDims);
//		///thrust::device_vector<int> d_CurrentMasked(nDims);
//		///thrust::device_vector<int> d_MarkClass(nPoints);
//
//		initMarkClass << <(nPoints+BLOCKDIM - 1)/ BLOCKDIM, BLOCKDIM >>> (c, nPoints, d_Class,d_MarkClass);
//
//		int npoints = scanKernel(d_MarkClass, nPoints);
//		//printf("npoints    ====== >    %d\n", npoints);
//
//		replace << <(nPoints + BLOCKDIM - 1) / BLOCKDIM, BLOCKDIM >> >(nPoints,d_MarkClass,d_PointsInThisClass);
//
//		const int nunmasked = Offset[cc];
//		const int nmasked = nDims - nunmasked;
//		int *d_CurrentUnmasked = (int *)&d_Current[cc * nDims];
//		int *d_CurrentMasked = (int *)&d_Current[cc * nDims + Offset[cc]];
//
//		float *d_cov;           gpuErrchk(cudaMalloc((void **)&d_cov, nunmasked * nunmasked * sizeof(float)));
//		float *d_dig;           gpuErrchk(cudaMalloc((void **)&d_dig, nmasked * sizeof(float)));
//		
//		//printf("nunmasked = %d,nmasked = %d, npoints = %d\n",nunmasked,nmasked,npoints);
//
//		//
//		//int *h_PointsInThisClass = (int *)malloc(nPoints * sizeof(int));
//		//gpuErrchk(cudaMemcpy(h_PointsInThisClass, d_PointsInThisClass, nPoints * sizeof(int), cudaMemcpyDeviceToHost));
//		//int *h_CurrentUnmasked = (int *)malloc(nDims * sizeof(int));
//		//gpuErrchk(cudaMemcpy(h_CurrentUnmasked, d_CurrentUnmasked, nDims * sizeof(int), cudaMemcpyDeviceToHost));
//		//
//		//Output("h_PointsInThisClass.size: %d\n", nPoints);
//		//for (int i = 0; i < nPoints; i++) std::cout << h_PointsInThisClass[i] << "  ";
//		//Output("\n\n\n");
//		//Output("h_CurrentUnmasked.size: %d\n", nDims);
//		//for (int i = 0; i < nDims; i++) std::cout << h_CurrentUnmasked[i] << "  ";
//		//Output("\n\n\n");
//
//		//printf("d_PointsInThisClass.size: %d\n", d_PointsInThisClass.size());
//		//for (int i = 0; i < npoints; i++) std::cout << d_PointsInThisClass[i] << "  ";
//		//printf("\n\n\n");
//		//printf("d_CurrentUnmasked.size: %d\n", d_CurrentUnmasked.size());
//		//for (int i = 0; i < nunmasked; i++) std::cout << d_CurrentUnmasked[i] << "  ";
//		//printf("\n\n\n");
//		
//		if (nunmasked > 0 && npoints > 0)
//		{
//			float *d_X;           gpuErrchk(cudaMalloc((void **)&d_X, npoints * nunmasked * sizeof(float)));
//			dim3 grid3(nunmasked/32 + 1, npoints/32 + 1);
//			matCopy << <grid3, block >> > (npoints, nunmasked, nDims,
//				d_PointsInThisClass,d_CurrentUnmasked,d_AllVector2Mean,d_X);
//
//			cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_T, nunmasked, nunmasked, npoints, &alpha,
//				d_X, nunmasked,d_X, nunmasked, &beta, d_cov, nunmasked);
//
//			cudaFree(d_X);
//		}
//
//		float factor = (nClassMembers[c] == 0) ? ((float)1.0 ): ((float)1.0 / (nClassMembers[c] + priorPoint - 1));
//		c_CorrectionTerm << < 1, BLOCKDIM >> > (nDims, nunmasked, nmasked, npoints,
//			d_CurrentUnmasked,d_CurrentMasked,d_cov,d_dig,
//			d_PointsInThisClass,d_CorrectionTerm,
//			priorPoint, d_NoiseVariance,factor);
//
//		//printf("factor  =  %f\n", factor);
//		//Output("d_cov.size: %d\n", nunmasked * nunmasked);
//		//float *cov =(float *)malloc(nunmasked * nunmasked * sizeof(float));
//		//gpuErrchk(cudaMemcpy(cov, d_cov, nunmasked * nunmasked * sizeof(float), cudaMemcpyDeviceToHost));
//		//float *dig = (float *)malloc(nmasked * sizeof(float));
//		//gpuErrchk(cudaMemcpy(dig, d_dig, nmasked * sizeof(float), cudaMemcpyDeviceToHost));
//		//for (int i = 0; i < nunmasked * nunmasked; i++) std::cout << cov[i] << "  ";
//		//Output("\n\n\n");
//		//Output("d_dig.size: %d\n", nmasked);
//		//for (int i = 0; i < nmasked; i++) std::cout << dig[i] << "  ";
//		//Output("\n\n\n");
//		
//		//========================================================E step===========================================================//
//
//		if (cc == 0){
//			initLogP << <nPoints/BLOCKDIM + 1, BLOCKDIM >> > (nPoints, MaxPossibleClusters,d_Weight,d_LogP);
//			continue;
//		}
//
//		////用于之后的计算points的值  ==========================================================================
//		dim3 grid4((nDims + 31)/ 32, (nPoints + 31) / 32);
//		extractPoints2Mean << < grid4, block ,nDims * sizeof(float)>> > (nPoints, nDims, c,d_Data,d_Mean,d_points2Mean);
//		float LogRootDet = 0.0;
//		/*==============================计算需要更新LogP的points的index=================================*/
//		UnSkipeednPoints<<<(nPoints+BLOCKDIM - 1)/BLOCKDIM, BLOCKDIM>>>(nPoints,c,FullStep, MaxPossibleClusters, DistThresh,
//			d_Class,d_OldClass,d_LogP,d_pIndex);
//
//		
//		//int nUpdatePoints = reduceInt<128>(d_pIndex, nPoints);
//		int nUpdatePoints = scanKernel(d_pIndex, nPoints);
//		//Output("nUpdatePoints == %d\n",nUpdatePoints);
//		calPointsList << <(nPoints+BLOCKDIM - 1)/BLOCKDIM, BLOCKDIM >> >(nPoints,d_pIndex,d_updatePointsList);
//
//		/*==============================================================================================*/
//
//		if (nunmasked > 0 && npoints > 0) {
//			/*处理unmasked部分的d_block，进行cholesky分解，判断是否奇异，并更新*/
//			// --- cuSOLVE input/output parameters/arrays
//			int work_size = 0;
//			int *devInfo;           cudaMalloc(&devInfo, sizeof(int));
//
//			// --- CUDA CHOLESKY initialization
//			cusolverDnSpotrf_bufferSize(solver_handle, CUBLAS_FILL_MODE_UPPER, nunmasked,
//				d_cov, nunmasked, &work_size);
//
//			// --- CUDA POTRF execution
//			float *work;   cudaMalloc(&work, work_size * sizeof(float));
//			cusolverDnSpotrf(solver_handle, CUBLAS_FILL_MODE_UPPER, nunmasked,
//				d_cov, nunmasked,work, work_size, devInfo);
//
//			int devInfo_h = 0;
//			cudaMemcpy(&devInfo_h, devInfo, sizeof(int), cudaMemcpyDeviceToHost);
//
//			cudaFree(devInfo);cudaFree(work);
//
//			if (devInfo_h != 0) {
//				//Output("Unmasked E-step: Deleting class %d (%d points): covariance matrix is singular \n", (int)c, (int)d_NumberInClass[c]);
//				Output("Unmasked E-step: Deleting class %d : covariance matrix is singular in the block!\n", (int)c);
//				ClassAlive[c] = 0;
//				continue;
//			}
//			
//
//
//			/*计算 logrootdet*/
//			makeCov << <1, BLOCKDIM >> > (nunmasked, d_cov,d_temp);
//
//			/*计算cov的逆矩阵，并通过逆矩阵更新d_InvCovDiag*/
//			float *d_unitMat;           gpuErrchk(cudaMalloc((void **)&d_unitMat, nunmasked*nunmasked * sizeof(float)));
//			makeUnitMat << < 1, BLOCKDIM >> > (nunmasked, d_unitMat);
//
//			cublasStrsm(handle, side, uplo, trans, diag,
//				nunmasked, nunmasked, &alpha, d_cov, nunmasked,d_unitMat, nunmasked);
//
//			pow2 << <nunmasked * nunmasked / BLOCKDIM + 1, BLOCKDIM >> >(nunmasked, nunmasked, d_unitMat);
//
//			float *d_solver;           gpuErrchk(cudaMalloc((void **)&d_solver,nunmasked * sizeof(float)));
//			cublasSgemv(handle, CUBLAS_OP_T, nunmasked, nunmasked, &alpha, d_unitMat, nunmasked,
//				d_ones, 1, &beta, d_solver, 1);
//
//			/*更新d_InvCovDiag*///----------------------------------------------------------------
//			transInvCov << <1, BLOCKDIM >> > (nunmasked,d_solver,d_InvCovDiag,d_CurrentUnmasked);
//
//			//=============================更新剪枝后的spike的值========================================
//			/*提取unmasked部分的（电位-均值）*/
//			float *d_unMaskedPoints2Mean;           gpuErrchk(cudaMalloc((void **)&d_unMaskedPoints2Mean, nUpdatePoints*nunmasked * sizeof(float)));
//			dim3 grid5(nunmasked / 32 + 1, nUpdatePoints / 32 + 1);
//			extractUnmaskedPoints2Mean << < grid5, block >> > (nUpdatePoints, nunmasked, nDims,
//				d_updatePointsList,d_CurrentUnmasked,d_points2Mean,d_unMaskedPoints2Mean);
//
//			/*计算每个Point的解*/
//			cublasStrsm(handle, side, uplo, trans, diag,
//				nunmasked, nUpdatePoints, &alpha, d_cov, nunmasked,d_unMaskedPoints2Mean, nunmasked);
//
//			pow2 << <((nUpdatePoints*nunmasked) / BLOCKDIM) + 1, BLOCKDIM >> >(nUpdatePoints, nunmasked, d_unMaskedPoints2Mean);
//			/*求每个Point的solver的平方和*/
//			cublasSgemv(handle, CUBLAS_OP_T, nunmasked, nUpdatePoints, &alpha, d_unMaskedPoints2Mean, nunmasked,
//				d_ones, 1, &beta, d_unmaskedSolver, 1);
//
//			cudaFree(d_unitMat);
//			cudaFree(d_solver);
//			cudaFree(d_unMaskedPoints2Mean);
//		}
//		//=============================================================================================
//		//=============================================================================================
//		//=============================================================================================
//
//		/*处理masked部分的d_dig，进行cholesky分解，判断是否奇异，并更新*/
//		/*
//		__global__ void checkDigSingular(int nunmasked, int *devInfo1, float *d_dig, float *d_choleskyDig) {
//		    int tid = blockDim.x * blockIdx.x + threadIdx.x;
//		    if (tid < nmasked) {
//		        if (d_dig[tid] <= 0) atomicAdd(&devInfo1[0], 1);
//		        else d_choleskyDig[tid] = (float)sqrt(d_dig[tid]);
//		    }
//		}*/
//		int *mark;           mark = (int *)malloc(sizeof(int));        mark[0] = 0;//memset(mark, 0, sizeof(mark));
//		int *d_mark;         cudaMalloc(&d_mark, sizeof(int));    cudaMemcpy(d_mark, mark, sizeof(int), cudaMemcpyHostToDevice);
//		float *d_choleskyDig;           gpuErrchk(cudaMalloc((void **)&d_choleskyDig, nmasked * sizeof(float)));
//
//		checkDigSingular << <1, BLOCKDIM >> > (nmasked, d_mark, d_dig,d_choleskyDig);
//		cudaMemcpy(mark, d_mark, sizeof(int), cudaMemcpyDeviceToHost);
//
//		cudaFree(d_mark);
//
//		if (mark[0] != 0) {
//			//Output("Unmasked E-step: Deleting class %d (%d points): covariance matrix is singular \n", (int)c, (int)NumberInClass[c]);
//			Output("Unmasked E-step: Deleting class %d : covariance matrix is singular in the dig!\n", (int)c);
//			ClassAlive[c] = 0;
//			cudaFree(d_choleskyDig);
//			continue;
//		}
//
//		/*通过cholesky分解之后的对角线更新LogRootDet*/
//		makeDig<<<1, BLOCKDIM >>>(nunmasked,nmasked,d_choleskyDig,d_temp);
//
//		//LogRootDet = reduceFlt<128>(d_temp, nDims);
//		cublasSasum(handle, nDims, d_temp, 1, &LogRootDet);
//		LogRootDet = -LogRootDet;
//
//		//Output("LogRootDet: %f\n", LogRootDet);
//
//		//cusolverDnSpotrs(solver_handle, uplo, nunmasked, nunmasked,
//			//thrust::raw_pointer_cast(&d_cov[0]), nunmasked,
//			//thrust::raw_pointer_cast(&d_unitMat[0]), nunmasked, devInfo);
//
//		transInvDig << <1, BLOCKDIM >> > ( nmasked,d_choleskyDig,d_InvCovDiag,d_CurrentMasked);
//
////======================================================================================================
////======================================================================================================
////======================================================================================================
//		/*===================计算unmasked部分的线性方程组的解，同时剪掉不用计算的nPoints===============*/
//		float *d_maskedPoints2Mean;           gpuErrchk(cudaMalloc((void **)&d_maskedPoints2Mean, nUpdatePoints*nmasked * sizeof(float)));
//		dim3 grid6(nmasked / 32 + 1, nUpdatePoints / 32 + 1);
//		maskedSolver << <grid6, block >> > (nUpdatePoints, nDims, nmasked,
//			d_updatePointsList,d_CurrentMasked,d_choleskyDig,d_points2Mean,d_maskedPoints2Mean);
//
//		float *d_maskedSolver;           gpuErrchk(cudaMalloc((void **)&d_maskedSolver, nUpdatePoints * sizeof(float)));
//		cublasSgemv(handle, CUBLAS_OP_T, nmasked, nUpdatePoints, &alpha, d_maskedPoints2Mean, nmasked,
//			d_ones, 1, &beta, d_maskedSolver, 1);
//
//
//		float *d_subMahal;           gpuErrchk(cudaMalloc((void **)&d_subMahal, nUpdatePoints*nDims * sizeof(float)));
//		dim3 grid7(nDims / 32 + 1, nUpdatePoints / 32 + 1);
//		calSubMahal << <grid7, block >> > (nUpdatePoints, nDims,d_updatePointsList,d_InvCovDiag,d_CorrectionTerm,d_subMahal);
//		float *d_subMahalSolver;           gpuErrchk(cudaMalloc((void **)&d_subMahalSolver, nUpdatePoints * sizeof(float)));
//		cublasSgemv(handle, CUBLAS_OP_T, nDims, nUpdatePoints, &alpha, d_subMahal, nDims,
//			d_ones, 1, &beta, d_subMahalSolver, 1);
//
//		if (nunmasked > 0 && npoints > 0)
//			calLogP1 << <(nUpdatePoints + BLOCKDIM - 1)/ BLOCKDIM, BLOCKDIM >> > (MaxPossibleClusters, c, nUpdatePoints, nDims, LogRootDet,
//				correction_factor, d_unmaskedSolver, d_maskedSolver,d_subMahalSolver,d_updatePointsList, d_Weight, d_LogP);
//		else
//			calLogP2 << <(nUpdatePoints + BLOCKDIM - 1) / BLOCKDIM, BLOCKDIM >> > (MaxPossibleClusters, c, nUpdatePoints, nDims, LogRootDet,
//				correction_factor, d_maskedSolver, d_subMahalSolver, d_updatePointsList, d_Weight, d_LogP);
//
//
//		cudaFree(d_cov);                cudaFree(d_dig);             cudaFree(d_choleskyDig);
//		cudaFree(d_maskedPoints2Mean);  cudaFree(d_maskedSolver);
//		cudaFree(d_subMahal);           cudaFree(d_subMahalSolver);
//		
//	}//for (int cc = 0; cc < nClustersAlive; cc++)
//
//	clock_t dif2 = clock() - clock2;
//	clock_t dif = clock() - clock1;
//	Output("---------------------------------------------->cost times:  %f ms\n", (float)dif / CLOCKS_PER_SEC);
//	Output("---------------------------------------------->for loop cost times:  %f ms\n", (float)dif2 / CLOCKS_PER_SEC);
//	Output("\n");
//}

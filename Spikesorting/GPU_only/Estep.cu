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

#include "dlinalg.cuh"

#include <stdio.h>
#include <iostream>

__global__ void initLogP(int MaxPossibleClusters, int nPoints, float *d_LogP, int *d_Weight, int *d_Class, int *d_NumberInClass) {
	int tid = blockDim.x * blockIdx.x + threadIdx.x;
	if (tid < nPoints) {
		d_LogP[tid*MaxPossibleClusters] = (float)-log(d_Weight[0]);
		//int ccc = d_Class[tid];
		atomicAdd(&d_NumberInClass[ d_Class[tid] ], 1);
	}
}

__global__ void copyAndCombin(int m, int n, float *A, float *B, float *C) {
	int tid = threadIdx.x;
	if (tid < m) {
		C[tid] = log(A[tid * m + tid]);
	}
	syncthreads();
	if (tid < n) {
		C[tid + m] = log(B[tid]);
	}
}

__global__ void minus(int nDims, float *A, float *B, float *C) {
	int tid = threadIdx.x;
	if (tid < nDims) {
		A[tid] = B[tid] - C[tid];
	}
}

void KK::EStep()
{
	int nSkipped;
	float LogRootDet; // log of square root of covariance determinant
	float correction_factor = (float)1; // for partial correction in distributional step
										//float InverseClusterNorm;
	std::vector<float> Chol(nDims2); // to store choleski decomposition
	std::vector<float> Vec2Mean(nDims); // stores data point minus class mean
	std::vector<float> Root(nDims); // stores result of Chol*Root = Vec
	std::vector<float> InvCovDiag(nDims);

	//----------------------------
	thrust::device_vector<float> d_Chol(nDims2); // to store choleski decomposition
	thrust::device_vector<float> d_Vec2Mean(nDims); // stores data point minus class mean
	thrust::device_vector<float> d_Root(nDims); // stores result of Chol*Root = Vec
	thrust::device_vector<float> d_InvCovDiag(nDims);
	//-----------------------------
	nSkipped = 0;

	if (Debug) { Output("Entering Unmasked Estep \n"); }

	// start with cluster 0 - uniform distribution over space
	// because we have normalized all dims to 0...1, density will be 1.
	std::vector<int> NumberInClass(MaxPossibleClusters);  // For finding number of points in each class
	thrust::device_vector<int> NumberInClass(MaxPossibleClusters);
	/*
	for (int p = 0; p<nPoints; p++)
	{
		LogP[p*MaxPossibleClusters + 0] = (float)-log(Weight[0]);
		int ccc = Class[p];
		NumberInClass[ccc]++;
	}
	*/
	initLogP << <128, 128 >> > (MaxPossibleClusters, nPoints,
		thrust::raw_pointer_cast(&d_LogP[0]),
		thrust::raw_pointer_cast(&d_Weight[0]),
		thrust::raw_pointer_cast(&d_Class[0]),
		thrust::raw_pointer_cast(&d_NumberInClass[0]));

	BlockPlusDiagonalMatrix *CurrentCov;
	BlockPlusDiagonalMatrix *CholBPD = NULL;

	BlockPlusDiagonalMatrix *d_CurrentCov;
	BlockPlusDiagonalMatrix *d_CholBPD = NULL;

	for (int cc = 1; cc<nClustersAlive; cc++)
	{
		int c = AliveIndex[cc];

		// calculate cholesky decomposition for class c
		int chol_return;
		CurrentCov = &(DynamicCov[cc]);
		d_CurrentCov = &(d_DynamicCov[cc]);
		/*if (CholBPD)
		{
			delete CholBPD;
			CholBPD = NULL;
		}*/
		if (d_CholBPD)
		{
			delete d_CholBPD;
			d_CholBPD = NULL;
		}
		CholBPD = new BlockPlusDiagonalMatrix(*(CurrentCov->Masked), *(CurrentCov->Unmasked));
		d_CholBPD = new d_BlockPlusDiagonalMatrix(*(d_CurrentCov->Masked), *(d_CurrentCov->Unmasked));
		chol_return = BPDCholesky(*CurrentCov, *CholBPD);
		chol_return = CusolverCholesky(*d_CurrentCov, *d_CholBPD);
		if (chol_return)
		{
			// If Cholesky returns 1, it means the matrix is not positive definite.
			// So kill the class.
			// Cholesky is defined in linalg.cpp
			Output("Unmasked E-step: Deleting class %d (%d points): covariance matrix is singular \n", (int)c, (int)NumberInClass[c]);
			//ClassAlive[c] = 0;
			d_ClassAlive[c] = 0;
			continue;
		}

		// LogRootDet is given by log of product of diagonal elements
		LogRootDet = 0;
		int m = CholBPD->NumUnmasked, n = CholBPD->NumMasked;
		thrust::device_vector<float> d_temp(m + n);
		copyAndCombin<<<1,128>>>(m, n, d_CholBPD->Block, d_CholBPD->Diagonal, thrust::raw_pointer_cast(&d_temp[0]));
		LogRootDet = thrust::count(d_temp.begin(), d_temp.end(), 1);
		/*for (int ii = 0; ii < CholBPD->NumUnmasked; ii++)
			LogRootDet += log(CholBPD->Block[ii*CholBPD->NumUnmasked + ii]);
		for (int ii = 0; ii < CholBPD->NumMasked; ii++)
			LogRootDet += log(CholBPD->Diagonal[ii]);*/

		// if distributional E step, compute diagonal of inverse of cov matrix
		vector<float> BasisVector(nDims,0.0);
		//for (int i = 0; i<nDims; i++)
		//	BasisVector[i] = (float)0;
		thrust::device_vector<float> d_BasisVector = BasisVector;

		
		for (int i = 0; i<nDims; i++)
		{
			d_BasisVector[i] = (float)1;
			// calculate Root vector - by Chol*Root = BasisVector
			CublasTriSolve(*d_CholBPD, d_BasisVector, d_Root);
			// add half of Root vector squared to log p
			float Sii = (float)0;
			for (int j = 0; j<nDims; j++)
				Sii += d_Root[j] * d_Root[j];
			d_InvCovDiag[i] = Sii;
			d_BasisVector[i] = (float)0;
		}

#pragma omp parallel for schedule(dynamic) firstprivate(Vec2Mean, Root) default(shared)
		for (int p = 0; p<nPoints; p++)
		{
			// to save time -- only recalculate if the last one was close
			/*
			if (
				!FullStep
				&& (Class[p] == OldClass[p])
				&& (LogP[p*MaxPossibleClusters + c] - LogP[p*MaxPossibleClusters + Class[p]] > DistThresh)
				)
			{
#pragma omp atomic
				nSkipped++;
				continue;
			}
			*/

			// to save time, skip points with mask overlap below threshold
			/*
			if (MinMaskOverlap > 0)
			{
				// compute dot product of point mask with cluster mask
				const float * __restrict PointMask = &(FloatMasks[p*nDims]);
				//const float * __restrict cm = &(ClusterMask[c*nDims]);
				float dotprod = 0.0;
				//// InverseClusterNorm is computed above, uncomment it if you uncomment any of this
				//for (i = 0; i < nDims; i++)
				//{
				//	dotprod += cm[i] * PointMask[i] * InverseClusterNorm;
				//	if (dotprod >= MinMaskOverlap)
				//		break;
				//}
				const int NumUnmasked = CurrentCov->NumUnmasked;
				if (NumUnmasked)
				{
					const int * __restrict cu = &((*(CurrentCov->Unmasked))[0]);
					for (int ii = 0; ii < NumUnmasked; ii++)
					{
						const int i = cu[ii];
						dotprod += PointMask[i];
						if (dotprod >= MinMaskOverlap)
							break;
					}
				}
				//dotprod *= InverseClusterNorm;
				if (dotprod < MinMaskOverlap)
				{
#pragma omp atomic
					nSkipped++;
					continue;
				}
			}
			*/


			// Compute Mahalanobis distance
			float Mahal = 0;

			// calculate data minus class mean
			//for (i = 0; i<nDims; i++)
			//	Vec2Mean[i] = Data[p*nDims + i] - Mean[c*nDims + i];
			/*
			float * __restrict Data_p = &(Data[p*nDims]);
			float * __restrict Mean_c = &(Mean[c*nDims]);
			float * __restrict v2m = &(Vec2Mean[0]);

			for (int i = 0; i < nDims; i++)
				v2m[i] = Data_p[i] - Mean_c[i];*/

			minus<<<1,128>>>(nDims, thrust::raw_pointer_cast(&d_Data[p*nDims]),
				thrust::raw_pointer_cast(&d_Mean[p*nDims]),
				thrust::raw_pointer_cast(&d_Vec2Mean[0]));

			// calculate Root vector - by Chol*Root = Vec2Mean
			CublasTriSolve(*d_CholBPD, d_Vec2Mean, d_Root);

			// add half of Root vector squared to log p
			for (int i = 0; i<nDims; i++)
				Mahal += d_Root[i] * d_Root[i];

			// if distributional E step, add correction term

			float subMahal = 0.0;
			for (int i = 0; i < nDims; i++)
				subMahal += InvCovDiag[i] * CorrectionTerm[p*nDims + i];

			Mahal += subMahal*correction_factor;

			// Score is given by Mahal/2 + log RootDet - log weight
			LogP[p*MaxPossibleClusters + c] = Mahal / 2
				+ LogRootDet
				- log(Weight[c])
				+ (float)(0.5*log(2 * M_PI))*nDims;

		} // for(p=0; p<nPoints; p++)
	} // for(cc=1; cc<nClustersAlive; cc++)
	if (CholBPD)
		delete CholBPD;
	if(d_CholBPD)
		delete d_CholBPD;
}
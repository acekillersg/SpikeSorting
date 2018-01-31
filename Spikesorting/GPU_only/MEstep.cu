#include "klustakwik.h"
#include "params.h"

#include "cuda_runtime.h"  
#include "device_launch_parameters.h"

#include <cublas_v2.h>
#include <cusolverDn.h>

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/generate.h>
#include <thrust/reduce.h>
#include <thrust/functional.h>
#include <thrust/random.h>
#include <thrust/sequence.h>
#include <thrust/iterator/constant_iterator.h>
#include <thrust/count.h>
#include <thrust/sort.h>
#include <thrust/copy.h>
#include <algorithm>

#include <stdio.h>
#include <iostream>

#define _USE_MATH_DEFINES
#include <math.h>

#define BLOCKDIM 128


//global kernel
//===========================================compute weight=======================================================//

__global__ void c_Weight(int nClustersAlive, int priorPoint, int nPoints, int NoisePoint, int *d_AliveIndex, int *d_nClassMembers, float *d_Weight) {
	int tid = blockDim.x * blockIdx.x + threadIdx.x;
	if (tid < nClustersAlive)
	{
		int c = d_AliveIndex[tid];
		if (c>0) d_Weight[c] = ((float)d_nClassMembers[c] + priorPoint) / (nPoints + NoisePoint + priorPoint*(nClustersAlive - 1));
		else d_Weight[c] = ((float)d_nClassMembers[c] + NoisePoint) / (nPoints + NoisePoint + priorPoint*(nClustersAlive - 1));
	}
}
//===========================================compute mean=======================================================//

__global__ void c_nClassMembers(int nPoints, int *d_Class, int *d_nClassMembers) {
	int tidx = blockDim.x * blockIdx.x + threadIdx.x;
	if (tidx < nPoints) {
		atomicAdd(&d_nClassMembers[d_Class[tidx]], 1);
	}
}
__global__ void matCopy(int npoints, int ndims, int nDims, int *d_rowId, int *d_colId, float *d_sourceMat, float *d_copyMat) {
	int tidx = blockDim.x * blockIdx.x + threadIdx.x;
	int tidy = blockDim.y * blockIdx.y + threadIdx.y;
	if (tidx < ndims && tidy < npoints) {
		int p = d_rowId[tidy];
		int nd = d_colId[tidx];
		//printf("in the kernel,this is the x: %d,y: %d,xx: %d, yy:%d\n", tidx, tidy, p, nd);
		d_copyMat[tidy * ndims + tidx] = d_sourceMat[p * nDims + nd];
	}

}

__global__ void c_CheckDead(int nClustersAlive, int *d_AliveIndex, int *d_nClassMembers, int *d_ClassAlive) {
	int tid = blockDim.x * blockIdx.x + threadIdx.x;
	if (tid <= nClustersAlive)
	{
		int c = d_AliveIndex[tid];
		if (c>0 && d_nClassMembers[c]<1)
			d_ClassAlive[c] = 0;
	}
}

__global__ void c_FeatureSum(int nPoints, int nDims, int *d_Class, float *d_Mean, float *d_Data) {
	int tidx = blockDim.x * blockIdx.x + threadIdx.x;
	int tidy = blockDim.y * blockIdx.y + threadIdx.y;
	if (tidx < nPoints && tidy < nDims) {
		int c = d_Class[tidx];
		atomicAdd(&d_Mean[c*nDims + tidy], d_Data[tidx*nDims + tidy]);
	}
}

__global__ void c_FeatureMean(int nClustersAlive, int nDims, int *d_AliveIndex, float *d_Mean, int *d_nClassMembers) {
	int tidx = blockDim.x * blockIdx.x + threadIdx.x;
	int tidy = blockDim.y * blockIdx.y + threadIdx.y;
	if (tidx < nClustersAlive && tidy < nDims) {
		int c = d_AliveIndex[tidx];
		d_Mean[c*nDims + tidy] /= d_nClassMembers[c];
	}
}

__global__ void c_AllVector2Mean(int nPoints, int nDims, float *d_AllVector2Mean, float* d_Mean, float *d_Data, int *d_Class) {
	int tidx = blockDim.x * blockIdx.x + threadIdx.x;
	int tidy = blockDim.y * blockIdx.y + threadIdx.y;
	if (tidx < nPoints && tidy < nDims) {
		int c = d_Class[tidx];
		d_AllVector2Mean[tidx *nDims + tidy] = d_Data[tidx*nDims + tidy] - d_Mean[c*nDims + tidy];
	}
}

__global__ void c_CorrectionTerm(int nDims, int NumUnmasked, int NumMasked, int NumPointsInThisClass,
	int *d_CurrentUnmasked, int *d_CurrentMasked, float *d_cov, float *d_dig,
	int *d_PointsInThisClass, float *d_CorrectionTerm) {
	int tid = blockDim.x * blockIdx.x + threadIdx.x;
	if (tid < NumUnmasked)
	{
		float ccf = 0.0;
		int i = d_CurrentUnmasked[tid];
		for (int q = 0; q<NumPointsInThisClass; q++)
		{
			const int p = d_PointsInThisClass[q];
			ccf += d_CorrectionTerm[p*nDims + i];
		}
		d_cov[tid*NumUnmasked + tid] += ccf;
	}
	__syncthreads();
	if (tid < NumMasked)
	{
		float ccf = 0.0;
		int i = d_CurrentMasked[tid];
		for (int q = 0; q<NumPointsInThisClass; q++)
		{
			const int p = d_PointsInThisClass[q];
			ccf += d_CorrectionTerm[p*nDims + i];
		}
		d_dig[tid] += ccf;
	}
}

__global__ void addNoiseVar(int NumUnmasked, int NumMasked, int priorPoint, float *d_NoiseVariance,
	int *d_CurrentUnmasked, int *d_CurrentMasked, float *d_cov, float *d_dig) {
	int tid = blockDim.x * blockIdx.x + threadIdx.x;
	if (tid < NumUnmasked) {
		d_cov[tid * NumUnmasked + tid] += priorPoint * d_NoiseVariance[d_CurrentUnmasked[tid]];
	}
	__syncthreads();
	if (tid < NumMasked) {
		d_dig[tid] += priorPoint * d_NoiseVariance[d_CurrentMasked[tid]];
	}
}
//=============================================================================//
__global__ void checkDigSingular(int nunmasked, int mark, float *d_dig, float *d_invdig) {
	int tid = blockDim.x * blockIdx.x + threadsIdx.x;
	if (tid < nmasked) {
		if (d_dig[tid] <= 0)
		{
			atomicAdd(&mark, 1);
			break;
		}
		else d_invdig[tid] = (float)sqrt(d_dig[tid]);
	}
}

__global__ void makeInvCov(int nunmasked, float *d_cov, float *d_invcov) {
	int tid = blockDim.x * blockIdx.x + threadsIdx.x;
	if (tid < nunmasked)
		d_invcov[tid] = d_cov[tid * nunmasked + tid];
}
__global__ void calSubMahal(int nDims, int pId, float *d_invDig, float *d_CorrectionTerm,
	float *d_subMah) {
	int tidx = blockDim.x * blockIdx.x + threadIdx.x;
	if (tidx < nDims)
		d_subMah[tidx] = d_invDig[tidx] * d_CorrectionTerm[pId*nDims + tidx];
}

__global__ void extractPoints2Mean(int nunmasked, int nPoints, int nDims, int clusterId,
	int *d_CurrentUnmasked,
	float *d_Data, float *d_Mean, float *d_points2mean) {
	int tidx = blockDim.x * blockIdx.x + threadIdx.x;
	int tidy = blockDim.y * blockIdx.y + threadIdx.y;
	if (tidx < nunmasked && tidy < nPoints) {
		int nd = d_CurrentUnmasked[tidx];
		d_points2mean[tidy * nunmasked + tidx] =
			d_Data[tidy * nDims + nd] - d_Mean[clusterId * nDims + nd];
	}
}


void KK::MEstep()
{
	clock_t clock1 = clock();

	// clear arrays
	//memset((void*)&nClassMembers.front(), 0, MaxPossibleClusters * sizeof(int));
	//memset((void*)&Mean.front(), 0, MaxPossibleClusters*nDims * sizeof(float));
	if (Debug) { Output("Entering Unmasked Mstep \n"); }

	thrust::device_vector<int> d_nClassMembers(MaxPossibleClusters, 0);
	thrust::device_vector<float>  d_Mean(MaxPossibleClusters*nDims, 0.0);

	thrust::device_vector<int> d_Class = Class;

	// Accumulate total number of points in each class
	int gridd = (nPoints / BLOCKDIM) + 1;
	c_nClassMembers << <gridd, BLOCKDIM >> > (nPoints,
		thrust::raw_pointer_cast(&d_Class[0]),
		thrust::raw_pointer_cast(&d_nClassMembers[0])
		);

	// check for any dead classes
	thrust::device_vector<int> d_AliveIndex = AliveIndex;
	thrust::device_vector<int>  d_ClassAlive = ClassAlive;

	c_CheckDead << <BLOCKDIM, BLOCKDIM >> > (nClustersAlive,
		thrust::raw_pointer_cast(&d_AliveIndex[0]),
		thrust::raw_pointer_cast(&d_nClassMembers[0]),
		thrust::raw_pointer_cast(&d_ClassAlive[0])
		);
	thrust::copy(d_ClassAlive.begin(), d_ClassAlive.end(), ClassAlive.begin());

	Reindex();
	thrust::copy(d_AliveIndex.begin(), d_AliveIndex.end(), AliveIndex.begin());

	// Normalize by total number of points to give class weight
	// Also check for dead classes

	thrust::device_vector<float> d_Weight(MaxPossibleClusters);
	c_Weight << <BLOCKDIM, BLOCKDIM >> > (nClustersAlive, priorPoint, nPoints, NoisePoint,
		thrust::raw_pointer_cast(&d_AliveIndex[0]),
		thrust::raw_pointer_cast(&d_nClassMembers[0]),
		thrust::raw_pointer_cast(&d_Weight[0]));
	thrust::copy(d_Weight.begin(), d_Weight.end(), Weight.begin());


	//================================================compute Cov=======================================//  

	thrust::device_vector<float> d_Data = Data;

	int dimx = 32;
	int dimy = 32;
	dim3 block(dimx, dimy);
	dim3 grid(512, 512);

	c_FeatureSum << <grid, block >> > (nPoints, nDims,
		thrust::raw_pointer_cast(&d_Class[0]),
		thrust::raw_pointer_cast(&d_Mean[0]),
		thrust::raw_pointer_cast(&d_Data[0])
		);

	//printf("d_Mean.size: %d\n", d_Mean.size());
	//for (int i = 0; i < 1000; i++) std::cout << d_Mean[i] << "  ";
	//printf("\n\n\n");

	c_FeatureMean << <grid, block >> >(nClustersAlive, nDims,
		thrust::raw_pointer_cast(&d_AliveIndex[0]),
		thrust::raw_pointer_cast(&d_Mean[0]),
		thrust::raw_pointer_cast(&d_nClassMembers[0])
		);

	//printf("d_Mean.size: %d\n", d_Mean.size());
	//for (int i = 0; i < 1000; i++) std::cout << d_Mean[i] << "  ";
	//printf("\n\n\n");

	thrust::device_vector<float> d_AllVector2Mean(nPoints*nDims);
	c_AllVector2Mean << <grid, block >> >(nPoints, nDims,
		thrust::raw_pointer_cast(&d_AllVector2Mean[0]),
		thrust::raw_pointer_cast(&d_Mean[0]),
		thrust::raw_pointer_cast(&d_Data[0]),
		thrust::raw_pointer_cast(&d_Class[0])
		);
	//printf("d_AllVector2Mean.size: %d\n", d_AllVector2Mean.size());
	//for (int i = 0; i <1000; i++) std::cout << d_AllVector2Mean[i] << "  ";
	//printf("\n\n\n");

	vector< vector<int> > PointsInClass(MaxPossibleClusters);
	for (int p = 0; p<nPoints; p++)
	{
		int c = Class[p];
		PointsInClass[c].push_back(p);
	}

	//printf("d_AllVector2Mean.size: %d\n", d_AllVector2Mean.size());
	//for (int i = 0; i < d_AllVector2Mean.size(); i++) std::cout << d_AllVector2Mean[i] << "  ";
	//printf("\n\n\n");

	//E step used
	thrust::copy(d_Mean.begin(), d_Mean.end(), Mean.begin());
	thrust::copy(d_ClassAlive.begin(), d_ClassAlive.end(), ClassAlive.begin());


	thrust::copy(d_AliveIndex.begin(), d_AliveIndex.end(), AliveIndex.begin());
	thrust::copy(d_nClassMembers.begin(), d_nClassMembers.end(), nClassMembers.begin());

	//=======================================================================================
	// Compute the cluster masks, used below to optimise the computation
	ComputeClusterMasks();
	// Empty the dynamic covariance matrices (we will fill it up as we go)
	DynamicCov.clear();

	for (int cc = 0; cc < nClustersAlive; cc++)
	{
		int c = AliveIndex[cc];
		vector<int> &CurrentUnmasked = ClusterUnmaskedFeatures[c];
		vector<int> &CurrentMasked = ClusterMaskedFeatures[c];
		DynamicCov.push_back(BlockPlusDiagonalMatrix(CurrentMasked, CurrentUnmasked));
	}

	thrust::device_vector<float> d_CorrectionTerm = CorrectionTerm;
	thrust::device_vector<float> d_NoiseVariance = NoiseVariance;

	//this is  a  big  kernel!!

	clock_t clock2 = clock();
	//printf("%d    %d\n", nClustersAlive,DynamicCov.size());
	//printf("\n");
	for (int cc = 0; cc < nClustersAlive; cc++)
	{
		const int c = AliveIndex[cc];
		const vector<int> &PointsInThisClass = PointsInClass[c];
		const vector<int> &CurrentUnmasked = ClusterUnmaskedFeatures[c];
		const vector<int> &CurrentMasked = ClusterMaskedFeatures[c];
		BlockPlusDiagonalMatrix &CurrentCov = DynamicCov[cc];

		const int npoints = (int)PointsInThisClass.size();
		const int nunmasked = (int)CurrentUnmasked.size();
		const int nmasked = (int)CurrentMasked.size();

		thrust::device_vector<int> d_PointsInThisClass = PointsInClass[c];
		thrust::device_vector<int> d_CurrentUnmasked = ClusterUnmaskedFeatures[c];
		thrust::device_vector<int> d_CurrentMasked = ClusterMaskedFeatures[c];
		thrust::device_vector<float> d_cov(nunmasked * nunmasked, 0.0);
		thrust::device_vector<float> d_dig(nmasked, 0.0);
		/*
		printf("%d    %d  \n", npoints)
		for (int q = 0; q < npoints; q++)
		{
		const int p = PointsInThisClass[q];
		const float * __restrict av2mp = &(AllVector2Mean[p*nDims]);
		for (int ii = 0; ii < nDims; ii++)
		{
		printf("%f  ", av2mp[ii]);
		}
		printf("\n");
		}
		printf("\n");

		printf("d_PointsInThisClass.size: %d\n", d_PointsInThisClass.size());
		for (int i = 0; i < d_PointsInThisClass.size(); i++) std::cout << d_PointsInThisClass[i] << "  ";
		printf("\n\n\n");

		printf("d_CurrentUnmasked.size: %d\n", d_CurrentUnmasked.size());
		for (int i = 0; i < d_CurrentUnmasked.size(); i++) std::cout << d_CurrentUnmasked[i] << "  ";
		printf("\n\n\n");
		*/


		if (nunmasked > 0 && npoints > 0)
		{
			thrust::device_vector<float> d_X(npoints * nunmasked);
			int dimx = 32;
			int dimy = 32;
			dim3 block(dimx, dimy);
			dim3 grid(512, 512);
			matCopy << <grid, block >> > (npoints, nunmasked, nDims,
				thrust::raw_pointer_cast(&d_PointsInThisClass[0]),
				thrust::raw_pointer_cast(&d_CurrentUnmasked[0]),
				thrust::raw_pointer_cast(&d_AllVector2Mean[0]),
				thrust::raw_pointer_cast(&d_X[0]));

			//cudaEvent_t start, stop;
			//float time;
			//cudaEventCreate(&start);
			//cudaEventCreate(&stop);

			cublasHandle_t handle;
			cublasCreate(&handle);
			float alpha = 1.f;
			float beta = 0.0f;

			//cudaEventRecord(start, 0);

			cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_T, nunmasked, nunmasked, npoints, &alpha,
				thrust::raw_pointer_cast(d_X.data()), nunmasked, thrust::raw_pointer_cast(d_X.data()), nunmasked, &beta,
				thrust::raw_pointer_cast(d_cov.data()), nunmasked);

			//cudaEventRecord(stop, 0);
			//cudaEventSynchronize(stop);

			cublasDestroy(handle);

			//cudaEventElapsedTime(&time, start, stop);
			//printf("---------------------------------------------->cublasSgemm cost times:  %f ms\n", time);
		}

		//int griddim = (nDims / BLOCKDIM) + 1;
		c_CorrectionTerm << < 1, BLOCKDIM >> > (nDims, nunmasked, nmasked, npoints,
			thrust::raw_pointer_cast(d_CurrentUnmasked.data()),
			thrust::raw_pointer_cast(d_CurrentMasked.data()),
			thrust::raw_pointer_cast(d_cov.data()),
			thrust::raw_pointer_cast(d_dig.data()),
			thrust::raw_pointer_cast(d_PointsInThisClass.data()),
			thrust::raw_pointer_cast(d_CorrectionTerm.data())
			);

		addNoiseVar << <1, BLOCKDIM >> > (nunmasked, nmasked, priorPoint,
			thrust::raw_pointer_cast(d_NoiseVariance.data()),
			thrust::raw_pointer_cast(d_CurrentUnmasked.data()),
			thrust::raw_pointer_cast(d_CurrentMasked.data()),
			thrust::raw_pointer_cast(d_cov.data()),
			thrust::raw_pointer_cast(d_dig.data())
			);

		const float factor = nClassMembers[c] + priorPoint - 1;
		thrust::transform(
			d_cov.begin(), d_cov.end(),
			thrust::make_constant_iterator((float)factor),
			d_cov.begin(),
			thrust::divides<float>());
		thrust::transform(
			d_dig.begin(), d_dig.end(),
			thrust::make_constant_iterator((float)factor),
			d_dig.begin(),
			thrust::divides<float>());
		/*
		printf("%d    %d\n", d_cov.size(), CurrentCov.Block.size());
		Output("d_cov.size: %d\n", d_cov.size());
		for (int i = 0; i < d_cov.size(); i++) std::cout << d_cov[i] << "  ";
		Output("\n\n\n");
		Output("CurrentCov.Block.size:%d\n", CurrentCov.Block.size());
		for (int i = 0; i < CurrentCov.Block.size(); i++) std::cout << CurrentCov.Block[i] << "  ";
		Output("\n\n\n");
		*/


		//thrust::copy(d_cov.begin(), d_cov.end(), CurrentCov.Block.begin());
		//thrust::copy(d_dig.begin(), d_dig.end(), CurrentCov.Diagonal.begin());

		/*
		Output("d_cov.size: %d\n", d_cov.size());
		for (int i = 0; i < d_cov.size(); i++) std::cout << d_cov[i] << "  ";
		Output("\n\n\n");
		Output("d_dig.size: %d\n", d_dig.size());
		for (int i = 0; i < d_dig.size(); i++) std::cout << d_dig[i] << "  ";
		Output("\n\n\n");

		Output("CurrentCov.Block.size:%d\n", CurrentCov.Block.size());
		for (int i = 0; i < CurrentCov.Block.size(); i++) std::cout << CurrentCov.Block[i] << "  ";
		Output("\n\n\n");
		Output("CurrentCov.Diagonal.size:%d\n", CurrentCov.Diagonal.size());
		for (int i = 0; i < CurrentCov.Diagonal.size(); i++) std::cout << CurrentCov.Diagonal[i] << "  ";
		Output("\n\n\n");

		Output("\n\n\n");
		*/



		//===============================E step=============================//

		// --- cuSOLVE input/output parameters/arrays
		int work_size = 0;
		int *devInfo;           cudaMalloc(&devInfo, sizeof(int));
		// --- CUDA solver initialization
		cusolverDnHandle_t solver_handle;
		cusolverDnCreate(&solver_handle);
		cublasFillMode_t uplo = CUBLAS_FILL_MODE_LOWER;

		// --- CUDA CHOLESKY initialization
		cusolverDnSpotrf_bufferSize(solver_handle, uplo, nunmasked,
			thrust::raw_pointer_cast(&d_cov[0]), nunmasked, &work_size);

		// --- CUDA POTRF execution
		//float *work;   cudaMalloc(&work, work_size * sizeof(float));
		thrust::device_vector<float> work(work_size);
		cusolverDnSpotrf(solver_handle, uplo, nunmasked,
			thrust::raw_pointer_cast(&d_cov[0]), nunmasked,
			thrust::raw_pointer_cast(&work[0]), work_size, devInfo);
		int devInfo_h = 0;
		cudaMemcpy(&devInfo_h, devInfo, sizeof(int), cudaMemcpyDeviceToHost);
		if (devInfo_h != 0) {
			Output("Unmasked E-step: Deleting class %d (%d points): covariance matrix is singular \n", (int)c, (int)NumberInClass[c]);
			ClassAlive[c] = 0;
			continue;
		}

		/*此处处理dig，判断是否奇异，并更新*/
		int mark = 0;
		thrust::device_vector<float> d_invdig(nmasked);
		/*
		__global__ void checkDigSingular(int nunmasked, int mark, float *d_dig, float *d_invdig) {
			int tid = blockDim.x * blockIdx.x + threadsIdx.x;
			if (tid < nmasked) {
				if (d_dig[tid] <= 0)
				{
					atomicAdd(&mark, 1);
					break;
				}
				else d_invdig[tid] = (float)sqrt(d_dig[tid]);
			}
		}*/
		checkDigSingular << <1, BLOCKDIM >> > (nunmasked, mark, thrust::raw_pointer_cast(&d_dig[0]),
			thrust::raw_pointer_cast(&d_invdig[0]));
		if (mark != 0) {
			Output("Unmasked E-step: Deleting class %d (%d points): covariance matrix is singular \n", (int)c, (int)NumberInClass[c]);
			ClassAlive[c] = 0;
			continue;
		}

		/*此处通过对角线更新LogRootDet*/
		float LogRootDet = thrust::reduce(d_invdig.begin(), d_invdig.end(), (float)0, thrust::plus<float>());
		thrust::device_vector<float> d_invcov(nunmasked);
		/*
		__global__ void makeInvCov(int nunmasked, float *d_cov, float *d_invcov) {
			int tid = blockDim.x * blockIdx.x + threadsIdx.x;
			if (tid < nunmasked)
				d_invcov[tid] = d_cov[tid * nunmasked + tid];
		}*/
		makeInvCov << <1, BLOCKDIM >> > (nunmasked,
			thrust::raw_pointer_cast(&d_cov[0])
			thrust::raw_pointer_cast(&d_invcov[0]));
		LogRootDet = thrust::reduce(d_invcov.begin(), d_invcov.end(), LogRootDet, thrust::plus<float>());



		/*此处计算cov的inverse的对角线元素，存在invDig中*/
		thrust::device_vector<float> d_unitMat(nunmasked*nunmasked, 0.0);
		/*
		__global__ void makeUnitMat(int nunmasked, float *d_unitMat) {
			int tid = blockDim.x * blockIdx.x + threadsIdx.x;
			if (tid < nunmasked)
				d_unitMat[tid * nunmasked + tid] = 1.0;
		}
		*/
		makeUnitMat << < 1, BLOCKDIM >> > (nunmasked, thrust::raw_pointer_cast(&d_unitMat[0]));
		cusolverDnSpotrs(solver_handle, uplo, nunmasked, nunmasked,
			thrust::raw_pointer_cast(&d_cov[0]), nunmasked,
			thrust::raw_pointer_cast(&d_unitMat[0]), nunmasked, devInfo);
		cudaDeviceSynchronize();

		thrust::device_vector<float> d_invDig(nDims);
		/*
		__global__ void transInvDig(int nunmasked, int nmasked,
			float *d_unitMat, float *d_invdig,
			float *d_CurrentUnmasked, float *d_CurrenMasked,
			float *d_invDig) {
			int tid = blockDim.x * blockIdx.x + threadsIdx.x;
			if (tid < nunmasked)
				d_invDig[d_CurrentUnmasked[tid]] = d_unitMat[tid * nunmasked + tid];
			__syncthreads();
			if (tid < nmasked)
				d_invDig[d_CurrentMasked[tid]] = -1.0 / d_invdig[tid];
		}
		*/
		transInvDig << <1, BLOCKDIM >> > (nunmasked, nmasked,
			thrust::raw_pointer_cast(&d_unitMat[0]),
			thrust::raw_pointer_cast(&d_invdig[0]),
			thrust::raw_pointer_cast(&d_CurrentUnmasked[0]),
			thrust::raw_pointer_cast(&d_CurrenMasked[0]))
			//...

			/*先进行数组转化，提取需要计算的right side*/
			thrust::device_vector<float> d_points2mean(nPoints*nunmasked);
		/*
		__global__ void extractPoints2Mean(int nunmasked,int nPoints,int nDims,int clusterId,
			int *d_CurrentUnmasked,
			float *d_Data,float *d_Mean,float *d_points2mean) {
			int tidx = blockDim.x * blockIdx.x + threadIdx.x;
			int tidy = blockDim.y * blockIdx.y + threadIdx.y;
			if (tidx < nunmasked && tidy < nPoints) {
				int nd = d_CurrentUnmasked[tidx];
				d_points2mean[tidy * nunmasked + tidx] =
					d_Data[tidy * nDims + nd] - d_Mean[clusterId * nDims + nd];
			}
		}*/
		extractPoints2Mean << < (512, 512), (32, 32) >> > (nunmasked, nPoints, nDims, c,
			thrust::raw_pointer_cast(&d_CurrentUnmasked[0]),
			thrust::raw_pointer_cast(&d_Data[0]),
			thrust::raw_pointer_cast(&d_Mean[0]),
			thrust::raw_pointer_cast(&d_points2mean[0]));

		/*此处计算齐次线性方程组，并求解每个point的解的平方和Mahal*/
		cusolverDnSpotrs(solver_handle, uplo, nunmasked, nPoints,
			thrust::raw_pointer_cast(&d_cov[0]), nunmasked,
			thrust::raw_pointer_cast(&d_points2mean[0]), nunmasked, devInfo);
		cudaDeviceSynchronize();
		cusolverDnDestroy(solver_handle);
		//...
		float Mahal = 0;

		/*此处利用correctionterm和invcovdig更新Mahal*/
		thrust::device_vector<float> d_subMah(nDims);
		/*
		__global__ void calSubMahal(int nDims, int pId, float *d_invDig, float *d_CorrectionTerm,
			float *d_subMah) {
			int tidx = blockDim.x * blockIdx.x + threadIdx.x;
			if (tidx < nDims)
				d_subMah[tidx] = d_invDig[tidx] * d_CorrectionTerm[pId*nDims + tidx];
		}
		*/
		calSubMahal << <1, BLOCKDIM >> > (nDims, cc,
			thrust::raw_pointer_cast(&d_invDig[0]),
			thrust::raw_pointer_cast(&d_CorrectionTerm[0]),
			thrust::raw_pointer_cast(&d_subMah[0]));
		Mahal = thrust::reduce(d_subMah.begin(), d_subMah.end(), Mahal, thrust::plus<float>());

		/*此处更新LogP*/
		d_LogP[cc*MaxPossibleClusters + c] = Mahal / 2.0
			+ LogRootDet
			- log(Weight[c])
			+ (float)(0.5*log(2 * M_PI))*nDims;
	}

	clock_t dif2 = clock() - clock2;
	clock_t dif = clock() - clock1;
	printf("---------------------------------------------->cost times:  %f ms\n", (float)dif / CLOCKS_PER_SEC);
	printf("---------------------------------------------->for loop cost times:  %f ms\n", (float)dif2 / CLOCKS_PER_SEC);
	printf("\n");
	//mtime += dif;
}
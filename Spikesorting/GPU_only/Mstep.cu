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

#define BLOCK_SIZE 128

//d_AliveIndex[0] = 0; d_nClustersAlive = 1;
__global__ void c_Reindex(int nClustersAlive, int MaxPossibleClusters, int *d_AliveIndex, int *d_ClassAlive, int *d_mark) {

	int tid = blockDim.x * blockIdx.x + threadIdx.x;

	if (tid == 0) {
		d_AliveIndex[0] = 0;
	    nClustersAlive = 1;
	}
	if (tid > 0 && tid + 1 < MaxPossibleClusters)
		if (d_ClassAlive[tid + 1] > 0)
			d_mark[tid + 1] = 1;
		else d_mark[tid + 1] = 0;
	__syncthreads();

	__shared__ float s[BLOCK_SIZE * 2];

	if (tid < MaxPossibleClusters) { s[threadIdx.x] = d_mark[i]; }

	for (unsigned int stride = 1; stride <= BLOCK_SIZE; stride *= 2) {
		__syncthreads();
		int index = (threadIdx.x + 1)*stride * 2 - 1;
		if (index < 2 * BLOCK_SIZE)
			s[index] += s[index - stride];
		__syncthreads();
	}
	// threadIdx.x+1 = 1,2,3,4....
	// stridek index = 1,3,5,7...
	for (unsigned int stride = BLOCK_SIZE / 2; stride > 0; stride /= 2) {
		__syncthreads();
		int index = (threadIdx.x + 1)*stride * 2 - 1;
		if (index < 2 * BLOCK_SIZE)
			s[index + stride] += s[index];
	}
	__syncthreads();
	if (tid < MaxPossibleClusters) 
		d_mark[tid] = s[threadIdx.x];
	__syncthreads();
	if (tid + 1 < MaxPossibleClusters)
	{
		if (d_mark[tid] > d_mark[tid - 1])
			d_AliveIndex[d_mark[tid]] = tid;
	}
	__syncthreads();
	if (tid == 0)
		nClustersAlive = d_mark[MaxPossibleClusters - 1];

}

__global__ void c_ComputeClassPenalties1(int MaxPossibleClusters, int nPoints,int penaltyK, int penaltyKLogN, 
	                                    float *d_ClassPenalty, int *d_NumberInClass,int *d_Class, float *UnMaskDims) {
	int tid = blockDim.x * blockIdx.x + threadIdx.x;

	if(tid < MaxPossibleClusters) d_ClassPenalty[tid] = (float)0;
	__syncthreads();

	// compute sum of nParams for each
	//vector<int> NumberInClass(MaxPossibleClusters);
	if (tid < nPoints)
	{
		int c = d_Class[tid];
		atomicAdd(&d_NumberInClass[c], 1);
		float n = UnMaskDims[tid];
		float nParams = n*(n + 1) / 2 + n + 1;
		atomicAdd(&d_ClassPenalty[c], nParams);
	}
}
__global__ void c_ComputeClassPenalties2(int MaxPossibleClusters, int nPoints, int penaltyK, int penaltyKLogN,
	float *d_ClassPenalty, int *d_NumberInClass, int *d_Class, float *UnMaskDims) {
	int tid = blockDim.x * blockIdx.x + threadIdx.x;
	if (tid < MaxPossibleClusters)
	{
		if(d_NumberInClass[tid]>0)
		d_ClassPenalty[tid] /= (float)d_NumberInClass[tid];

		d_ClassPenalty[tid] = penaltyK*(float)(d_ClassPenalty[tid] * 2)
			+ penaltyKLogN * ((float)d_ClassPenalty[tid] * (float)log((float)nPoints) / 2);
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

__global__ void c_Weight(int nClustersAlive, int priorPoint, int nPoints, int NoisePoint, int *d_AliveIndex, int *d_nClassMembers, int *d_Weight) {
	int tid = blockDim.x * blockIdx.x + threadIdx.x;
	if (tid < nClustersAlive)
	{
		int c = d_AliveIndex[tid];
		if (c) d_Weight[c] = ((float)d_nClassMembers[c] + priorPoint) / (nPoints + NoisePoint + priorPoint*(nClustersAlive - 1));
		else d_Weight[c] = ((float)d_nClassMembers[c] + NoisePoint) / (nPoints + NoisePoint + priorPoint*(nClustersAlive - 1));
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
		d_Mean[c*nDims + tidy]/= d_nClassMembers[c];
	}
}

/*
__global__ void PointsInClassResize(int nClustersAlive, int nPoints, thrust::device_vector< thrust::device_vector<int> > d_PointsInClass,int *d_nClassMembers) {
int tid = blockDim.x * blockIdx.x + threadIdx.x;
if (nPoints < tid)
{
int c = d_Class[tid];
atomicAdd(d_mark[c], 1);
d_PointsInClass[d_mark[c]] = tid;
}
}
*/

__global__ void CComputeClusterMasks1(int nPoints, int nDims, int *d_Class, float *d_ClusterMask, float *d_FloatMasks)
{
	int tidx = blockDim.x * blockIdx.x + threadIdx.x;
	int tidy = blockDim.y * blockIdx.y + threadIdx.y;
	// Initialise cluster mask to 0
	//for (int i = 0; i < nDims*MaxPossibleClusters; i++)
	//	ClusterMask[i] = 0;

	// Compute cluster mask
	if (tidx < nPoints && tidy < nDims) {
		int c = d_Class[tidx];
		atomicAdd(&d_ClusterMask[c*nDims + tidy], d_FloatMasks[tidx*nDims + tidy]);
	}
}
void CComputeClusterMasks2() {
	ClusterUnmaskedFeatures.clear();
	ClusterUnmaskedFeatures.resize(MaxPossibleClusters);
	ClusterMaskedFeatures.clear();
	ClusterMaskedFeatures.resize(MaxPossibleClusters);
	// fill them in
	for (int cc = 0; cc<nClustersAlive; cc++)
	{
		int c = AliveIndex[cc];
		vector<int> &CurrentUnmasked = ClusterUnmaskedFeatures[c];
		vector<int> &CurrentMasked = ClusterMaskedFeatures[c];
		for (int i = 0; i < nDims; i++)
		{
			if (ClusterMask[c*nDims + i] >= PointsForClusterMask)
				CurrentUnmasked.push_back(i);
			else
				CurrentMasked.push_back(i);
		}
	}
}
	
/*Mstep update mean and cov*/
/*************************************/
/* CONVERT LINEAR INDEX TO ROW INDEX */
/*************************************/
template <typename T>
struct linear_index_to_row_index : public thrust::unary_function<T, T> {

	T Ncols; // --- Number of columns

	__host__ __device__ linear_index_to_row_index(T Ncols) : Ncols(Ncols) {}

	__host__ __device__ T operator()(T i) { return i / Ncols; }
};

void MstepMain(int nPoints, int nDims, thrust::device_vector<float> d_X, thrust::device_vector<float> d_cov) {
	// --- cuBLAS handle creation
	cublasHandle_t handle;
	cublasCreate(&handle);

	/*************************************************/
	/* CALCULATING THE MEANS OF THE RANDOM VARIABLES */
	/*************************************************/
	// --- Array containing the means multiplied by nPoints
	thrust::device_vector<float> d_means(nDims);

	thrust::device_vector<float> d_ones(nPoints, 1.f);

	float alpha = 1.f / (float)nPoints;
	float beta = 0.f;
	cublasSgemv(handle, CUBLAS_OP_T, nPoints, nDims, &alpha, thrust::raw_pointer_cast(d_X.data()), nPoints,
		thrust::raw_pointer_cast(d_ones.data()), 1, &beta, thrust::raw_pointer_cast(d_means.data()), 1);

	/**********************************************/
	/* SUBTRACTING THE MEANS FROM THE MATRIX ROWS */
	/**********************************************/
	thrust::transform(
		d_X.begin(), d_X.end(),
		thrust::make_permutation_iterator(
			d_means.begin(),
			thrust::make_transform_iterator(thrust::make_counting_iterator(0), linear_index_to_row_index<int>(nPoints))),
		d_X.begin(),
		thrust::minus<float>());

	/*************************************/
	/* CALCULATING THE COVARIANCE MATRIX */
	/*************************************/

	//thrust::device_vector<float> d_cov(nDims * nDims);

	alpha = 1.f;
	cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N, nDims, nDims, nPoints, &alpha,
		thrust::raw_pointer_cast(d_X.data()), nPoints, thrust::raw_pointer_cast(d_X.data()), nPoints, &beta,
		thrust::raw_pointer_cast(d_cov.data()), nDims);

	// --- Final normalization by nPoints - 1
	thrust::transform(
		d_cov.begin(), d_cov.end(),
		thrust::make_constant_iterator((float)(nPoints - 1)),
		d_cov.begin(),
		thrust::divides<float>());

	//for (int i = 0; i < nDims * nDims; i++) std::cout << d_cov[i] << "\n";
}

__global__ void matCopy(int npoints, int ndims, int nPoints,int *d_rowId, int *d_colId, float *d_sourceMat, float *d_copyMat) {
	int tidx = blockDim.x * blockIdx.x + threadIdx.x;
	int tidy = blockDim.y * blockIdx.y + threadIdx.y;
	{
		if (tidx < npoints && tidy < ndims) {
			int p = d_rowId[tidx];
			int nd = d_colId[tidy];
			d_copyMat[tidy * npoints + tidx] = d_sourceMat[tidx * nPoints + tidy];
		}
	}
}

__global__ void arrayCopy(int num, int *d_rowId, int colId, float *d_sourceArray, float *d_copyArray) {
	int tid = blockDim.x * blockIdx.x + threadIdx.x;
	if (tid < num) {
		int p = d_rowId[tid];
		d_copyArray[]

	}
	for (int q = 0; q<NumPointsInThisClass; q++)
	{
		const int p = PointsInThisClass[q];
		ccf += CorrectionTerm[p*nDims + i];
	}
}

__global__ void addNoiseVar(int NumUnmasked, int NumMasked, int priorPoint, float *d_NoiseVariance, 
	int *d_CurrentUnmasked, int *d_CurrentMasked, float *d_cov, float *d_dig) {
	int tid = threadIdx.x;
	if (tid < NumUnmasked) {
		d_cov[tid * nunmasked + tid] += priorPoint * d_NoiseVariance[d_CurrentUnmasked[tid]];
	}
	__syncthreads();
	if (tid < NumMasked) {
		d_dig[tid] += priorPoint * d_NoiseVariance[d_CurrentMasked[tid]];
	}
}

__global__ void CorrectionTerm(int NumUnmasked, int NumMasked, int NumPointsInThisClass, int *d_CurrentUnmasked, int *d_CurrentMasked,
	float *d_cov, float *d_dig, int *d_PointsInThisClass, float *d_CorrectionTerm) {
	int tidx = threadIdx.x;
	if (tidx < NumUnmasked)
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
	if (tidx < NumMasked)
	{
		float ccf = 0.0;
		int i = d_CurrentMasked[tid];
		for (int q = 0; q<NumPointsInThisClass; q++)
		{
			const int p = d_PointsInThisClass[q];
			ccf += d_CorrectionTerm[p*nDims + i];
		}
		d_dig[i] += ccf;
	}
}

void KK::MStep()
{
	thrust::device_vector<float> d_Vec2Mean(nDims);
	vector<float> Vec2Mean(nDims);

	// clear arrays
	memset((void*)&nClassMembers.front(), 0, MaxPossibleClusters * sizeof(int));
	memset((void*)&Mean.front(), 0, MaxPossibleClusters*nDims * sizeof(float));

	thrust::device_vector<int> d_nClassMembers = nClassMembers;
	thrust::device_vector<int> d_Mean = Mean;


	if (Debug) { Output("Entering Unmasked Mstep \n"); }

	// Accumulate total number of points in each class

	for (int p = 0; p<nPoints; p++) nClassMembers[Class[p]]++;
	thrust::device_vector<int> d_Class = Class;

	// check for any dead classes
	/*if (UseDistributional)
	{
		for (int cc = 0; cc<nClustersAlive; cc++)
		{
			int c = AliveIndex[cc];
			if (Debug) { Output("DistributionalMstep: Class %d contains %d members \n", (int)c, (int)nClassMembers[c]); }
			if (c>0 && nClassMembers[c]<1)//nDims)
			{
				ClassAlive[c] = 0;
				if (Debug) { Output("UnmaskedMstep_dist: Deleted class %d: no members\n", (int)c); }
			}
		}
	}*/
	thrust::device_vector<int> d_AliveIndex = AliveIndex;
	thrust::device_vector<int> d_ClassAlive = ClassAlive;
	int *p_AliveIndex = thrust::raw_pointer_cast(&d_AliveIndex[0]);
	int *p_ClassAlive = thrust::raw_pointer_cast(&d_ClassAlive[0]);
	int *p_nClassMembers = thrust::raw_pointer_cast(&d_nClassMembers[0]);
	c_CheckDead << <128, 128 >> > (nClustersAlive,p_AliveIndex,p_nClassMembers,p_ClassAlive);

	//Reindex();
	thrust::device_vector<int> d_mark(MaxPossibleClusters);
	c_Reindex(nClustersAlive,MaxPossibleClusters,
		thrust::raw_pointer_cast(&d_AliveIndex[0]),
		thrust::raw_pointer_cast(&d_ClassAlive[0]),
		thrust::raw_pointer_cast(&d_mark[0]) );

	// Normalize by total number of points to give class weight
	// Also check for dead classes
	/*if (UseDistributional)
	{
		for (int cc = 0; cc<nClustersAlive; cc++)
		{
			int c = AliveIndex[cc];
			//Output("DistributionalMstep: PriorPoint on weights ");
			// add "noise point" to make sure Weight for noise cluster never gets to zero
			if (c == 0)
			{
				Weight[c] = ((float)nClassMembers[c] + NoisePoint) / (nPoints + NoisePoint + priorPoint*(nClustersAlive - 1));
			}
			else
			{
				Weight[c] = ((float)nClassMembers[c] + priorPoint) / (nPoints + NoisePoint + priorPoint*(nClustersAlive - 1));
			}
		}
	}*/
	thrust::device_vector<int> d_Weight = Weight;
	int *p_Weight = thrust::raw_pointer_cast(&d_Weight[0]);
	c_Weight << <128, 128 >> > (nClustersAlive, priorPoint, nPoints, NoisePoint, p_AliveIndex, p_nClassMembers, p_Weight);

	//Reindex();
	c_Reindex(nClustersAlive, MaxPossibleClusters,
		thrust::raw_pointer_cast(&d_AliveIndex[0]),
		thrust::raw_pointer_cast(&d_ClassAlive[0]),
		thrust::raw_pointer_cast(&d_mark[0]));

	// Accumulate sums for mean calculation
	/*for (int p = 0; p<nPoints; p++)
	{
		int c = Class[p];
		for (int i = 0; i<nDims; i++)
		{
			Mean[c*nDims + i] += GetData(p, i);
		}
	}

	// and normalize
	for (int cc = 0; cc<nClustersAlive; cc++)
	{
		int c = AliveIndex[cc];
		for (int i = 0; i<nDims; i++) Mean[c*nDims + i] /= nClassMembers[c];
	}
	

	/*if ((int)AllVector2Mean.size() < nPoints*nDims)
	{
		//mem.add((nPoints*nDims-AllVector2Mean.size())*sizeof(float));
		AllVector2Mean.resize(nPoints*nDims);
	}
	thrust::device_vector<int> d_AllVector2Mean(nPoints*nDims);*/

	thrust::device_vector< thrust::device_vector<int> > d_PointsInClass(MaxPossibleClusters);
	std::vector<std::vector<int>> PointsInClass(MaxPossibleClusters);

	/*for (int p = 0; p<nPoints; p++)
	{
		int c = Class[p];
		PointsInClass[c].push_back(p);
		for (int i = 0; i < nDims; i++)
			AllVector2Mean[p*nDims + i] = GetData(p, i) - Mean[c*nDims + i];
	}*/

	if (UseDistributional)
	{
		// Compute the cluster masks, used below to optimise the computation
		ComputeClusterMasks();
		// Empty the dynamic covariance matrices (we will fill it up as we go)
		DynamicCov.clear();
		d_DynamicCov.clear();

		for (int cc = 0; cc < nClustersAlive; cc++)
		{
			int c = AliveIndex[cc];
			vector<int> &CurrentUnmasked = ClusterUnmaskedFeatures[c];
			vector<int> &CurrentMasked = ClusterMaskedFeatures[c];
			DynamicCov.push_back(BlockPlusDiagonalMatrix(CurrentMasked, CurrentUnmasked));
		}
		d_DynamicCov = DynamicCov;

//#pragma omp parallel for schedule(dynamic)
		for (int cc = 0; cc<nClustersAlive; cc++)
		{
			const int c = AliveIndex[cc];
			const std::vector<int> &PointsInThisClass = PointsInClass[c];
			thrust::device_vector<int> d_PointsInThisClass = PointsInThisClass;
			const int NumPointsInThisClass = PointsInThisClass.size();
			const std::vector<int> &CurrentUnmasked = ClusterUnmaskedFeatures[c];
			thrust::device_vector<int> d_CurrentUnmasked = CurrentUnmasked;

			const std::vector<int> &CurrentMasked = ClusterMaskedFeatures[c];
			thrust::device_vector<int> d_CurrentMasked = CurrentMasked;

			//const vector<int> &CurrentMasked = ClusterMaskedFeatures[c];
			d_BlockPlusDiagonalMatrix &d_CurrentCov = d_DynamicCov[cc];

			
			const int npoints = (int)PointsInThisClass.size();
			const int nunmasked = (int)CurrentUnmasked.size();
			const int nmasked = (int)CurrentMasked.size();
			thrust::device_vector<float> d_cov(nunmasked *nunmasked);
			if (CurrentUnmasked.size() > 0)
			{

				if (npoints > 0 && nunmasked > 0)
				{
					thrust::device_vector<float> d_X(npoints * nunmasked);

					matCopy<<<(512,32),(32,32)>>>(npoints, nunmasked, nPoints,
						thrust::raw_pointer_cast(&d_PointsInThisClass[0]),
						thrust::raw_pointer_cast(&d_CurrentUnmasked[0]), 
						thrust::raw_pointer_cast(&d_Data[0]),
						thrust::raw_pointer_cast(&d_X[0]));
					/*for (int i = 0; i < npoints; i++) {
						int p = PointsInThisClass[i];
						for (int j = 0;j < nunmasked;j++)
						{
							int nd = CurrentUnmasked[j];
							d_X[j*npoints + i] = Data[i][j];
						}*/
					MstepMain(npoints, nunmasked, d_X,d_cov);
				}
			}




			//
			CorrectionTerm << <1, 256 >> > (nunmasked, nmasked, npoints,
				thrust::raw_pointer_cast(d_CurrentUnmasked[0]),
				thrust::raw_pointer_cast(&d_CurrentMasked[0]),
				thrust::raw_pointer_cast(d_cov[0]), 
				thrust::raw_pointer_cast(d_dig[0]),
				thrust::raw_pointer_cast(d_PointsInThisClass[0]), 
				thrust::raw_pointer_cast(d_CorrectionTerm[0]));

			/*for (int ii = 0; ii<CurrentCov.NumUnmasked; ii++)
			{
				const int i = (*CurrentCov.Unmasked)[ii];
				float ccf = 0.0; // class correction factor
				for (int q = 0; q<NumPointsInThisClass; q++)
				{
					const int p = PointsInThisClass[q];
					ccf += CorrectionTerm[p*nDims + i];
				}
				CurrentCov.Block[ii*CurrentCov.NumUnmasked + ii] += ccf;
			}
			for (int ii = 0; ii<CurrentCov.NumMasked; ii++)
			{
				const int i = (*CurrentCov.Masked)[ii];
				float ccf = 0.0; // class correction factor
				for (int q = 0; q<NumPointsInThisClass; q++)
				{
					const int p = PointsInThisClass[q];
					ccf += CorrectionTerm[p*nDims + i];
				}
				CurrentCov.Diagonal[ii] += ccf;
			}*/

			//
			addNoiseVar << <1, 256 >> > (nunmasked, nmasked, priorPoint, 
				thrust::raw_pointer_cast(d_NoiseVariance[0]),
				thrust::raw_pointer_cast(d_CurrentUnmasked[0]),
				thrust::raw_pointer_cast(&d_CurrentMasked[0]),
				thrust::raw_pointer_cast(d_cov[0]), 
				thrust::raw_pointer_cast(d_dig[0]));
			/*for (int ii = 0; ii < CurrentCov.NumUnmasked; ii++)
				CurrentCov.Block[ii*CurrentCov.NumUnmasked + ii] += priorPoint*NoiseVariance[(*CurrentCov.Unmasked)[ii]];
			for (int ii = 0; ii < CurrentCov.NumMasked; ii++)
				CurrentCov.Diagonal[ii] += priorPoint*NoiseVariance[(*CurrentCov.Masked)[ii]];*/

			//

			const float factor = (float)1.0 / (nClassMembers[c] + priorPoint - 1);
			/*for (int i = 0; i < (int)CurrentCov.Block.size(); i++)
				CurrentCov.Block[i] *= factor;
			for (int i = 0; i < (int)CurrentCov.Diagonal.size(); i++)
				CurrentCov.Diagonal[i] *= factor;*/
			cublasHandle_t handle;
			cublasCreate(&handle);
			cublasSscal(handle, (nunmasked*nunmasked), &factor, thrust::raw_pointer_cast(d_cov[0]), 1);
			cublasSscal(handle, nmasked, &factor, thrust::raw_pointer_cast(d_dig[0]), 1);
			cublasDestroy(handle);
		}
	}

}

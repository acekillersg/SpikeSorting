// MaskedKlustaKwik2.C
//
// Fast clustering using the CEM algorithm with Masks.

# pragma warning (disable:4819)

#ifndef VERSION
#define VERSION "0.3.0-nogit"
#endif

// Disable some Visual Studio warnings
#define _CRT_SECURE_NO_WARNINGS


#include "cuda_runtime.h"  
#include "device_launch_parameters.h"

#include <cublas_v2.h>
#include <cusolverDn.h>
#include <stdio.h>
#include <iostream>

#include "klustakwik.h"
#include "util.h"
#include<stdlib.h>
#define _USE_MATH_DEFINES
#include<math.h>

#define BLOCKDIM 128

#ifdef _OPENMP
#include<omp.h>
#endif

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort = false) {
	if (code != cudaSuccess) {
		Output("GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
		if (abort) exit(code);
	}
	else {
		//printf("cuda returned code == cudaSuccess\n");
	}
}

// GLOBAL VARIABLES
FILE *Distfp;
int global_numiterations = 0;
float iteration_metric2 = (float)0;
float iteration_metric3 = (float)0;
clock_t Clock0;
float timesofar;

//===========================================================================================

template<class T>
inline void resize_and_fill_with_zeros(vector<T> &x, int newsize)
{
	if (x.size() == 0)
	{
		x.resize((unsigned int)newsize);
		return;
	}
	if (x.size() > (unsigned int)newsize)
	{
		fill(x.begin(), x.end(), (T)0);
		x.resize((unsigned int)newsize);
	}
	else
	{
		x.resize((unsigned int)newsize);
		fill(x.begin(), x.end(), (T)0);
	}
}

//===========================================init d_ones========================================================//
__global__ void init_dones(int nDims, float *d_ones) {
	int tid = blockDim.x * blockIdx.x + threadIdx.x;
	if (tid < 2000) d_ones[tid] = 1.0;
}
// Sets storage for KK class.  Needs to have nDims and nPoints defined
void KK::AllocateArrays() {

	nDims2 = nDims*nDims;
	NoisePoint = 1; // Ensures that the mixture weight for the noise cluster never gets to zero
	// Set sizes for arrays
	resize_and_fill_with_zeros(Data, nPoints * nDims);
	resize_and_fill_with_zeros(Masks, nPoints * nDims);
	resize_and_fill_with_zeros(FloatMasks, nPoints * nDims);
	resize_and_fill_with_zeros(UnMaskDims, nPoints); //SNK Number of unmasked dimensions for each data point when using float masks $\sum m_i$
	resize_and_fill_with_zeros(Weight, MaxPossibleClusters);
	resize_and_fill_with_zeros(Mean, MaxPossibleClusters*nDims);
	resize_and_fill_with_zeros(LogP, MaxPossibleClusters*nPoints);
	resize_and_fill_with_zeros(Class, nPoints);
	resize_and_fill_with_zeros(OldClass, nPoints);
	resize_and_fill_with_zeros(Class2, nPoints);
	resize_and_fill_with_zeros(BestClass, nPoints);
	resize_and_fill_with_zeros(ClassAlive, MaxPossibleClusters);
	resize_and_fill_with_zeros(AliveIndex, MaxPossibleClusters);
	resize_and_fill_with_zeros(ClassPenalty, MaxPossibleClusters);
	resize_and_fill_with_zeros(nClassMembers, MaxPossibleClusters);

	resize_and_fill_with_zeros(CorrectionTerm, nPoints * nDims);
	resize_and_fill_with_zeros(ClusterMask, MaxPossibleClusters*nDims);

	resize_and_fill_with_zeros(Offset, MaxPossibleClusters);

	//==============================GPU Allocate==============================
	//int sizeI = sizeof(int);
	//int sizeF = sizeof(float);

	//gpuErrchk(cudaMalloc((void **)&d_Class, nPoints*sizeI));
	//gpuErrchk(cudaMalloc((void **)&d_Data, nPoints*nDims*sizeF));
	//gpuErrchk(cudaMalloc((void **)&d_Masks, nPoints*nDims*sizeof(int)));
	//gpuErrchk(cudaMalloc((void **)&d_nClassMembers, MaxPossibleClusters*sizeI));

	//gpuErrchk(cudaMalloc((void **)&d_ClassAlive, MaxPossibleClusters*sizeI));
	//gpuErrchk(cudaMalloc((void **)&d_AliveIndex, MaxPossibleClusters*sizeI));
	//gpuErrchk(cudaMalloc((void **)&d_NoiseMean, nDims*sizeF));
	//gpuErrchk(cudaMalloc((void **)&d_NoiseVariance, nDims*sizeF));
	//gpuErrchk(cudaMalloc((void **)&d_CorrectionTerm, nPoints*nDims*sizeF));
	//gpuErrchk(cudaMalloc((void **)&d_FloatMasks, nPoints*nDims*sizeF));
	//gpuErrchk(cudaMalloc((void **)&d_UnMaskDims, nPoints*sizeF));
	//gpuErrchk(cudaMalloc((void **)&d_ClusterMask, MaxPossibleClusters*nDims*sizeF));

	//gpuErrchk(cudaMalloc((void **)&d_Mean, MaxPossibleClusters*nDims*sizeF));
	//gpuErrchk(cudaMalloc((void **)&d_Weight, MaxPossibleClusters*sizeF));
	//gpuErrchk(cudaMalloc((void **)&d_LogP, MaxPossibleClusters*nPoints*sizeF));
	//gpuErrchk(cudaMalloc((void **)&d_OldClass, nPoints*sizeI));
	//gpuErrchk(cudaMalloc((void **)&d_ClassPenalty, MaxPossibleClusters*sizeF));
	//gpuErrchk(cudaMalloc((void **)&d_Class2, nPoints*sizeI));
	//gpuErrchk(cudaMalloc((void **)&d_BestClass, nPoints*sizeI));

	////temp variables
	//gpuErrchk(cudaMalloc((void **)&d_ClassAliveTemp, MaxPossibleClusters*sizeI));
	//gpuErrchk(cudaMalloc((void **)&d_DeletionLoss, MaxPossibleClusters*sizeF));
	//gpuErrchk(cudaMalloc((void **)&d_tempSubtraction, MaxPossibleClusters*sizeF));
	//gpuErrchk(cudaMalloc((void **)&d_tempLogP, nPoints*sizeF));
	//gpuErrchk(cudaMalloc((void **)&d_tempOldClass, nPoints*sizeF));
	////MEstep
	//gpuErrchk(cudaMalloc((void **)&d_unmaskedSolver, nPoints*sizeF));
	//gpuErrchk(cudaMalloc((void **)&d_AllVector2Mean, nPoints*nDims * sizeof(float)));
	//gpuErrchk(cudaMalloc((void **)&d_Current, MaxPossibleClusters * nDims * sizeof(int)));
	//gpuErrchk(cudaMalloc((void **)&d_PointsInThisClass, nPoints * sizeof(int)));
	//gpuErrchk(cudaMalloc((void **)&d_MarkClass, nPoints * sizeof(int)));
	//gpuErrchk(cudaMalloc((void **)&d_Offset, MaxPossibleClusters * sizeof(int)));
	////for loop E step
	//gpuErrchk(cudaMalloc((void **)&d_pIndex, nPoints * sizeof(int)));
	//gpuErrchk(cudaMalloc((void **)&d_points2Mean, nPoints*nDims * sizeof(float)));
	//gpuErrchk(cudaMalloc((void **)&d_InvCovDiag, nDims * sizeof(float)));
	//gpuErrchk(cudaMalloc((void **)&d_temp, nDims * sizeof(float)));
	//gpuErrchk(cudaMalloc((void **)&d_updatePointsList, nPoints * sizeof(int)));
	//
	gpuErrchk(cudaMalloc((void **)&d_ones, 2000 * sizeof(float)));
	init_dones << <(2000 + BLOCKDIM - 1) / BLOCKDIM, BLOCKDIM >> > (nDims, d_ones);
}

// Penalty for standard CEM
// Penalty(nAlive) returns the complexity penalty for that many clusters
// bearing in mind that cluster 0 has no free params except p.
float KK::Penalty(int n)
{
	int nParams;
	if (n == 1)
		return 0;
	nParams = (nDims*(nDims + 1) / 2 + nDims + 1)*(n - 1); // each has cov, mean, &p 

	float p = penaltyK*(float)(nParams) // AIC units (Spurious factor of 2 removed from AIC units on 09.07.13)
		+penaltyKLogN*((float)nParams*(float)log((float)nPoints) / 2); // BIC units
	return p;
}

//======================================ComputeClassPenalties================================================
__global__ void c_nnClassMembers(int nPoints, int *d_Class, int *d_nClassMembers) {
	int tidx = blockDim.x * blockIdx.x + threadIdx.x;
	if (tidx < nPoints) {
		atomicAdd(&d_nClassMembers[d_Class[tidx]], 1);
	}
}
__global__ void initClassPenalty(int MaxPossibleClusters, float *d_ClassPenalty) {
	int tidx = blockDim.x * blockIdx.x + threadIdx.x;
	if (tidx < MaxPossibleClusters) 
		d_ClassPenalty[tidx] = (float)0;
}
__global__ void updateClassPenalty(int nPoints, int *d_Class, float *d_UnMaskDims,float *d_ClassPenalty) {
	int tidx = blockDim.x * blockIdx.x + threadIdx.x;
	if (tidx < nPoints) {
		float n = d_UnMaskDims[tidx];
		atomicAdd(&d_ClassPenalty[d_Class[tidx]], (n*(n + 1) / 2 + n + 1));
	}
}
__global__ void computeClassPenalty(int MaxPossibleClusters, int nPoints,float penaltyK,float penaltyKLogN,
	int *d_nClassMembers, float *d_ClassPenalty) {
	int tidx = blockDim.x * blockIdx.x + threadIdx.x;
	if (tidx < MaxPossibleClusters) {
		if (d_nClassMembers[tidx]>0)
			d_ClassPenalty[tidx] /= (float)d_nClassMembers[tidx];
		float nParams = d_ClassPenalty[tidx];
		d_ClassPenalty[tidx] = penaltyK*(float)(nParams * 2)
			+ penaltyKLogN*((float)nParams*(float)log((float)nPoints) / 2);
	}
}
// Penalties for Masked CEM
void KK::ComputeClassPenalties()
{
	if (useCpu) {
		// Output("ComputeClassPenalties: Correct if UseDistributional only");
		for (int c = 0; c < MaxPossibleClusters; c++)
			ClassPenalty[c] = (float)0;
		// compute sum of nParams for each
		vector<int> NumberInClass(MaxPossibleClusters);
		for (int p = 0; p < nPoints; p++)
		{
			int c = Class[p];
			NumberInClass[c]++;
			//    int n = UnmaskedInd[p+1]-UnmaskedInd[p]; // num unmasked dimensions
			float n = UnMaskDims[p];
			float nParams = n*(n + 1) / 2 + n + 1;
			ClassPenalty[c] += nParams;
		}
		// compute mean nParams for each cluster
		for (int c = 0; c < MaxPossibleClusters; c++)
			if (NumberInClass[c] > 0)
				ClassPenalty[c] /= (float)NumberInClass[c];
		// compute penalty for each cluster
		for (int c = 0; c < MaxPossibleClusters; c++)
		{
			float nParams = ClassPenalty[c];
			ClassPenalty[c] = penaltyK*(float)(nParams * 2)
				+ penaltyKLogN*((float)nParams*(float)log((float)nPoints) / 2);
		}
	}
	//=======================================GPU code======================================
	else {
		initClassPenalty << <(MaxPossibleClusters+BLOCKDIM - 1)/ BLOCKDIM, BLOCKDIM >> > (MaxPossibleClusters, d_ClassPenalty);
		updateClassPenalty << <(nPoints+ BLOCKDIM - 1)/ BLOCKDIM, BLOCKDIM >> > (nPoints,d_Class,d_UnMaskDims,d_ClassPenalty);
		computeClassPenalty << <(MaxPossibleClusters + BLOCKDIM - 1) / BLOCKDIM, BLOCKDIM >> > (MaxPossibleClusters, nPoints, penaltyK, penaltyKLogN,
			d_nClassMembers,d_ClassPenalty);

		//Output("d_ClassPenalty.size: %d\n", d_ClassPenalty.size());
		//for (int i = 0; i < d_ClassPenalty.size(); i++) std::cout << d_ClassPenalty[i] << "  ";
		//Output("\n");
	}
	
}

//===========================================CStep==================================================
__global__ void d_cstep(int MaxPossibleClusters, int nPoints, bool allow_assign_to_noise, int  nClustersAlive, float HugeScore,
	int *d_OldClass, int *d_Class, int *d_Class2, int *d_AliveIndex, float *d_LogP) {
	int tid = blockDim.x * blockIdx.x + threadIdx.x;
	if (tid < nPoints)
	{
		d_OldClass[tid] = d_Class[tid];
		float BestScore = HugeScore;
		float SecondScore = HugeScore;
		float ThisScore;
		int TopClass = 0;
		int SecondClass = 0;

		int ccstart = 0, c;
		if (!allow_assign_to_noise)
			ccstart = 1;
		for (int cc = ccstart; cc<nClustersAlive; cc++)
		{
			c = d_AliveIndex[cc];
			ThisScore = d_LogP[tid*MaxPossibleClusters + c];
			if (ThisScore < BestScore)
			{
				SecondClass = TopClass;
				TopClass = c;
				SecondScore = BestScore;
				BestScore = ThisScore;
			}
			else if (ThisScore < SecondScore)
			{
				SecondClass = c;
				SecondScore = ThisScore;
			}
		}
		d_Class[tid] = TopClass;
		d_Class2[tid] = SecondClass;
	}
}

// Choose best class for each point (and second best) out of those living
void KK::CStep(bool allow_assign_to_noise)
{
	if (useCpu) {
		int p, c, cc, TopClass, SecondClass;
		int ccstart = 0;
		if (!allow_assign_to_noise)
			ccstart = 1;
		float ThisScore, BestScore, SecondScore;

		for (p = 0; p < nPoints; p++)
		{
			OldClass[p] = Class[p];
			BestScore = HugeScore;
			SecondScore = HugeScore;
			TopClass = SecondClass = 0;
			for (cc = ccstart; cc < nClustersAlive; cc++)
			{
				c = AliveIndex[cc];
				ThisScore = LogP[p*MaxPossibleClusters + c];
				if (ThisScore < BestScore)
				{
					SecondClass = TopClass;
					TopClass = c;
					SecondScore = BestScore;
					BestScore = ThisScore;
				}
				else if (ThisScore < SecondScore)
				{
					SecondClass = c;
					SecondScore = ThisScore;
				}
			}
			Class[p] = TopClass;
			Class2[p] = SecondClass;
		}
	}
	//=====================================GPUcode=======================================
	else {
		d_cstep << <(nPoints + BLOCKDIM - 1)/BLOCKDIM, BLOCKDIM >> > (MaxPossibleClusters, nPoints, allow_assign_to_noise, nClustersAlive, HugeScore,
			d_OldClass,d_Class,d_Class2,d_AliveIndex,d_LogP);

		/*
		Output("d_OldClass.size: %d\n", d_OldClass.size());
		for (int i = 0; i < d_OldClass.size(); i++) std::cout << d_OldClass[i] << "  ";
		Output("\n");
		Output("d_Class.size: %d\n", d_Class.size());
		for (int i = 0; i < d_Class.size(); i++) std::cout << d_Class[i] << "  ";
		Output("\n");
		Output("d_Class2.size: %d\n", d_Class2.size());
		for (int i = 0; i < d_Class2.size(); i++) std::cout << d_Class2[i] << "  ";
		Output("\n");
		*/
	}

}

//======================================ConsiderDeletion============================================
__global__ void initDeletionLoss(int MaxPossibleClusters, float HugeScore, int *d_ClassAlive,float *d_DeletionLoss) {
	int tidx = blockDim.x * blockIdx.x + threadIdx.x;
	if (tidx > 0 && tidx < MaxPossibleClusters) {
		if (d_ClassAlive[tidx]) d_DeletionLoss[tidx] = 0;
		else d_DeletionLoss[tidx] = HugeScore;
	}
}
__global__ void computeDeletionLoss(int nPoints, int MaxPossibleClusters,int *d_Class, int *d_Class2,float *d_LogP, float *d_DeletionLoss) {
	int tidx = blockDim.x * blockIdx.x + threadIdx.x;
	if (tidx < nPoints) {
		float s = d_LogP[tidx*MaxPossibleClusters + d_Class2[tidx]] - d_LogP[tidx*MaxPossibleClusters + d_Class[tidx]];
		atomicAdd(&d_DeletionLoss[d_Class[tidx]], s);
	}
}
__global__ void subtractionLoss(int MaxPossibleClusters, float HugeScore,float *d_ClassPenalty,float *d_DeletionLoss,float *d_tempSubtraction) {
	int tidx = blockDim.x * blockIdx.x + threadIdx.x;
	if (tidx < MaxPossibleClusters) {
		if (tidx == 0)d_tempSubtraction[tidx] = HugeScore;
		else d_tempSubtraction[tidx] = d_DeletionLoss[tidx] - d_ClassPenalty[tidx];
	}
}
__global__ void reallocatePoints(int nPoints, int CandidateClass, int* d_ClassAlive, int *d_Class, int *d_Class2){
	int tidx = blockDim.x * blockIdx.x + threadIdx.x;
	if (tidx < nPoints) {
		if (d_Class[tidx] == CandidateClass) 
			d_Class[tidx] = d_Class2[tidx];
		if(tidx == 0)
			d_ClassAlive[CandidateClass] = 0;
	}
}

struct KeyValue {
	int id;
	float v;
};

//<<<1,512>>>  so the array max length is 1024,single block;
__global__ void findMin(int n, float *d_s, KeyValue *d_result) {
	extern __shared__ KeyValue mixdata[];
	int tid = threadIdx.x;
	mixdata[tid].id = tid;
	mixdata[tid].v = 100000000.0;
	if (tid + blockDim.x < n) {
		if (d_s[tid] > d_s[tid + blockDim.x]) mixdata[tid].v = d_s[tid + blockDim.x], mixdata[tid].id = mixdata[tid + blockDim.x].id;
		else mixdata[tid].v = d_s[tid];
	}__syncthreads();

	if (tid < 256) {
		if (mixdata[tid].v > mixdata[tid + 256].v) mixdata[tid].v = mixdata[tid + 256].v, mixdata[tid].id = mixdata[tid + 256].id;
	} __syncthreads();

	if (tid < 128) {
		if (mixdata[tid].v > mixdata[tid + 128].v) mixdata[tid].v = mixdata[tid + 128].v, mixdata[tid].id = mixdata[tid + 128].id;
	} __syncthreads();

	if (tid <64) {
		if (mixdata[tid].v > mixdata[tid + 64].v) mixdata[tid].v = mixdata[tid + 64].v, mixdata[tid].id = mixdata[tid + 64].id;
	} __syncthreads();

	if (tid < 32) {
		if (mixdata[tid].v > mixdata[tid + 32].v) mixdata[tid].v = mixdata[tid + 32].v, mixdata[tid].id = mixdata[tid + 32].id;

		if (mixdata[tid].v > mixdata[tid + 16].v) mixdata[tid].v = mixdata[tid + 16].v, mixdata[tid].id = mixdata[tid + 16].id;

		if (mixdata[tid].v > mixdata[tid + 8].v) mixdata[tid].v = mixdata[tid + 8].v, mixdata[tid].id = mixdata[tid + 8].id;

		if (mixdata[tid].v > mixdata[tid + 4].v) mixdata[tid].v = mixdata[tid + 4].v, mixdata[tid].id = mixdata[tid + 4].id;

		if (mixdata[tid].v > mixdata[tid + 2].v) mixdata[tid].v = mixdata[tid + 2].v, mixdata[tid].id = mixdata[tid + 2].id;

		if (mixdata[tid].v > mixdata[tid + 1].v) mixdata[tid].v = mixdata[tid + 1].v, mixdata[tid].id = mixdata[tid + 1].id;

	}
	if (tid == 0) d_result[0].v = mixdata[0].v, d_result[0].id = mixdata[0].id;
}

KeyValue findResult(int n, float *d_s) {
	KeyValue *d_result;
	cudaMalloc((void **)&d_result, sizeof(KeyValue));
	findMin<<<1,512, 512 * sizeof(KeyValue) >>>(n, d_s, d_result);
	KeyValue h_result;
	cudaMemcpy(&h_result, d_result, sizeof(KeyValue), cudaMemcpyDeviceToHost);
	cudaFree(d_result);
	return h_result;
}
// Sometimes deleting a cluster will improve the score, when you take into account
// the BIC. This function sees if this is the case.  It will not delete more than
// one cluster at a time.
void KK::ConsiderDeletion()
{
	if(useCpu){
		int c, p, CandidateClass = 0;
		float Loss, DeltaPen;
		vector<float> DeletionLoss(MaxPossibleClusters); // the increase in log P by deleting the cluster
		if (Debug)
			Output(" Entering ConsiderDeletion: ");
		for (c = 1; c < MaxPossibleClusters; c++){
			if (ClassAlive[c]) DeletionLoss[c] = 0;
			else DeletionLoss[c] = HugeScore; // don't delete classes that are already there
		}
		// compute losses by deleting clusters
		vector<int> NumberInClass(MaxPossibleClusters);
		for (p = 0; p < nPoints; p++){
			DeletionLoss[Class[p]] += LogP[p*MaxPossibleClusters + Class2[p]] - LogP[p*MaxPossibleClusters + Class[p]];
			int ccc = Class[p];
			NumberInClass[ccc]++;  // For computing number of points in each class
		}

		// find class with smallest increase in total score
		Loss = HugeScore;
		if (UseDistributional) //For UseDistribution, we use the ClusterPenalty
		{
			for (c = 1; c < MaxPossibleClusters; c++){
				if ((DeletionLoss[c] - ClassPenalty[c]) < Loss){
					Loss = DeletionLoss[c] - ClassPenalty[c];
					CandidateClass = c;
				}
			}

		}// or in the case of fixed penalty find class with least to lose

		 // what is the change in penalty?
		if (UseDistributional) //For the distributional algorithm we need to use the ClusterPenalty
			DeltaPen = ClassPenalty[CandidateClass];

		//Output("cand Class %d would lose %f gain is %f\n", (int)CandidateClass, Loss, DeltaPen);
		// is it worth it?
		//06/12/12 fixing bug introduced which considered DeltaPen twice!
		if (UseDistributional) //For the distributional algorithm we need to use the ClusterPenalty
		{
			if (Loss < 0){
				Output("Deleting Class %d (%d points): Lose %f but Gain %f\n", (int)CandidateClass, (int)NumberInClass[CandidateClass], DeletionLoss[CandidateClass], DeltaPen);
				// set it to dead
				ClassAlive[CandidateClass] = 0;

				// re-allocate all of its points
				for (p = 0; p < nPoints; p++) if (Class[p] == CandidateClass) Class[p] = Class2[p];
				// recompute class penalties
				ComputeClassPenalties();
			}
		}
		Reindex();
	}
	
	//=============================================GPU code=======================================
	else {
		initDeletionLoss << <(MaxPossibleClusters + BLOCKDIM - 1)/BLOCKDIM, BLOCKDIM >> > (MaxPossibleClusters, HugeScore, d_ClassAlive,d_DeletionLoss);
		computeDeletionLoss << <(nPoints + BLOCKDIM - 1)/BLOCKDIM, BLOCKDIM >> > (nPoints, MaxPossibleClusters,d_Class,d_Class2,d_LogP,d_DeletionLoss);
		//compute minloss and index
		subtractionLoss << <MaxPossibleClusters/BLOCKDIM + 1, BLOCKDIM >> > (MaxPossibleClusters, HugeScore,
			d_ClassPenalty, d_DeletionLoss, d_tempSubtraction);
		KeyValue result = findResult(MaxPossibleClusters, d_tempSubtraction);

		//float minLoss = result.v;int CandidateClass = result.id;
		if (result.v < 0) {
			//Output("Deleting Class %d (%d points): Lose %f but Gain %f\n", (int)CandidateClass, (int)d_NumberInClass[CandidateClass], d_DeletionLoss[CandidateClass], d_ClassPenalty[CandidateClass]);
			reallocatePoints << <nPoints/BLOCKDIM + 1, BLOCKDIM >> > (nPoints, result.id,
				d_ClassAlive, d_Class, d_Class2);
			ComputeClassPenalties();
		}
		Reindex();

		cudaMemcpy(&AliveIndex[0], d_AliveIndex, MaxPossibleClusters*sizeof(int), cudaMemcpyDeviceToHost);
		cudaMemcpy(&nClassMembers[0], d_nClassMembers, MaxPossibleClusters * sizeof(int), cudaMemcpyDeviceToHost);
		
		/*
		Output("this is in the ConsiderDeletion:=======================\n");
		Output("d_AliveIndex:  %d\n", d_AliveIndex.size());
		for (int i = 0;i < d_AliveIndex.size();i++) cout << d_AliveIndex[i] << "  ";
		Output("\n");
		Output("d_ClassAlive:  %d\n", d_ClassAlive.size());
		for (int i = 0;i < d_ClassAlive.size();i++) cout << d_ClassAlive[i] << "  ";
		Output("\n");
		*/
	}
	
}


// LoadClu(CluFile)
void KK::LoadClu(char *CluFile)
{
	FILE *fp;
	int p, c;
	int val; // read in from %d
	int status;


	fp = fopen_safe(CluFile, "r");
	status = fscanf(fp, "%d", &nStartingClusters);
	nClustersAlive = nStartingClusters;// -1;
	for (c = 0; c<MaxPossibleClusters; c++) ClassAlive[c] = (c<nStartingClusters);

	for (p = 0; p<nPoints; p++)
	{
		status = fscanf(fp, "%d", &val);
		if (status == EOF) Error("Error reading cluster file");
		Class[p] = val - 1;
	}
}

// for each cluster, try to split it in two.  if that improves the score, do it.
// returns 1 if split was successful
int KK::TrySplits()
{
	int c, cc, c2, p, p2, DidSplit = 0;
	float Score, NewScore, UnsplitScore, SplitScore;
	int UnusedCluster;
	//KK K2; // second KK structure for sub-clustering
	//KK K3; // third one for comparison

	if (nClustersAlive >= MaxPossibleClusters - 1)
	{
		Output("Won't try splitting - already at maximum number of clusters\n");
		return 0;
	}

	// set up K3 and remember to add the masks
	//KK K3(*this);
	if (!AlwaysSplitBimodal)
	{
		if (KK_split == NULL)
		{
			KK_split = new KK(*this);
		}
		else
		{
			// We have to clear these to bypass the debugging checks
			// in precomputations.cpp
			KK_split->Unmasked.clear();
			KK_split->UnmaskedInd.clear();
			KK_split->SortedMaskChange.clear();
			KK_split->SortedIndices.clear();
			// now we treat it as empty
			KK_split->ConstructFrom(*this);
		}
	}
	//KK &K3 = *KK_split;
#define K3 (*KK_split)

	Output("Compute initial score before splitting: ");
	Score = ComputeScore();

	// loop the clusters, trying to split
	for (cc = 1; cc<nClustersAlive; cc++)
	{
		c = AliveIndex[cc];

		// set up K2 structure to contain points of this cluster only

		vector<int> SubsetIndices;
		for (p = 0; p<nPoints; p++)
			if (Class[p] == c)
				SubsetIndices.push_back(p);
		if (SubsetIndices.size() == 0)
			continue;

		if (K2_container)
		{
			// We have to clear these to bypass the debugging checks
			// in precomputations.cpp
			K2_container->Unmasked.clear();
			K2_container->UnmaskedInd.clear();
			K2_container->SortedMaskChange.clear();
			K2_container->SortedIndices.clear();
			//K2_container->AllVector2Mean.clear();
			// now we treat it as empty
			K2_container->ConstructFrom(*this, SubsetIndices);
		}
		else
		{
			K2_container = new KK(*this, SubsetIndices);
		}
		//KK K2(*this, SubsetIndices);
		KK &K2 = *K2_container;

		// find an unused cluster
		UnusedCluster = -1;
		for (c2 = 1; c2<MaxPossibleClusters; c2++){
			if (!ClassAlive[c2]){
				UnusedCluster = c2;
				break;
			}
		}
		if (UnusedCluster == -1)
		{
			Output("No free clusters, abandoning split");
			return DidSplit;
		}

		// do it
		if (Verbose >= 1) Output("\n Trying to split cluster %d (%d points) \n", (int)c, (int)K2.nPoints);
		K2.nStartingClusters = 2; // (2 = 1 clusters + 1 unused noise cluster)
		UnsplitScore = K2.CEM(NULL, 0, 1, false);
		K2.nStartingClusters = 3; // (3 = 2 clusters + 1 unused noise cluster)
		SplitScore = K2.CEM(NULL, 0, 1, false);

		// Fix by Michaël Zugaro: replace next line with following two lines
		// if(SplitScore<UnsplitScore) {
		if (K2.nClustersAlive<2) Output("\n Split failed - leaving alone\n");
		if ((SplitScore<UnsplitScore) && (K2.nClustersAlive >= 2)) {
			if (AlwaysSplitBimodal)
			{
				DidSplit = 1;
				Output("\n We are always splitting bimodal clusters so it's getting split into cluster %d.\n", (int)UnusedCluster);
				p2 = 0;
				for (p = 0; p < nPoints; p++)
				{
					if (Class[p] == c)
					{
						if (K2.Class[p2] == 1) Class[p] = c;
						else if (K2.Class[p2] == 2) Class[p] = UnusedCluster;
						else Error("split should only produce 2 clusters\n");
						p2++;
					}
					ClassAlive[Class[p]] = 1;
				}
			}
			else
			{
				// will splitting improve the score in the whole data set?

				// assign clusters to K3
				for (c2 = 0; c2 < MaxPossibleClusters; c2++) K3.ClassAlive[c2] = 0;
				//   Output("%d Points in class %d in KKobject K3 ", (int)c2, (int)K3.nClassMembers[c2]);
				p2 = 0;
				for (p = 0; p < nPoints; p++)
				{
					if (Class[p] == c)
					{
						if (K2.Class[p2] == 1) K3.Class[p] = c;
						else if (K2.Class[p2] == 2) K3.Class[p] = UnusedCluster;
						else Error("split should only produce 2 clusters\n");
						p2++;
					}
					else K3.Class[p] = Class[p];
					K3.ClassAlive[K3.Class[p]] = 1;
				}
				K3.Reindex();

				// compute scores

				K3.MEstep();
				//K3.MStep();
				//K3.EStep();

				//Output("About to compute K3 class penalties");
				if (UseDistributional) K3.ComputeClassPenalties(); //SNK Fixed bug: Need to compute the cluster penalty properly, cluster penalty is only used in UseDistributional mode
				NewScore = K3.ComputeScore();
				Output("\nSplitting cluster %d changes total score from %f to %f\n", (int)c, Score, NewScore);

				if (NewScore < Score)
				{
					DidSplit = 1;
					Output("\n So it's getting split into cluster %d.\n", (int)UnusedCluster);
					// so put clusters from K3 back into main KK struct (K1)
					for (c2 = 0; c2 < MaxPossibleClusters; c2++) ClassAlive[c2] = K3.ClassAlive[c2];
					for (p = 0; p < nPoints; p++) Class[p] = K3.Class[p];
				}
				else
				{
					Output("\n So it's not getting split.\n");
				}
			}
		}
	}
	return DidSplit;
#undef K3
}

//=========================================ComputeScore==============================================
__global__ void copyLogP(int nPoints, int MaxPossibleClusters, int *d_Class, float *d_LogP, float *d_tempLogP) {
	int tidx = blockDim.x * blockIdx.x + threadIdx.x;
	if (tidx < nPoints)
		d_tempLogP[tidx] = d_LogP[tidx * MaxPossibleClusters + d_Class[tidx]];
}
// ComputeScore() - computes total score.  Requires M, E, and C steps to have been run
float KK::ComputeScore()
{
	if (useCpu) {
		int p;
		// int debugadd;
		float penalty = (float)0;
		if (UseDistributional)  // For distributional algorithm we require the cluster penalty
			for (int c = 0; c < MaxPossibleClusters; c++)
				penalty += ClassPenalty[c];
		else
			penalty = Penalty(nClustersAlive);
		float Score = penalty;
		for (p = 0; p < nPoints; p++)
		{    //debugadd = LogP[p*MaxPossibleClusters + Class[p]];
			Score += LogP[p*MaxPossibleClusters + Class[p]];
			// Output("point %d: cumulative score %f adding%f\n", (int)p, Score, debugadd);
		}
		//Error("Score: %f Penalty: %f\n", Score, penalty);

		Output("  Score: Raw %f + Penalty %f = %f\n", Score - penalty, penalty, Score);

		if (Debug) {
			int c, cc;
			float tScore;
			for (cc = 0; cc < nClustersAlive; cc++) {
				c = AliveIndex[cc];
				tScore = 0;
				for (p = 0; p < nPoints; p++) if (Class[p] == c) tScore += LogP[p*MaxPossibleClusters + Class[p]];
				Output("class %d has subscore %f\n", c, tScore);
			}
		}
		return Score;
	}
	//====================================GPU code=========================
	else {
		float penalty;// = reduceFlt<128>(d_ClassPenalty, MaxPossibleClusters);
		cublasSasum(handle, MaxPossibleClusters, d_ClassPenalty, 1, &penalty);
		copyLogP << <(nPoints+BLOCKDIM - 1)/BLOCKDIM, BLOCKDIM >> > (nPoints, MaxPossibleClusters, d_Class,
			d_LogP,d_tempLogP);
		float Score;
		cublasSasum(handle, nPoints, d_tempLogP, 1, &Score);
		Score = -Score;
		//float Score = reduceFlt<128>(d_tempLogP, nPoints);
		Output("  Score: Raw %f + Penalty %f = %f\n", Score, penalty, Score + penalty);
		return Score + penalty;
	}
}


// Initialise starting conditions by selecting unique masks at random
void KK::StartingConditionsFromMasks()
{
	int nClusters2start = 0; //SNK To replace nStartingClusters within this variable only

							 //if (Debug)
							 //    Output("StartingConditionsFromMasks: ");
	Output("Starting initial clusters from distinct float masks \n ");

	if (nStartingClusters <= 1) // If only 1 starting clutser has been requested, assign all the points to cluster 0
	{
		for (int p = 0; p<nPoints; p++)
			Class[p] = 0;
	}
	else
	{
		int num_masks = 0;
		for (int p = 0; p<nPoints; p++)
			num_masks += (int)SortedMaskChange[p];

		if ((nStartingClusters - 1)>num_masks)
		{
			Error("Not enough masks (%d) to generate starting clusters (%d), "
				"so starting with (%d) clusters instead.\n", (int)num_masks,
				(int)nStartingClusters, (int)(num_masks + 1));
			nClusters2start = num_masks + 1;
			//return;
		}
		else
		{
			nClusters2start = nStartingClusters;
		}
		// Construct the set of all masks
		vector<bool> MaskUsed;
		vector<int> MaskIndex(nPoints);
		vector<int> MaskPointIndex;
		int current_mask_index = -1;
		for (int q = 0; q<nPoints; q++)
		{
			int p = SortedIndices[q];
			if (q == 0 || SortedMaskChange[p])
			{
				current_mask_index++;
				MaskUsed.push_back(false);
				MaskPointIndex.push_back(p);
			}
			MaskIndex[p] = current_mask_index;
		}
		// Select points at random until we have enough masks
		int masks_found = 0;
		vector<int> MaskIndexToUse;
		vector<int> FoundMaskIndex(num_masks);
		while (masks_found<nClusters2start - 1)
		{
			int p = irand(0, nPoints - 1);
			int mask_index = MaskIndex[p];
			if (!MaskUsed[mask_index])
			{
				MaskIndexToUse.push_back(mask_index);
				MaskUsed[mask_index] = true;
				FoundMaskIndex[mask_index] = masks_found;
				masks_found++;
			}
		}
		// Assign points to clusters based on masks
		for (int p = 0; p<nPoints; p++)
		{
			if (MaskUsed[MaskIndex[p]]) // we included this points mask
				Class[p] = FoundMaskIndex[MaskIndex[p]] + 1; // so assign class to mask index
			else // this points mask not included
			{
				// so find closest match
				int closest_index = 0;
				int distance = nDims + 1;
				vector<int> possibilities;
				for (int mi = 0; mi<nClusters2start - 1; mi++)
				{
					int mip = MaskPointIndex[MaskIndexToUse[mi]];
					// compute mask distance
					int curdistance = 0;
					for (int i = 0; i<nDims; i++)
						if (GetMasks(p*nDims + i) != GetMasks(mip*nDims + i))
							curdistance++;
					if (curdistance<distance)
					{
						possibilities.clear();
						distance = curdistance;
					}
					if (curdistance == distance)
						possibilities.push_back(mi);
				}
				if ((MaskStarts > 0) || AssignToFirstClosestMask)
					closest_index = possibilities[0];
				else
					closest_index = possibilities[irand(0, possibilities.size() - 1)];
				Class[p] = closest_index + 1;
			}
		}
		// print some info
		Output("Assigned %d initial classes from %d unique masks.\n",
			(int)nClusters2start, (int)num_masks);
		// Dump initial random classes to a file - knowledge of maskstart configuration may be useful
		// TODO: remove this for final version - SNK: actually it is a nice idea to keep this
		char fname[STRLEN];
		FILE *fp;
		sprintf(fname, "%s.initialclusters.%d.clu.%d", FileBase, (int)nClusters2start, (int)ElecNo);
		fp = fopen_safe(fname, "w");
		fprintf(fp, "%d\n", (int)nClusters2start);
		for (int p = 0; p<nPoints; p++)
			fprintf(fp, "%d\n", (int)Class[p]);
		fclose(fp);
	}
	for (int c = 0; c<MaxPossibleClusters; c++)
		ClassAlive[c] = (c<nClusters2start);

	cudaMemcpy(d_Class, &Class[0], nPoints*sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(d_ClassAlive, &ClassAlive[0], MaxPossibleClusters* sizeof(int), cudaMemcpyHostToDevice);
}

//======================================CEM step======================================================
__global__ void updataTempOldClass(int nPoints, int *d_Class, int *d_OldClass, float *d_tempOldClass){
	int tidx = blockDim.x * blockIdx.x + threadIdx.x;
	if (tidx < nPoints) {
		if (d_Class[tidx] != d_OldClass[tidx])
			d_tempOldClass[tidx] = 1.0;
		else d_tempOldClass[tidx] = 0.0;
	}
}
// CEM(StartFile) - Does a whole CEM algorithm from a random start or masked start
// whereby clusters are assigned according to the similarity of their masks
// optional start file loads this cluster file to start iteration
// if Recurse is 0, it will not try and split.
// if InitRand is 0, use cluster assignments already in structure
float KK::CEM(char *CluFile, int Recurse, int InitRand,
	bool allow_assign_to_noise)
{
	int p;
	int nChanged;
	int Iter;
	//vector<int> OldClass(nPoints);
	//thrust::device_vector<int> d_OldClass(nPoints);
	float Score, OldScore;
	int LastStepFull; // stores whether the last step was a full one
	int DidSplit = 0;

	if (Debug)
	{
		Output("Entering CEM \n");
	}

	int time1 = clock();

	if (CluFile && *CluFile)
		LoadClu(CluFile);
	else if (InitRand)
	{
		// initialize data to random
		if ((MaskStarts || UseMaskedInitialConditions) && (UseDistributional) && Recurse)
			StartingConditionsFromMasks();//2.5s
	}
	
	CopyHostToDevice();
	float preComputeTime = (clock() - time1) / (float)CLOCKS_PER_SEC;
	Output("==========================preComputeTime is : %f\n", preComputeTime);

	//int sizeI = sizeof(int);
	//int sizeF = sizeof(float);

	//gpuErrchk(cudaMemcpy(d_nClassMembers, &nClassMembers[0], MaxPossibleClusters*sizeI, cudaMemcpyHostToDevice));//==
	//gpuErrchk(cudaMemcpy(d_AliveIndex, &AliveIndex[0], MaxPossibleClusters*sizeI, cudaMemcpyHostToDevice));//==
	//gpuErrchk(cudaMemcpy(d_ClusterMask, &ClusterMask[0], MaxPossibleClusters*nDims*sizeF, cudaMemcpyHostToDevice));

	//gpuErrchk(cudaMemcpy(d_Mean, &Mean[0], MaxPossibleClusters*nDims*sizeF, cudaMemcpyHostToDevice));//==
	//gpuErrchk(cudaMemcpy(d_Weight, &Weight[0], MaxPossibleClusters*sizeF, cudaMemcpyHostToDevice));//==
	//gpuErrchk(cudaMemcpy(d_LogP, &LogP[0], MaxPossibleClusters*nPoints*sizeF, cudaMemcpyHostToDevice));//==
	//gpuErrchk(cudaMemcpy(d_OldClass, &OldClass[0], nPoints*sizeI, cudaMemcpyHostToDevice));//==
	//gpuErrchk(cudaMemcpy(d_ClassPenalty, &ClassPenalty[0], MaxPossibleClusters*sizeF, cudaMemcpyHostToDevice));//==
	//gpuErrchk(cudaMemcpy(d_Class2, &Class2[0], nPoints*sizeI, cudaMemcpyHostToDevice));//==
	//gpuErrchk(cudaMemcpy(d_BestClass, &BestClass[0], nPoints*sizeI, cudaMemcpyHostToDevice));//==


	// set all classes to alive
	Reindex();

	// main loop
	Iter = 0;
	FullStep = 1;
	Score = 0.0;
	do {
		Output("this is %d cycle...\n", Iter);
		//========================
		if (useCpu) for (p = 0; p < nPoints; p++) OldClass[p] = Class[p];
		else 
			cudaMemcpy(d_OldClass, d_Class, nPoints*sizeof(int), cudaMemcpyDeviceToDevice);
		//========================
		// M-step - calculate class weights, means, and covariance matrices for each class
		// E-step - calculate scores for each point to belong to each class

		int time3 = clock();
		MEstep();
		Output("==========================MEstep : %f\n", (clock() - time3) / (float)CLOCKS_PER_SEC);

		int time4 = clock();
		// C-step - choose best class for each
		CStep(allow_assign_to_noise);

		// Compute class penalties
		ComputeClassPenalties();

		// Would deleting any classes improve things?
		if (Recurse) ConsiderDeletion();

		//================================
		// Calculate number changed
		nChanged = 0;
		if(useCpu) for (p = 0; p < nPoints; p++) nChanged += (OldClass[p] != Class[p]);
		else {
			updataTempOldClass << < (nPoints+BLOCKDIM - 1)/ BLOCKDIM,BLOCKDIM>> > (nPoints,d_Class,d_OldClass,d_tempOldClass);
			float nchanged;
			cublasSasum(handle, nPoints,d_tempOldClass, 1, &nchanged);
			nChanged = (int)nchanged;//reduceInt<128>(d_tempOldClass, nPoints);
		}
		//===============================

		//Compute elapsed time
		timesofar = (clock() - Clock0) / (float)CLOCKS_PER_SEC;
		//Output("\nTime so far%f seconds.\n", timesofar);

		//Write start of Output to klg file
		if (Verbose >= 1)
		{
			if (Recurse == 0) Output("\t\tSP:");
			if ((Recurse != 0) || (SplitInfo == 1 && Recurse == 0))
				Output("Iteration %d%c (%f sec): %d clusters\n",
				(int)Iter, FullStep ? 'F' : 'Q', timesofar, (int)nClustersAlive);
		}

		// Calculate score
		OldScore = Score;
		Score = ComputeScore();

		Output("==========================other step : %f\n", (clock() - time4) / (float)CLOCKS_PER_SEC);
		int time5 = clock();

		//Finish Output to klg file with Score already returned via the ComputeScore() function
		if (Verbose >= 1)
		{
			Output("  nChanged %d\n", (int)nChanged);
		}

		//if(Verbose>=1)
		//{
		//    if(Recurse==0) Output("\t");
		//    Output(" Iteration %d%c: %d clusters Score %.7g nChanged %d\n",
		//        (int)Iter, FullStep ? 'F' : 'Q', (int)nClustersAlive, Score, (int)nChanged);
		//}

		Iter++;
		numiterations++;
		global_numiterations++;
		iteration_metric2 += (float)(nDims*nDims)*(float)(nPoints);
		iteration_metric3 += (float)(nDims*nDims)*(float)(nDims*nPoints);


		//if (Debug)
		//{
		//	for (p = 0; p<nPoints; p++) BestClass[p] = Class[p];
		//	SaveOutput();
		//	Output("Press return");
		//	getchar();
		//}

		// Next step a full step?
		LastStepFull = FullStep;
		FullStep = (
			nChanged > ChangedThresh*nPoints
			|| nChanged == 0
			|| Iter%FullStepEvery == 0
			|| Score > OldScore // SNK: Resurrected
								//SNK    Score decreases ARE because of quick steps!
			);
		if (Iter > MaxIter)
		{
			Output("Maximum iterations exceeded\n");
			break;
		}

		//Save a temporary clu file when not splitting
		if ((SaveTempCluEveryIter && Recurse) && (OldScore > Score))
		{

			//SaveTempOutput(); //SNK Saves a temporary Output clu file on each iteration
			Output("Writing temp clu file \n");
			Output("Because OldScore, %f, is greater than current (better) Score,%f  \n ", OldScore, Score);
		}

		// try splitting
		//int mod = (abs(Iter-SplitFirst))%SplitEvery;
		//Output("\n Iter mod SplitEvery = %d\n",(int)mod);
		//Output("Iter-SplitFirst %d \n",(int)(Iter-SplitFirst));

		//if ((Recurse && SplitEvery > 0) && (Iter == SplitFirst || (Iter >= SplitFirst + 1 && (Iter - SplitFirst) % SplitEvery == SplitEvery - 1) || (nChanged == 0 && LastStepFull)))
		//{
		//	if (OldScore > Score) //This should be trivially true for the first run of KlustaKwik
		//	{
		//		//SaveTempOutput(); //SNK Saves a temporary Output clu file before each split
		//		Output("Writing temp clu file \n");
		//		Output("Because OldScore, %f, is greater than current (better) Score,%f \n ", OldScore, Score);
		//	}
		//	DidSplit = TrySplits();
		//}
		//else DidSplit = 0;

		//Output("==========================trysplit : %f\n", (clock() - time5) / (float)CLOCKS_PER_SEC);

	} while (nChanged > 0 || !LastStepFull || DidSplit);
	if (DistDump) fprintf(Distfp, "\n");
	return Score;
}

// does the two-step clustering algorithm:
// first make a subset of the data, to SubPoints points
// then run CEM on this
// then use these clusters to do a CEM on the full data
// It calls CEM whenever there is no initialization clu file (i.e. the most common usage)
float KK::Cluster(char *StartCluFile = NULL)
{
	if (Debug)
	{
		Output("Entering Cluster \n");
	}



	if (Subset <= 1)
	{ // don't subset
		Output("------ Clustering full data set of %d points ------\n", (int)nPoints);
		return CEM(NULL, 1, 1);
	}

	//// otherwise run on a subset of points
	//int sPoints = nPoints / Subset; // number of subset points - int division will round down

	//vector<int> SubsetIndices(sPoints);
	//for (int i = 0; i<sPoints; i++)
	//	// choose point to include, evenly spaced plus a random offset
	//	SubsetIndices[i] = Subset*i + irand(0, Subset - 1);
	//KK KKSub = KK(*this, SubsetIndices);

	//// run CEM algorithm on KKSub
	//Output("------ Running on subset of %d points ------\n", (int)sPoints);
	//KKSub.CEM(NULL, 1, 1);

	//// now copy cluster shapes from KKSub to main KK
	//Weight = KKSub.Weight;
	//Mean = KKSub.Mean;
	//Cov = KKSub.Cov;
	//DynamicCov = KKSub.DynamicCov;
	//ClassAlive = KKSub.ClassAlive;
	//nClustersAlive = KKSub.nClustersAlive;
	//AliveIndex = KKSub.AliveIndex;

	//// Run E and C steps on full data set
	//Output("------ Evaluating fit on full set of %d points ------\n", (int)nPoints);
	//if (UseDistributional)
	//	ComputeClusterMasks(); // needed by E-step normally computed by M-step
	////EStep();
	//CStep();

	//// compute score on full data set and leave
	//return ComputeScore();
}

// Initialise by loading data from files
KK::KK(char *FileBase, int ElecNo, char *UseFeatures,
	float PenaltyK, float PenaltyKLogN, int PriorPoint)
{
	cublasCreate(&handle);
	cusolverDnCreate(&solver_handle);
	side = CUBLAS_SIDE_LEFT;
	uplo = CUBLAS_FILL_MODE_UPPER;
	trans = CUBLAS_OP_T;
	diag = CUBLAS_DIAG_NON_UNIT;

	KK_split = NULL;
	K2_container = NULL;
	penaltyK = PenaltyK;
	penaltyKLogN = PenaltyKLogN;
	LoadData(FileBase, ElecNo, UseFeatures);
	priorPoint = PriorPoint;

	//NOTE: penaltyK, penaltyKlogN, priorPoint, lower case versions of global variable PenaltyK PenaltyKLogN and PriorPoint

	DoInitialPrecomputations();//Now DoPrecomputations is only invoked in the initialization
	numiterations = 0;
	init_type = 0;
}

// This function is used by both of the constructors below, it initialises
// the data from a source KK object with a subset of the indices.
//used trysplit() --
void KK::ConstructFrom(const KK &Source, const vector<int> &Indices)
{
	KK_split = NULL;
	K2_container = NULL;
	nDims = Source.nDims;
	nDims2 = nDims*nDims;
	nPoints = Indices.size();
	penaltyK = Source.penaltyK;
	penaltyKLogN = Source.penaltyKLogN;
	priorPoint = Source.priorPoint;
	nStartingClusters = Source.nStartingClusters;
	NoisePoint = Source.NoisePoint;
	FullStep = Source.FullStep;
	nClustersAlive = Source.nClustersAlive;
	numiterations = Source.numiterations;

	//define cublas and cusolver handle
	handle = Source.handle;
	solver_handle = Source.solver_handle;
	side = Source.side;
	uplo = Source.uplo;
	trans = Source.trans;
	diag = Source.diag;

	AllocateArrays(); // ===========================Set storage for all the arrays such as Data, FloatMasks, Weight, Mean, Cov, etc.

	if (Debug)
	{
		Output("Entering ConstructFrom: \n");
	}

	// fill with a subset of points
	for (int p = 0; p<nPoints; p++)
	{
		int psource = Indices[p];
		//copy data and masks
		for (int d = 0; d<nDims; d++)
			Data[p*nDims + d] = Source.Data[psource*nDims + d];
		if (Source.Masks.size()>0)
		{
			for (int d = 0; d<nDims; d++)
				Masks[p*nDims + d] = Source.Masks[psource*nDims + d];
		}
		if (UseDistributional)
		{
			for (int d = 0; d<nDims; d++)
			{
				FloatMasks[p*nDims + d] = Source.FloatMasks[psource*nDims + d];
			}
		}

		UnMaskDims[p] = Source.UnMaskDims[psource];

	}



	//Output(" Printing Source.NoiseVariance[2] = %f",Source.NoiseVariance[2]);

	if (UseDistributional)
	{
		NoiseMean.resize(nDims);
		NoiseVariance.resize(nDims);
		nMasked.resize(nDims);


		for (int d = 0; d<nDims; d++)
		{
			NoiseMean[d] = Source.NoiseMean[d];
			NoiseVariance[d] = Source.NoiseVariance[d];
			nMasked[d] = Source.nMasked[d];

		}
	}

	DoPrecomputations();

	//Output(" Printing Source.NoiseMean[2] = %f",NoiseVariance[2]);

	numiterations = 0;
}

void KK::ConstructFrom(const KK &Source)
{
	vector<int> Indices(Source.nPoints);
	for (int i = 0; i<Source.nPoints; i++)
		Indices[i] = i;
	ConstructFrom(Source, Indices);
}

KK::KK(const KK &Source, const vector<int> &Indices)
{
	ConstructFrom(Source, Indices);
	init_type = 2;
}

// If we don't specify an index subset, use everything.
//invoke in the trysplit step
KK::KK(const KK &Source)
{
	ConstructFrom(Source);
	init_type = 1;
}

KK::~KK()
{
	if (KK_split) delete KK_split;
	KK_split = NULL;
	if (K2_container) delete K2_container;
	K2_container = NULL;

	//cublasDestroy(handle);
	//cusolverDnDestroy(solver_handle);
}

// Main loop
//int main(int argc, char **argv)
int main()
{
	float Score;
	float BestScore = HugeScore;
	int p, i;

	char fname[STRLEN];
	if (Log) {
		sprintf(fname, "%s.klg.%d", FileBase, (int)ElecNo);
		logfp = fopen_safe(fname, "w");
	}
	//SetupParams((int)argc, argv); // This function is defined in parameters.cpp

	//getchar();

	Output("Starting KlustaKwik. Version: %s\n", VERSION);
	//if (RamLimitGB == 0.0)
	//{
	//	RamLimitGB = (1.0*total_physical_memory()) / (1024.0*1024.0*1024.0);
	//	Output("Setting RAM limit to total physical memory, %.2f GB.\n", (double)RamLimitGB);
	//}
	//else if (RamLimitGB < 0.0)
	//{
	//	RamLimitGB = 1e20;
	//	Output("WARNING: You have chosen not to set a RAM limit, this may cause problems.\n");
	//}

	//clock_t Clock0 = clock();
	Clock0 = clock();
	clock_t etime = 0;
	clock_t mtime = 0;

#ifdef _OPENMP
	double start_time = omp_get_wtime();
#endif


	// The main KK object, loads the data and does some precomputations
	KK K1(FileBase, ElecNo, UseFeatures, PenaltyK, PenaltyKLogN, PriorPoint);

	int constructor = (clock() - Clock0) / (float)CLOCKS_PER_SEC;
	Output("Time taken for constructor:%f seconds.\n", constructor);

	if (UseDistributional && SaveSorted) //Bug fix (Classical KK would terminate here)
		K1.SaveSortedData();

	Output("\nFileBase  :  %s\n ----------------------------------------------------------------\n", FileBase);

	// Seed random number generator
	srand((unsigned int)RandomSeed);

	// open distance dump file if required
	if (DistDump) Distfp = fopen("DISTDUMP", "w");

	// start with provided file, if required
	if (*StartCluFile)
	{
		Output("\nStarting from cluster file %s\n", StartCluFile);

		float iterationtime = (float)clock();
		BestScore = K1.CEM(StartCluFile, 1, 1);  //Main computation
		iterationtime = (clock() - iterationtime) / (float)CLOCKS_PER_SEC;
		Output("Time taken for this iteration:%f seconds.\n", iterationtime);

		Output(" %d->%d Clusters: Score %f\n\n", (int)K1.nStartingClusters, (int)K1.nClustersAlive, BestScore);
		for (p = 0; p<K1.nPoints; p++)
			K1.BestClass[p] = K1.Class[p];
		K1.SaveOutput();
	}
	else
	{
		// loop through numbers of clusters ...
		K1.nStartingClusters = MaskStarts;

		// do CEM iteration
		Output("\nStarting from %d clusters...\n", (int)K1.nStartingClusters);
		float iterationtime = (float)clock();
		Score = K1.Cluster(); //Main computation
		iterationtime = (clock() - iterationtime) / (float)CLOCKS_PER_SEC;
		Output("Time taken for this iteration:%f seconds.\n", iterationtime);

		Output(" %d->%d Clusters: Score %f, best is %f\n", (int)K1.nStartingClusters, (int)K1.nClustersAlive, Score, BestScore);
		if (Score < BestScore)
		{
			Output("THE BEST YET!\n"); // New best classification found
			BestScore = Score;
			cudaMemcpy(&K1.BestClass[0], K1.d_Class, K1.nPoints * sizeof(int), cudaMemcpyDeviceToHost);
			//for (p = 0; p < K1.nPoints; p++)
				//K1.BestClass[p] = K1.Class[p];
			K1.SaveOutput();
		}
		Output("\n");
	}

	K1.SaveOutput();
	K1.FreeArray();
	cudaDeviceReset();

#ifdef _OPENMP
	float tottime = omp_get_wtime() - start_time;
#else
	float tottime = (clock() - Clock0) / (float)CLOCKS_PER_SEC;
#endif

	Output("E step: %d (time per iteration =%f ms)\n",
		(int)K1.numiterations,
		1e3*etime / (float)CLOCKS_PER_SEC / K1.numiterations);

	Output("M step: %d (time per iteration =%f ms)\n",
		(int)K1.numiterations,
		1e3*mtime / (float)CLOCKS_PER_SEC / K1.numiterations);

	Output("Main iterations: %d (time per iteration =%f ms)\n",
		(int)K1.numiterations,
		1e3*tottime / K1.numiterations);
	Output("Total iterations: %d (time per iteration =%f ms)\n",
		(int)global_numiterations,
		1e3*tottime / global_numiterations);
	Output("\nDef. Iteration metric 2:\nIteration_metric2 += (float)(nDims*nDims)*(float)(nPoints)\n");
	Output("Iterations metric 2: %f (time per metric unit =%fns)\n",
		iteration_metric2,
		1e9*tottime / iteration_metric2);
	Output("\nDef. Iteration metric 3:\nIteration_metric3 += (float)(nDims*nDims)*(float)(nDims*nPoints)\n");
	Output("Iterations metric 3: %f (time per metric unit=%fps)\n",
		iteration_metric3,
		1e12*tottime / iteration_metric3);
	Output("\nThat took %f seconds.\n", tottime);
	if (DistDump) fclose(Distfp);

	//Output("maxsize = %d\n", maxsize);
	//getchar();
	return 0;
}

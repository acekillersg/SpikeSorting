#include "klustakwik.h"
#include<algorithm>


#include "cuda_runtime.h"  
#include "device_launch_parameters.h"
#include "util.h"

#include <cusolverDn.h>
#include <cublas_v2.h>
#include <cuda_runtime_api.h>


using namespace std;

// Handles doing all the Initial precomputations once the data has been loaded into
// the class.
//invoked KK(char *FileBase, int ElecNo, char *UseFeatures,float PenaltyK, float PenaltyKLogN, int PriorPoint)
void KK::DoInitialPrecomputations()
{
	if (UseDistributional)
	{
		// Precompute the indices of the unmasked dimensions for each point
		//ComputeUnmasked();
		// Compute the order of points to consider that minimises the number of
		// times the mask changes
		ComputeSortIndices();
		// Now compute the points at which the mask changes in sorted order
		ComputeSortedUnmaskedChangePoints();
		// Compute the sum of the masks/float masks for each point (used for computing the cluster penalty)
		PointMaskDimension();
		// Precompute the noise means and variances
		ComputeNoiseMeansAndVariances();
		ComputeCorrectionTermsAndReplaceData();
	}
}

// Handles doing all the precomputations once the data has been loaded into
// the class. Note that this function has to be called after Data or Masks
// has changed. This is called by TrySplits() and Cluster()
// invoked ConstructFrom(const KK &Source, const vector<int> &Indices)
void KK::DoPrecomputations()
{
	if (UseDistributional)
	{
		// Precompute the indices of the unmasked dimensions for each point
		//ComputeUnmasked();
		// Compute the order of points to consider that minimises the number of
		// times the mask changes
		ComputeSortIndices();
		// Now compute the points at which the mask changes in sorted order
		ComputeSortedUnmaskedChangePoints();
		ComputeCorrectionTermsAndReplaceData();
	}
}

void KK::ComputeUnmasked()
{
	int i = 0;
	if (Unmasked.size() || UnmaskedInd.size())
	{
		Output("Precomputations have already been done, this indicates a bug.\n");
		Output("Error occurred in ComputeUnmasked().\n");
		abort();
	}
	for (int p = 0; p<nPoints; p++)
	{
		UnmaskedInd.push_back(i);
		for (int j = 0; j<nDims; j++)
		{
			if (GetMasks(p*nDims + j))
			{
				Unmasked.push_back(j);
				i++;
			}
		}
	}
	UnmaskedInd.push_back(i);
}

///////////////// SORTING /////////////////////////////////////////////////
// Comparison class, the operator()(i, j) function is used to provide the
// comparison i<j passed to stl::sort. Here i and j are the indices of two
// points, and i<j means that mask_i < mask_j in lexicographical order (we
// could change the ordering, as long as different masks are not considered
// equal).
class KKSort
{
public:
	KK *kk;
	KKSort(KK *kk) : kk(kk) {};
	__host__ __device__ bool operator()(const int i, const int j) const;

	//struct less_than_or_eq_zero
	//{
	//	__host__ __device__ bool operator() (double x) { return x <= 0.; }
	//};
};

// Less than operator for KK.Masks, it's just a lexicographical comparison
bool KKSort::operator()(const int i, const int j) const
{
	int nDims = kk->nDims;
	for (int k = 0; k<nDims; k++)
	{
		int x = kk->GetMasks(i*nDims + k);
		int y = kk->GetMasks(j*nDims + k);
		if (x<y) return true;
		if (x>y) return false;
	}
	return false;
}

/*
* This function computes the order in which indices should be considered in
* order to minimise the number of times the mask changes. We do this simply
* by creating an array SortedIndices=[0,1,2,...,nPoints-1] and then sorting
* this array where i<j if mask_i<mask_j in lexicographical order. The sorting
* is performed by stl::sort and the comparison function is provided by the
* KKSort class above.
*
* The optional force flag forces a recomputation of the sorted indices, which
* is necessary only in TrySplits(), but we should probably change this by
* refactoring.
*/
void KK::ComputeSortIndices()
{
	KKSort kksorter(this);
	if (SortedIndices.size())
	{
		Output("Precomputations have already been done, this indicates a bug.\n");
		Output("Error occurred in ComputeSortIndices().\n");
		abort();
	}
	SortedIndices.resize(nPoints);
	for (int i = 0; i<nPoints; i++)
		SortedIndices[i] = i;
	stable_sort(SortedIndices.begin(), SortedIndices.end(), kksorter);
}

// This function computes the points at which the mask changes if we iterate
// through the points in sorted order defined by ComputeSortIndices(). After
// the function is called, SortedMaskChange[SortedIndices[q]] is true if
// the mask for SortedIndices[q] is different from the mask for
// SortedIndices[q-1]
void KK::ComputeSortedUnmaskedChangePoints()
{
	if (SortedMaskChange.size()>0)
	{
		Output("Precomputations have already been done, this indicates a bug.\n");
		Output("Error occurred in ComputeSortedUnmaskedChangePoints().\n");
		abort();
	}
	SortedMaskChange.resize(nPoints);
	// The first point when we iterate through the points in sorted order is
	// SortedIndices[0] and we consider the mask as having 'changed' for this
	// first point, because we use the mask having changed to signal that
	// we should recompute the matrices that depend on the masks.
	SortedMaskChange[SortedIndices[0]] = true;

	int oldmask_offset = SortedIndices[0] * nDims;
	int numchanged = 0;
	for (int q = 1; q<nPoints; q++)
	{
		int p = SortedIndices[q];

		int newmask_offset = p*nDims;
		bool changed = false;
		for (int i = 0; i<nDims; i++)
		{
			if (GetMasks(newmask_offset + i) != GetMasks(oldmask_offset + i))
			{
				oldmask_offset = newmask_offset;
				changed = true;
				numchanged++;
				break;
			}
		}
		SortedMaskChange[p] = changed;
	}
}

//PointMaskDimension() computes the sum of the masks/float masks for each point 
void KK::PointMaskDimension()
{
	
	int i, p;
	for (p = 0; p < nPoints; p++)
	{
		UnMaskDims[p] = 0;
		for (i = 0; i < nDims; i++)
			UnMaskDims[p] += FloatMasks[p*nDims + i];
	}
	if (Debug) Output("UnmaskDims[%d] = %f ", (int)p, UnMaskDims[p]);
	
	//float alpha = 1.f;
	//float beta = 0.f;
	//cublasSgemv(handle, CUBLAS_OP_T, nDims, nPoints, &alpha, d_FloatMasks, nDims,
	//	d_ones, 1, &beta, d_UnMaskDims, 1);

}

//============================================================================================================================================
__global__ void init_NoiseMean(int nPoints, int nDims, float *d_FloatMasks, float *d_Data, float *d_tempNoiseMean,float *d_tempnMasked) {
	int tidx = blockDim.x * blockIdx.x + threadIdx.x;
	int tidy = blockDim.y * blockIdx.y + threadIdx.y;
	if (tidx < nDims && tidy < nPoints) {
		if (d_FloatMasks[tidy*nDims + tidx] < (float)1) {
			d_tempNoiseMean[tidy * nDims + tidx] = d_Data[tidy * nDims + tidx];
			d_tempnMasked[tidy * nDims + tidx] = 1.0;
		}
		else {
			d_tempNoiseMean[tidy * nDims + tidx] = 0.0;
			d_tempnMasked[tidy * nDims + tidx] = 0.0;
		}
	}
}
__global__ void init_NoiseVariance(int nPoints, int nDims, float *d_FloatMasks, float *d_Data, float *d_NoiseMean, float *d_tempNoiseVariance) {
	int tidx = blockDim.x * blockIdx.x + threadIdx.x;
	int tidy = blockDim.y * blockIdx.y + threadIdx.y;
	if (tidx < nDims && tidy < nPoints) {
		if (d_FloatMasks[tidy*nDims + tidx] < (float)1) {
			float x = d_Data[tidy * nDims + tidx] - d_NoiseMean[tidx];
			d_tempNoiseVariance[tidy * nDims + tidx] = x*x;
		}
		else
			d_tempNoiseVariance[tidy * nDims + tidx] = 0;
	}
}
__global__ void MeanValue(int n, float *d_Arrays, float *d_Nums) {
	int tidx = blockDim.x * blockIdx.x + threadIdx.x;
	if (tidx < n && d_Nums[tidx] > 0)
		d_Arrays[tidx] /= d_Nums[tidx];
}
void KK::ComputeNoiseMeansAndVariances()
{
	Output("Masked EM: Computing Noise Means and Variances \n ---------------------------------------------------");
	//float alpha = 1.0;
	//float beta = 0.0;
	//float *d_tempNoiseMean;
	//cudaMalloc((void **)&d_tempNoiseMean, nPoints*nDims * sizeof(float));
	//float *d_tempnMasked;
	//cudaMalloc((void **)&d_tempnMasked, nPoints*nDims * sizeof(float));
	//float *d_tempNoiseVariance;
	//cudaMalloc((void **)&d_tempNoiseVariance, nPoints*nDims * sizeof(float));
	//float *d_nMasked;
	//cudaMalloc((void **)&d_nMasked, nDims * sizeof(float));

	//dim3 block(32, 32);
	//dim3 grid((nDims + 31) / 32, (nPoints + 31) / 32);
	//init_NoiseMean << < grid,block>> > (nPoints, nDims, d_FloatMasks, d_Data, d_tempNoiseMean, d_tempnMasked);
	//cublasSgemv(handle, CUBLAS_OP_N, nDims, nPoints, &alpha, d_tempNoiseMean, nDims,
	//	d_ones, 1, &beta, d_NoiseMean, 1);
	//cublasSgemv(handle, CUBLAS_OP_N, nDims, nPoints, &alpha, d_tempnMasked, nDims,
	//	d_ones, 1, &beta, d_nMasked, 1);
	//MeanValue << <(nDims + 31) / 32, 32 >> > (nDims, d_NoiseMean, d_nMasked);
	//init_NoiseVariance<<<grid,block>>>(nPoints, nDims, d_FloatMasks, d_Data, d_NoiseMean, d_tempNoiseVariance);
	//cublasSgemv(handle, CUBLAS_OP_N, nDims, nPoints, &alpha, d_tempNoiseVariance, nDims,
	//	d_ones, 1, &beta, d_NoiseVariance, 1);
	//MeanValue << <(nDims + 31) / 32, 32 >> > (nDims, d_NoiseVariance, d_nMasked);

	///*vector<float> h_NoiseMean(nDims);
	//vector<float> h_NoiseVariance(nDims);
	//vector<float> h_nMasked(nDims);
	//cudaMemcpy(&h_NoiseMean[0], d_NoiseMean, nDims * sizeof(float), cudaMemcpyDeviceToHost);
	//cudaMemcpy(&h_NoiseVariance[0], d_NoiseVariance, nDims * sizeof(float), cudaMemcpyDeviceToHost);
	//cudaMemcpy(&h_nMasked[0], d_nMasked, nDims * sizeof(float), cudaMemcpyDeviceToHost);*/

	//cudaFree(d_tempNoiseMean);
	//cudaFree(d_tempnMasked);
	//cudaFree(d_tempNoiseVariance);
	//cudaFree(d_nMasked);
	
	
	NoiseMean.resize(nDims);
	NoiseVariance.resize(nDims);
	nMasked.resize(nDims);
	for (int p = 0; p<nPoints; p++)
		for (int i = 0; i<nDims; i++)
			if (!GetMasks(p*nDims + i))
			{
				float thisdata = GetData(p, i);
				NoiseMean[i] += thisdata;
				nMasked[i]++;
			}

	for (int i = 0; i<nDims; i++)
	{
		if (nMasked[i] == 0) {
			NoiseMean[i] = 0.0;
		}
		else
			NoiseMean[i] /= (float)nMasked[i];
	}

	for (int p = 0; p<nPoints; p++)
		for (int i = 0; i<nDims; i++)
			if (!GetMasks(p*nDims + i)) {
				float thisdata = GetData(p, i);
				NoiseVariance[i] += (thisdata - NoiseMean[i])*(thisdata - NoiseMean[i]);
			}

	for (int i = 0; i<nDims; i++)
	{
		if (nMasked[i] == 0)
			NoiseVariance[i] = 0;
		else
			NoiseVariance[i] /= (float)nMasked[i];
	}

	//cudaMemcpy(d_NoiseMean, &NoiseMean[0], nDims * sizeof(float), cudaMemcpyHostToDevice);
	//cudaMemcpy(d_NoiseVariance, &NoiseVariance[0], nDims * sizeof(float), cudaMemcpyHostToDevice);
	
}

__global__ void d_PointMaskDimension(int nPoints, int nDims, float *d_Data, float *d_FloatMasks,float *d_NoiseMean,
	float *d_NoiseVariance,float *d_CorrectionTerm) {
	int tidx = blockDim.x * blockIdx.x + threadIdx.x;
	int tidy = blockDim.y * blockIdx.y + threadIdx.y;
	if (tidx < nDims && tidy < nPoints) {
		int offset = tidy * nDims + tidx;
		float x = d_Data[offset];
		float w = d_FloatMasks[offset];
		float nu = d_NoiseMean[tidx];
		float sigma2 = d_NoiseVariance[tidx];
		float y = w * x + (1 - w) * nu;
		float z = w * x * x + (1 - w) * (nu * nu + sigma2);
		d_CorrectionTerm[offset] = z - y * y;
		d_Data[offset] = y;
	}
}

void KK::ComputeCorrectionTermsAndReplaceData()
{
	
	for (int p = 0; p<nPoints; p++)
		for (int i = 0; i<nDims; i++)
		{
			float x = GetData(p, i);
			float w = FloatMasks[p*nDims + i];
			float nu = NoiseMean[i];
			float sigma2 = NoiseVariance[i];
			float y = w*x + (1 - w)*nu;
			float z = w*x*x + (1 - w)*(nu*nu + sigma2);
			CorrectionTerm[p*nDims + i] = z - y*y;
			Data[p*nDims + i] = y;
		}
		
	//dim3 block(32, 32);
	//dim3 grid((nDims + 31) / 32, (nPoints + 31) / 32);
	//d_PointMaskDimension<<<grid,block>>>(nPoints, nDims, d_Data, d_FloatMasks, d_NoiseMean,
	//	d_NoiseVariance, d_CorrectionTerm);
	
}
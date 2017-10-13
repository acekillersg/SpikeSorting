#include "klustakwik.h"
#include<algorithm>

using namespace std;

// Handles doing all the Initial precomputations once the data has been loaded into
// the class.
void KK::DoInitialPrecomputations()
{
	if (UseDistributional)
	{
		// Precompute the indices of the unmasked dimensions for each point
		ComputeUnmasked();
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

void KK::DoPrecomputations()
{
	if (UseDistributional)
	{
		// Precompute the indices of the unmasked dimensions for each point
		ComputeUnmasked();
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
	bool operator()(const int i, const int j) const;
};

// Less than operator for KK.Masks, it's just a lexicographical comparison
bool KKSort::operator()(const int i, const int j) const
{
	int nDims = kk->nDims;
	for (int k = 0; k<nDims; k++)
	{
		char x = kk->GetMasks(i*nDims + k);
		char y = kk->GetMasks(j*nDims + k);
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

void KK::ComputeNoiseMeansAndVariances()
{
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
		if (nMasked[i] == 0){
			NoiseMean[i] = 0.0;
			NoiseVariance[i] = 0;
		}
		else
			NoiseMean[i] /= (float)nMasked[i];
	}

	for (int p = 0; p<nPoints; p++)
		for (int i = 0; i<nDims; i++)
			if (!GetMasks(p*nDims + i)){
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
}

//PointMaskDimension() computes the sum of the masks/float masks for each point 
void KK::PointMaskDimension()
{
	int i, p;

	for (p = 0; p<nPoints; p++)
	{
		UnMaskDims[p] = 0;
		for (i = 0; i<nDims; i++)
			UnMaskDims[p] += FloatMasks[p*nDims + i];
	}
}

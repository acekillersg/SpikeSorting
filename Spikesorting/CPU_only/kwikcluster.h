#pragma once
#pragma execution_character_set("utf-8")

#ifndef KWIKCLUSTER__H__
#define KWIKCLUSTER__H__

#include<iostream>
#include<vector>
#include"kwikparameters.h"

using namespace std;

class KK{
public:
	/*------------------------------VARIABLES --------------------------------------------------*/
	Params1 p1;
	float PointsForClusterMask,HugeScore;
	int MaxPossibleClusters;
	int MaskStarts, AssignToFirstClosestMask;
	//KK *KK_split, *K2_container; // used for splitting
	int nDims, nDims2; // nDims2 is nDims squared and the mean of the unmasked dimensions.
	int nStartingClusters; // total # starting clusters, including clu 0, the noise cluster (int because read in from file)
	int nClustersAlive; // nClustersAlive is total number with points in, excluding noise cluster
	int nPoints;
	int priorPoint; // Prior for regularization NOTE: separate from global variabl PriorPoint (capitalization)
	int NoisePoint; // number of fake points always in noise cluster to ensure noise weight>0
	int FullStep; // Indicates that the next E-step should be a full step (no time saving)
	float  penaltyK, penaltyKLogN;

	std::vector<vector<float> > Data; // Data[p*nDims + d] = Input data for point p, dimension d

	// We sort the points into an order where the corresponding mask changes as
	// infrequently as possible, this vector is used to store the sorted indices,
	// see the function KK::ComputeSortIndices() in sortdata.cpp to see how this
	// sorting works. In the case of the TrySplits() function this sorting needs
	// to be recomputed when we change the underlying data.
	vector<int> SortedIndices;

	vector<float> Masks; //SNK: Masks[p*nDims + d] = Input masks for point p, dimension d

	vector<float> FloatMasks; // as above but for floating point masks

	// We store just the indices of the unmasked points in this sparse array
	// structure. For point p, the segment Unmasked[UnmaskedInd[p]] to
	// Unmasked[UnmaskedInd[p+1]] contains the indices i where Masks[i]==1.
	// This allows for a nice contiguous piece of memory without a separate
	// memory allocation for each point (and there can be many points). The
	// precomputation is performed by the function ComputeUnmasked(), and is
	// automatically called by LoadData() at the appropriate time, although
	// in the case of TrySplits() it has to be called explicitly.
	vector<int> Unmasked;
	vector<int> UnmaskedInd;
	// We also store a bool that tells us if the mask has changed when we go
	// through the points in sorted order. Computed by function
	// ComputeSortedUnmaskedChangePoints()
	vector<int> SortedMaskChange;

	vector<float> UnMaskDims; //SNK: Number of unmasked dimensions for each data point. In the Float masks case, the sum of the weights.-------------------
	vector<int> nClassMembers;

	vector<float> Weight; // Weight[c] = Class weight for class c
	vector<vector<float> > Mean; // Mean[c*nDims + d] = cluster mean for cluster c in dimension d
	vector<vector<vector<float> > > Cov; // Cov[c*nDims*nDims + i*nDims + j] = Covariance for cluster C, entry i,j
	// NB covariances are stored in upper triangle (j>=i)
	vector<BlockPlusDiagonalMatrix> DynamicCov; // Covariance matrices, DynamicCov[cc] where cc is alive cluster index
	vector<float> LogP; // LogP[p*MaxClusters + c] = minus log likelihood for point p in cluster c
	vector<int> Class; // Class[p] = best cluster for point p
	vector<int> OldClass; // Class[p] = previous cluster for point p
	vector<int> Class2; // Class[p] = second best cluster for point p
	vector<int> BestClass; // BestClass = best classification yet achieved
	vector<int> ClassAlive; // contains 1 if the class is still alive - otherwise 0
	vector<int> AliveIndex; // a list of the alive classes to iterate over

	// Used for distributional optimisations
	vector<float> ClusterMask;
	vector< vector<int> > ClusterUnmaskedFeatures;
	vector< vector<int> > ClusterMaskedFeatures;

	// Used in EStep(), but this will probably change later
	vector<float> AllVector2Mean;
	// used in M step
	vector<float> NoiseMean;
	vector<float> NoiseVariance;
	vector<float> nMasked;
	// used in distribution EM steps
	vector<float> CorrectionTerm;
	// used in ComputeScore and ConsiderDeletion
	vector<float> ClassPenalty;

	/*-----------------------------------functions---------------------------------------------------*/
	KK(char *FileBase, int ElecNo, char *UseFeatures,
	float PenaltyK, float PenaltyKLogN, int PriorPoint);
	// Construct from subset of existing KK object
	void ConstructFrom(const KK &Source, const vector<int> &Indices);
	void ConstructFrom(const KK &Source);
	KK(const KK &Source, const vector<int> &Indices);
	// Make an entire copy of existing KK object
	KK(const KK &Source);
	// Destructor
	~KK();
	//=========================================================
	//void MemoryCheck();
	//void AllocateArrays();
	void Reindex();
	// Random initial starting conditions functions
	void StartingConditionsRandom();
	void StartingConditionsFromMasks();
	// Precomputation related functions
	void DoInitialPrecomputations();

	void PointMaskDimension();//SNK PointMaskDimension() computes the sum of the masks/float masks for each point
	// Precomputations for cluster masks
	void ComputeClusterMasks();
	// Score and penalty functions
	float ComputeScore();
	float Penalty(int n);
	void ComputeClassPenalties();
	// Main algorithm functions
	//void MStep();
	//void EStep();
	void CStep(bool allow_assign_to_noise = true);
	void ConsiderDeletion();
	//integer TrySplits();
};
#endif
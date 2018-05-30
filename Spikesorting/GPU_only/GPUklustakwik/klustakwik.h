#pragma once
/*
* Main header file
*/

#ifndef MASKED_KLUSTA_KWIK_2_H_
#define MASKED_KLUSTA_KWIK_2_H_

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <stdarg.h>
#include <iostream>
#include <fstream>
#include <vector>
#include <string.h>
#include "params.h"
#include "log.h"
#include "util.h"
//#include "numerics.h"
#include "linalg.h"
//#include "memorytracking.h"

#include "cuda_runtime.h"  
#include "device_launch_parameters.h"

#include <cublas_v2.h>
#include <cusolverDn.h>

using namespace std;

class KK {
public:
	/////////////// CONSTRUCTORS ///////////////////////////////////////////////
	// Construct from file
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
	/////////////// FUNCTIONS //////////////////////////////////////////////////
	//void MemoryCheck();
	void AllocateArrays();
	void Reindex();
	// Random initial starting conditions functions
	//void StartingConditionsRandom();
	void StartingConditionsFromMasks();
	// Precomputation related functions
	void DoInitialPrecomputations();
	// Functions called by DoPrecomputations
	void DoPrecomputations();
	//SNK Precomputation for TrySplits to avoid changing NoiseMean and NoiseVariance and nMasked
	void ComputeUnmasked();
	void ComputeSortIndices();
	void ComputeSortedUnmaskedChangePoints();
	void ComputeNoiseMeansAndVariances();
	void ComputeCorrectionTermsAndReplaceData();
	void PointMaskDimension();//SNK PointMaskDimension() computes the sum of the masks/float masks for each point
							  // Precomputations for cluster masks
	void ComputeClusterMasks();//===============================
	// Score and penalty functions
	float ComputeScore();
	float Penalty(int n);
	void ComputeClassPenalties();
	// Main algorithm functions
	void MStep();
	void EStep();
	void MEstep();
	void CStep(bool allow_assign_to_noise = true);
	void ConsiderDeletion();
	int TrySplits();//===============================
	float CEM(char *CluFile, int recurse, int InitRand, bool allow_assign_to_noise = true);
	float Cluster(char *CluFile);
	// IO related functions
	void LoadData(char *FileBase, int ElecNo, char *UseFeatures);
	void LoadClu(char *StartCluFile);
	void SaveOutput();
	void SaveTempOutput();
	void SaveSortedData();
	void SaveSortedClu();
	void SaveCovMeans();

	//====================================================
	void CopyHostToDevice();
	void FreeArray();
public:
	/////////////// VARIABLES //////////////////////////////////////////////////
	KK *KK_split, *K2_container; // used for splitting
	int nDims, nDims2; // nDims2 is nDims squared and the mean of the unmasked dimensions.
	int nStartingClusters; // total # starting clusters, including clu 0, the noise cluster (int because read in from file)
	int nClustersAlive; // nClustersAlive is total number with points in, excluding noise cluster
	int nPoints;
	int priorPoint; // Prior for regularization NOTE: separate from global variabl PriorPoint (capitalization)
	int NoisePoint; // number of fake points always in noise cluster to ensure noise weight>0
	int FullStep; // Indicates that the next E-step should be a full step (no time saving)
	float penaltyK, penaltyKLogN;

	vector<float> Data; // Data[p*nDims + d] = Input data for point p, dimension d

						// We sort the points into an order where the corresponding mask changes as
						// infrequently as possible, this vector is used to store the sorted indices,
						// see the function KK::ComputeSortIndices() in sortdata.cpp to see how this
						// sorting works. In the case of the TrySplits() function this sorting needs
						// to be recomputed when we change the underlying data.
	vector<int> SortedIndices;

	vector<int> Masks; //SNK: Masks[p*nDims + d] = Input masks for point p, dimension d

	inline int GetMasks(int i) { return Masks[i]; }

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

	vector<float> UnMaskDims; //SNK: Number of unmasked dimensions for each data point. In the Float masks case, the sum of the weights.
	vector<int> nClassMembers;

	vector<float> Weight; // Weight[c] = Class weight for class c
	vector<float> Mean; // Mean[c*nDims + d] = cluster mean for cluster c in dimension d
	vector<float> Cov; // Cov[c*nDims*nDims + i*nDims + j] = Covariance for cluster C, entry i,j
					   // NB covariances are stored in upper triangle (j>=i)
	vector<float> LogP; // LogP[p*MaxClusters + c] = minus log likelihood for point p in cluster c
	vector<int> Class; // Class[p] = best cluster for point p
	vector<int> OldClass; // Class[p] = previous cluster for point p
	vector<int> Class2; // Class[p] = second best cluster for point p
	vector<int> BestClass; // BestClass = best classification yet achieved
	vector<int> ClassAlive; // contains 1 if the class is still alive - otherwise 0
	vector<int> AliveIndex; // a list of the alive classes to iterate over

	vector<float> ClusterMask;
	// Used in EStep(), but this will probably change later
	vector<float> AllVector2Mean;
	// used in M step
	vector<float> NoiseMean;
	vector<float> NoiseVariance;
	vector<int> nMasked;
	vector<int> Offset;
	// used in distribution EM steps
	vector<float> CorrectionTerm;
	inline float GetData(int p, int i){return Data[p*nDims + i];};
	// used in ComputeScore and ConsiderDeletion
	vector<float> ClassPenalty;
	// debugging info
	int numiterations;
	int init_type;
	//=======================cuda library define====================================
	cublasHandle_t handle;
	cusolverDnHandle_t solver_handle;

	cublasSideMode_t side = CUBLAS_SIDE_LEFT;
	cublasFillMode_t uplo = CUBLAS_FILL_MODE_UPPER;
	cublasOperation_t trans = CUBLAS_OP_T;
	cublasDiagType_t diag = CUBLAS_DIAG_NON_UNIT;
	//=======================naive c define array===================================
	//constructor
	float* d_Data;
	//precompution
	float* d_CorrectionTerm;
	float* d_NoiseMean;
	float* d_NoiseVariance;
	float* d_FloatMasks;
	float* d_UnMaskDims;
	float* d_ClusterMask;
	float* d_Mean;
	int *d_Masks;

	int* d_AliveIndex;
	int* d_nClassMembers;

	int* d_Class;
	int* d_ClassAlive;
	float* d_ClassPenalty;
	float* d_Weight;
	float* d_LogP;
	int* d_OldClass;
	int* d_Class2;
	int* d_BestClass;

	//=======================temp variables in GPU kernel=============================
	//reindex
	int *d_ClassAliveTemp;
	//ConsiderDeletion
	float* d_DeletionLoss;
	float* d_tempSubtraction;
	//ComputeScore()
	float* d_tempLogP;
	//MECstep
	float *d_tempOldClass;
	//Mstep
	float *d_AllVector2Mean;
	int *d_Current;
	int *d_PointsInThisClass;
	int *d_MarkClass;
	int *d_Offset;

	float *d_cov;
	float *d_dig;
	float *d_X;

	//for loop E step
	float* d_unmaskedSolver;
	int *d_pIndex;
	float *d_points2Mean;
	float *d_InvCovDiag;
	float *d_temp;
	int *d_updatePointsList;
	//cublas Auxiliary array
	float *d_ones;

	float *d_solver;
	float *d_unMaskedPoints2Mean;
	float *d_choleskyDig;
	float *d_maskedPoints2Mean;
	float *d_subMahal;
	float *d_maskedSolver;
	float *d_subMahalSolver;


	int *devInfo;
	float *work;
	int *d_mark;
	float *d_unitMat;
	//scan kernel.
	//d_blocksum :grid1 = (nPoints + blocksize - 1) / blocksize * sizeof(int);
	//d_tt:       grid2 = (grid1 + blocksize - 1) / blocksize * sizeof(int);
	int *d_blocksum;
	int *d_tt;
	//===============================================
	
};

#endif /* MASKED_KLUSTA_KWIK_2_H_ */

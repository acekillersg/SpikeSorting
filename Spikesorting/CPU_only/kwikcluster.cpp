#pragma once
#pragma execution_character_set("utf-8")

#include"kwikcluster.h"
#include"kwikparameters.h"
#include<iostream>
#include<vector>
#include <cstdlib>
using namespace std;
//更新索引和有活力的簇
void KK::Reindex()
{
	int c;
	AliveIndex[0] = 0;
	nClustersAlive = 1;
	for (c = 1; c<p1.MaxPossibleClusters; c++)
	{
		if (ClassAlive[c])
		{
			AliveIndex[nClustersAlive] = c;
			nClustersAlive++;
		}
	}
}
//计算每个簇的惩罚
void KK::ComputeClassPenalties()
{
	// Output("ComputeClassPenalties: Correct if UseDistributional only");
	int MaxPossibleClusters = p1.MaxPossibleClusters;
	for (int c = 0; c<MaxPossibleClusters; c++)
		ClassPenalty[c] = 0.0;
	// compute sum of nParams for each
	vector<int> NumberInClass(MaxPossibleClusters);
	for (int p = 0; p<nPoints; p++)
	{
		int c = Class[p];
		NumberInClass[c]++;
		//    integer n = UnmaskedInd[p+1]-UnmaskedInd[p]; // num unmasked dimensions
		float n = UnMaskDims[p];
		float nParams = n*(n + 1) / 2 + n + 1;
		ClassPenalty[c] += nParams;
	}
	// compute mean nParams for each cluster
	for (int c = 0; c<MaxPossibleClusters; c++)
		if (NumberInClass[c]>0)
			ClassPenalty[c] /= (float)NumberInClass[c];
	// compute penalty for each cluster
	for (int c = 0; c<MaxPossibleClusters; c++)
	{
		float nParams = ClassPenalty[c];
		ClassPenalty[c] = penaltyK*(float)(nParams * 2)  // AIC units (Spurious factor of 2 removed from AIC units
			+ penaltyKLogN*((float)nParams*(float)log((float)nPoints) / 2);  // BIC units
	}
}

void KK::ComputeClusterMasks(){
	int MaxPossibleClusters = p1.MaxPossibleClusters;
	Reindex();
	// Initialise cluster mask to 0
	for (int i = 0; i<nDims*MaxPossibleClusters; i++)
		ClusterMask[i] = 0;

	// Compute cluster mask
	for (int p = 0; p < nPoints; p++){
		int c = Class[p];
		for (int i = 0; i < nDims; i++)
			ClusterMask[c*nDims + i] += FloatMasks[p*nDims + i];
	}
	// Compute the set of masked/unmasked features for each cluster

	// reset all the subvectors to empty
	ClusterUnmaskedFeatures.clear();
	ClusterUnmaskedFeatures.resize(MaxPossibleClusters);
	ClusterMaskedFeatures.clear();
	ClusterMaskedFeatures.resize(MaxPossibleClusters);
	// fill them in
	for (int cc = 0; cc<nClustersAlive; cc++)
	{
		//c为当前spike的所属簇编号
		int c = AliveIndex[cc];
		vector<int> &CurrentUnmasked = ClusterUnmaskedFeatures[c];
		vector<int> &CurrentMasked = ClusterMaskedFeatures[c];
		for (int i = 0; i < nDims; i++)
		{
			if (ClusterMask[c*nDims + i] >= PointsForClusterMask)//PointsForClusterMask为一个阈值
				CurrentUnmasked.push_back(i);
			else
				CurrentMasked.push_back(i);
		}
	}
}
//依据属于每个簇的概率去分类，找出最优簇和次优簇
void KK::CStep(bool allow_assign_to_noise)
{
	int MaxPossibleClusters = p1.MaxPossibleClusters;
	int p, c, cc, TopClass, SecondClass;
	int ccstart = 0;
	if (!allow_assign_to_noise)
			ccstart = 1;
	float ThisScore, BestScore, SecondScore;

	for (p = 0; p<nPoints; p++)
	{
		OldClass[p] = Class[p];
		BestScore = HugeScore;//HugeScore为初始score
		SecondScore = HugeScore;
		TopClass = SecondClass = 0;
		for (cc = ccstart; cc<nClustersAlive; cc++)
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

//根据惩罚和得分计算loss，从而确定是否删除该簇，同一时间只能删除一个簇
void KK::ConsiderDeletion()
{

	int c, p, CandidateClass = 0;
	float Loss, DeltaPen;
	vector<float> DeletionLoss(MaxPossibleClusters); // the increase in log P by deleting the cluster

	for (c = 1; c<MaxPossibleClusters; c++)
	{
		if (ClassAlive[c]) DeletionLoss[c] = 0;
		else DeletionLoss[c] = HugeScore; // don't delete classes that are already there
	}

	// compute losses by deleting clusters
	vector<int> NumberInClass(MaxPossibleClusters);
	for (p = 0; p<nPoints; p++)
	{
		DeletionLoss[Class[p]] += LogP[p*MaxPossibleClusters + Class2[p]] - LogP[p*MaxPossibleClusters + Class[p]];
		int ccc = Class[p];
		NumberInClass[ccc]++;  // For computing number of points in each class
	}

	// find class with smallest increase in total score
	Loss = HugeScore;

	for (c = 1; c<MaxPossibleClusters; c++)
	{
		if ((DeletionLoss[c] - ClassPenalty[c])<Loss)
		{
			Loss = DeletionLoss[c] - ClassPenalty[c];
			CandidateClass = c;
		}
	}

	// what is the change in penalty?
	DeltaPen = ClassPenalty[CandidateClass];

	if (Loss<0)
	{
		ClassAlive[CandidateClass] = 0;

		// re-allocate all of its points
		for (p = 0; p<nPoints; p++) if (Class[p] == CandidateClass) Class[p] = Class2[p];
		// recompute class penalties
		ComputeClassPenalties();
	}

	Reindex();
}

//通过penalty和log_p来计算score，其中penalty在ComputeClassPenalties中计算，log_p在E_step计算
float KK::ComputeScore()
{
	int p;
	float penalty =0.0;
	for (int c = 0; c<MaxPossibleClusters; c++)
		penalty += ClassPenalty[c];
	float Score = penalty;
	for (p = 0; p<nPoints; p++)
		Score += LogP[p*MaxPossibleClusters + Class[p]];
	return Score;
}

void KK::StartingConditionsFromMasks()
{
	int nClusters2start = 0; //SNK To replace nStartingClusters within this variable only

	if (nStartingClusters <= 1) // If only 1 starting clutser has been requested, assign all the points to cluster 0
	{
		for (int p = 0; p<nPoints; p++)
			Class[p] = 0;
	}
	else
	{
		int num_masks = 0;
		for (int p = 0; p<nPoints; p++)
			num_masks += (int)SortedMaskChange[p];//完全不同的spike（masks不相同）的个数

		if ((nStartingClusters - 1)>num_masks)
		{
			cout << "Not enough masks (" << num_masks << ") to generate starting clusters " << nStartingClusters << "so starting with " << num_masks + 1 << ") clusters instead.\n"<< endl;
			nClusters2start = num_masks + 1;
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
			int p = rand() % (nPoints - 1);
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
						if (Masks[p, i] != Masks[mip, i];
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
					closest_index = possibilities[rand()%(possibilities.size() - 1)];
				Class[p] = closest_index + 1;
			}
		}
	}
	for (int c = 0; c<MaxPossibleClusters; c++)
		ClassAlive[c] = (c<nClusters2start);
}
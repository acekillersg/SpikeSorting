#include <cublas_v2.h>

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/generate.h>
#include <thrust/reduce.h>
#include <thrust/functional.h>
#include <thrust/random.h>
#include <thrust/sequence.h>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <iostream>

__global__ void d_cstep(int MaxPossibleClusters, int nPoints, int allow_assign_to_noise, int  nClustersAlive, int HugeScore,
	                    int *d_OldClass, int *d_Class, int *d_Class2, int *d_AliveIndex, int *d_LogP) {
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
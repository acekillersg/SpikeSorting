#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <stdlib.h>
#define N 15
//-----------------------------------------klustakwik.py�����ݴ����в���-----------------------------------------------------------------------------------//
////�������Spike��ÿ�����������ֵ����Сֵ�����ֵ
//__global__ void MaxMinValue(float *featuresmax, float *featuresmin, float *masks, int *shape)
//{
//	int bdx = blockDim.x;
//	int bdy = blockDim.y;
//	int bidx = blockIdx.x;
//	int bidy = blockIdx.y;
//	int tidx = threadIdx.x;
//	int tidy = threadIdx.y;
//
//	int idy = (blockIdx.x * blockDim.x) + threadIdx.x;
//	int idx = (blockIdx.y * blockDim.y) + threadIdx.y;
//	int nums = shape[0] * shape[1];
//
//	if (idx < (shape[0] + 1) / 2 && idy < shape[1]){
//		int x = (shape[0] + 1) / 2;
//		int x_max = shape[0];
//		while (x != 0){
//			if (idx < x && idx + x < x_max){
//				if (featuresmax[idy + idx * shape[1]] < featuresmax[idy + (idx + x)*shape[1]])
//					featuresmax[idy + idx * shape[1]] = featuresmax[idy + (idx + x)*shape[1]];
//				if (featuresmin[idy + idx * shape[1]] > featuresmin[idy + (idx + x)*shape[1]])
//					featuresmin[idy + idx * shape[1]] = featuresmin[idy + (idx + x)*shape[1]];
//				//printf("features[%d] %f  :  features[%d] %f \n ", idy + idx * shape[1], features[idy + idx * shape[1]], idy + (idx + x)*shape[1], features[idy + (idx + x)*shape[1]]);
//			}
//			__syncthreads();
//			x_max = x;
//			x = (x + 1) / 2;
//		}
//	}
//	__syncthreads();
//}
////����ÿ�����������ֵ����Сֵ�Ĳ�ֵ�����ڹ�һ��features
//__global__ void vdiff(float *Vmax, float *Vmin, int *shape){
//	int tid = threadIdx.x;
//	if (tid < shape[1]){
//		Vmax[tid] = Vmax[tid] - Vmin[tid];
//		if (Vmax[tid] == 0) vmax[tid] = 1;
//	}
//}
////�������Spike��ÿ�����������ֵ����Сֵ�����ֵ
__global__ void MaxMinV(float *features, float *maxValue, float *minValue, float *diff, int L, int H)
{
	int idx = (blockIdx.x * blockDim.x) + threadIdx.x;

	if (idx < L){
		maxValue[idx] = 0.0;
		minValue[idx] = 999999.0;
	}
	__syncthreads();
	if (idx < 2 * L){
		if (idx < L){
			for (int i = 0; i < H; i++)
				if (maxValue[idx] < features[idx + i * L])
					maxValue[idx] = features[idx + i * L];
			__syncthreads();
		}
		else{
			for (int i = 0; i < H; i++)
				if (minValue[idx - L] > features[idx - L + i * L])
					minValue[idx - L] = features[idx - L + i * L];
			__syncthreads();
		}
	}
	if (idx < L)
		diff[idx] = maxValue[idx] - minValue[idx];
}
//���²�����
__global__ void preSparedata(float *features, float *masks, int *shape, float *diff, float *minValue, int sum_inds, int *inds, float *fetsum, float *fet2sum, int *nsum){
	int bdx = blockDim.x;
	int bdy = blockDim.y;
	int bidx = blockIdx.x;
	int bidy = blockIdx.y;
	int tidx = threadIdx.x;
	int tidy = threadIdx.y;

	int idy = (blockIdx.x * blockDim.x) + threadIdx.x;
	int idx = (blockIdx.y * blockDim.y) + threadIdx.y;

	int sum_inds = 0;
	if (idx < shape[0] && idy < shape[1]){
		//��ÿ��spike���������й�һ��
		features[idy + idx * shape[1]] = (features[idy + idx * shape[1]] - minValue[idy]) / diff[idy];
		//ͳ������δ���ڱε���������Ŀ��ÿ��spikeδ���ڱε���������Ŀ
		if (masks[idy + idx * shape[1]] > 0){
			atomicAdd(&sum_inds, 1);
			atomicAdd(&inds[idy], 1);
		}
		//ͨ���ڱ���������fetsum��fet2sum��nsum,�������ڼ��������ľ�ֵ�ͷ���
		else{
			float t = features[idy + idx * shape[1]]
				atomicAdd(&fetsum[idy], t);
			atomicAdd(&fet2sum[idy], t*t);
			atomicAdd(&nsum[idy], 1);
		}
	}
	__syncthreads();
}
//���������ľ�ֵ�ͷ���
__global__ void noise_arg(float *fetsum, float *fet2sum, int *nsum, int *shape, float *noise_mean, float *noise_variance){
	int tid = threadsIdx.x;
	if (tid < shape[1]){
		if (nsum[tid] == 0) nsum[tid] == 1;
		noise_mean[tid] = fetsum[tid] / nsum[tid];
		noise_variance[tid] = fet2sum[tid] / nsum[tid] - noise_mean[tid] * noise_mean[tid];
	}
	__syncthreads();
}
//����spike��offsets������inds��ǰ׺��
__global__ void prescan(float *indsOffsets, float *inds, int *shape)
{
	int n = shape[0];
	extern __shared__ float temp[];// allocated on invocation
	int thid = threadIdx.x;
	int offset = 1;
	temp[2 * thid] = inds[2 * thid]; // load input into shared memory
	temp[2 * thid + 1] = inds[2 * thid + 1];
	for (int d = n >> 1; d > 0; d >>= 1) // build sum in place up the tree
	{
		__syncthreads();
		if (thid < d)
		{
			int ai = offset*(2 * thid + 1) - 1;
			int bi = offset*(2 * thid + 2) - 1;
			temp[bi] += temp[ai];
		}
		offset *= 2;
	}
	if (thid == 0) { temp[n - 1] = 0; } // clear the last element
	for (int d = 1; d < n; d *= 2) // traverse down tree & build scan
	{
		offset >>= 1;
		__syncthreads();
		if (thid < d)
		{
			int ai = offset*(2 * thid + 1) - 1;
			int bi = offset*(2 * thid + 2) - 1;
			float t = temp[ai];
			temp[ai] = temp[bi];
			temp[bi] += t;
		}
	}
	__syncthreads();
	indsOffsets[2 * thid] = temp[2 * thid]; // write results to device memory
	indsOffsets[2 * thid + 1] = temp[2 * thid + 1];
}
//����δ���ڱεĲ���all_features, all_masks, all_unmasked
__global__ void unmasked_arg(float *features, float *masks, int *shape, int *inds, int *indsOffsets, float *all_features, float *all_masks, int *all_unmasked)
{
	int tid = threadsIdx.x;
	if (tid < shape[0]){
		//��ΪҪ��֤ÿ��Spike��unmasked��˳���ԣ����Բ���forѭ������
		int cnt = 1;
		for (int i = 0; i < shape[1]; i++)
		{
			if (mask[tid* shape[0] + i] > 0){
				all_features[indsOffsets[tid] - cnt] = features[tid* shape[0] + i];
				all_masks[indsOffsets[tid] - cnt] = masks[tid* shape[0] + i];
				all_unmasked[indsOffsets[tid] - cnt] = i;
			}
			cnt += 1;
		}
	}
}
//-------------------------------------------------------------------------------------------------------------------//
//-----------------------------------------data.py ����Ԥ������-------------------------------------------------//
_global__ void compute_correction_terms_and_replace_data(float *all_features, float *all_masks, int *all_unmasked, float *noise_mean, float *noise_variance, int sum_inds, float *fet, float *correction_terms)
{
	int tid = threadsIdx.x;
	if (tid < sum_inds)
	{
		//��δ���ڱεĲ��ε����� �����滻����
		fet[tid] = all_masks[tid] * all_features[tid] + (1 - all_masks[tid]) * noise_mean[tid];
		//�������ϵ��
		correction_terms[tid] = all_masks[tid] * all_features[tid] * all_features[tid] + (1 - all_masks[tid]) * (noise_mean[tid] * noise_mean[tid] + noise_variance[tid]) - fet[tid] * fet[tid];
	}
}

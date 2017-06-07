#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
/*
waveform.cu:包含的函数主要是对应SpikeDetect部分的waveform的一些操作
函数作用如下：
comps_wave()：对于detect部分提取到的components，从变换后的波形data_t中提取对应的wave
normalize()：对于波形中的电位值，通过高阈值ts和低阈值tw进行归一化，方便之后计算masks和尖峰的中心时间
compute_masks():对于每一个提取到的wave，计算其masks的值
*/
/*******************************************************copy the components to the wave**************************************************************/
__global__ void comps_wave(int **wave, int s_min, int s_max, int *flit_ary, size_t num)
{
	int tidx = blockIdx.x * blockDim.x + threadIdx.x;
	int tidy = blockIdx.y * blockDim.y + threadIdx.y;
	if (tidx < s_max - s_min && tidy < 32)
	{
		wave[tidx][tidy] = flit_ary[(tidx + s_min) * 32 + tidy];
	}
}
/****************************************************normalize操作*************************************************************/
__global__ void normalize(float *nor_ary, float *flit_ary,float tw,float ts, size_t N)
{
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	if (tid < N)
	{
		if (flit_ary[tid] >= ts) nor_ary[tid] = 1;
		else if (nor_ary[tid] < tw) nor_ary[tid] = 0;
		else nor_ary[tid] = (flit_ary[tid] - tw) / (ts - tw);
	}
}
/****************************************************compute_masks操作*************************************************************/
__global__ void compute_masks(float **wave, float *mask_bin, float *masks, float tw, float ts, size_t num)
{
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	if (tid < 32) masks[tid] = wave[0][tid];

	if (tid < num/32)
	for (int i = 0; i < 32; i++)
	{
		if (wave[tid][i] > masks[tid])
			masks[tid] = wave[tid][i];
	}
	__syncthreads();
	if (tid < 32)
	{
		if (mask_bin[tid] == 0)
			masks[tid] = 0;
	}
}
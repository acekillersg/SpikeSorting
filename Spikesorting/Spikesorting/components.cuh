#include <stdio.h>
#include <stdlib.h>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#define Num 1000
//表示一个坐标
struct point{
	int x, y;
};

/*
mark数组：表示所有超过高阈值点的坐标
crossing数组：表示对于每个（s,ch）点处的阈值状态，超过高阈值为2，低于低阈值为0，其余为1
sample数组：表示每个component的sample的边界
chanMax:表示每一个componment的每个channel的最大值，用于计算masks，shape = （n，32）
adjlabel：表示与当前component相邻的component的id
data_t：表示原始电位数据
n：表示mark数组的长度，即初始有多少个点超过高阈值
N：表示一共的点的个数（Sample*channel）
*/
__global__ void components(struct point *mark, int *crossing, struct point *sample, float chanMax[][32], int adjlabel[][Num], float * data_t, size_t n, size_t N)
{
	extern __shared__ struct point queue[];
	int tid = threadIdx.x + blockDim.x * blockIdx.x;
	
	if (tid < n)
	{
		atomicAdd(&crossing[mark[tid].x * 32 + mark[tid].y], tid + 3);

		//sample记录每个component的时间的上界和下界，用于之后从原始数据提取波形
		sample[tid].x = mark[tid].x;
		sample[tid].y = mark[tid].y;
		atomicMax(&chanMax[tid][mark[tid].y], data_t[mark[tid].x * 32 + mark[tid].y] );

		//用数组queue和lptr与rptr模拟队列进行BFS
		queue[0] = mark[tid];
		int lptr = 0, rptr = 1,num = 0;
		while (lptr < rptr){
			int x = queue[lptr].x;
			int y = queue[lptr].y;

			//左边的第二个坐标
			if (y - 2 >= 0 && crossing[x * 32 + y - 2] >= 1){
				if (crossing[x * 32 + y - 2] == 1){
					atomicAdd(&crossing[x * 32 + y - 2], tid + 4);
					queue[rptr].x = x;
					queue[rptr].y = y - 2;
					rptr += 1;
					//更新当前component中x channel的最大电位值
					atomicMax(&chanMax[tid][y - 2], data_t[x * 32 + y - 2]);
				}
				//记录与当前component邻接的component的id
				else{
					adjlabel[tid][num] = crossing[x * 32 + y - 2];
				    num += 1;
				}
			}
			//左边的第一个坐标
			if (y - 1 >= 0 && crossing[x * 32 + y - 1] >= 1){
				if (crossing[x * 32 + y - 1] == 1){
					atomicAdd(&crossing[x * 32 + y - 1], tid + 4);
				    queue[rptr].x = x;
				    queue[rptr].y = y - 1;
				    rptr += 1;
				    //更新当前component中x channel的最大电位值
				    atomicMax(&chanMax[tid][y-1], data_t[x * 32 + y - 1]);
				}
				else{
					adjlabel[tid][num] = crossing[x * 32 + y - 1];
					num += 1;
				}
				
			}
			//上边的坐标
			if (x-1 >= 0 && crossing[(x-1) * 32 + y] >= 1){
				if (crossing[(x - 1) * 32 + y] == 1){
					atomicAdd(&crossing[(x-1) * 32 + y], tid + 4);
				    queue[rptr].x = x-1;
				    queue[rptr].y = y;
				    rptr += 1;
				    //更新当前component中上边界
				    atomicMin(&sample[tid].x, x - 1);
				    //更新当前component中x channel的最大电位值
				    atomicMax(&chanMax[tid][y], data_t[ (x-1) * 32 + y]);
				}
				else{
					adjlabel[tid][num] = crossing[(x-1) * 32 + y];
					num += 1;
				}
			}
			//右边的第一个坐标
			if (y + 1 < 32 && crossing[x * 32 + y + 1] >= 1){
				if (crossing[x * 32 + y + 1] == 1){
					atomicAdd(&crossing[x * 32 + y +1], tid + 4);
				    queue[rptr].x = x;
				    queue[rptr].y = y + 1;
				    rptr += 1;
				    //更新当前component中x channel的最大电位值
				    atomicMax(&chanMax[tid][y+1], data_t[x * 32 + y +1]);
				}
				else{
					adjlabel[tid][num] = crossing[x * 32 + y + 1];
					num += 1;
				}
			}
			//右边的第二个坐标
			if (y + 2 < 32 && crossing[x * 32 + y + 2] >= 1){
				if (crossing[x * 32 + y + 2] >= 1){
					atomicAdd(&crossing[x * 32 + y + 2], tid + 4);
				    queue[rptr].x = x;
				    queue[rptr].y = y + 2;
				    rptr += 1;
				    //更新当前component中x channel的最大电位值
				    atomicMax(&chanMax[tid][y + 2], data_t[x * 32 + y + 2]);
				}
				else{
					adjlabel[tid][num] = crossing[x * 32 + y + 2];
					num += 1;
				}
			}
			//上边的坐标
			if (x + 1 <  N/32 && crossing[(x + 1) * 32 + y] >= 1){
				if (crossing[(x + 1) * 32 + y] == 1){
					atomicAdd(&crossing[(x + 1) * 32 + y], tid + 4);
				    queue[rptr].x = x + 1;
				    queue[rptr].y = y;
				    rptr += 1;
				    //更新当前component中下边界
				    atomicMax(&sample[tid].y, x + 1);
				    //更新当前component中x channel的最大电位值
				    atomicMax(&chanMax[tid][y], data_t[(x+1) * 32 + y]);
				}
				else{
					adjlabel[tid][num] = crossing[(x + 1) * 32 + y];
					num += 1;
				}
			}
			lptr += 1;
		}
	}
	__syncthreads();
	
	//......
}
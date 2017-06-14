#include <stdio.h>
#include <stdlib.h>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#define Num 1000
//��ʾһ������
struct point{
	int x, y;
};

/*
mark���飺��ʾ���г�������ֵ�������
crossing���飺��ʾ����ÿ����s,ch���㴦����ֵ״̬����������ֵΪ2�����ڵ���ֵΪ0������Ϊ1
sample���飺��ʾÿ��component��sample�ı߽�
chanMax:��ʾÿһ��componment��ÿ��channel�����ֵ�����ڼ���masks��shape = ��n��32��
adjlabel����ʾ�뵱ǰcomponent���ڵ�component��id
data_t����ʾԭʼ��λ����
n����ʾmark����ĳ��ȣ�����ʼ�ж��ٸ��㳬������ֵ
N����ʾһ���ĵ�ĸ�����Sample*channel��
*/
__global__ void components(struct point *mark, int *crossing, struct point *sample, float chanMax[][32], int adjlabel[][Num], float * data_t, size_t n, size_t N)
{
	extern __shared__ struct point queue[];
	int tid = threadIdx.x + blockDim.x * blockIdx.x;
	
	if (tid < n)
	{
		atomicAdd(&crossing[mark[tid].x * 32 + mark[tid].y], tid + 3);

		//sample��¼ÿ��component��ʱ����Ͻ���½磬����֮���ԭʼ������ȡ����
		sample[tid].x = mark[tid].x;
		sample[tid].y = mark[tid].y;
		atomicMax(&chanMax[tid][mark[tid].y], data_t[mark[tid].x * 32 + mark[tid].y] );

		//������queue��lptr��rptrģ����н���BFS
		queue[0] = mark[tid];
		int lptr = 0, rptr = 1,num = 0;
		while (lptr < rptr){
			int x = queue[lptr].x;
			int y = queue[lptr].y;

			//��ߵĵڶ�������
			if (y - 2 >= 0 && crossing[x * 32 + y - 2] >= 1){
				if (crossing[x * 32 + y - 2] == 1){
					atomicAdd(&crossing[x * 32 + y - 2], tid + 4);
					queue[rptr].x = x;
					queue[rptr].y = y - 2;
					rptr += 1;
					//���µ�ǰcomponent��x channel������λֵ
					atomicMax(&chanMax[tid][y - 2], data_t[x * 32 + y - 2]);
				}
				//��¼�뵱ǰcomponent�ڽӵ�component��id
				else{
					adjlabel[tid][num] = crossing[x * 32 + y - 2];
				    num += 1;
				}
			}
			//��ߵĵ�һ������
			if (y - 1 >= 0 && crossing[x * 32 + y - 1] >= 1){
				if (crossing[x * 32 + y - 1] == 1){
					atomicAdd(&crossing[x * 32 + y - 1], tid + 4);
				    queue[rptr].x = x;
				    queue[rptr].y = y - 1;
				    rptr += 1;
				    //���µ�ǰcomponent��x channel������λֵ
				    atomicMax(&chanMax[tid][y-1], data_t[x * 32 + y - 1]);
				}
				else{
					adjlabel[tid][num] = crossing[x * 32 + y - 1];
					num += 1;
				}
				
			}
			//�ϱߵ�����
			if (x-1 >= 0 && crossing[(x-1) * 32 + y] >= 1){
				if (crossing[(x - 1) * 32 + y] == 1){
					atomicAdd(&crossing[(x-1) * 32 + y], tid + 4);
				    queue[rptr].x = x-1;
				    queue[rptr].y = y;
				    rptr += 1;
				    //���µ�ǰcomponent���ϱ߽�
				    atomicMin(&sample[tid].x, x - 1);
				    //���µ�ǰcomponent��x channel������λֵ
				    atomicMax(&chanMax[tid][y], data_t[ (x-1) * 32 + y]);
				}
				else{
					adjlabel[tid][num] = crossing[(x-1) * 32 + y];
					num += 1;
				}
			}
			//�ұߵĵ�һ������
			if (y + 1 < 32 && crossing[x * 32 + y + 1] >= 1){
				if (crossing[x * 32 + y + 1] == 1){
					atomicAdd(&crossing[x * 32 + y +1], tid + 4);
				    queue[rptr].x = x;
				    queue[rptr].y = y + 1;
				    rptr += 1;
				    //���µ�ǰcomponent��x channel������λֵ
				    atomicMax(&chanMax[tid][y+1], data_t[x * 32 + y +1]);
				}
				else{
					adjlabel[tid][num] = crossing[x * 32 + y + 1];
					num += 1;
				}
			}
			//�ұߵĵڶ�������
			if (y + 2 < 32 && crossing[x * 32 + y + 2] >= 1){
				if (crossing[x * 32 + y + 2] >= 1){
					atomicAdd(&crossing[x * 32 + y + 2], tid + 4);
				    queue[rptr].x = x;
				    queue[rptr].y = y + 2;
				    rptr += 1;
				    //���µ�ǰcomponent��x channel������λֵ
				    atomicMax(&chanMax[tid][y + 2], data_t[x * 32 + y + 2]);
				}
				else{
					adjlabel[tid][num] = crossing[x * 32 + y + 2];
					num += 1;
				}
			}
			//�ϱߵ�����
			if (x + 1 <  N/32 && crossing[(x + 1) * 32 + y] >= 1){
				if (crossing[(x + 1) * 32 + y] == 1){
					atomicAdd(&crossing[(x + 1) * 32 + y], tid + 4);
				    queue[rptr].x = x + 1;
				    queue[rptr].y = y;
				    rptr += 1;
				    //���µ�ǰcomponent���±߽�
				    atomicMax(&sample[tid].y, x + 1);
				    //���µ�ǰcomponent��x channel������λֵ
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
#define _CRT_SECURE_NO_WARNINGS
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

//#include "parameters.h"

#define CHANNEL  32
#define SAMPLE 4000
#define NUMS CHANNEL*SAMPLE

extern "C"
void mixGPU(int *crossing, float *ary, int *mark, size_t N);

int main()
{
	/***************************���ļ�************************************/
	FILE *fp;
	float traces_f[NUMS];
	fp = fopen("traces_f.txt", "rt");
	if (fp == NULL)
	{ 
		printf("cannot open file");
		getchar();
	}
	for (int i = 0; i < NUMS; i++)
		fscanf(fp, "%f", &traces_f[i]);
	fclose(fp);
	/*****************************GPU����*******************************************/
	int mark[2];
	mark[0] = 1; mark[1] = 1;
	int crossing[NUMS];
	memset(crossing, 0, sizeof(crossing));

	mixGPU(crossing,traces_f, mark,NUMS);
	/*****************************����*******************************************/
	getchar();
	return 0;
}
